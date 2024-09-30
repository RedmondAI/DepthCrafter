import gc
import os
import numpy as np
import torch
import argparse
from diffusers.training_utils import set_seed
import time
from typing import List
import shutil
import subprocess

from depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
from depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
from depthcrafter.utils import (
    vis_sequence_depth,
    read_image_sequence,
    save_png_sequence,
    read_video_frames,
    get_frame_length,
)

def create_quicktime(outdir):
    # Define the command
    command = [
        'ffmpeg', 
        '-framerate', '24',  # Set frame rate to 24fps
        '-pattern_type', 'glob', 
        '-i', os.path.join(outdir, '*.png'),  # Input files
        '-c:v', 'libx264',  # Video codec
        '-pix_fmt', 'yuv420p',  # Pixel format
        '-crf', '12',  # Lower CRF for higher quality (lower is better, 18 is visually lossless)
        os.path.join(outdir, 'depth.mp4')  # Output file
    ]

    # Execute the command
    subprocess.run(command, check=True)
    print("QuickTime video created:", os.path.join(outdir, 'sbs_high_quality.mp4'))

class DepthCrafterDemo:
    def __init__(
        self,
        unet_path: str,
        pre_train_path: str,
        cpu_offload: str = "model",
    ):
        unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
            unet_path,
            # subfolder="unet",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        # Load weights of other components from the provided checkpoint
        self.pipe = DepthCrafterPipeline.from_pretrained(
            pre_train_path,
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
        )

        # For saving memory, we can offload the model to CPU or run the model sequentially to save more memory
        if cpu_offload is not None:
            if cpu_offload == "sequential":
                # This will be slower but save more memory
                self.pipe.enable_sequential_cpu_offload()
            elif cpu_offload == "model":
                self.pipe.enable_model_cpu_offload()
            else:
                raise ValueError(f"Unknown CPU offload option: {cpu_offload}")
        else:
            self.pipe.to("cuda")

        # Enable attention slicing
        self.pipe.enable_attention_slicing()

    def infer(
        self,
        input_path: str,
        num_denoising_steps: int,
        guidance_scale: float,
        save_folder: str = "./demo_output",
        window_size: int = 110,
        overlap: int = 25,
        max_res: int = 1024,
        target_fps: int = 15,
        seed: int = 42,
        track_time: bool = True,
        save_npz: bool = False,
        input_type: str = "video",
        original_sizes: List[tuple] = None,
        start_frame: int = 0,
        end_frame: int = 999999999,
        gain: float = 1.0,
        denoise: bool = False,  # Add this line
    ):
        set_seed(seed)

        if input_type == "video":
            frames, target_fps = read_video_frames(
                input_path, target_fps, max_res, start_frame, end_frame
            )
        elif input_type == "image_sequence":
            frames, original_sizes = read_image_sequence(
                input_path, max_res, start_frame, end_frame, gain, denoise  # Add denoise here
            )
        else:
            raise ValueError(f"Unknown input type: {input_type}")

        print("Frame length: ", len(frames))

        # Inference the depth map using the DepthCrafter pipeline
        with torch.inference_mode():
            res = self.pipe(
                frames,
                height=frames.shape[1],
                width=frames.shape[2],
                output_type="np",
                guidance_scale=guidance_scale,
                num_inference_steps=num_denoising_steps,
                window_size=window_size,
                overlap=overlap,
                track_time=track_time,
            ).frames[0]
        # Convert the three-channel output to a single-channel depth map
        res = res.sum(-1) / res.shape[-1]
        # Normalize the depth map to [0, 1] across the whole sequence
        res = (res - res.min()) / (res.max() - res.min())
        # Visualize the depth map and save the results
        vis = vis_sequence_depth(res)
        # Save the depth map and visualization as 16-bit PNGs
        save_path = os.path.join(save_folder, os.path.basename(input_path))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_png_sequence(res, save_path + "_depth", original_sizes, dtype=np.float16)
        return [
            save_path + "_input",
        ]


if __name__ == "__main__":
    # Running configs
    parser = argparse.ArgumentParser(description="DepthCrafter")
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to the input video file(s) or image sequence directory",
    )
    parser.add_argument(
        "--input-type",
        type=str,
        default="video",
        choices=["video", "image_sequence"],
        help="Type of input: 'video' or 'image_sequence'",
    )
    parser.add_argument(
        "--save-folder",
        type=str,
        default="./demo_output",
        help="Folder to save the output",
    )
    parser.add_argument(
        "--unet-path",
        type=str,
        default="tencent/DepthCrafter",
        help="Path to the UNet model",
    )
    parser.add_argument(
        "--pre-train-path",
        type=str,
        default="stabilityai/stable-video-diffusion-img2vid-xt",
        help="Path to the pre-trained model",
    )
    parser.add_argument(
        "--cpu-offload",
        type=str,
        default="model",
        choices=["model", "sequential", None],
        help="CPU offload option",
    )
    parser.add_argument("--target-fps", type=int, default=15, help="Target FPS for the output")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--num-inference-steps", type=int, default=25, help="Number of inference steps"
    )
    parser.add_argument("--guidance-scale", type=float, default=1.2, help="Guidance scale")
    parser.add_argument("--window-size", type=int, default=110, help="Window size")
    parser.add_argument("--overlap", type=int, default=25, help="Overlap size")
    parser.add_argument("--max-res", type=int, default=1024, help="Maximum resolution")
    parser.add_argument("--save_npz", type=bool, default=True, help="Save npz file")
    parser.add_argument("--track_time", type=bool, default=False, help="Track time")
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Starting frame index (inclusive)",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=999999999,
        help="Ending frame index (inclusive)",
    )
    parser.add_argument(
        "--create-quicktime",
        action="store_true",
        help="Create a QuickTime video from the output frames",
    )
    parser.add_argument("--gain", type=float, default=1.0, help="Gain to apply to input images (default: 1.0)")
    parser.add_argument("--denoise", action="store_true", help="Apply denoising to each frame")

    args = parser.parse_args()

    if os.path.exists(args.save_folder):
        shutil.rmtree(args.save_folder)
    os.makedirs(args.save_folder, exist_ok=True)

    start_time = time.time()
    depthcrafter_demo = DepthCrafterDemo(
        unet_path=args.unet_path,
        pre_train_path=args.pre_train_path,
        cpu_offload=args.cpu_offload,
    )

    # Process each input (can be multiple paths separated by commas)
    input_paths = args.input_path.split(",")
    for input_path in input_paths:
        depthcrafter_demo.infer(
            input_path,
            args.num_inference_steps,
            args.guidance_scale,
            save_folder=args.save_folder,
            window_size=args.window_size,
            overlap=args.overlap,
            max_res=args.max_res,
            target_fps=args.target_fps,
            seed=args.seed,
            track_time=args.track_time,
            input_type=args.input_type,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            gain=args.gain,
            denoise=args.denoise,  # Add this line
        )
        # Clear the cache for the next input
        gc.collect()
        torch.cuda.empty_cache()

    end_time = time.time()
    try:
        frame_length = get_frame_length(input_path)
    except:
        frame_length = 1

    gc.collect()
    torch.cuda.empty_cache()

    if args.create_quicktime:
        create_quicktime(args.save_folder)
    # The rest of your script (e.g., video creation with ffmpeg) remains unchanged