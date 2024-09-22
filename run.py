import gc
import os
import numpy as np
import torch
import argparse
from diffusers.training_utils import set_seed
import time
from typing import List
import shutil

from depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
from depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
from depthcrafter.utils import vis_sequence_depth, read_image_sequence, save_png_sequence, read_image_sequence_frames


def read_video_frames(input_path, process_length, target_fps, max_res):
    # Placeholder implementation for reading video frames
    # You need to replace this with actual video reading logic
    import cv2

    cap = cv2.VideoCapture(input_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    frames = np.array(frames)
    return frames, target_fps


def get_frame_length(input_path: str) -> int:
    """
    Quickly count the number of image files in the input directory.
    """
    supported_formats = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    return len([fname for fname in os.listdir(input_path) if fname.lower().endswith(supported_formats)])


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
        # load weights of other components from the provided checkpoint
        self.pipe = DepthCrafterPipeline.from_pretrained(
            pre_train_path,
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
        )

        # for saving memory, we can offload the model to CPU, or even run the model sequentially to save more memory
        if cpu_offload is not None:
            if cpu_offload == "sequential":
                # This will slow, but save more memory
                self.pipe.enable_sequential_cpu_offload()
            elif cpu_offload == "model":
                self.pipe.enable_model_cpu_offload()
            else:
                raise ValueError(f"Unknown cpu offload option: {cpu_offload}")
        else:
            self.pipe.to("cuda")

        # enable attention slicing and xformers memory efficient attention
        # try:
        #     self.pipe.enable_xformers_memory_efficient_attention()
        # except Exception as e:
        #     print(e)
        #     print("Xformers is not enabled")

        self.pipe.enable_attention_slicing()

    def infer(
        self,
        input_path: str,
        num_denoising_steps: int,
        guidance_scale: float,
        save_folder: str = "./demo_output",
        window_size: int = 110,
        process_length: int = None,
        overlap: int = 25,
        max_res: int = 1024,
        target_fps: int = 15,
        seed: int = 42,
        track_time: bool = True,
        save_npz: bool = False,
        input_type: str = "video",
        original_sizes: List[tuple] = None,
    ):
        set_seed(seed)

        if input_type == "video":
            frames, target_fps = read_video_frames(input_path, process_length, target_fps, max_res)
        elif input_type == "image_sequence":
            frames, original_sizes = read_image_sequence(input_path, max_res)
        else:
            raise ValueError(f"Unknown input type: {input_type}")

        print("frame length: ", len(frames))
        # Determine process_length if not provided
        if process_length is None:
            process_length = len(frames)

        print(f"==> Input path: {input_path}, frames shape: {frames.shape}")

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
        # Convert the three-channel output to a single channel depth map
        res = res.sum(-1) / res.shape[-1]
        # Normalize the depth map to [0, 1] across the whole sequence
        res = (res - res.min()) / (res.max() - res.min())
        # Visualize the depth map and save the results
        vis = vis_sequence_depth(res)
        # Save the depth map and visualization as 16-bit PNGs
        save_path = os.path.join(save_folder, os.path.basename(input_path))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_png_sequence(res, save_path + "_depth", original_sizes, dtype=np.float16)
        # save_png_sequence(vis, save_path + "_vis", original_sizes, dtype=np.float16)
        # save_png_sequence(frames, save_path + "_input", original_sizes, dtype=np.float16)
        return [
            save_path + "_input",
        ]

    def run(
        self,
        input_folder,
        num_denoising_steps,
        guidance_scale,
        max_res=1024,
    ):
        frames, original_sizes = read_image_sequence(input_folder, max_res)  # Capture original sizes
        process_length = len(frames)
        res_path = self.infer(
            input_folder,
            num_denoising_steps,
            guidance_scale,
            process_length=None,
            original_sizes=original_sizes,
        )
        gc.collect()
        return res_path[:2]


if __name__ == "__main__":
    # Running configs
    parser = argparse.ArgumentParser(description="DepthCrafter")
    parser.add_argument(
        "--input-path", type=str, required=True, help="Path to the input video file(s) or image sequence directory"
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
    parser.add_argument("--num-inference-steps", type=int, default=25, help="Number of inference steps")
    parser.add_argument("--guidance-scale", type=float, default=1.2, help="Guidance scale")
    parser.add_argument("--window-size", type=int, default=110, help="Window size")
    parser.add_argument("--overlap", type=int, default=25, help="Overlap size")
    parser.add_argument("--max-res", type=int, default=1024, help="Maximum resolution")
    parser.add_argument("--save_npz", type=bool, default=True, help="Save npz file")
    parser.add_argument("--track_time", type=bool, default=False, help="Track time")

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
            process_length=None,  # Pass None to infer method
            overlap=args.overlap,
            max_res=args.max_res,
            target_fps=args.target_fps,
            seed=args.seed,
            track_time=args.track_time,
            save_npz=args.save_npz,
            input_type=args.input_type,
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

    # Empty the output directory before running inference

    # Create high-quality mp4 from sorted PNGs
    import subprocess
    from glob import glob

    # Get sorted list of PNGs
    png_files = sorted(glob(os.path.join(args.save_folder, "*.png")))

    # Write the list to a temporary file for ffmpeg
    list_file = os.path.join(args.save_folder, "file_list.txt")
    with open(list_file, "w") as f:
        for png in png_files:
            f.write(f"file '{os.path.abspath(png)}'\n")

    # Generate MP4 using ffmpeg with the sorted PNGs
    output_mp4 = os.path.join(args.save_folder, "output.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-r",
            "24",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_file,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            output_mp4,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    print(f"Time taken: {end_time - start_time} seconds")
    print("Time per frame: ", (end_time - start_time) / frame_length, " seconds")
