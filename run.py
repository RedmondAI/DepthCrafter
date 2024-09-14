import gc
import os
import numpy as np
import torch
import argparse
from diffusers.training_utils import set_seed
import time

from depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
from depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
from depthcrafter.utils import vis_sequence_depth, save_video, read_video_frames, read_image_sequence, save_png_sequence

class DepthCrafterDemo:
    def __init__(
        self,
        unet_path: str,
        pre_train_path: str,
        cpu_offload: str = "model",
    ):
        unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
            unet_path,
            subfolder="unet",
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
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(e)
            print("Xformers is not enabled")
        self.pipe.enable_attention_slicing()

    def infer(
        self,
        image_folder: str,
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
    ):
        set_seed(seed)

        frames = read_image_sequence(image_folder, max_res)
        process_length = len(frames)
        print(f"==> image folder: {image_folder}, number of frames: {process_length}")

        # inference the depth map using the DepthCrafter pipeline
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
        # convert the three-channel output to a single channel depth map
        res = res.sum(-1) / res.shape[-1]
        # normalize the depth map to [0, 1] across the whole sequence
        res = (res - res.min()) / (res.max() - res.min())
        # visualize the depth map and save the results
        vis = vis_sequence_depth(res)
        # save the depth map and visualization as 16-bit PNGs
        save_path = os.path.join(
            save_folder, os.path.basename(image_folder)
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_png_sequence(res, save_path + "_depth", dtype=np.float16)
        save_png_sequence(vis, save_path + "_vis", dtype=np.float16)
        save_png_sequence(frames, save_path + "_input", dtype=np.float16)
        return [
            save_path + "_input",
            save_path + "_vis",
            save_path + "_depth",
        ]

    def run(
        self,
        input_folder,
        num_denoising_steps,
        guidance_scale,
        max_res=1024,
    ):
        frames = read_image_sequence(input_folder, max_res)
        process_length = len(frames)
        res_path = self.infer(
            input_folder,
            num_denoising_steps,
            guidance_scale,
            process_length=process_length,
        )
        # clear the cache for the next input
        gc.collect()
        torch.cuda.empty_cache()
        return res_path[:2]

if __name__ == "__main__":
    # running configs
    parser = argparse.ArgumentParser(description="DepthCrafter")
    parser.add_argument(
        "--image-folder", type=str, required=True, help="Path to the input image sequence folder"
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
    parser.add_argument(
        "--target-fps", type=int, default=15, help="Target FPS for the output"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--num-inference-steps", type=int, default=25, help="Number of inference steps"
    )
    parser.add_argument(
        "--guidance-scale", type=float, default=1.2, help="Guidance scale"
    )
    parser.add_argument("--window-size", type=int, default=110, help="Window size")
    parser.add_argument("--overlap", type=int, default=25, help="Overlap size")
    parser.add_argument("--max-res", type=int, default=1024, help="Maximum resolution")
    parser.add_argument("--save_npz", type=bool, default=True, help="Save npz file")
    parser.add_argument("--track_time", type=bool, default=False, help="Track time")

    args = parser.parse_args()
    start_time = time.time()
    depthcrafter_demo = DepthCrafterDemo(
        unet_path=args.unet_path,
        pre_train_path=args.pre_train_path,
        cpu_offload=args.cpu_offload,
    )
    depthcrafter_demo.run(
        args.image_folder,
        args.num_inference_steps,
        args.guidance_scale,
        max_res=args.max_res,
    )
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print("Time per frame: ", (end_time - start_time) / len(frames), " seconds")
    gc.collect()
    torch.cuda.empty_cache()
