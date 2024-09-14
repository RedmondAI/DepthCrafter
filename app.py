import gc
import os
from copy import deepcopy

import gradio as gr
import numpy as np
import torch
from diffusers.training_utils import set_seed

from depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
from depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
from depthcrafter.utils import read_image_sequence, vis_sequence_depth, save_png_sequence
from run import DepthCrafterDemo

examples = [
    ["examples/images_example_01", 25, 1.2],
]

def construct_demo():
    with gr.Blocks(analytics_enabled=False) as depthcrafter_iface:
        gr.Markdown(
            """
            <div align='center'> <h1> DepthCrafter: Generating Consistent Long Depth Sequences for Open-world Images </span> </h1>
                        <h2 style='font-weight: 450; font-size: 1rem; margin: 0rem'>\
                        <a href='https://wbhu.github.io'>Wenbo Hu</a>, \
                        <a href='https://scholar.google.com/citations?user=qgdesEcAAAAJ&hl=en'>Xiangjun Gao</a>, \
                        <a href='https://xiaoyu258.github.io/'>Xiaoyu Li</a>, \
                        <a href='https://scholar.google.com/citations?user=tZ3dS3MAAAAJ&hl=en'>Sijie Zhao</a>, \
                        <a href='https://vinthony.github.io/academic'> Xiaodong Cun</a>, \
                        <a href='https://yzhang2016.github.io'>Yong Zhang</a>, \
                        <a href='https://home.cse.ust.hk/~quan'>Long Quan</a>, \
                        <a href='https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en'>Ying Shan</a>\
                    </h2> \
                    <a style='font-size:18px;color: #000000'>If you find DepthCrafter useful, please help star the </a>\
                    <a style='font-size:18px;color: #FF5DB0' href='https://github.com/wbhu/DepthCrafter'>[Github Repo]</a>\
                    <a style='font-size:18px;color: #000000'>, which is important to Open-Source projects. Thanks!</a>\
                        <a style='font-size:18px;color: #000000' href='https://arxiv.org/abs/2409.02095'> [ArXiv] </a>\
                        <a style='font-size:18px;color: #000000' href='https://depthcrafter.github.io/'> [Project Page] </a> </div>
            """
        )
        # demo
        depthcrafter_demo = DepthCrafterDemo(
            unet_path="tencent/DepthCrafter",
            pre_train_path="stabilityai/stable-video-diffusion-img2vid-xt",
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                input_folder = gr.Textbox(label="Input Image Folder", placeholder="Path to image sequence folder")
                upload_btn = gr.File(label="Upload Images", file_count="multiple", file_types=[".png", ".jpg", ".jpeg"])

            with gr.Column(scale=2):
                output_folder = gr.Textbox(label="Output Folder", placeholder="Path to save output images")
                output_video_1 = gr.Gallery(
                    label="Preprocessed images",
                    interactive=False,
                )
                output_video_2 = gr.Gallery(
                    label="Generated Depth Images",
                    interactive=False,
                )

        with gr.Row():
            num_denoising_steps = gr.Slider(1, 50, value=25, step=1, label="Number of Inference Steps")
            guidance_scale = gr.Slider(0.1, 10.0, value=1.2, step=0.1, label="Guidance Scale")

        generate_btn = gr.Button("Generate Depth")

        gr.Examples(
            examples=examples,
            inputs=[
                input_folder,
                num_denoising_steps,
                guidance_scale,
            ],
            outputs=[output_video_1, output_video_2],
            fn=depthcrafter_demo.run,
            cache_examples=False,
        )

        generate_btn.click(
            fn=depthcrafter_demo.run,
            inputs=[
                input_folder,
                num_denoising_steps,
                guidance_scale,
            ],
            outputs=[output_video_1, output_video_2],
        )

    return depthcrafter_iface


demo = construct_demo()

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=80, debug=True)
