## ___***DepthCrafter: Generating Consistent Long Depth Sequences for Open-world Videos***___
<div align="center">
<img src='https://depthcrafter.github.io/img/logo.png' style="height:140px"></img>

<a href='https://arxiv.org/abs/2409.02095'><img src='https://img.shields.io/badge/arXiv-2409.02095-b31b1b.svg'></a> &nbsp;
<a href='https://depthcrafter.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;

...

## 🚀 Quick Start

### 🛠️ Installation
1. Clone this repo:
    ```bash
    git clone https://github.com/Tencent/DepthCrafter.git
    ```
2. Install dependencies (please refer to [requirements.txt](requirements.txt)):
    ```bash
    pip install -r requirements.txt
    ```

## 🤗 Model Zoo
[DepthCrafter](https://huggingface.co/tencent/DepthCrafter) is available in the Hugging Face Model Hub.

### 🏃‍♂️ Inference
#### 1. Using Image Sequence, requires appropriate GPU memory based on resolution:
- Full inference (~0.6 fps on A100, recommended for high-quality results):
    ```bash
    python run.py --image-folder examples/images_example_01
    ```

- Fast inference through 4-step denoising and without classifier-free guidance (~2.3 fps on A100):
    ```bash
    python run.py --image-folder examples/images_example_01 --num-inference-steps 4 --guidance-scale 1.0
    ```

## 🤖 Gradio Demo
We provide a local Gradio demo for DepthCrafter, which can be launched by running:
```bash
gradio app.py
```

## 🤝 Contributing
- Welcome to open issues and pull requests.
- Welcome to optimize the inference speed and memory usage, e.g., through model quantization, distillation, or other acceleration techniques.
