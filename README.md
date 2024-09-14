## ___***DepthCrafter: Generating Consistent Long Depth Sequences for Open-world Videos***___
<div align="center">
<img src='https://depthcrafter.github.io/img/logo.png' style="height:140px"></img>

<a href='https://arxiv.org/abs/2409.02095'><img src='https://img.shields.io/badge/arXiv-2409.02095-b31b1b.svg'></a> &nbsp;
<a href='https://depthcrafter.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;

<<<<<<< HEAD
...
=======

 <a href='https://arxiv.org/abs/2409.02095'><img src='https://img.shields.io/badge/arXiv-2409.02095-b31b1b.svg'></a> &nbsp;
 <a href='https://depthcrafter.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;


_**[Wenbo Hu<sup>1* &dagger;</sup>](https://wbhu.github.io), 
[Xiangjun Gao<sup>2*</sup>](https://scholar.google.com/citations?user=qgdesEcAAAAJ&hl=en), 
[Xiaoyu Li<sup>1* &dagger;</sup>](https://xiaoyu258.github.io), 
[Sijie Zhao<sup>1</sup>](https://scholar.google.com/citations?user=tZ3dS3MAAAAJ&hl=en), 
[Xiaodong Cun<sup>1</sup>](https://vinthony.github.io/academic), <br>
[Yong Zhang<sup>1</sup>](https://yzhang2016.github.io), 
[Long Quan<sup>2</sup>](https://home.cse.ust.hk/~quan), 
[Ying Shan<sup>3, 1</sup>](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en)**_
<br><br>
<sup>1</sup>Tencent AI Lab
<sup>2</sup>The Hong Kong University of Science and Technology
<sup>3</sup>ARC Lab, Tencent PCG

arXiv preprint, 2024

</div>

## 🔆 Introduction
🤗 DepthCrafter can generate temporally consistent long depth sequences with fine-grained details for open-world videos, 
without requiring additional information such as camera poses or optical flow.

## 🎥 Visualization
We provide some demos of unprojected point cloud sequences, with reference RGB and estimated depth videos. 
Please refer to our [project page](https://depthcrafter.github.io) for more details.


https://github.com/user-attachments/assets/62141cc8-04d0-458f-9558-fe50bc04cc21



>>>>>>> parent of 6e06eb5 (update)

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
