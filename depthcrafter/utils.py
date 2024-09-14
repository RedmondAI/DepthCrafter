import numpy as np
import cv2
import matplotlib.cm as cm
import torch
import os
from typing import List

def read_image_sequence(folder_path: str, max_res: int):
    image_files = sorted([
        os.path.join(folder_path, img) for img in os.listdir(folder_path)
        if img.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    frames = []
    original_sizes = []
    for img_path in image_files:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img.dtype != np.float32:
            img = img.astype("float32") / 255.0
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_height, original_width = img.shape[:2]
        original_sizes.append((original_width, original_height))
        scale = min(max_res / original_height, max_res / original_width, 1)
        # Ensure dimensions are divisible by 16
        factor = 16  # Adjust based on network's downsampling layers
        new_width = (int(original_width * scale) // factor) * factor
        new_height = (int(original_height * scale) // factor) * factor
        new_size = (new_width, new_height)
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        frames.append(img)
        print(f"Resized image dimensions: {new_width} x {new_height}")
    frames = np.array(frames)
    return frames, original_sizes

def save_png_sequence(frames: np.ndarray, save_path_prefix: str, original_sizes: List[tuple], dtype=np.float16):
    os.makedirs(os.path.dirname(save_path_prefix), exist_ok=True)
    for idx, frame in enumerate(frames):
        if dtype == np.float16:
            frame_to_save = (frame * 65535).astype(np.uint16)
        else:
            frame_to_save = (frame * 255).astype(np.uint8)
        
        # Resize back to original size
        original_width, original_height = original_sizes[idx]
        frame_to_save = cv2.resize(frame_to_save, (original_width, original_height), interpolation=cv2.INTER_AREA)
        
        cv2.imwrite(f"{save_path_prefix}_{idx:04d}.png", frame_to_save)
    return save_path_prefix

class ColorMapper:
    # a color mapper to map depth values to a certain colormap
    def __init__(self, colormap: str = "inferno"):
        self.colormap = torch.tensor(cm.get_cmap(colormap).colors)

    def apply(self, image: torch.Tensor, v_min=None, v_max=None):
        # assert len(image.shape) == 2
        if v_min is None:
            v_min = image.min()
        if v_max is None:
            v_max = image.max()
        image = (image - v_min) / (v_max - v_min)
        image = (image * 255).long()
        image = self.colormap[image]
        return image


def vis_sequence_depth(depths: np.ndarray, v_min=None, v_max=None):
    visualizer = ColorMapper()
    if v_min is None:
        v_min = depths.min()
    if v_max is None:
        v_max = depths.max()
    res = visualizer.apply(torch.tensor(depths), v_min=v_min, v_max=v_max).numpy()
    return res
