import numpy as np
import glob
import cv2
import matplotlib.cm as cm
import torch
import os
from typing import List, Tuple

def read_image_sequence(folder_path: str, max_res: int) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    image_files = sorted([
        os.path.join(folder_path, img) for img in os.listdir(folder_path)
        if img.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    frames = []
    original_sizes = []
    for img_path in image_files:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: Unable to read image {img_path}, skipping.")
            continue
        if img.dtype != np.float32:
            img = img.astype("float32") / 255.0
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        original_height, original_width = img.shape[:2]
        original_sizes.append((original_width, original_height))  # Collect original sizes
        
        # Resize frame if necessary to be multiples of 64
        if max(original_height, original_width) > max_res:
            scale = max_res / max(original_height, original_width)
            height = int(round(original_height * scale / 64) * 64)
            width = int(round(original_width * scale / 64) * 64)
        else:
            height = int(round(original_height / 64) * 64)
            width = int(round(original_width / 64) * 64)
        
        # Ensure dimensions are at least 64
        height = max(height, 64)
        width = max(width, 64)
        
        frame = cv2.resize(img, (width, height))
        frames.append(frame.astype('uint8'))
    
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

def read_image_sequence_frames(image_sequence_path, process_length, target_fps, max_res):
    # Get a sorted list of image file paths
    supported_formats = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff')
    image_files = []
    for ext in supported_formats:
        image_files.extend(glob.glob(os.path.join(image_sequence_path, ext)))
    image_files = sorted(image_files)
    
    if not image_files:
        raise ValueError(f"No images found in the directory: {image_sequence_path}")

    # Optionally limit to process_length
    # if process_length > 0:
    #     image_files = image_files[:process_length]

    # Read and preprocess images
    frames = []
    for img_path in image_files:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Warning: Unable to read image {img_path}, skipping.")
            continue  # Skip if frame is not read correctly

        original_height, original_width = frame.shape[:2]

        # Resize frame if necessary
        if max(original_height, original_width) > max_res:
            scale = max_res / max(original_height, original_width)
            height = int(round(original_height * scale / 64) * 64)
            width = int(round(original_width * scale / 64) * 64)
        else:
            height = int(round(original_height / 64) * 64)
            width = int(round(original_width / 64) * 64)

        frame = cv2.resize(frame, (width, height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame.astype('uint8'))

    frames = np.array(frames)
    return frames, target_fps
