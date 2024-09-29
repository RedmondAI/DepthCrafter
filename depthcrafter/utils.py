import numpy as np
import glob
import cv2
import matplotlib.cm as cm
import torch
import os
from typing import List, Tuple

def read_image_sequence(
    folder_path: str,
    max_res: int,
    start_frame: int = 0,
    end_frame: int = 999999999,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    image_files = sorted(
        [
            os.path.join(folder_path, img)
            for img in os.listdir(folder_path)
            if img.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
    )

    # Slice the list of image files using start_frame and end_frame
    image_files = image_files[start_frame : end_frame + 1]

    frames = []
    original_sizes = []
    for img_path in image_files:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: Unable to read image {img_path}, skipping.")
            continue

        # Convert image to float32 scaled to [0.0, 1.0]
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        elif img.dtype == np.uint16:
            img = img.astype(np.float32) / 65535.0
        elif img.dtype == np.float32:
            img = np.clip(img, 0.0, 1.0)
        else:
            print(f"Warning: Unexpected image dtype {img.dtype} in {img_path}, skipping.")
            continue

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
        
        frame = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    
    frames = np.array(frames, dtype=np.float32)
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

def read_video_frames(
    video_path: str,
    target_fps: int,
    max_res: int,
    start_frame: int = 0,
    end_frame: int = 999999999,
) -> Tuple[np.ndarray, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")

    # Get original FPS of the video
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Adjust end_frame if it exceeds the total number of frames
    end_frame = min(end_frame, frame_count - 1)

    # Calculate frame interval for FPS adjustment
    frame_interval = max(1, int(round(original_fps / target_fps)))

    frames = []
    current_frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames before start_frame
        if current_frame_idx < start_frame:
            current_frame_idx += 1
            continue

        # Stop if we've reached the end_frame
        if current_frame_idx > end_frame:
            break

        # Downsample FPS by skipping frames
        if (current_frame_idx - start_frame) % frame_interval != 0:
            current_frame_idx += 1
            continue

        original_height, original_width = frame.shape[:2]

        # Resize frame if necessary
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

        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame.astype('uint8'))

        current_frame_idx += 1

    cap.release()
    frames = np.array(frames, dtype=np.uint8)
    return frames, target_fps