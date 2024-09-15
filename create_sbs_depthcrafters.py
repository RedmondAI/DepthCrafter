
import torch
import os
import cv2
import numpy as np
import time
import glob
import math
import cupy as cp
import cupyx
import subprocess
from PIL import Image
import shutil
import argparse
import json
from collections import deque
from tqdm import tqdm
from skimage.transform import resize  # Scikit-image works with NumPy arrays
from multiprocessing import Value
from ctypes import c_int
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Manager
import cupy as cp
from cupyx.scipy.ndimage import grey_dilation, gaussian_filter
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('--input_rgb', type=str, help='Path to the directory containing RGB images')
parser.add_argument('--input_depth', type=str, help='Path to the directory containing depth images')
parser.add_argument('--output_dir', type=str, help='Path to the directory for output images')
parser.add_argument('--deviation', type=int, default=20, help='How many pixels of deviation on a 1920x1080 image. Default is 20')
parser.add_argument('--blur', type=int, default=10, help='How many pixels of blur on a 1920x1080 image. Default is 10')
parser.add_argument('--dilate', type=int, default=2, help='How many iterations to dilate the depth with a kernal of 4. Default is 2')
parser.add_argument('--gamma', type=float, default=1.4, help='How much to gamma the depth. Default is 1.4')
parser.add_argument('--max_workers', type=int, default=16, help='How many max workers to run at once to make the distorted images. Default is 16')
parser.add_argument('--offset', type=float, default=0.3, help='How much to offset the image. 0.3 means 30 percent of the depth is positive. Default is 0.3')
parser.add_argument('--extend_depth', type=int, default=3, help='How many pixels to extend the depth pixels to the left or right of each eye. Default is 3')
parser.add_argument('--scale_factor', type=int, default=3, help='How much to scale up the rgb image before doing the warp. Higher values give higher quality warping but you might run out of memory Default is 3')
parser.add_argument('--watermark', type=bool, default=False, help='true or false. When enabled this will add a watermark to the output SBS and spatial video.')
args = parser.parse_args()

# deviation, dilate, and blur are stored as a percentage of the total pixels in a 1920x1080 image. This way if the input image gets bigger or smaller the reletive amount deviation, blur, and dilation will stay the same
scale_factor = args.scale_factor
max_scale = scale_factor*1920
total_pixels = 1920*1080
deviation = args.deviation/1920
dilate_size = args.dilate/total_pixels
blur = args.blur/total_pixels
max_workers = args.max_workers
extend_depth = args.extend_depth
offset = args.offset
gamma = args.gamma
watermark = args.watermark
input_rgb = args.input_rgb
input_depth = args.input_depth
output_dir = args.output_dir

# Printing the values of the variables
print(f"scale_factor: {scale_factor}")
print(f"max_scale: {max_scale}")
print(f"total_pixels: {total_pixels}")
print(f"deviation: {deviation}")
print(f"dilate_size: {dilate_size}")
print(f"blur: {blur}")
print(f"max_workers: {max_workers}")
print(f"extend_depth: {extend_depth}")
print(f"offset: {offset}")
print(f"gamma: {gamma}")
print(f"watermark: {watermark}")
print(f"input_rgb: {input_rgb}")
print(f"input_depth: {input_depth}")
print(f"output_dir: {output_dir}")

def resize_image(image, DESIRED_WIDTH, DESIRED_HEIGHT, interpolation=cv2.INTER_LINEAR_EXACT):
    """Resize the image to the given width and height, converting to 16-bit if necessary.
    
    Args:
        image (numpy.ndarray): The input image.
        DESIRED_WIDTH (int): The desired width.
        DESIRED_HEIGHT (int): The desired height.
        interpolation (int): Interpolation method. Defaults to cv2.INTER_LINEAR.
    
    Returns:
        numpy.ndarray: The resized image, in its original bit depth.
    """
    # Detect the original bit depth of the image
    original_depth = image.dtype


    # Convert image to 16-bit if it's not already
    if original_depth != np.uint16:
        if original_depth == np.uint8:
            image = np.uint16(image) << 8
        elif original_depth in [np.int16, np.int32, np.float32, np.float64]:
            # Normalize the image to the 16-bit range
            norm_image = cv2.normalize(image, None, 0, 65535, cv2.NORM_MINMAX)
            image = np.uint16(norm_image)
        else:
            raise TypeError(f"Unsupported image data type: {original_depth}")

    # Resize the image
    resized_image_16bit = cv2.resize(image, (DESIRED_WIDTH, DESIRED_HEIGHT), interpolation=interpolation)

    # Convert the resized image back to its original bit depth
    if original_depth == np.uint8:
        resized_image = np.uint8(resized_image_16bit >> 8)
    elif original_depth in [np.int16, np.int32, np.float32, np.float64]:
        # Convert back by normalizing to the original depth's range
        resized_image = cv2.normalize(resized_image_16bit, None, 0, np.iinfo(original_depth).max, cv2.NORM_MINMAX).astype(original_depth)
    else:
        resized_image = resized_image_16bit

    return resized_image

def create_quicktime(outdir):
    # Define the command
    command = [
        'ffmpeg', 
        '-framerate', '30',  # Change this to the desired frame rate
        '-pattern_type', 'glob', 
        '-i', os.path.join(outdir, '*.png'),  # Input files
        '-c:v', 'libx264',  # Video codec
        '-pix_fmt', 'yuv420p',  # Pixel format
        '-crf', '25',  # Constant Rate Factor (lower is higher quality)
        os.path.join(outdir, 'sbs.mp4')  # Output file
    ]

    # Execute the command
    subprocess.run(command, check=True)
    print("quicktime created",os.path.join(outdir, 'depth.mp4'))

def extend_depth_map(normalized_depth, side, max_displacement):
    """
    Extend the depth map to the left or right where pixels are brighter using GPU.
    """
    extended_depth = normalized_depth.copy()
    # max_displacement = int(max_displacement/4)
    inverse=True
    inverse=False
    
    if side == 'left':
        # Invert the depth map
        if inverse == True:
            normalized_depth = 1 - normalized_depth

        # Generate shifted depth maps and compare with the original
        for shift in range(1, (max_displacement) + 1):
            shifted_depth = cp.roll(normalized_depth, shift, axis=-1)
            # Propagate the depth value to the left if the shifted pixel is brighter
            mask = shifted_depth > extended_depth
            extended_depth[mask] = shifted_depth[mask]

        # Invert the depth map back
        if inverse == True:
            extended_depth = 1 - extended_depth
            
    elif side == 'right':
        # Generate shifted depth maps and compare with the original
        for shift in range(1, max_displacement + 1):
            shifted_depth = cp.roll(normalized_depth, -shift, axis=1)
            # Propagate the depth value to the right if the shifted pixel is brighter
            mask = shifted_depth > extended_depth
            extended_depth[mask] = shifted_depth[mask]

    return extended_depth

def shift_image_gpu(image, max_displacement, normalized_depth, side, extend_depth, scale_factor=3):
    """
    Shifts the image pixels horizontally based on the normalized depth value using GPU,
    filling gaps with the nearest pixel value and ensuring no overlaps.
    Adjusts the scale_factor to ensure the scaled image does not exceed 5760 pixels in width.
    """
    image = image.astype(cp.float32)
    # print(f"Minimum value in the image image_01: {np.min(image)} Maximum: {np.max(image)}")
    normalized_depth = normalized_depth.astype(cp.float32)

    if side in ['left', 'right']:
        normalized_depth = extend_depth_map(normalized_depth, side, extend_depth)

    # Calculate the scaled width and adjust scale_factor if necessary
    _, original_width, _ = image.shape
    scaled_width = original_width * scale_factor
    if scaled_width > 5760:
        scale_factor = 5760 / original_width
        #print(f"Adjusted scale_factor to {scale_factor} to prevent exceeding 5760 pixels in width.")

    # Scale up the image, normalized_depth, and max_displacement by scale_factor

    original_min = cp.min(image)
    original_max = cp.max(image)

    # Normalize the image to [0, 1]
    #image_normalized = (image - original_min) / (original_max - original_min)
    # Scale image up by scale factor
    # print(f"Minimum value in the image image: {np.min(image)} Maximum: {np.max(image)}")
    # image_scaled = cupyx.scipy.ndimage.zoom(image_normalized, (scale_factor, scale_factor, 1))
    image = cupyx.scipy.ndimage.zoom(image, (scale_factor, scale_factor, 1), order=1)  # Linear interpolation
    # set back to orginal range
    #image = image_scaled * (original_max - original_min) + original_min
    #image = image/256

    normalized_depth = cupyx.scipy.ndimage.zoom(normalized_depth, (scale_factor, scale_factor))
    max_displacement *= scale_factor

    # print(f"Minimum value in the image image_03: {np.min(image)} Maximum: {np.max(image)}")

    h, w, _ = image.shape
    shifted_image = cp.zeros_like(image)
    shift = (1 - normalized_depth) * max_displacement

    x, y = cp.meshgrid(cp.arange(w), cp.arange(h))
    if side == 'right':
        new_x = (x + shift).astype(cp.int32)
    elif side == 'left':
        new_x = (x - shift).astype(cp.int32)

    # Ensure that new_x does not cause out-of-bounds access
    new_x = cp.clip(new_x, 0, w - 1)

    # Flatten the arrays for vectorized operations
    flat_new_x = new_x.flatten()
    flat_y = y.flatten()
    flat_image = image.reshape(-1, 3)  # Assuming a 3-channel image

    # Sort pixels by depth, from farthest to closest
    depth_flat = normalized_depth.flatten()
    sorted_indices = cp.argsort(-depth_flat)  # Negative for descending order
    sorted_flat_new_x = flat_new_x[sorted_indices]
    sorted_flat_y = flat_y[sorted_indices]
    sorted_flat_image = flat_image[sorted_indices]

    # Scatter operation to place the pixels in the new positions
    shifted_image[sorted_flat_y, sorted_flat_new_x] = sorted_flat_image

    # print(f"Minimum value in the image shifted_image_01: {np.min(shifted_image)} Maximum: {np.max(shifted_image)}")

    # Fill gaps by propagating the last valid pixel to the right or left
    if side == 'right':
        shifted_image = fill_gaps_custom_kernel(shifted_image)
        shifted_image = fill_gaps_custom_kernel_left_eye(shifted_image)
    elif side == 'left':
        shifted_image = fill_gaps_custom_kernel_left_eye(shifted_image)
        shifted_image = fill_gaps_custom_kernel(shifted_image)

    # Scale down the shifted_image back to its original size
    # print(f"Minimum value in the image shifted_image_02: {np.min(shifted_image)} Maximum: {np.max(shifted_image)}")
    shifted_image = cupyx.scipy.ndimage.zoom(shifted_image, (1/scale_factor, 1/scale_factor, 1), order=1)
    # print(f"Minimum value in the image shifted_image_03: {np.min(shifted_image)} Maximum: {np.max(shifted_image)}")
    return shifted_image

def fill_gaps_custom_kernel(image, threshold=1):
    """
    Fills the black or dark gaps in the image from right to left based on a threshold using GPU.
    """
    height, width, channels = image.shape

    # Create a mask where True indicates a gap (dark pixel)
    mask = cp.all(image < threshold, axis=-1)

    # Prepare the kernel
    fill_gaps_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void fill_gaps_kernel(const float* input, float* output, const bool* mask, 
                          int width, int height, int channels) {
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) return;

        // Calculate the linear index for the arrays
        unsigned int idx = (y * width + x) * channels;

        // If this is not a gap or it's the first column, just copy the pixel
        if (!mask[y * width + x] || x == 0) {
            for (int c = 0; c < channels; c++) {
                output[idx + c] = input[idx + c];
            }
        } else {
            // Propagate the value from the nearest non-gap pixel to the left
            for (int x_left = x - 1; x_left >= 0; x_left--) {
                if (!mask[y * width + x_left]) {
                    for (int c = 0; c < channels; c++) {
                        output[idx + c] = input[(y * width + x_left) * channels + c];
                    }
                    break;
                }
            }
        }
    }
    ''', 'fill_gaps_kernel')

    # Allocate output image on the GPU
    output_image = cp.zeros_like(image)

    # Grid and block sizes
    threads_per_block = (32, 32)
    blocks_per_grid_x = (width + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (height + threads_per_block[1] - 1) // threads_per_block[1]

    # Launch the kernel
    fill_gaps_kernel((blocks_per_grid_x, blocks_per_grid_y), threads_per_block, 
                     (image, output_image, mask, width, height, channels))

    return output_image

def fill_gaps_custom_kernel_left_eye(image, threshold=1):
    """
    Fills the black or dark gaps in the image from left to right based on a threshold using GPU.
    """
    height, width, channels = image.shape

    # Create a mask where True indicates a gap (dark pixel)
    mask = cp.all(image < threshold, axis=-1)

    # Prepare the kernel
    fill_gaps_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void fill_gaps_kernel(const float* input, float* output, const bool* mask, 
                          int width, int height, int channels) {
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x >= width || y >= height) return;

        // Calculate the linear index for the arrays
        unsigned int idx = (y * width + x) * channels;

        // If this is not a gap or it's the last column, just copy the pixel
        if (!mask[y * width + x] || x == (width - 1)) {
            for (int c = 0; c < channels; c++) {
                output[idx + c] = input[idx + c];
            }
        } else {
            // Propagate the value from the nearest non-gap pixel to the right
            for (int x_right = x + 1; x_right < width; x_right++) {
                if (!mask[y * width + x_right]) {
                    for (int c = 0; c < channels; c++) {
                        output[idx + c] = input[(y * width + x_right) * channels + c];
                    }
                    break;
                }
            }
        }
    }
    ''', 'fill_gaps_kernel')

    # Allocate output image on the GPU
    output_image = cp.zeros_like(image)

    # Grid and block sizes
    threads_per_block = (32, 32)
    blocks_per_grid_x = (width + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (height + threads_per_block[1] - 1) // threads_per_block[1]

    # Launch the kernel
    fill_gaps_kernel((blocks_per_grid_x, blocks_per_grid_y), threads_per_block, 
                     (image, output_image, mask, width, height, channels))

    return output_image

def generate_stereo_images8_gpu(left_img, depth, width, height):
    
    deviation_computed = int((width)*deviation)

    depth = cp.asarray(depth)

    

    left_img_normalized = left_img.astype(np.uint16)
    #left_img_normalized = np.clip(left_img_normalized + 500, 0, 65500)
    left_img_normalized = np.clip(left_img_normalized, 500, None)


    # print(f"Minimum value in the image left_img_normalized: {np.min(left_img_normalized)} Maximum: {np.max(left_img_normalized)}")

    
    # Now convert the normalized image to a CuPy array
    rgb_image_gpu = cp.asarray(left_img_normalized/255)
    #rgb_image_gpu = np.clip(rgb_image_gpu, 0, 256)
    # print(f"Minimum value in the image rgb_image_gpu: {np.min(rgb_image_gpu)} Maximum: {np.max(rgb_image_gpu)}")
    # Normalize the depth map using GPU
    normalized_depth = cp.interp(depth.astype(cp.float32), 
                                cp.array([0.0, 65535]), 
                                cp.array([0.0, 1.0]))
    normalized_depth = normalized_depth+offset


    # Generate shifted images for both views
    shift_amount = deviation_computed

    shifted_image_gpu = shift_image_gpu(rgb_image_gpu, shift_amount, normalized_depth,side="right",extend_depth=extend_depth,scale_factor=scale_factor)
    inverted_shift_image_gpu = shift_image_gpu(rgb_image_gpu, shift_amount, normalized_depth,side="left",extend_depth=extend_depth,scale_factor=scale_factor)

    # shifted_image_gpu = np.clip(shifted_image_gpu, 0, 255)
    # inverted_shift_image_gpu = np.clip(inverted_shift_image_gpu, 0, 255)

    # print(f"Minimum value in the image shifted_image_gpu: {np.min(shifted_image_gpu)} Maximum: {np.max(shifted_image_gpu)}")

    filled_shifted_image_gpu = fill_gaps_custom_kernel_left_eye(shifted_image_gpu)
    filled_inverted_shift_image_gpu = fill_gaps_custom_kernel(inverted_shift_image_gpu)
    filled_inverted_shift_image_gpu = fill_gaps_custom_kernel_left_eye(filled_inverted_shift_image_gpu)

    # print(f"Minimum value in the image filled_shifted_image_gpu: {np.min(filled_shifted_image_gpu)} Maximum: {np.max(filled_shifted_image_gpu)}")

    filled_shifted_image_right = cp.asnumpy(filled_shifted_image_gpu)
    filled_shifted_image_left = cp.asnumpy(filled_inverted_shift_image_gpu)

    # filled_shifted_image_right = cp.asnumpy(shifted_image_gpu)
    # filled_shifted_image_left = cp.asnumpy(inverted_shift_image_gpu)

    return filled_shifted_image_left,filled_shifted_image_right

def zoom_image(image, pixel_scale):
    """
    Scales an image up while keeping the same resolution.

    Args:
        image (cupy.ndarray): The input image.
        pixel_scale (int): The number of pixels to scale up by.

    Returns:
        cupy.ndarray: The scaled image.
    """

    if not isinstance(image, cp.ndarray):
        image = cp.asarray(image)
    # Calculate the new dimensions
    height, width = image.shape[:2]
    new_width = width - (2 * pixel_scale)
    new_height = height - (2 * pixel_scale)

    # Ensure the new dimensions are valid
    if new_width <= 0 or new_height <= 0:
        raise ValueError("Pixel scale is too large for the given image size.")

    # Calculate the cropping coordinates
    left = pixel_scale
    right = width - pixel_scale
    top = pixel_scale
    bottom = height - pixel_scale

    # Crop the image
    cropped_image = image[top:bottom, left:right]

    # Calculate the zoom factors
    zoom_factor_y = height / new_height
    zoom_factor_x = width / new_width

    # Resize the image using cupyx.scipy.ndimage.zoom
    scaled_image = cupyx.scipy.ndimage.zoom(cropped_image, (zoom_factor_y, zoom_factor_x, 1))
    scaled_image_np = cp.asnumpy(scaled_image)

    return scaled_image_np

def add_watermark(input_image, watermark_image_path="watermark_dropshadow_light.png"):
    """
    Adds a watermark to the input image. The watermark is resized to be 20% of the input image's width
    and placed in the bottom right corner of the input image.

    Args:
        input_image (numpy.ndarray): The input image as a NumPy array.
        watermark_image_path (str): Path to the watermark image. Defaults to "watermark_dropshadow.png".

    Returns:
        numpy.ndarray: The input image with the watermark added.
    """
    # Assume input_image is already loaded as a numpy array, so we don't need to read it again
    input_height, input_width = input_image.shape[:2]

    # Load the watermark image
    watermark_image = cv2.imread(watermark_image_path, cv2.IMREAD_UNCHANGED)

    # Calculate the new width and height of the watermark
    if input_width <= input_height:
        watermark_width = int(input_width * 0.4)
        watermark_height = int((watermark_width / watermark_image.shape[1]) * watermark_image.shape[0])
    else:
        watermark_width = int(input_width * 0.2)
        watermark_height = int((watermark_width / watermark_image.shape[1]) * watermark_image.shape[0])

    # Resize the watermark
    watermark_resized = cv2.resize(watermark_image, (watermark_width, watermark_height), interpolation=cv2.INTER_AREA)

    # Calculate the position to place the watermark
    x_pos = int((input_width - watermark_width)-(input_width*.05))
    y_pos = int((input_height - watermark_height)-(input_height*.05))

    # If the watermark has transparency (indicated by having 4 channels)
    if watermark_resized.shape[2] == 4:
        # Separate the watermark into its color and alpha channels
        watermark_color = watermark_resized[:, :, :3]
        alpha_channel = watermark_resized[:, :, 3] / 255.0

        # Create a mask for blending based on the alpha channel
        alpha_inv = 1.0 - alpha_channel

        # Extract the region of the input image where the watermark will be placed
        roi = input_image[y_pos:y_pos+watermark_height, x_pos:x_pos+watermark_width]

        # Blend the watermark with the input image based on the alpha channel
        for c in range(0, 3):
            roi[:, :, c] = (alpha_channel * watermark_color[:, :, c] +
                            alpha_inv * roi[:, :, c])

        # Place the blended watermark back into the input image
        input_image[y_pos:y_pos+watermark_height, x_pos:x_pos+watermark_width] = roi
    else:
        # If the watermark does not have an alpha channel, simply overlay it
        input_image[y_pos:y_pos+watermark_height, x_pos:x_pos+watermark_width] = watermark_resized

    return input_image

def process_file(depth_filename, rgb_filename):
    try:
        # Check if files exist
        if not os.path.exists(depth_filename):
            raise FileNotFoundError(f"Depth file not found: {depth_filename}")
        if not os.path.exists(rgb_filename):
            raise FileNotFoundError(f"RGB file not found: {rgb_filename}")

        frame = os.path.basename(rgb_filename)
        combined_output_file = os.path.join(output_dir, frame)
        
        # Read the RGB image and resize it
        rgb_img = cv2.imread(rgb_filename)
        rgb_img = np.uint16(rgb_img) * 257

        DESIRED_HEIGHT, DESIRED_WIDTH = rgb_img.shape[:2]
        deviation_computed = int(DESIRED_WIDTH*deviation)

        # Read the depth image and resize it
        depth_img = cv2.imread(depth_filename, cv2.IMREAD_UNCHANGED)
        depth_img = resize_image(depth_img,DESIRED_WIDTH,DESIRED_HEIGHT)

        # gamma the image
        # min_depth_img = np.min(depth_img)
        # max_depth_img = np.max(depth_img)
        # depth_img_gamma = np.power(depth_img, gamma)
        # depth_img_gamma = ((depth_img_gamma - min_depth_img) / (max_depth_img - min_depth_img)) * (max_depth_img - min_depth_img) + min_depth_img
        # depth_img = depth_img_gamma

        # Define the kernel for dilation
        totalpixels = DESIRED_WIDTH * DESIRED_HEIGHT
        dilate_size_computed = int(totalpixels * dilate_size)
        kernel_size = 4
        kernel = np.ones((kernel_size, kernel_size), np.uint8)  # np.uint16 is not necessary

        # Perform dilation
        depth_img = cv2.dilate(depth_img, kernel, iterations=dilate_size_computed)

        # # Normalize the depth image to the range [0.0, 1.0] for 32-bit float conversion
        # min_depth = 0
        # max_depth = 65535
        # depth_img_normalized = (depth_img - min_depth) / (max_depth - min_depth)
        # depth_img_32f = depth_img_normalized.astype('float32')

        
        # # Scale the blurred image back to the full 16-bit range [0, 65535] and convert to 16-bit
        # depth_img = (depth_img_32f * 65535).astype('uint16')

        # Blur the depth
        blur_size_computed = int(totalpixels*blur)
        if blur_size_computed != 0:
            blur_size_computed = max(3, blur_size_computed | 1)  # Make sure it's odd and at least 3
        if blur_size_computed != 0:
            depth_img = cv2.GaussianBlur(depth_img, (blur_size_computed, 3), 0)


        # Create the stereo pair using the generate_stereo function. It will also save a png of the final depth map if depth_save_path is not None
        rgb_img,right_img_fixed = generate_stereo_images8_gpu(rgb_img, depth_img,DESIRED_WIDTH,DESIRED_HEIGHT)
        
        rgb_img = zoom_image(rgb_img, deviation_computed)
        right_img_fixed = zoom_image(right_img_fixed, deviation_computed)

        # Check the resolution of rgb_img and right_img_fixed
        # If any dimension is not divisible by 2 then resize the image to make sure height and width are divisible by 2
        if rgb_img.shape[0] % 2 != 0 or rgb_img.shape[1] % 2 != 0:
            rgb_img = cv2.resize(rgb_img, (rgb_img.shape[1] - rgb_img.shape[1] % 2, rgb_img.shape[0] - rgb_img.shape[0] % 2))
        if right_img_fixed.shape[0] % 2 != 0 or right_img_fixed.shape[1] % 2 != 0:
            right_img_fixed = cv2.resize(right_img_fixed, (right_img_fixed.shape[1] - right_img_fixed.shape[1] % 2, right_img_fixed.shape[0] - right_img_fixed.shape[0] % 2))

        if watermark:
            rgb_img = add_watermark(rgb_img)
            right_img_fixed = add_watermark(right_img_fixed)

        # Combine the left and right images into one image (double width)
        combined_img = np.concatenate((rgb_img, right_img_fixed), axis=1)

        #print("saving",combined_output_file)
        cv2.imwrite(combined_output_file, combined_img)

        #print("elapsed time",elapsed_time)
    except Exception as e:
        import traceback
        print("Error occurred:", e)
        print(traceback.format_exc())
        raise

def create_stereo_pair(file_pairs):
    total_files = len(file_pairs)
    print("total number of image files", total_files)
    
    # Initialize progress bar
    progress_bar = tqdm(total=total_files, unit='file', desc='Processing files', leave=True)
    
    # Use ThreadPoolExecutor to avoid multiprocessing issues
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, depth_file, rgb_file) for depth_file, rgb_file in file_pairs]
        for future in concurrent.futures.as_completed(futures):
            progress_bar.update(1)  # Update progress bar for each completed task
    
    progress_bar.close()
    print("Stereo pair creation process completed.")



if __name__ == '__main__':


    print("deviation",args.deviation)
    print("blur",args.blur)
    print("dilate",args.dilate)
    print("gamma",args.gamma)
    print("offset",args.offset)
    print("extend_depth",args.extend_depth)
    print("Number of workers", max_workers)
    
    # Ensure the output directory exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    # Get list of depth image files
    depth_filenames = [f for f in os.listdir(input_depth) if f.endswith('.png')]
    depth_filenames.sort()
    depth_filepaths = [os.path.join(input_depth, f) for f in depth_filenames]
    
    # Get list of RGB image files
    rgb_filenames = [f for f in os.listdir(input_rgb) if f.endswith('.png')]
    rgb_filenames.sort()
    rgb_filepaths = [os.path.join(input_rgb, f) for f in rgb_filenames]
    
    # Ensure both directories have the same number of files
    assert len(depth_filepaths) == len(rgb_filepaths), "Mismatch between number of depth and RGB files."
    
    # Create a list of (depth_file, rgb_file) pairs
    file_pairs = list(zip(depth_filepaths, rgb_filepaths))
    
    # Update 'outdir' variable used in functions
    outdir = output_dir
    
    # Process the images
    create_stereo_pair(file_pairs)


    