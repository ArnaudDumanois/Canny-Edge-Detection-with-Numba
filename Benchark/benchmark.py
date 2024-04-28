import argparse
import os
from tabulate import tabulate
import numpy as np
from PIL import Image
import time
from numba import cuda
from env import PROJECT_FILE, LOW_THRESHOLD, HIGH_THRESHOLD
import math

project_gpu = __import__(PROJECT_FILE.replace(".py", ""))

def compute_thread_blocks(imagetab, block_size):
    height, width = imagetab.shape[:2]
    blockspergrid_x = math.ceil(width / block_size[0])
    blockspergrid_y = math.ceil(height / block_size[1])
    blockspergrid = (blockspergrid_y, blockspergrid_x)
    return blockspergrid

def print_results(execution_times, file_names):
    headers = ["File", "BW (s)", "Gaussian (s)", "Sobel (s)", "Threshold (s)", "Hysteresis (s)", "Total (s)"]
    data = []
    for i, execution_time in enumerate(execution_times):
        file_name = file_names[i]  # Vous pouvez remplacer cela par le nom du fichier
        bw_time = execution_time["BW Execution Time (s)"]
        gaussian_time = execution_time["Gaussian Execution Time (s)"]
        sobel_time = execution_time["Sobel Execution Time (s)"]
        threshold_time = execution_time["Threshold Execution Time (s)"]
        hysteresis_time = execution_time["Hysteresis Execution Time (s)"]
        total_time = sum([bw_time, gaussian_time, sobel_time, threshold_time, hysteresis_time])
        data.append([file_name, bw_time, gaussian_time, sobel_time, threshold_time, hysteresis_time, total_time])
    print(tabulate(data, headers=headers, tablefmt="grid"))


def run_benchmark(input_path, output_path, args):
    start_time = time.time()

    # Load the input image
    input_image = np.array(Image.open(input_path))

    # Set the thread block size
    block_size = (args.tb, args.tb)

    # Compute the grid size
    grid_size = compute_thread_blocks(input_image, block_size)

    # Convert the image to black and white
    s_image = cuda.to_device(input_image)
    d_image = cuda.device_array((input_image.shape[0], input_image.shape[1]), dtype=np.uint8)
    project_gpu.bw_kernel[grid_size, block_size](s_image, d_image)
    cuda.synchronize()
    bw_image = d_image.copy_to_host()

    bw_execution_time = time.time() - start_time
    start_time = time.time()

    d_bw_image = cuda.to_device(bw_image)
    d_blurred_image = cuda.device_array((bw_image.shape[0], bw_image.shape[1]), dtype=np.uint8)
    d_kernel = cuda.to_device(project_gpu.gaussian_kernel)
    project_gpu.gauss_kernel[grid_size, block_size](d_bw_image, d_blurred_image, d_kernel)
    cuda.synchronize()
    blurred_image = d_blurred_image.copy_to_host()

    gauss_execution_time = time.time() - start_time
    start_time = time.time()

    d_blurred_image = cuda.to_device(blurred_image)
    d_magnitude = cuda.device_array((blurred_image.shape[:2]), dtype=np.float32)
    project_gpu.sobel_kernel[grid_size, block_size](d_blurred_image, d_magnitude)
    cuda.synchronize()
    magnitude = d_magnitude.copy_to_host()

    sobel_execution_time = time.time() - start_time
    start_time = time.time()

    # Apply thresholding
    d_magnitude = cuda.to_device(magnitude)
    d_threshold = cuda.device_array((magnitude.shape[:2]), dtype=np.uint8)
    project_gpu.threshold_kernel[grid_size, block_size](d_magnitude, d_threshold, LOW_THRESHOLD, HIGH_THRESHOLD)
    cuda.synchronize()
    threshold = d_threshold.copy_to_host()

    threshold_execution_time = time.time() - start_time
    start_time = time.time()

    d_threshold = cuda.to_device(threshold)
    d_output = cuda.device_array((threshold.shape[:2]), dtype=np.uint8)
    project_gpu.hysteresis_kernel[grid_size, block_size](d_threshold, d_output)
    cuda.synchronize()
    output = d_output.copy_to_host()

    hysteresis_execution_time = time.time() - start_time

    output = Image.fromarray(output)
    output.save(output_path)

    return {
        "BW Execution Time (s)": bw_execution_time,
        "Gaussian Execution Time (s)": gauss_execution_time,
        "Sobel Execution Time (s)": sobel_execution_time,
        "Threshold Execution Time (s)": threshold_execution_time,
        "Hysteresis Execution Time (s)": hysteresis_execution_time
    }


def main():
    parser = argparse.ArgumentParser(description='Canny Edge Detector')
    parser.add_argument('input_folder', help='Input image')
    parser.add_argument('output_folder', help='Output image')
    parser.add_argument('--tb', type=int, default=32, help='Thread block size for all operations')
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    execution_times = []
    file_names = []

    for image in os.listdir(args.input_folder):
        if image.endswith(".jpg") or image.endswith(".png") or image.endswith(".jpeg"):
            input_path = os.path.join(args.input_folder, image)
            output_path = os.path.join(args.output_folder, f'output_{image}')
            execution_time = run_benchmark(input_path, output_path, args)
            execution_times.append(execution_time)
            file_names.append(image.replace(".jpg", "").replace(".png", "").replace(".jpeg", ""))

    print_results(execution_times, file_names)


if __name__ == '__main__':
    main()