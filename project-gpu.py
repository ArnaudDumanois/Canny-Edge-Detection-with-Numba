import argparse
from numba import cuda
import numpy as np
from PIL import Image
import math
import time

# Define Gaussian kernel
gaussian_kernel = np.array([[1, 4, 6, 4, 1],
                            [4, 16, 24, 16, 4],
                            [6, 24, 36, 24, 6],
                            [4, 16, 24, 16, 4],
                            [1, 4, 6, 4, 1]], dtype=np.float32)

# Define Sobel kernels
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)

sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]], dtype=np.float32)

# Define low and high threshold
low_threshold = 51
high_threshold = 102


# Function to compute the number of thread blocks
def compute_thread_blocks(imagetab, block_size):
    height, width = imagetab.shape[:2]
    blockspergrid_x = math.ceil(width / block_size[0])
    blockspergrid_y = math.ceil(height / block_size[1])
    blockspergrid = (blockspergrid_y, blockspergrid_x)
    return blockspergrid

@cuda.jit
def bw_kernel(input, output):
    # Convert image to black and white
    i, j = cuda.grid(2)
    if i < input.shape[0] and j < input.shape[1]:
        output[i, j] = 0.3 * input[i, j, 0] + 0.59 * input[i, j, 1] + 0.11 * input[i, j, 2]

@cuda.jit
def gauss_kernel(input, output, kernel):
    x, y = cuda.grid(2)
    if x < input.shape[0] and y < input.shape[1]:
        kernel_sum = 0
        weighted_sum = 0
        for a in range(kernel.shape[0]):
            for b in range(kernel.shape[1]):
                nx = x + a - kernel.shape[0] // 2
                ny = y + b - kernel.shape[1] // 2
                if nx >= 0 and ny >= 0 and nx < input.shape[0] and ny < input.shape[1]:
                    kernel_sum += kernel[a, b]
                    weighted_sum += kernel[a, b] * input[nx, ny]
        output[x, y] = weighted_sum // kernel_sum

@cuda.jit
def sobel_kernel(input, output_magnitude):
    x, y = cuda.grid(2)
    if x < input.shape[0] and y < input.shape[1]:
        Gx = 0
        Gy = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                nx = x + i
                ny = y + j
                if nx >= 0 and ny >= 0 and nx < input.shape[0] and ny < input.shape[1]:
                    Gx += input[nx, ny] * sobel_x[i + 1, j + 1]
                    Gy += input[nx, ny] * sobel_y[i + 1, j + 1]
        magnitude = math.sqrt(Gx ** 2 + Gy ** 2)
        if magnitude > 175:
            magnitude = 175  # Clamp the magnitude to 175
        output_magnitude[x, y] = magnitude

@cuda.jit
def threshold_kernel(input, output, low, high):
    x, y = cuda.grid(2)
    if x < input.shape[0] and y < input.shape[1]:
        if input[x, y] < low:
            output[x, y] = 0
        elif input[x, y] > high:
            output[x, y] = 255
        else:
            output[x, y] = input[x, y]

@cuda.jit
def hysteresis_kernel(input, output, low, high):
    x, y = cuda.grid(2)
    if x < input.shape[0] and y < input.shape[1]:
        if input[x, y] >= high:
            output[x, y] = 255
        elif input[x, y] >= low and is_connected_to_strong_edge(input, x, y, low):
            output[x, y] = 255
        else:
            output[x, y] = 0

@cuda.jit
def is_connected_to_strong_edge(input, x, y, low):
    for i in range(-1, 2):
        for j in range(-1, 2):
            nx = x + i
            ny = y + j
            if nx >= 0 and ny >= 0 and nx < input.shape[0] and ny < input.shape[1]:
                if input[nx, ny] >= low:
                    return True
    return False




def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Canny Edge Detector')
    parser.add_argument('input', help='Input image')
    parser.add_argument('output', help='Output image')
    parser.add_argument('--tb', type=int, default=32, help='Thread block size for all operations')
    parser.add_argument('--bw', action='store_true', help='Convert the image to black and white')
    parser.add_argument('--gauss', action='store_true', help='Apply a Gaussian blur to the image')
    parser.add_argument('--sobel', action='store_true', help='Apply the Sobel operator to the image')
    parser.add_argument('--threshold', action='store_true', help='Apply a threshold to the image')

    args = parser.parse_args()

    # Load the input image
    input_image = np.array(Image.open(args.input))
    print(input_image.shape)

    # Set the thread block size
    block_size = (args.tb, args.tb)

    # Compute the grid size
    grid_size = compute_thread_blocks(input_image, block_size)

    # Convert the image to black and white
    s_image = cuda.to_device(input_image)
    d_image = cuda.device_array((input_image.shape[0], input_image.shape[1]), dtype=np.uint8)
    bw_kernel[grid_size, block_size](s_image, d_image)
    cuda.synchronize()
    bw_image = d_image.copy_to_host()

    # Convert the image to black and white if specified
    if args.bw:
        bw_image = Image.fromarray(bw_image)
        bw_image.save(args.output)

        end_time = time.time()
        print("Execution time: ", end_time - start_time, "s")
        return

    d_bw_image = cuda.to_device(bw_image)
    d_blurred_image = cuda.device_array((bw_image.shape[0], bw_image.shape[1]), dtype=np.uint8)
    d_kernel = cuda.to_device(gaussian_kernel)
    gauss_kernel[grid_size, block_size](d_bw_image, d_blurred_image, d_kernel)
    cuda.synchronize()
    blurred_image = d_blurred_image.copy_to_host()

    # Apply Gaussian blur if specified
    if args.gauss:
        """
        # Convert the black and white image to device array
        d_bw_image = cuda.to_device(bw_image)

        # Allocate device array for blurred image
        d_blurred_image = cuda.device_array_like(d_bw_image)

        # Convert the Gaussian kernel to device array
        d_kernel = cuda.to_device(gaussian_kernel)

        # Apply Gaussian blur
        gauss_kernel[grid_size, block_size](d_bw_image, d_blurred_image, d_kernel)
        cuda.synchronize()

        # Copy blurred image back to host and save it
        blurred_image = d_blurred_image.copy_to_host()
        """
        blurred_image = Image.fromarray(blurred_image)
        blurred_image.save(args.output)

        end_time = time.time()
        print("Execution time: ", end_time - start_time, "s")
        return

    d_blurred_image = cuda.to_device(blurred_image)
    d_magnitude = cuda.device_array((blurred_image.shape[:2]), dtype=np.float32)
    sobel_kernel[grid_size, block_size](d_blurred_image, d_magnitude)
    cuda.synchronize()
    magnitude = d_magnitude.copy_to_host()

    if args.sobel:
        magnitude = Image.fromarray((magnitude).astype(np.uint8))
        magnitude.save(args.output)

        end_time = time.time()
        print("Execution time: ", end_time - start_time, "s")
        return


    # Apply thresholding
    d_magnitude = cuda.to_device(magnitude)
    d_threshold = cuda.device_array((magnitude.shape[:2]), dtype=np.uint8)
    threshold_kernel[grid_size, block_size](d_magnitude, d_threshold, low_threshold, high_threshold)
    cuda.synchronize()
    threshold = d_threshold.copy_to_host()


    if args.threshold:
        threshold = Image.fromarray(threshold)
        threshold.save(args.output)

        end_time = time.time()
        print("Execution time: ", end_time - start_time, "s")
        return

    d_threshold = cuda.to_device(threshold)
    d_output = cuda.device_array((threshold.shape[:2]), dtype=np.uint8)
    hysteresis_kernel[grid_size, block_size](d_threshold, d_output, low_threshold, high_threshold)
    cuda.synchronize()
    output = d_output.copy_to_host()

    output = Image.fromarray(output)
    output.save(args.output)

    end_time = time.time()
    print("Execution time: ", end_time - start_time, "s")
    return


if __name__ == '__main__':
    main()
