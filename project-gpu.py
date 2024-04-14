import argparse
from numba import cuda
import numpy as np
from PIL import Image
import math

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


# Function to compute the number of thread blocks
def compute_thread_blocks(imagetab, block_size):
    height, width = imagetab.shape[:2]
    blockspergrid_x = math.ceil(width / block_size[0])
    blockspergrid_y = math.ceil(height / block_size[1])
    blockspergrid = (blockspergrid_y, blockspergrid_x)
    return blockspergrid

@cuda.jit
def bw_kernel(input, output):
    i, j = cuda.grid(2)
    if i < input.shape[0] and j < input.shape[1]:
        output[i, j] = 0.3 * input[i, j, 0] + 0.59 * input[i, j, 1] + 0.11 * input[i, j, 2]

@cuda.jit
def gaussian_kernel(input, output, kernel):
    pass

@cuda.jit
def sobel_kernel(input, output_magnitude, output_angle):
    pass

@cuda.jit
def threshold_kernel(input, output, low, high):
    pass

@cuda.jit
def hysteresis_kernel(input, output, low, high):
    pass


def main():
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

    # Set the thread block size
    block_size = (args.tb, args.tb)

    # Compute the grid size
    grid_size = compute_thread_blocks(input_image, block_size)

    # Convert the image to black and white because we need for all the other operations
    s_image = cuda.to_device(input_image)
    d_image = cuda.device_array((input_image.shape[0], input_image.shape[1]), dtype=np.uint8)
    bw_kernel[grid_size, block_size](s_image, d_image)


    # apply the different kernel in fonction of the arguments
    if args.bw:
        bw_image = d_image.copy_to_host()
        bw_image = Image.fromarray(bw_image)
        bw_image.save(args.output)
        return
    if args.gauss:
        pass
    if args.sobel:
        pass
    if args.threshold:
        pass

if __name__ == '__main__':
    main()
