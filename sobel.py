import numpy as np
from PIL import Image
from numba import cuda
import math
import cv2

sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])

blur_image = np.array(Image.open("./mona_blur.jpeg"))

@cuda.jit
def apply_sobel_kernel(input, output_magnitude, output_angle):
    x, y = cuda.grid(2)
    if x < input.shape[0] and y < input.shape[1]:
        Gx = 0
        Gy = 0
        for c in range(input.shape[2]):
            for i in range(-1, 2):
                for j in range(-1, 2):
                    nx = x + i
                    ny = y + j
                    if nx >= 0 and ny >= 0 and nx < input.shape[0] and ny < input.shape[1]:
                        Gx += input[nx, ny, c] * sobel_x[i + 1, j + 1]
                        Gy += input[nx, ny, c] * sobel_y[i + 1, j + 1]
        output_magnitude[x, y] = math.sqrt(Gx ** 2 + Gy ** 2)
        output_angle[x, y] = math.atan2(Gy, Gx)

def compute_thread_blocks(imagetab, block_size):
    height, width = imagetab.shape[:2]
    blockspergrid_x = math.ceil(width / block_size[0])
    blockspergrid_y = math.ceil(height / block_size[1])
    blockspergrid = (blockspergrid_y, blockspergrid_x)
    return blockspergrid

def call_sobel_kernel():
    block_size = (32, 32)
    grid_size = compute_thread_blocks(blur_image, block_size)

    d_blur_image = cuda.to_device(blur_image)
    d_magnitude = cuda.device_array((blur_image.shape[:2]), dtype=np.float32)
    d_angle = cuda.device_array((blur_image.shape[:2]), dtype=np.float32)

    apply_sobel_kernel[grid_size, block_size](d_blur_image, d_magnitude, d_angle)
    cuda.synchronize()

    magnitude = d_magnitude.copy_to_host()
    angle = d_angle.copy_to_host()

    return magnitude, angle

def apply_sobel_to_image(image):

    _, binary_magnitude = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)


magnitude, angle = call_sobel_kernel()

# on enregistre les images
magnitude = Image.fromarray((magnitude * 255).astype(np.uint8))

magnitude.save("mona_magnitude.jpeg")
angle.save("mona_angle.jpeg")



