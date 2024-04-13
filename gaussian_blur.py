import numpy as np
from PIL import Image
from numba import cuda
import math


"""
np.array([[1, 4, 6, 4, 1],
                            [4, 16, 24, 16, 4],
                            [6, 24, 36, 24, 6],
                            [4, 16, 24, 16, 4],
                            [1, 4, 6, 4, 1]])
"""
rgb_image = np.array(Image.open("./mona.jpeg"))
print(rgb_image.shape)


def generate_gaussian_kernel(width, sigma):
    kernel = np.zeros((width, width))
    center = width // 2
    if width % 2 == 0:
        raise ValueError("Width must be an odd number")
    for i in range(width):
        for j in range(width):
            kernel[i, j] = np.exp(-((i - center) ** 2 + (j - center) ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()

gaussian_kernel = generate_gaussian_kernel(25,120)

@cuda.jit
def apply_gaussian_kernel(input, output, kernel):
      x, y = cuda.grid(2)
      if x < input.shape[0] and y < input.shape[1]:
          for c in range(input.shape[2]):
            kernel_sum = 0
            weighted_sum = 0
            for a in range(kernel.shape[0]):
              for b in range(kernel.shape[1]):
                nx = x + a - kernel.shape[0] // 2
                ny = y + b - kernel.shape[1] // 2
                if nx >= 0 and ny >= 0 and nx < input.shape[0] and ny < input.shape[1]:  # Correction ici : vérification des limites de l'image
                    kernel_sum += kernel[a, b]
                    weighted_sum += kernel[a, b] * input[nx, ny, c]
            output[x,y,c] = weighted_sum // kernel_sum


def compute_thread_blocks(imagetab, block_size):
    height, width = imagetab.shape[:2]
    blockspergrid_x = math.ceil(width / block_size[0])
    blockspergrid_y = math.ceil(height / block_size[1])
    blockspergrid = (blockspergrid_y, blockspergrid_x)
    return blockspergrid


def call_gaussian_kernel():
    block_size = (32, 32)
    grid_size = compute_thread_blocks(rgb_image, block_size)

    #print("Size de la grid",grid_size)
    d_rgb_image = cuda.to_device(rgb_image)
    d_blurred_image = cuda.device_array((rgb_image.shape[0], rgb_image.shape[1], rgb_image.shape[2]), dtype=np.uint8)
    d_kernel = cuda.to_device(gaussian_kernel)

    # Call the kernel
    apply_gaussian_kernel[grid_size, block_size](d_rgb_image, d_blurred_image, d_kernel)
    cuda.synchronize()

    bw_image = d_blurred_image.copy_to_host()
    bw_image = Image.fromarray(bw_image)
    #bw_image.save("mona_blur.png")
    bw_image.save("mona_blur.jpeg")

call_gaussian_kernel()