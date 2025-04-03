import os
from PIL import Image
import numpy as np


def process_image(image_path, output_path):
    image = Image.open(image_path)
    alpha = image.split()[-1]
    binary_mask = alpha.point(lambda p: p > 0 and 255)

    binary_mask_np = np.array(binary_mask)
    binary_mask_np[binary_mask_np == 255] = 1  # Convert to binary (1 for white_images, 0 for black)
    binary_mask_np = binary_mask_np.astype(np.uint8)  # Make sure it's an 8-bit image

    binary_mask_img = Image.fromarray(binary_mask_np * 255)  # Convert from 1 to 255 for white_images
    binary_mask_img.save(output_path)


def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.png'):
            file_path = os.path.join(input_folder, file_name)
            output_file_path = os.path.join(output_folder, file_name)
            process_image(file_path, output_file_path)
            print(f'Processed {file_name}')


# Set your input and output folder paths
input_folder = 'Product/mask'
output_folder = 'Product/white_mask'

process_folder(input_folder, output_folder)
