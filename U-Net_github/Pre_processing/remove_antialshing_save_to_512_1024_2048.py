import os
from PIL import Image
import numpy as np

def process_image(original_image, size):
    # Resize the image
    resized_image = original_image.resize((size, size))

    alpha = resized_image.split()[-1]  # Extracting the alpha channel
    binary_mask = alpha.point(lambda p: 255 if p > 0 else 0)

    binary_mask_np = np.array(binary_mask) // 255  # Convert to binary (1 for white, 0 for black)
    binary_mask_np = binary_mask_np.astype(np.uint8)  # Ensure it's an 8-bit image

    # Convert from 1 to 255 for white
    binary_mask_img = Image.fromarray(binary_mask_np * 255, mode='L')
    return binary_mask_img

def process_folder(input_folder, output_folder):
    # Define the sizes for resizing
    sizes = [512, 1024, 2048]

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.png'):
            file_path = os.path.join(input_folder, file_name)
            original_image = Image.open(file_path)

            for size in sizes:
                processed_image = process_image(original_image, size)

                # Create output folder for each size if it doesn't exist
                current_output_folder = os.path.join(output_folder, str(size))
                if not os.path.exists(current_output_folder):
                    os.makedirs(current_output_folder)

                output_file_path = os.path.join(current_output_folder, file_name)
                processed_image.save(output_file_path)
                print(f'Processed and saved {file_name} at resolution {size}')

# Set your input and output folder paths
input_folder = 'iiiii'  # Replace with your input folder path
output_folder = 'output'  # Replace with your output folder path

process_folder(input_folder, output_folder)
