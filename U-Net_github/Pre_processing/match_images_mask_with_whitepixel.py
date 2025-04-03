import os
import shutil
from PIL import Image
import numpy as np


def copy_images_with_non_empty_masks(image_folder, mask_folder, new_image_folder, new_mask_folder):

    os.makedirs(new_image_folder, exist_ok=True)
    os.makedirs(new_mask_folder, exist_ok=True)

    # Iterate over mask files
    for mask_file in os.listdir(mask_folder):
        mask_path = os.path.join(mask_folder, mask_file)

        # Open the mask image
        with Image.open(mask_path) as mask_img:
            mask_array = np.array(mask_img)

            # Check if the mask contains at least one white pixel
            if np.any(mask_array > 0):  # Assuming white is represented by values greater than 0
                # Corresponding image file
                image_file = mask_file  # Adjust this if the naming convention differs
                image_path = os.path.join(image_folder, image_file)

                # Check if the corresponding image file exists
                if os.path.exists(image_path):
                    # Copy the image and mask to the new folders
                    shutil.copy(image_path, os.path.join(new_image_folder, image_file))
                    shutil.copy(mask_path, os.path.join(new_mask_folder, mask_file))


# Example usage
image_folder = 'input_images'
mask_folder = 'input_mask'
new_image_folder = 'output_images'
new_mask_folder = 'output_maskk'

# Call the function
copy_images_with_non_empty_masks(image_folder, mask_folder, new_image_folder, new_mask_folder)
