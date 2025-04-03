import cv2
import numpy as np
import os

# RGB mapping for the classes
classes_rgb_mapping = {
    "skin": (227,155,0),
    "nose": (227,155,0),
    "eye_g": (0, 255, 0),
    "l_eye": (0, 128, 0),
    "r_eye": (0, 128, 0),
    "l_brow": (255, 0, 255),
    "r_brow": (255, 0, 255),
    "l_ear": (227,155,0),
    "r_ear": (227,155,0),
    "mouth": (0, 255, 222),
    "u_lip": (84, 255, 169),
    "l_lip": (84, 255, 169),
    "hair": (0, 0, 255),
    "hat": (53, 111, 216),
    "ear_r": (227,155,0),
    "neck_l": (227,155,0),
    "neck": (227,155,0),
    "cloth": (0, 85, 255),
    "necklace": (227,155,0),
}

def combine_masks_for_image(base_filename, directory, output_directory):
    # Adjusted the mask filenames format to: base_filename_classname.png
    mask_files = [f"{directory}/{base_filename}_{classname}.png" for classname in classes_rgb_mapping.keys()]

    # Initialize an RGB image with zeros
    combined_mask = np.zeros((512, 512, 3), dtype=np.uint8)

    for mask_file, (classname, color) in zip(mask_files, classes_rgb_mapping.items()):
        if os.path.exists(mask_file):
            # Load the binary mask
            binary_mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            # Wherever the mask is activated, set the corresponding RGB color
            combined_mask[binary_mask > 0] = color

    # Save the combined RGB mask
    output_path = os.path.join(output_directory, f"{base_filename}.png")
    cv2.imwrite(output_path, combined_mask)

def process_directory(directory, output_directory):
    # Get a list of unique base filenames (without the class suffixes)
    filenames = [f.split('_')[0] for f in os.listdir(directory) if f.endswith('.png')]
    unique_base_filenames = set(filenames)

    for base_filename in unique_base_filenames:
        combine_masks_for_image(base_filename, directory, output_directory)

# Example usage:
process_directory("0", "out")