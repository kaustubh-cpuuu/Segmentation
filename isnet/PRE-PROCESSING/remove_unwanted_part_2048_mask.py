import os
import cv2
import numpy as np
from PIL import Image

# Define paths for the input, new image, and output folders
input_folder = "/home/ml2/Desktop/Vscode/Background_removal/DIS/demo_datasets/original_512"
new_image_folder = "/home/ml2/Desktop/Vscode/Background_removal/DIS/demo_datasets/original_mask_2048"  # Update with your new images path
output_folder = "demo_datasets/selection_masks"

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

# Get a list of all input mask files
input_mask_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]

for mask_file in input_mask_files:
    try:
        # Paths for the current mask and corresponding new image
        input_mask_path = os.path.join(input_folder, mask_file)
        new_mask_path = os.path.join(new_image_folder, mask_file)  # Assuming the same filename

        # Open the original mask and resize it
        mask = Image.open(input_mask_path).convert('L')
        resized_mask = mask.resize((1700, 1700))

        # Convert to NumPy array and create a binary mask
        resized_mask_np = np.array(resized_mask)
        binary_mask = np.where(resized_mask_np > 0, 255, 0).astype(np.uint8)

        # Dilation to expand the mask
        kernel = np.ones((10, 10), np.uint8)
        expanded_mask = cv2.dilate(binary_mask, kernel, iterations=1)

        # Load the corresponding new mask
        new_mask = cv2.imread(new_mask_path, cv2.IMREAD_UNCHANGED)
        
        # Ensure new mask is the same size as expanded mask
        if new_mask.shape[0] != 1700 or new_mask.shape[1] != 1700:
            new_mask = cv2.resize(new_mask, (1700, 1700))

        # Create a black image for the selection
        selection_mask = np.zeros_like(new_mask)

        # Copy the region inside the expanded mask from the new mask to the black image
        selection_mask[expanded_mask == 255] = new_mask[expanded_mask == 255]

        # Define the output path for the selection mask
        output_selection_path = os.path.join(output_folder, mask_file)

        # Save the selection result
        cv2.imwrite(output_selection_path, selection_mask)
        print(f"Selection mask saved at {output_selection_path}")

    except FileNotFoundError:
        print(f"Error: The file {input_mask_path} or {new_mask_path} was not found.")
    except Exception as e:
        print(f"An error occurred while processing {mask_file}: {e}")