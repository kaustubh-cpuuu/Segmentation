import cv2
import numpy as np
import os
from multiprocessing import Pool

# Gamma correction function
def gamma_correction(image_path_and_output_path):
    image_path, output_path = image_path_and_output_path
    try:
        # Read the image in BGR format
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return
        
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize the image to [0, 1]
        image = image / 255.0
        
        # Apply gamma correction
        gamma = 5.2
        adjust_image = np.power(image, gamma)
        
        # Scale back to [0, 255] and convert to uint8
        adjust_image = (adjust_image * 255.0).astype(np.uint8)
        
        # Save the adjusted image
        cv2.imwrite(output_path, cv2.cvtColor(adjust_image, cv2.COLOR_RGB2BGR))
        print(f"Processed and saved: {output_path}")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# Process images in a folder
def process_images_in_folder(input_folder, output_folder, num_processes=4):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get all image paths in the input folder
    image_paths = [
        os.path.join(input_folder, filename)
        for filename in os.listdir(input_folder)
        if filename.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    # Create corresponding output paths
    output_paths = [
        os.path.join(output_folder, os.path.basename(image_path))
        for image_path in image_paths
    ]

    # Combine image paths and output paths
    image_path_and_output_path = list(zip(image_paths, output_paths))

    # Use multiprocessing to process images
    with Pool(processes=num_processes) as pool:
        pool.map(gamma_correction, image_path_and_output_path)

# Main function
if __name__ == "__main__":
    input_folder = "/home/ml2/Desktop/Vscode/Background_removal/DIS/demo_datasets/DOT"
    output_folder = "/home/ml2/Desktop/Vscode/Background_removal/DIS/demo_datasets/level"
    process_images_in_folder(input_folder, output_folder, num_processes=4)
    