
from PIL import Image
import os

def remove_anti_aliasing(image_path, save_path, threshold=128):
    # Load the image and convert it to grayscale
    image = Image.open(image_path).convert("L")

    # Process the image by applying a threshold
    processed_image = image.point(lambda p: 255 if p > threshold else 0)

    # Save the processed image
    processed_image.save(save_path)

    return save_path

def process_directory(directory_path, output_folder):
    # Ensure the output folder exists, create if it doesn't
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(directory_path):
        if filename.endswith(".png"):  # Adjust based on your file types
            image_path = os.path.join(directory_path, filename)
            save_path = os.path.join(output_folder, filename)  # Use original filename in output folder
            remove_anti_aliasing(image_path, save_path, threshold=128)
            print(f"Processed and saved to output folder: {filename}")

# Example usage - specify the paths to your folder and output folder
directory_path = "webstar_data/gtt"  # Adjust this to your directory
output_folder = "webstar_data/gttt"  # Adjust to your desired output directory
process_directory(directory_path, output_folder)
