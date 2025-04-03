
import os
import shutil
from PIL import Image

def is_corrupt(image_path):
    """Check if an image is truncated, corrupted, or unreadable."""
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify the integrity of the image
            img.close()
        return False
    except Exception as e:
        print(f"Corrupt image found: {image_path}, Error: {e}")
        return True

def has_four_channels(image_path):
    """Check if an image has 4 channels (RGBA)."""
    try:
        with Image.open(image_path) as img:
            return img.mode == "RGBA"
    except Exception as e:
        print(f"Error checking channels for {image_path}: {e}")
        return False

def move_corrupt_images(source_folder, destination_folder):
    """Move corrupt images or images with 4 channels from source folder to destination folder."""
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    for filename in os.listdir(source_folder):
        file_path = os.path.join(source_folder, filename)
        if os.path.isfile(file_path) and (is_corrupt(file_path) or has_four_channels(file_path)):
            shutil.move(file_path, os.path.join(destination_folder, filename))
            print(f"Moved: {filename}")

# Example usage
source_directory = "/home/ml2/Desktop/Vscode/Background_removal/DIS/DIS5K/training/im"
destination_directory = "/home/ml2/Desktop/Vscode/Background_removal/DIS/DIS5K/training/imm"
move_corrupt_images(source_directory, destination_directory)