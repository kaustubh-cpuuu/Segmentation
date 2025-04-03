
import os
from PIL import Image
import shutil
import multiprocessing

def resize_image_and_mask(image_path, mask_path, save_image_path, save_mask_path, max_size=256):
    
    try:
        # Open the image and mask
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        # Ensure images are fully loaded (helps detect truncated files early)
        image.load()
        mask.load()

        # Get original image dimensions
        width, height = image.size

        # Skip processing if the shortest side is less than or equal to max_size
        if min(width, height) <= max_size:
            print(f"Skipped (short side ≤ max_size): {image_path} and its mask.")
            return

        # Function to calculate new dimensions keeping aspect ratio
        def calculate_new_size(width, height, max_size):
            scale = max_size / min(width, height)
            return int(width * scale), int(height * scale)

        # Calculate new dimensions
        new_size = calculate_new_size(width, height, max_size)

        # Resize and save the image and mask
        resized_image = image.resize(new_size, Image.Resampling.LANCZOS)
        resized_mask = mask.resize(new_size, Image.Resampling.LANCZOS)
        resized_image.save(save_image_path)
        resized_mask.save(save_mask_path)
        print(f"Processed (resized): {image_path} and its mask.")

    except (OSError, IOError) as e:
        print(f"Error processing {image_path} or {mask_path}: {e}")



def process_image_mask_pair(image_name, image_folder, mask_folder, save_image_folder, save_mask_folder, max_size=256):
    """
    Processes an individual image and its corresponding mask.
    """
    # Construct the paths for the image and mask
    image_path = os.path.join(image_folder, image_name)
    mask_name = image_name.replace('.jpg', '.png')  # Assuming mask filenames match image names with a .png extension
    mask_path = os.path.join(mask_folder, mask_name)

    # Construct the paths for saving resized images and masks
    save_image_path = os.path.join(save_image_folder, image_name)
    save_mask_path = os.path.join(save_mask_folder, mask_name)

    if os.path.exists(mask_path):
        resize_image_and_mask(image_path, mask_path, save_image_path, save_mask_path, max_size)
    else:
        print(f"Mask for {image_name} not found.")


def process_folders(image_folder, mask_folder, save_image_folder, save_mask_folder, max_size=256):
    """
    Processes all images and their masks in the specified folders.
    Uses multiprocessing for efficiency.
    """
    # Create the save directories if they don't exist
    os.makedirs(save_image_folder, exist_ok=True)
    os.makedirs(save_mask_folder, exist_ok=True)

    # Get a list of image files in the image folder
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

    # Create a pool of worker processes
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.starmap(
            process_image_mask_pair,
            [
                (image_name, image_folder, mask_folder, save_image_folder, save_mask_folder, max_size)
                for image_name in image_files
            ]
        )


if __name__ == "__main__":
    image_folder = "/home/ml2/Desktop/Vscode/Background_removal/DIS/element_remove_256/Image"
    mask_folder = "/home/ml2/Desktop/Vscode/Background_removal/DIS/element_remove_256/Mask"

    save_image_folder = "/home/ml2/Desktop/Vscode/Background_removal/DIS/element_remove_256/New_256/im"
    save_mask_folder = "/home/ml2/Desktop/Vscode/Background_removal/DIS/element_remove_256/New_256/gt"

    print("Starting processing...")
    process_folders(image_folder, mask_folder, save_image_folder, save_mask_folder)
    print("Processing complete.")
