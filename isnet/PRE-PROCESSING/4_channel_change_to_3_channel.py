from PIL import Image
import os
from multiprocessing import Pool, cpu_count

def process_image(args):
    """
    Process a single image: converts RGBA images to RGB and strips incorrect ICC profiles.
    
    Args:
        args (tuple): A tuple containing image_path and output_path.
    """
    image_path, output_path = args

    try:
        # Open the image using Pillow
        with Image.open(image_path) as img:
            if img.mode == 'RGBA':
                print(f"{os.path.basename(image_path)} has 4 channels (RGBA)")

            # Strip the ICC profile if present
            if 'icc_profile' in img.info:
                img.info.pop('icc_profile')

            # Convert to RGB and save as JPEG
            img.convert('RGB').save(output_path, 'JPEG')

    except Exception as e:
        print(f"Error processing {os.path.basename(image_path)}: {e}")

def process_images(folder_path, output_folder):
    """
    Process images in the given folder and save them in the output folder.
    Uses multiprocessing to process multiple images concurrently.
    
    Args:
        folder_path (str): Path to the folder containing input images.
        output_folder (str): Path to the folder to save processed images.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create a list of image paths and corresponding output paths
    image_paths = [(os.path.join(folder_path, filename), os.path.join(output_folder, filename))
                   for filename in os.listdir(folder_path)
                   if os.path.isfile(os.path.join(folder_path, filename))]

    # Use a Pool to process images concurrently
    with Pool(processes=cpu_count()) as pool:
        pool.map(process_image, image_paths)

# Paths to the folders
image_folder = r'/home/ml2/Desktop/Vscode/Background_removal/DIS/demo_datasets/trial_filter_new'
output_image_folder = r'/home/ml2/Desktop/Vscode/Background_removal/DIS/demo_datasets/trial_filter_new'

# Process the images
if __name__ == "__main__":
    process_images(image_folder, output_image_folder)
