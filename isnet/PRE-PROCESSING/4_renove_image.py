import os
import shutil
from PIL import Image
from multiprocessing import Pool, cpu_count

def process_image(args):
    """
    Process a single image to check its channels and move it if it has 4 channels.

    Args:
        args (tuple): Tuple containing input_path and output_folder_4_channels.

    Returns:
        str: Status message about the processing.
    """
    input_path, output_folder_4_channels = args
    filename = os.path.basename(input_path)

    try:
        # Open the image and check its mode (channels)
        with Image.open(input_path) as img:
            num_channels = len(img.getbands())

            if num_channels == 4:
                # Move to 4-channel output folder
                shutil.move(input_path, os.path.join(output_folder_4_channels, filename))
                return f"Moved {filename}: 4-channel image."
            else:
                return f"Skipped {filename}: Not a 4-channel image ({num_channels} channels)."

    except Exception as e:
        return f"Error processing {filename}: {e}"

def move_images_by_channel(input_folder, output_folder_4_channels):
    """
    Identifies images with 4 channels and moves them to the specified folder using multiprocessing.

    Args:
        input_folder (str): Path to the folder containing input images.
        output_folder_4_channels (str): Path to store images with 4 channels (e.g., RGBA).
    """
    # Create output folder if it does not exist
    os.makedirs(output_folder_4_channels, exist_ok=True)

    # Prepare a list of image paths
    image_paths = [os.path.join(input_folder, filename) for filename in os.listdir(input_folder)]

    # Prepare arguments for multiprocessing
    args = [(path, output_folder_4_channels) for path in image_paths]

    # Use multiprocessing to process images
    with Pool(cpu_count()) as pool:
        results = pool.map(process_image, args)

    # Print results
    for result in results:
        print(result)

# Example usage
input_folder = "/home/ml2/Desktop/Vscode/Background_removal/DIS/2_classes/training/im"
output_folder_4_channels = "/home/ml2/Desktop/Vscode/Background_removal/DIS/2_classes/training/imm"

move_images_by_channel(input_folder, output_folder_4_channels)