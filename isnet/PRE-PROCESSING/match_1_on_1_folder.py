import os
import shutil
from glob import glob
from multiprocessing import Pool, cpu_count

def move_matched_images(args):
    """Helper function to move matched images to unmatched folder."""
    name, image_path, unmatched_folder_path = args
    shutil.move(image_path, unmatched_folder_path)

def match_and_remove_images_by_name(folder1, folder2, unmatched_folder_path):

    # Ensure the unmatched folder exists
    os.makedirs(unmatched_folder_path, exist_ok=True)

    # Get list of image files from both directories
    folder1_images = glob(os.path.join(folder1, '*'))
    folder2_images = glob(os.path.join(folder2, '*'))

    # Extract filenames without extension
    folder1_names = {os.path.splitext(os.path.basename(img))[0]: img for img in folder1_images}
    folder2_names = {os.path.splitext(os.path.basename(img))[0]: img for img in folder2_images}

    # Find common image names (those that exist in both folders)
    common_names = set(folder1_names.keys()) & set(folder2_names.keys())

    # Prepare arguments for multiprocessing to move the matched images
    args = [(name, folder1_names[name], unmatched_folder_path) for name in common_names]

    # Use multiprocessing to move matched images
    with Pool(cpu_count()) as pool:
        pool.map(move_matched_images, args)

    return len(common_names)

# Example usage
unmatched_folder_path = 'unmatched'  # Update with the actual path

matched_images_moved = match_and_remove_images_by_name(
    r"/home/ml2/Desktop/Vscode/Background_removal/DIS/backdrop_dataset/im",
    r"/home/ml2/Desktop/Vscode/Background_removal/DIS/backdrop_dataset/gt",
    unmatched_folder_path
)

print(f"Moved {matched_images_moved} matched images to {unmatched_folder_path}.")
