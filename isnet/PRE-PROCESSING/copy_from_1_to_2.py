import os
import shutil
from multiprocessing import Pool

def copy_image(file_path, destination_folder):
    """Copies an image file to the destination folder."""
    try:
        shutil.copy(file_path, destination_folder)
        print(f"Copied {file_path} to {destination_folder}")
    except Exception as e:
        print(f"Failed to copy {file_path}: {e}")

def main(source_folder, destination_folder, num_processes=None):
    """Copies all image files from source folder to destination folder using multiprocessing."""
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Get all image files in the source folder
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    image_files = [os.path.join(source_folder, f) for f in os.listdir(source_folder) 
                   if os.path.splitext(f)[1].lower() in image_extensions]

    print(f"Found {len(image_files)} image files to copy.")

    # Set up multiprocessing pool
    if num_processes is None:
        num_processes = os.cpu_count()  # Default to number of CPU cores

    with Pool(num_processes) as pool:
        pool.starmap(copy_image, [(file, destination_folder) for file in image_files])

    print("All images have been copied.")

if __name__ == "__main__":
    source_folder = "/home/ml2/Desktop/Vscode/Background_removal/DIS/bg_removal_all_in_one/validation/gtt"  # Replace with your source folder path
    destination_folder = "/home/ml2/Desktop/Vscode/Background_removal/DIS/bg_removal_all_in_one/validation/gt"  # Replace with your destination folder path

    main(source_folder, destination_folder)
