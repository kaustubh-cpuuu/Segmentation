import os
import shutil
import random
from multiprocessing import Pool, cpu_count

def copy_image_task(args):
    """Helper function to copy a single image."""
    src_path, dest_path = args
    try:
        shutil.copy(src_path, dest_path)
        return f"Copied: {os.path.basename(src_path)}"
    except Exception as e:
        return f"Error copying {os.path.basename(src_path)}: {e}"

def copy_random_images(input_folder, output_folder, num_images=15000):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get a list of all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    # If there are fewer images than num_images, adjust num_images to the available number
    num_images = min(num_images, len(image_files))

    # Select random images
    selected_images = random.sample(image_files, num_images)

    # Prepare arguments for multiprocessing
    copy_tasks = [
        (os.path.join(input_folder, image), os.path.join(output_folder, image))
        for image in selected_images
    ]

    # Use multiprocessing to copy images
    with Pool(cpu_count()) as pool:
        results = pool.map(copy_image_task, copy_tasks)

    # Print the results
    for result in results:
        print(result)

    print(f"Copied {num_images} images to {output_folder} using multiprocessing.")

# Example usage
if __name__ == "__main__":
    input_folder = r'/mnt/qnap_share/organized/livestock/before'
    output_folder = r'/home/ml2/Desktop/Vscode/Background_removal/DIS/white_images/ls/im'
    copy_random_images(input_folder, output_folder)



