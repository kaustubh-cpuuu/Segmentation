
import os
import shutil
import random

def copy_same_basename_different_extension(input_folder1, input_folder2, output_folder1, output_folder2, num_images=5500):
    # Create the output folders if they don't exist
    os.makedirs(output_folder1, exist_ok=True)
    os.makedirs(output_folder2, exist_ok=True)
    
    # Get a list of all image files in the first input folder
    image_files1 = [f for f in os.listdir(input_folder1) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    image_files2 = [f for f in os.listdir(input_folder2) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    # Create a set of basenames for the images in the first folder
    basenames1 = {os.path.splitext(f)[0]: f for f in image_files1}
    
    # Create a set of basenames for the images in the second folder
    basenames2 = {os.path.splitext(f)[0]: f for f in image_files2}
    
    # Find common basenames in both folders
    common_basenames = set(basenames1.keys()).intersection(basenames2.keys())

    # If there are fewer images than num_images, adjust num_images to the available number
    num_images = min(num_images, len(common_basenames))

    # Select the same random basenames
    selected_basenames = random.sample(common_basenames, num_images)

    # Copy the selected images from the first input folder to the first output folder
    for basename in selected_basenames:
        src_path1 = os.path.join(input_folder1, basenames1[basename])
        dest_path1 = os.path.join(output_folder1, basenames1[basename])

        try:
            shutil.copy(src_path1, dest_path1)
            print(f"Copied {basenames1[basename]} to {output_folder1}")
        except Exception as e:
            print(f"Error copying {basenames1[basename]} to {output_folder1}: {e}")

    # Copy the same selected images from the second input folder to the second output folder
    for basename in selected_basenames:
        src_path2 = os.path.join(input_folder2, basenames2[basename])
        dest_path2 = os.path.join(output_folder2, basenames2[basename])

        try:
            shutil.copy(src_path2, dest_path2)
            print(f"Copied {basenames2[basename]} to {output_folder2}")
        except Exception as e:
            print(f"Error copying {basenames2[basename]} to {output_folder2}: {e}")

    print(f"Copied {num_images} images to {output_folder1} and {output_folder2}")

# Example usage
input_folder1 = '/home/ml2/Desktop/Vscode/Background_removal/DIS/bg_removal_all_in_one/validation/im'
input_folder2 = '/home/ml2/Desktop/Vscode/Background_removal/DIS/bg_removal_all_in_one/validation/gt'

output_folder1 = '/home/ml2/Desktop/Vscode/Background_removal/DIS/bg_removal_all_in_one/training/im'
output_folder2 = '/home/ml2/Desktop/Vscode/Background_removal/DIS/bg_removal_all_in_one/training/gt'

copy_same_basename_different_extension(input_folder1, input_folder2, output_folder1, output_folder2)
