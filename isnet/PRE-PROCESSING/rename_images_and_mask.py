import os
import shutil

def rename_and_copy_files(image_dir, mask_dir, new_name_prefix, output_image_dir, output_mask_dir):
    # Ensure the provided directories exist
    if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
        raise ValueError("Provided image or mask directories do not exist.")
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
    if not os.path.exists(output_mask_dir):
        os.makedirs(output_mask_dir)

    # Get the list of files in the directories
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))

    # Ensure we have the same number of images and masks
    if len(image_files) != len(mask_files):
        raise ValueError("The number of images and masks do not match.")

    # Rename and copy files
    for i, (image_file, mask_file) in enumerate(zip(image_files, mask_files)):
        # Construct the new name
        new_name = f"{new_name_prefix}_{i+1}"

        # Construct full paths
        old_image_path = os.path.join(image_dir, image_file)
        old_mask_path = os.path.join(mask_dir, mask_file)

        new_image_name = f"{new_name}.jpg"
        new_mask_name = f"{new_name}.png"

        new_image_path = os.path.join(output_image_dir, new_image_name)
        new_mask_path = os.path.join(output_mask_dir, new_mask_name)

        # Copy and rename files to the output directories
        shutil.copy(old_image_path, new_image_path)
        shutil.copy(old_mask_path, new_mask_path)

        print(f"Copied '{image_file}' to '{new_image_path}'")
        print(f"Copied '{mask_file}' to '{new_mask_path}'")

# Example usage
image_directory =  r"/home/ml2/Desktop/Vscode/Background_removal/DIS/multiclass/onepiece/imm"
mask_directory = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/multiclass/onepiece/gtt"
prefix = 'onepiece'
output_image_directory =r"/home/ml2/Desktop/Vscode/Background_removal/DIS/2_classes_bottom_onepiece/training/im"
output_mask_directory =r"/home/ml2/Desktop/Vscode/Background_removal/DIS/2_classes_bottom_onepiece/training/gt"

rename_and_copy_files(image_directory, mask_directory, prefix, output_image_directory, output_mask_directory)




