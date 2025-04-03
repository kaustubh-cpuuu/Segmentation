import os
from shutil import copyfile

def rename_and_save_images(input_dir, output_dir, suffix='_+30'):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over each file in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # Construct the new filename
            base_name, file_extension = os.path.splitext(filename)
            new_filename = base_name + suffix + file_extension

            # Construct the full input and output paths
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, new_filename)

            # Copy and rename the file to the output directory
            copyfile(input_path, output_path)

# Example usage
input_directory = "216_images/validation/gt"
output_directory = "216_images/validation/gt_bright"
rename_and_save_images(input_directory, output_directory)




