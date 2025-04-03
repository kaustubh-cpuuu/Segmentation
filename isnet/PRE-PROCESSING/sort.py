import os
import shutil


def rename_and_copy_images(root_folder):
    # Iterate over each folder in the root folder
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)

        # Check if it's a directory
        if os.path.isdir(folder_path):
            # Iterate over each file in the folder
            for filename in os.listdir(folder_path):
                if filename.endswith('.png'):
                    # Construct new filename
                    new_filename = f"{folder_name}_{filename}"

                    # Rename the file
                    os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))

                    # Copy images to different paths
                    if filename == 'Background.png':
                        shutil.copy(os.path.join(folder_path, new_filename), 'webstart_output/original')
                    elif filename == 'j.png':
                        shutil.copy(os.path.join(folder_path, new_filename), 'webstart_output/jacket')
                    elif filename == 't.png':
                        shutil.copy(os.path.join(folder_path, new_filename), 'webstart_output/tshirt')


# Replace 'root_folder' with the path to your parent folder containing all the child folders
rename_and_copy_images('webster_png')
