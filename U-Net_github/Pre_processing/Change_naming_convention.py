import os

def rename_images_in_folder(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return

    for filename in os.listdir(folder_path):
        # Constructing the old file path
        old_file_path = os.path.join(folder_path, filename)

        # Check if it's a file and not a directory
        if os.path.isfile(old_file_path):
            # Split the file name and its extension
            file_base, file_extension = os.path.splitext(filename)

            # Add underscore to the end of the basename
            new_filename = file_base + "_" + file_extension
            new_file_path = os.path.join(folder_path, new_filename)

            # Renaming the file
            os.rename(old_file_path, new_file_path)
            print(f"Renamed {filename} to {new_filename}")

# Replace these with your actual folder paths
folder_path_1 = "image_fol"
folder_path_2 = "image_mask"

# Renaming images in both folders
rename_images_in_folder(folder_path_1)
rename_images_in_folder(folder_path_2)
