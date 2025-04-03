import os
import random

def rename_and_move_subfolders(parent_folder_path, output_folder_path):
    """
    Renames each subfolder in the parent folder by appending a random number to its name,
    and moves them to the output parent folder.
    """
    if not os.path.exists(parent_folder_path):
        print(f"The parent folder '{parent_folder_path}' does not exist.")
        return

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
        print(f"Created output folder: {output_folder_path}")

    for folder_name in os.listdir(parent_folder_path):
        folder_path = os.path.join(parent_folder_path, folder_name)

        # Check if it's a directory
        if os.path.isdir(folder_path):
            random_number = random.randint(1000, 9999)  # Generate a random number between 1000 and 9999
            new_folder_name = f"{folder_name}_{random_number}"
            new_folder_path = os.path.join(output_folder_path, new_folder_name)

            # Move and rename the subfolder
            os.rename(folder_path, new_folder_path)
            print(f"Subfolder '{folder_name}' moved and renamed to: {new_folder_name}")


# Example usage
parent_folder = "New folder"  # Replace with the path to your parent folder
output_folder = "extra_output"  # Replace with the path to your output folder
rename_and_move_subfolders(parent_folder, output_folder)
