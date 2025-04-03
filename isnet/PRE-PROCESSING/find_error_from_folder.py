import os
import shutil
from PIL import Image, UnidentifiedImageError

# Set the path to the folder containing the images
input_folder_path = '/home/ml2/Desktop/Vscode/Background_removal/DIS/bg_removal_all_in_one/validation/gt'
output_folder_path = '/home/ml2/Desktop/Vscode/Background_removal/DIS/bg_removal_all_in_one/training/im_error'

# Create the output folder if it doesn't exist
os.makedirs(output_folder_path, exist_ok=True)

# Iterate through all files in the folder
for filename in os.listdir(input_folder_path):
    file_path = os.path.join(input_folder_path, filename)
    
    # Check if it's a file
    if os.path.isfile(file_path):
        try:
            # Attempt to open the image
            with Image.open(file_path) as img:
                img.verify()  # Verify that the image file is valid
        except UnidentifiedImageError:
            print(f"Unidentified image file found and will be moved: {filename}")
            # Move the file to the output folder
            shutil.move(file_path, os.path.join(output_folder_path, filename))
        except Exception as e:
            print(f"An error occurred with file {filename}: {e}")

print("Processing complete.")
