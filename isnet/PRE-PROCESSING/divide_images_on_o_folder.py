import os
import random
import shutil

# Define the source folder and the number of folders to divide into
source_folder = '/home/ml2/Desktop/Vscode/Background_removal/DIS/white_images/olivila/im_256_new'  # Replace with your image folder path
num_folders = 5
output_folders = [f'folder_{i+1}' for i in range(num_folders)]

# Create the output folders if they don't exist
for folder in output_folders:
    os.makedirs(folder, exist_ok=True)

# Get all the image files in the source folder
image_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

# Shuffle the image files randomly
random.shuffle(image_files)

# Divide the image files into the folders
for i, image_file in enumerate(image_files):
    folder = output_folders[i % num_folders]  # Distribute images evenly
    shutil.copy(os.path.join(source_folder, image_file), os.path.join(folder, image_file))

# Print the path of the folders where the images have been saved
for folder in output_folders:
    print(f"Images have been saved in: {os.path.abspath(folder)}")

print("Images have been randomly divided into 6 folders.")
