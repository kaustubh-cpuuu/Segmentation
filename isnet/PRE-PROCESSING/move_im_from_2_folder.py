import os
import shutil

# Define folder paths
first_folder = "/home/ml2/Desktop/Vscode/Background_removal/DIS/bg_removal_all_in_one/validation/im"
second_folder = "/home/ml2/Desktop/Vscode/Background_removal/DIS/bg_removal_all_in_one/training/Extra"
output_folder = "/home/ml2/Desktop/Vscode/Background_removal/DIS/bg_removal_all_in_one/training/imm"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Get base names from second folder
second_folder_basenames = {os.path.splitext(f)[0] for f in os.listdir(second_folder) if f.lower().endswith(('.jpg', '.png'))}

# Move matching images from first folder to output folder
for file in os.listdir(first_folder):
    name, ext = os.path.splitext(file)
    if name in second_folder_basenames:
        src = os.path.join(first_folder, file)
        dst = os.path.join(output_folder, file)
        shutil.move(src, dst)
        print(f"Moved: {file}")

print("Process completed!")
