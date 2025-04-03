import os
import shutil

parent_folder = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/backdrop"  # Change this to the path of your parent folder
before_dir = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/backdrop_dataset/im"
top_dir = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/backdrop_dataset/gt"

# Create the directories if they do not exist
os.makedirs(before_dir, exist_ok=True)
os.makedirs(top_dir, exist_ok=True)

for subdir, dirs, files in os.walk(parent_folder):
    for file in files:
        # Build the source path
        src_path = os.path.join(subdir, file)

        # Determine if it's a 'before.jpg', 'top.png', or 'tshirt.png' file
        if file == 'Background.jpg':
            dest_path = os.path.join(before_dir, os.path.basename(subdir) + '.jpg')
            shutil.copy2(src_path, dest_path)
            print(f'Copied {src_path} to {dest_path}')

        elif file == 'Layer-1.png' or file == 'Color-Fill-1.png':
            dest_path = os.path.join(top_dir, os.path.basename(subdir)  + '.png')
            shutil.copy2(src_path, dest_path)
            print(f'Copied {src_path} to {dest_path}') 




