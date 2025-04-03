import os
import shutil

parent_folder = r"E:\Kaustubh\All_Raw_data\Multimodal_multiclass\All_in_one_multimodel\14_clients\multi_modal\158"  # Change this to the path of your parent folder
before_dir = r"C:\Users\ml2au\PycharmProjects\Background_removal\DIS\shoes\otherclient\158\im"  # Change this to your desired directory for before.png images
top_dir = r"C:\Users\ml2au\PycharmProjects\Background_removal\DIS\shoes\otherclient\158\gt"  # Change this to your desired directory for top.png images

# Create the directories if they do not exist
os.makedirs(before_dir, exist_ok=True)
os.makedirs(top_dir, exist_ok=True)

for subdir, dirs, files in os.walk(parent_folder):
    for file in files:
        # Build the source path
        src_path = os.path.join(subdir, file)

        # Determine if it's a 'before.jpg', 'top.png', or 'tshirt.png' file
        if file == 'Background.jpg' :
            dest_path = os.path.join(before_dir, os.path.basename(subdir) + '.jpg')
            shutil.copy2(src_path, dest_path)
            print(f'Copied {src_path} to {dest_path}')

        elif file == 'Shoes.png' :
            dest_path = os.path.join(top_dir, os.path.basename(subdir)  + '.png')
            shutil.copy2(src_path, dest_path)
            print(f'Copied {src_path} to {dest_path}')




