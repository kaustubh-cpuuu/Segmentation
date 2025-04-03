import os
import shutil

parent_folder = r"D:\Kaustubh\HBX_For_binary_multimodal\RAW_HBX\batch_3_4_1600"  # Change this to the path of your parent folder
before_dir = r'C:\Users\ml2au\PycharmProjects\Background_removal\DIS\1600_top\im'  # Change this to your desired directory for before.png images
top_dir = r'C:\Users\ml2au\PycharmProjects\Background_removal\DIS\1600_top\gt'  # Change this to your desired directory for top.png images

# Create the directories if they do not exist
os.makedirs(before_dir, exist_ok=True)
os.makedirs(top_dir, exist_ok=True)

for subdir, dirs, files in os.walk(parent_folder):
    # Check if 'upperweare.png' is present in the current folder
    upperwear_present = 'upperweare.png' in files

    for file in files:
        # Build the source path
        src_path = os.path.join(subdir, file)

        # Determine if it's a 'before.jpg' or 'before.png' file
        if file == 'before.png' or file == 'before.jpg':
            dest_path = os.path.join(before_dir, os.path.basename(subdir) + '.jpg')
            shutil.copy2(src_path, dest_path)
            print(f'Copied {src_path} to {dest_path}')
        # Check if 'upperweare.png' is present and copy it to 'top_dir'
        elif file == 'upperwaer.png' or  file == 'upperwear.png' or file == 'upperweare.png':
            dest_path = os.path.join(top_dir, os.path.basename(subdir) + '.png')
            shutil.copy2(src_path, dest_path)
            print(f'Copied {src_path} to {dest_path}')


        # If 'upperweare.png' has been copied, skip 'top.png', 'shirt.png', and 'jacket.png'
        elif upperwear_present and (file == 'top.png' or file == 'shirt.png' or file == 'jacket.png' or  file == 'Top.png' or file == 'hoddie.png' or file == 'hoodie.png'
        or file == 'sweater.png' or file == 't shirt.png' or file == 't-shairt.png' or  file == 't-shirt-.png' or file == 't-shirt.png' or file == '-t-shirt.png'
        or file == 'tshart.png' or file == 'tshirt.png' or file == 'vest.png'):
            continue


        # Copy 'top.png', 'shirt.png', or 'jacket.png' if 'upperweare.png' hasn't been copied
        elif file == 'top.png' or file == 'shirt.png' or file == 'jacket.png'   or file == 'TOP.png' or file == 'hoddie.png' or file == 'hoodie.png'\
                or file == 'sweater.png' or file == 't shirt.png' or file == 't-shairt.png'   or file == 't-shirt-.png' or file == 't-shirt.png' or file == '-t-shirt.png'\
                or file == 'tshart.png' or file == 'tshirt.png' or file == 'vest.png':


            dest_path = os.path.join(top_dir, os.path.basename(subdir) + '.png')
            shutil.copy2(src_path, dest_path)
            print(f'Copied {src_path} to {dest_path}')





