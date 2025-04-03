import os
import shutil

# Define paths to image and mask folders
image_folder = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/giglo/training/im"
mask_folder = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/giglo/training/gt"

# Define paths to the directories where you want to save the selected images and masks
output_image_folder = r"C:\Users\ml2au\PycharmProjects\Background_removal\DIS\HBX_Full_body\uncropped\im_7000"
output_mask_folder = r"C:\Users\ml2au\PycharmProjects\Background_removal\DIS\HBX_Full_body\uncropped\gt_7000"
# Ensure the output folders exist, if not, create

os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_mask_folder, exist_ok=True)

image_files = sorted(os.listdir(image_folder))
# Get list of files in mask folder
mask_files = sorted(os.listdir(mask_folder))

# Ensure the number of images and masks are consistent
if len(image_files) != len(mask_files):
    raise ValueError("Number of images and masks don't match.")

# Select the images from 50 to 100 and corresponding masks
selected_images = image_files[6000:7386]
selected_masks = mask_files[6000:7386]

# Copy selected images to output image folde2
for img, msk in zip(selected_images, selected_masks):
    img_path = os.path.join(image_folder, img)
    msk_path = os.path.join(mask_folder, msk)

    # Ensure the corresponding mask exists
    if os.path.exists(msk_path):
        shutil.copy(img_path, os.path.join(output_image_folder, img))
        shutil.copy(msk_path, os.path.join(output_mask_folder, msk))
    else:
        print(f"Corresponding mask not found for image: {img}")

print("Images from 50 to 100 and their masks copied to", output_image_folder, "and", output_mask_folder)
