from PIL import Image
import cv2
import os

def process_images(folder_path, output_folder):
    """
    Process images in the given folder and save them in the output folder.
    Converts RGBA images to RGB and strips incorrect ICC profiles.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            # Open the image using Pillow to strip the ICC profile
            with Image.open(image_path) as img:
                if 'icc_profile' in img.info:
                    img.info.pop('icc_profile')  # Remove the ICC profile
                img.convert('RGB').save(output_path, 'JPEG')  # Convert to RGB and save

            # Read the image using OpenCV
            image = cv2.imread(output_path, cv2.IMREAD_UNCHANGED)

            if image is not None and image.shape[2] == 4:  # Check if image has 4 channels (RGBA)
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

            cv2.imwrite(output_path, image)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Paths to the folders
image_folder = 'Jwellery_dataset/DIS5K/DIS-VD/im'
mask_folder = 'Jwellery_dataset/DIS5K/DIS-VD/gt'
output_image_folder = 'Jwellery_dataset/DIS5K/DIS-VD/out_image'
output_mask_folder = 'Jwellery_dataset/DIS5K/DIS-VD/out_mask'


# Process the images and masks
process_images(image_folder, output_image_folder)
process_images(mask_folder, output_mask_folder)

