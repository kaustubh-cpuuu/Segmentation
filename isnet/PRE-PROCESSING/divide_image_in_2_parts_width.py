from PIL import Image
import os

def crop_image_and_mask(image_path, mask_path, save_dir_images, save_dir_masks):
    # Load image and mask
    image = Image.open(image_path)
    mask = Image.open(mask_path)

    # Calculate the midpoint of the width
    width, height = image.size
    midpoint = width // 2
    box1 = (0, 0, midpoint, height)
    box2 = (midpoint, 0, width, height)
    

    # Crop the image and mask
    image1, image2 = image.crop(box1), image.crop(box2)
    mask1, mask2 = mask.crop(box1), mask.crop(box2)

    # Save the cropped images and masks
    base_filename = os.path.basename(image_path)
    filename_without_ext = os.path.splitext(base_filename)[0]

    image1.save(os.path.join(save_dir_images, f"{filename_without_ext}_1.jpg"))
    image2.save(os.path.join(save_dir_images, f"{filename_without_ext}_2.jpg"))
    mask1.save(os.path.join(save_dir_masks, f"{filename_without_ext}_1.png"))
    mask2.save(os.path.join(save_dir_masks, f"{filename_without_ext}_2.png"))

def process_folder(images_dir, masks_dir, save_dir_images, save_dir_masks):
    # Make sure the save directories exist
    os.makedirs(save_dir_images, exist_ok=True)
    os.makedirs(save_dir_masks, exist_ok=True)

    # Process each image in the images directory
    for image_filename in os.listdir(images_dir):
        image_path = os.path.join(images_dir, image_filename)

        # Construct the corresponding mask path with PNG extension
        filename_without_ext = os.path.splitext(image_filename)[0]
        mask_path = os.path.join(masks_dir, f"{filename_without_ext}.png")

        # Check if the mask file exists
        if not os.path.exists(mask_path):
            print(f"Mask for {image_filename} not found, skipping.")
            continue

        print(f"Processing {image_filename}...")
        crop_image_and_mask(image_path, mask_path, save_dir_images, save_dir_masks)

# Define your directories
images_dir = r"C:\Users\ml2au\PycharmProjects\Background_removal\DIS\bg_product_test\im"
masks_dir = r"C:\Users\ml2au\PycharmProjects\Background_removal\DIS\bg_product_test\gt"
save_dir_images = r"C:\Users\ml2au\PycharmProjects\Background_removal\DIS\bg_product_test\im_height"
save_dir_masks = r"C:\Users\ml2au\PycharmProjects\Background_removal\DIS\bg_product_test\gt_height"

process_folder(images_dir, masks_dir, save_dir_images, save_dir_masks)
