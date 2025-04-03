import os
from PIL import Image, ImageOps
from multiprocessing import Pool, cpu_count

# Directories
images_folder = r'/home/ml2/Desktop/Vscode/Background_removal/DIS/combine_all_in_one/im'
masks_folder = r'/home/ml2/Desktop/Vscode/Background_removal/DIS/combine_all_in_one/gt'
output_images_folder = r'/home/ml2/Desktop/Vscode/Background_removal/DIS/combine_all_in_one/128/im'
output_masks_folder = r'/home/ml2/Desktop/Vscode/Background_removal/DIS/combine_all_in_one/128/gt'

# Create output directories if they don't exist
os.makedirs(output_images_folder, exist_ok=True)
os.makedirs(output_masks_folder, exist_ok=True)

# Resize dimensions
resize_to = 25

# Function to resize and maintain aspect ratio, ensuring the smaller dimension is target_size
def resize_with_padding(image, target_size, padding_color):
    # Calculate the new size maintaining aspect ratio with the smaller side as target_size
    ratio = target_size / min(image.size)
    new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
    image = image.resize(new_size, Image.LANCZOS)
    
    # Calculate padding to make the image square
    delta_w = target_size - image.size[0] if image.size[0] < target_size else 0
    delta_h = target_size - image.size[1] if image.size[1] < target_size else 0
    padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
    
    return ImageOps.expand(image, padding, fill=padding_color)

# Function to process a single image and its corresponding mask
def process_image(filename):
    if filename.endswith('.jpg'):
        img_path = os.path.join(images_folder, filename)
        mask_path = os.path.join(masks_folder, filename.replace('.jpg', '.png'))

        try:
            # Open and resize image with white padding
            with Image.open(img_path) as img:
                img_resized = resize_with_padding(img, resize_to, padding_color='white')
                img_resized.save(os.path.join(output_images_folder, filename))

            # Check if mask file exists before processing
            if os.path.exists(mask_path):
                with Image.open(mask_path) as mask:
                    mask_resized = resize_with_padding(mask, resize_to, padding_color='black')
                    mask_resized.save(os.path.join(output_masks_folder, filename.replace('.jpg', '.png')))
            else:
                print(f"Mask file {mask_path} not found. Skipping.")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Main function to use multiprocessing
if __name__ == "__main__":
    # Get list of all image filenames
    image_filenames = [f for f in os.listdir(images_folder) if f.endswith('.jpg')]

    # Set up multiprocessing with the number of processes as the number of CPU cores
    num_processes = cpu_count()

    # Use a pool of workers to process images in parallel
    with Pool(processes=num_processes) as pool:
        pool.map(process_image, image_filenames)

    print("Resizing completed!")
