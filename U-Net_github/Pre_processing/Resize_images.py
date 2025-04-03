import cv2
import os
from glob import glob


def resize_images(input_dir, output_dir, height=2048, width=2048):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List all image files in the input directory
    image_paths = glob(os.path.join(input_dir, "*"))

    # Process each image
    for img_path in image_paths:
        # Read the image
        img = cv2.imread(img_path)

        # Resize the image
        resized_img = cv2.resize(img, (width, height))

        # Construct the path for the output image
        base_name = os.path.basename(img_path)
        output_path = os.path.join(output_dir, base_name)

        # Save the resized image
        cv2.imwrite(output_path, resized_img)

        print(f"Resized and saved: {output_path}")


# Define your input and output directories
input_dir = "images"  # Update with your input folder path
output_dir = "2048/training/im"  # Update with your output folder path

# Run the function
resize_images(input_dir, output_dir)
