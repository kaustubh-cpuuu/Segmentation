import cv2
import os
import numpy as np


def make_image_square_with_black_padding(input_path, output_path):
    """
    Add black padding to an image to make it square and save it to the specified path.

    Parameters:
    - input_path: Path to the input image.
    - output_path: Path to save the output image with padding.
    """
    # Read the image
    image = cv2.imread(input_path)
    if image is None:
        print(f"Error: Could not read image from {input_path}")
        return

    # Calculate the size for padding
    h, w = image.shape[:2]
    max_side = max(h, w)

    # Create a square canvas of size max_side filled with black pixels
    square_canvas = np.zeros((max_side, max_side, 3), dtype=np.uint8)

    # Calculate the center offset
    x_offset = (max_side - w) // 2
    y_offset = (max_side - h) // 2

    # Place the original image in the center of the square canvas
    square_canvas[y_offset:y_offset + h, x_offset:x_offset + w, :] = image

    # Save the squared image
    cv2.imwrite(output_path, square_canvas)
    print(f"Saved squared image to {output_path}")


def process_folder(input_folder, output_folder):
    """
    Process all images in the input folder, making them square with black padding,
    and save them to the output folder.

    Parameters:
    - input_folder: Path to the input folder containing images.
    - output_folder: Path to the output folder to save squared images.
    """
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            make_image_square_with_black_padding(input_path, output_path)


if __name__ == "__main__":
    input_folder = "04/aa"
    output_folder = "04/aaa"

    process_folder(input_folder, output_folder)
