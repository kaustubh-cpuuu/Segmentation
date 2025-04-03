import os
from PIL import Image

def fill_background(input_folder, output_folder, background_color=(0, 0, 0, 255)):

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each file in the input directory
    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            file_path = os.path.join(input_folder, filename)

            # Open the image and ensure it has an alpha channel
            image = Image.open(file_path)
            if image.mode != 'RGBA':
                image = image.convert('RGBA')

            # Create a new image with the specified background color and the same size as the original
            background = Image.new('RGBA', image.size, background_color)

            # Paste the original image onto the background
            background.paste(image, (0, 0), image)

            # Save the new image to the output folder
            background.save(os.path.join(output_folder, filename))

    return f"Processed images have been saved to {output_folder}"

# Example usage:
input_folder_path = 'PNG'
output_folder_path = 'out_mask'

fill_background(input_folder_path, output_folder_path)
