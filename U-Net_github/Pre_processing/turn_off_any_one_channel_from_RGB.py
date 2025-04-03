
import os
from PIL import Image

def turn_off_color_channel(input_dir, output_dir, channel_to_turn_off):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over each file in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # Load the image
            with Image.open(input_path) as img:
                # Convert the image to RGB if it's not
                img = img.convert("RGB")

                # Get data of the image
                data = img.getdata()

                new_data = []
                for pixel in data:
                    new_pixel = list(pixel)

                    # Set the specified channel's value to 0
                    new_pixel[channel_to_turn_off] = 0

                    # Append the modified pixel
                    new_data.append(tuple(new_pixel))

                # Update image data
                img.putdata(new_data)

                # Save the transformed image as PNG
                img.save(output_path, "PNG")

# Example usage
input_directory = "216_images/training/im"
output_directory = "216_images/training/im_RG"
# To turn off the Red channel, use 0
# To turn off the Green channel, use 1
# To turn off the Blue channel, use 2
turn_off_color_channel(input_directory, output_directory, 2)
