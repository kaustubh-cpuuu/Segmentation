from PIL import Image
import os

# Function to change blue pixels to white
def change_color_to_white(img):
    pixels = img.load()
    width, height = img.size
    for x in range(width):
        for y in range(height):
            r, g, b, a = pixels[x, y]
            if b > 100 and r < 100 and g < 100:  # Assuming blue pixels are the ones with high blue value
                pixels[x, y] = (255, 255, 255, a)  # Change blue pixels to white

# Input and output folder paths
input_folder = "combine_inner_wear_to_upperwear_batch_1&2/blue"
output_folder = "combine_inner_wear_to_upperwear_batch_1&2/blue_gt"

# Ensure output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".png"):
        # Open image
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path).convert("RGBA")

        # Change blue to white
        change_color_to_white(img)

        # Save modified image to output folder
        output_path = os.path.join(output_folder, filename)
        img.save(output_path)

        print(f"Processed {filename} and saved as {output_path}")
