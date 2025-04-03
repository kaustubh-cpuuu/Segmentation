import os
from PIL import Image

def replace_colors(image_path, output_folder, color_mappings):
    # Open an image file
    with Image.open(image_path) as img:
        # Convert image to RGB (if it's not already in that format)
        img = img.convert('RGB')
        # Load pixels
        pixels = img.load()

        # Get image dimensions
        width, height = img.size

        # Replace target colors
        for x in range(width):
            for y in range(height):
                current_color = pixels[x, y]
                if current_color in color_mappings:
                    pixels[x, y] = color_mappings[current_color]

        # Create output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Save the modified image in the output folder
        new_image_path = os.path.join(output_folder, os.path.basename(image_path))
        img.save(new_image_path)
        return new_image_path

def process_folder(folder_path, output_folder, color_mappings):
    # Process all PNG images in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            new_image_path = replace_colors(image_path, output_folder, color_mappings)
            print(f"Processed {filename} -> {new_image_path}")

# Define the color mappings as a dictionary
# Add your color mappings here
color_mappings = {
    (7, 7, 7): (255, 0, 0),
    (16, 16, 16): (0, 0, 255),
    (2, 2, 2): (255, 255, 0),
    (14, 14, 14): (255, 0, 255)
}

# Path to the folder containing images
folder_path = 'image'
output_folder = 'out'

# Call the function with the path to your folder, output folder, and the color mappings
process_folder(folder_path, output_folder, color_mappings)
