import os
from PIL import Image

# Input and output folders
input_folder =r"shoes/batch_5_6/gt"
output_folder = "shoes/batch_5_6/gtt"

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List all files in the input folder
input_files = os.listdir(input_folder)

# Process each image
for file_name in input_files:
    # Check if the file is an image
    valid_extensions = (".png", ".jpg", ".jpeg")
    if file_name.lower().endswith(valid_extensions):
        # Open the image
        input_path = os.path.join(input_folder, file_name)
        image = Image.open(input_path)

        # Convert the image to RGB mode (if it's not already in RGB mode)
        image = image.convert("RGBA")  # Convert to RGBA mode to handle transparency

        # Create a new image with black background
        new_image = Image.new("RGBA", image.size, "black")

        # Paste the original image onto the new black background
        new_image.paste(image, (0, 0), image)

        # Convert the image back to RGB mode (if needed)
        new_image = new_image.convert("RGB")

        # Save the modified image to the output folder
        output_path = os.path.join(output_folder, file_name)
        new_image.save(output_path)

        print(f"Processed: {file_name}")

print("All images processed and saved to the output folder.")

