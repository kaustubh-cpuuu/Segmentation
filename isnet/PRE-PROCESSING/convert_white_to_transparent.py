from PIL import Image

# Load the image
image_path = "trial/im/CZ0775-133_jordan_1_low_og_white__oxidized_green___sail_1.tif"  # Replace with your image path
img = Image.open(image_path).convert("RGBA")

# Define the RGB color to be replaced (R, G, B)
target_color = (247,247,247)  # Replace with your specific RGB color code

# Replace the target color with transparency
data = img.getdata()

new_data = []
for item in data:
    # Change all pixels that match target_color to transparent
    if item[:3] == target_color:  # Compare RGB only
        new_data.append((255, 255, 255, 0))  # (R, G, B, A) for transparency
    else:
        new_data.append(item)

# Update image data
img.putdata(new_data)

# Save the image with transparency
img.save("output_image.png", "PNG")
