import os
from PIL import Image

def rename_and_resize_images(input_folder, output_folder, target_width=2048):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # List all files in the input folder
    for idx, filename in enumerate(os.listdir(input_folder)):
        # Check if the file is an image
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # Construct full file path
            img_path = os.path.join(input_folder, filename)
            
            # Open the image
            with Image.open(img_path) as img:
                # Calculate the new height to maintain aspect ratio
                aspect_ratio = img.height / img.width
                new_height = int(target_width * aspect_ratio)

                # Resize the image
                resized_img = img.resize((target_width, new_height))

                # Define the new filename
                new_filename = f"1024_{idx + 1}.jpg"  # Adding index for uniqueness
                
                # Save the resized image to the output folder
                resized_img.save(os.path.join(output_folder, new_filename))
                print(f'Renamed and resized: {img_path} to {os.path.join(output_folder, new_filename)}')

# Define your input and output folders
input_folder = r'/home/ml2/Desktop/Vscode/Background_removal/DIS/demo_datasets/yolo_input'
output_folder = r'/home/ml2/Desktop/Vscode/Background_removal/DIS/demo_datasets/2048'

# Call the function
rename_and_resize_images(input_folder, output_folder)
