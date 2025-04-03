import os
import cv2
import imageio
from PIL import Image, UnidentifiedImageError

input_folder_path = 'out_mask'
output_folder_path = 'output_png'

if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

for filename in os.listdir(input_folder_path):
    if filename.lower().endswith('.jpg'):
        input_file_path = os.path.join(input_folder_path, filename)
        success = False

        try:
            # Try opening with PIL
            img = Image.open(input_file_path)
            img.load()
            success = True
        except (UnidentifiedImageError, IOError):
            try:
                # If PIL fails, try opening with OpenCV
                cv_img = cv2.imread(input_file_path)
                if cv_img is not None:
                    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(cv_img)
                    success = True
            except Exception:
                pass  # Continue to the next attempt

        if not success:
            try:
                # If OpenCV fails, try opening with imageio
                io_img = imageio.imread(input_file_path)
                img = Image.fromarray(io_img)
                success = True
            except Exception as e:
                print(f'Exception while processing {filename}: {e}')
                continue

        # Convert and save the image in PNG format
        new_filename = filename.rsplit('.', 1)[0] + '.png'
        output_file_path = os.path.join(output_folder_path, new_filename)
        img.save(output_file_path, 'PNG')

        print(f'Converted {filename} to {new_filename} in the output folder')

print('Conversion complete.')
