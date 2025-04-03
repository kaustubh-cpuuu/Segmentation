
# import cv2
# import numpy as np

# # Load image
# image = cv2.imread("/home/ml2/Desktop/Vscode/Background_removal/DIS/demo_datasets/ret/im/jenna-125.jpg")

# # Load mask in grayscale
# mask = cv2.imread("/home/ml2/Desktop/Vscode/Background_removal/DIS/demo_datasets/ret/mask/jenna-125.png", cv2.IMREAD_GRAYSCALE)

# # Ensure mask has the same size as the image
# mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

# cv2.imwrite('/home/ml2/Desktop/Vscode/Background_removal/DIS/demo_datasets/ret//output/threshold.png', mask)

# # _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)


# # Apply inpainting
# inpainted = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

# # Save result
# cv2.imwrite("/home/ml2/Desktop/Vscode/Background_removal/DIS/demo_datasets/ret//output/jenna-125.jpg", inpainted)


#Original code

import cv2
import numpy as np
import os

# Define input and output directories
image_folder = "/home/ml2/Desktop/Vscode/Background_removal/DIS/demo_datasets/multi"
mask_folder = "/home/ml2/Desktop/Vscode/Background_removal/DIS/demo_datasets/ret/1024"
output_folder = "/home/ml2/Desktop/Vscode/Background_removal/DIS/demo_datasets/ret/output"

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Process all images in the folder
for image_name in os.listdir(image_folder):
    if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_folder, image_name)
        mask_path = os.path.join(mask_folder, os.path.splitext(image_name)[0] + ".png")
        output_path = os.path.join(output_folder, image_name)   
        
        # Check if the corresponding mask exists
        if not os.path.exists(mask_path):
            print(f"Mask not found for {image_name}, skipping...")
            continue
        
        # Load image and mask
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Ensure mask has the same size as the image
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Apply thresholding to mask
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Save thresholded mask for verification
        mask_output_path = os.path.join(output_folder, os.path.splitext(image_name)[0] + "_mask.png")
        cv2.imwrite(mask_output_path, mask)
        
        # Apply inpainting
        inpainted = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        
        # Save the inpainted result
        cv2.imwrite(output_path, inpainted)
        print(f"Processed {image_name} and saved results.")
        
print("Processing complete.")


