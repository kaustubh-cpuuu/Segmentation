# import cv2
# import numpy as np
# import os

# image_folder = "/home/ml2/Desktop/Vscode/Background_removal/DIS/demo_datasets/multi"
# mask_folder = "/home/ml2/Desktop/Vscode/Background_removal/DIS/demo_datasets/ret"
# output_folder = "/home/ml2/Desktop/Vscode/Background_removal/DIS/demo_datasets/out_inpaint"

# os.makedirs(output_folder, exist_ok=True)

# for image_name in os.listdir(image_folder):
#     if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
#         image_path = os.path.join(image_folder, image_name)
#         mask_path = os.path.join(mask_folder, os.path.splitext(image_name)[0] + ".png")
#         output_path = os.path.join(output_folder, image_name)
        
 
#         if not os.path.exists(mask_path):
#             print(f"Mask not found for {image_name}, skipping...")
#             continue
        
#         image = cv2.imread(image_path)
#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#         mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
#         _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
#         mask_output_path = os.path.join(output_folder, os.path.splitext(image_name)[0] + "_mask.png")
#         cv2.imwrite(mask_output_path, mask)
#         inpainted = cv2.inpaint(image, mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA)
#         cv2.imwrite(output_path, inpainted)
#         print(f"Processed {image_name} and saved results.")

# print("Processing complete.")






import cv2
import numpy as np
import os

image_folder = "/home/ml2/Desktop/Vscode/Background_removal/DIS/demo_datasets/multi"
mask_folder = "/home/ml2/Desktop/Vscode/Background_removal/DIS/demo_datasets/ret"
output_folder = "/home/ml2/Desktop/Vscode/Background_removal/DIS/demo_datasets/out_inpaint"

os.makedirs(output_folder, exist_ok=True)

for image_name in os.listdir(image_folder):
    if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_folder, image_name)
        mask_path = os.path.join(mask_folder, os.path.splitext(image_name)[0] + ".png")
        output_path = os.path.join(output_folder, image_name)
        
        if not os.path.exists(mask_path):
            print(f"Mask not found for {image_name}, skipping...")
            continue
        
        image = cv2.imread(image_path)
        mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_gray = cv2.resize(mask_gray, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        _, mask_binary = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)

        # Save the mask (optional)
        mask_output_path = os.path.join(output_folder, os.path.splitext(image_name)[0] + "_mask.png")
        cv2.imwrite(mask_output_path, mask_binary)

        # Prepare destination image
        inpainted = np.zeros_like(image)

        # Use xphoto inpaint — now with correct mask
        cv2.xphoto.inpaint(image, mask_binary, inpainted, cv2.xphoto.INPAINT_SHIFTMAP)

        cv2.imwrite(output_path, inpainted)
        print(f"Processed {image_name} with xphoto.inpaint() and saved results.")

