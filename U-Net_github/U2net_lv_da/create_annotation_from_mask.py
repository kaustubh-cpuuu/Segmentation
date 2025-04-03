import os
import cv2
import numpy as np
import pandas as pd

# Function to convert RGB mask to RLE
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]  # Transpose and flatten to row-major order
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return ' '.join(map(str, run_lengths))

# Function to process an RGB mask
def process_mask(image_id, mask, color_class_dict):
    annotations = []
    for color, class_id in color_class_dict.items():
        class_mask = np.all(mask == np.array(color), axis=-1)
        if np.any(class_mask):  # Check if the class color is present in the mask
            rle = rle_encoding(class_mask)
            annotations.append({
                "ImageId": image_id + '.png',  # Add the file extension if needed
                "EncodedPixels": rle,
                "Height": mask.shape[0],
                "Width": mask.shape[1],
                "ClassId": class_id
            })
    return annotations

# Function to process all mask images in a folder
def process_folder(folder_path, color_class_dict):
    annotations = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(folder_path, filename)
            mask = cv2.imread(file_path)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            image_id = os.path.splitext(filename)[0]
            annotations.extend(process_mask(image_id, mask, color_class_dict))
    return annotations

# Placeholder color to class mapping, replace with actual mappings
color_class_dict = {

    (0, 0, 0):1,      
    (255, 0, 0):2,   
    (255, 228, 0):3 ,   
    (0, 216, 255):4, 
    (0, 255, 0):5,    

}

folder_path = '/home/ml2/Desktop/Vscode/U-Net/U2net_levain_Dabhi/annotation_dataset/training/gt'  # Example: '/user/images/masks/'
all_annotations = process_folder(folder_path, color_class_dict)
df_all_annotations = pd.DataFrame(all_annotations)
df_all_annotations = df_all_annotations[['ImageId', 'EncodedPixels', 'Height', 'Width', 'ClassId']]
csv_filename = '/home/ml2/Desktop/Vscode/U-Net/U2net_levain_Dabhi/annotation_dataset/training/mask_annotations.csv'
df_all_annotations.to_csv(csv_filename, index=False)
print(df_all_annotations.head())
print(f'Annotations CSV file saved as: {csv_filename}')
