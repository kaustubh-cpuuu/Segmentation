
# import os
# import cv2
# from ultralytics import YOLO
# import shutil

# def move_image(source_path, destination_path):
#     """ 
#     Move the entire image from the source to the destination path.
#     """
#     try:
#         shutil.copy(source_path, destination_path)
#         print(f"Moved {source_path} to {destination_path}")
#     except Exception as e:
#         print(f"Failed to move {source_path}: {e}")

# def process_folder(folder_path, destination_with_person, destination_without_person, model):
#     os.makedirs(destination_with_person, exist_ok=True)
#     os.makedirs(destination_without_person, exist_ok=True)

#     for root, dirs, files in os.walk(folder_path):
#         for file_name in files:
#             file_path = os.path.join(root, file_name)
#             try:
#                 result = model.predict(source=file_path, conf=0.50, device='cuda', iou=0.5)
#                 predictions = result[0]
#                 classes = predictions.boxes.cls.cpu().detach().numpy().astype(int)

#                 if 0 in classes:
#                     destination_path = os.path.join(destination_with_person, file_name)
#                     move_image(file_path, destination_path)
#                 else:
#                     destination_path = os.path.join(destination_without_person, file_name)
#                     move_image(file_path, destination_path)

#             except Exception as e:
#                 print(f"Error processing file {file_path}: {e}")
                
# def process_images(source, destination_with_person, destination_without_person, model):
#     if not os.path.exists(source):
#         print(f"Source directory {source} does not exist.")
#         return

#     folders_list = os.listdir(source)
#     for folder_name in folders_list:
#         folder_path = os.path.join(source, folder_name)
#         if os.path.isdir(folder_path):
#             process_folder(folder_path, destination_with_person, destination_without_person, model)
#         else:
#             process_folder(source, destination_with_person, destination_without_person, model)
#             break  # Processing is complete

# # Example usage
# source = r"/mnt/qnap_share/organized/giglio/after"
# destination_with_person = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/JD/human"
# destination_without_person = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/JD/product"

# model_path = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/saved_models/yolov8x.pt"

# try:
#     model = YOLO(model_path, task="detect")
#     process_images(source, destination_with_person, destination_without_person, model)
# except Exception as e:
#     print(f"Error loading model or processing images: {e}")




import os
import cv2
from ultralytics import YOLO
import shutil

def move_image(source_path, destination_path):
    """ 
    Move the image from the source to the destination path.
    """
    try:
        shutil.copy(source_path, destination_path)
        print(f"Moved {source_path} to {destination_path}")
    except Exception as e:
        print(f"Failed to move {source_path}: {e}")

def process_folder(folder_path, destination_with_person, model):
    os.makedirs(destination_with_person, exist_ok=True)

    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            try:
                # Make prediction using YOLO model
                result = model.predict(source=file_path, conf=0.60, device='cuda', iou=0.5)
                predictions = result[0]
                classes = predictions.boxes.cls.cpu().detach().numpy().astype(int)

                # Check if any human (class 0) is detected in the image
                if 0 in classes:
                    destination_path = os.path.join(destination_with_person, file_name)
                    move_image(file_path, destination_path)
                else:
                    print(f"No human detected in {file_name}, skipping.")

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                
def process_images(source, destination_with_person, model):
    if not os.path.exists(source):
        print(f"Source directory {source} does not exist.")
        return

    folders_list = os.listdir(source)
    for folder_name in folders_list:
        folder_path = os.path.join(source, folder_name)
        if os.path.isdir(folder_path):
            process_folder(folder_path, destination_with_person, model)
        else:
            process_folder(source, destination_with_person, model)
            break  # Processing is complete

# Example usage
source = r"/mnt/qnap_share/organized/size/after"
destination_with_person = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/40k_Human/size_human/Human"

model_path = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/saved_models/yolov8x.pt"

try:
    model = YOLO(model_path, task="detect")
    process_images(source, destination_with_person, model)
except Exception as e:
    print(f"Error loading model or processing images: {e}")

