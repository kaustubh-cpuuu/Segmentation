import os
import cv2
from ultralytics import YOLO


def crop_and_save_image(image, box, expansion_pixels, destination_path):
    """
    Crop the image based on the bounding box with expansion and save it to the specified path.
    """
    x1, y1, x2, y2 = [int(coord) for coord in box]  # Convert box coordinates to integer

    # Expand the bounding box by the specified number of pixels
    x1 = max(0, x1 - expansion_pixels)
    y1 = max(0, y1 - expansion_pixels)
    x2 = min(image.shape[1], x2 + expansion_pixels)
    y2 = min(image.shape[0], y2 + expansion_pixels)

    cropped_image = image[y1:y2, x1:x2]  # Crop image
    if cropped_image.size == 0:
        print("Empty cropped image, skipping.")
        return False
    cv2.imwrite(destination_path, cropped_image)  # Save cropped image
    return True


def save_classified_image(image_path, destination, boxes):
    """
    Save the cropped T-shirt images.
    """
    base_image_name = os.path.basename(image_path)
    image_name_without_extension, extension = os.path.splitext(base_image_name)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return False

    saved = False
    for i, box in enumerate(boxes):  # Assuming multiple T-shirts could be present
        new_image_name = f"{image_name_without_extension}_tshirt_{i}{extension}"
        destination_path = os.path.join(destination, new_image_name)
        expansion_pixels = 20
        if crop_and_save_image(image, box, expansion_pixels ,  destination_path ):
            print(f"Saved {destination_path}")
            saved = True

    return saved



def process_folder(folder_path, destination, model, names):
    os.makedirs(destination, exist_ok=True)  # Ensure the destination directory exists
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            try:
                result = model.predict(source=file_path, conf=0.20)
                predictions = result[0]
                classes = predictions.boxes.cls.cpu().detach().numpy().astype(int)
                boxes = predictions.boxes.xyxy.cpu().detach().numpy()  # Get bounding boxes

                tshirt_boxes = [box for cls, box in zip(classes, boxes) if cls == 1 and names.get(cls) == "t-shirt"]  # Filter for T-shirts

                if not save_classified_image(file_path, destination, tshirt_boxes):
                    print(f"No T-shirts found for {file_name}, skipping save.")

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")


def process_images(source, destination, model, names):
    if not os.path.exists(source):
        print(f"Source directory {source} does not exist.")
        return

    if not os.path.exists(destination):
        os.makedirs(destination)

    folders_list = os.listdir(source)
    for folder_name in folders_list:
        folder_path = os.path.join(source, folder_name)
        if os.path.isdir(folder_path):
            process_folder(folder_path, destination, model, names)
        else:
            process_folder(source, destination, model, names)
            break  # Processing is complete


# Example usage
source = "demo_datasets/yolo_input"
destination = "demo_datasets/yolo_output"
names = {0: "none", 1: "t-shirt", 2: "jacket", 3: "top", 4: "bottom", 5: "shoes", 6: "shirt"}

model_path = "saved_models/best.pt"
# model = model_path
try:
    model = YOLO(model_path, task="detect").to("cuda")
    # model.export(format='onnx')
    process_images(source, destination, model, names)
except Exception as e:
    print(f"Error loading model or processing images: {e}")



