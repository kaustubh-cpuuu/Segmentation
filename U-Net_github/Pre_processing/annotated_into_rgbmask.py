import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def process_folder(input_folder, csv_path, output_folder):


    # Ensure the output1 directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load annotations
    annotations = load_annotations(csv_path)

    # Iterate over each image in the input folder
    for image_filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_filename)

        if not os.path.isfile(image_path):  # skip if not a file (e.g., subdirectory)
            continue

        # Filter annotations for the current image
        annotations_for_image = annotations[annotations["ImageId"] == image_filename]

        # Generate the mask for this image
        mask = generate_mask(image_path, annotations_for_image)

        # Save the mask to the output1 folder
        mask_save_path = os.path.join(output_folder, image_filename)
        cv2.imwrite(mask_save_path, mask)

        print(f"Processed and saved mask for {image_filename}")
def rle_decode(mask_rle, shape):
    """
    Decode RLE.
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')

def load_annotations(csv_path):
    """
    Load annotations from the CSV file.
    """
    return pd.read_csv(csv_path)


def generate_color_palette(num_classes=42):
    """
    Generate distinct colors for each class.
    """
    # Using a predefined color palette, expand if needed.
    palette = np.array([
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
        [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
        [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
        [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
        [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
        # ... add more if needed.
    ])

    if num_classes <= palette.shape[0]:
        return palette[:num_classes]
    else:
        # If you have more classes than the palette, generate random colors for the remaining ones.
        additional_colors = np.random.randint(0, 256, size=(num_classes - palette.shape[0], 3))
        return np.vstack((palette, additional_colors))


colors = generate_color_palette()


def generate_mask(image_path, annotations):
    """
    Generate an RGB mask for an image based on the annotations.
    """
    img = cv2.imread(image_path)
    mask = np.zeros_like(img)

    for _, row in annotations.iterrows():
        label = int(row['ClassId']) % len(colors)
        shape = (row['Height'], row['Width'])
        binary_mask = rle_decode(row['EncodedPixels'], shape)

        print(f"label: {label}")
        color_mask = colors[label].reshape(1, 1, 3) * binary_mask[:, :, np.newaxis]

        color_mask = colors[label].reshape(1, 1, 3) * binary_mask[:, :, np.newaxis]


        # Combine with the overall mask
        mask = cv2.addWeighted(mask, 1, color_mask.astype(np.uint8), 1, 0)

    return mask


def visualize(image_path, mask):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.title("RGB Mask")
    plt.show()


def main(image_path, csv_path):
    annotations = load_annotations(csv_path)
    annotations_for_image = annotations[
        annotations["ImageId"] == image_path.split('/')[-1]]  # assuming image_path ends with the image filename

    mask = generate_mask(image_path, annotations_for_image)

    visualize(image_path, mask)


def visualize(image_path, mask):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.title("RGB Mask")
    plt.show()

def main(image_path, csv_path):
    annotations = load_annotations(csv_path)
    annotations_for_image = annotations[annotations["ImageId"] == image_path.split('/')[-1]]
    mask = generate_mask(image_path, annotations_for_image)
    visualize(image_path, mask)

if __name__ == "__main__":
    process_folder("train", "train.csv", "output")






