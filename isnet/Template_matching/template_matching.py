import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Set paths for the original and cropped images folders
original_images_folder = '/home/ml2/Desktop/Vscode/Background_removal/DIS/trial/im'
cropped_images_folder = '/home/ml2/Desktop/Vscode/Background_removal/DIS/trial/gt'
output_directory = '/home/ml2/Desktop/Vscode/Background_removal/DIS/trial/res'

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Get list of all images in the original and cropped images folders
original_images = [f for f in os.listdir(original_images_folder) if f.endswith(('.jpg', '.png'))]
cropped_images = [f for f in os.listdir(cropped_images_folder) if f.endswith(('.jpg', '.png'))]

# Check if the numbers of images match
if len(original_images) != len(cropped_images):
    raise ValueError("The number of original images and cropped images must match.")

# Process each pair of images
for original_image_name, cropped_image_name in zip(original_images, cropped_images):
    # Load images
    original_image_path = os.path.join(original_images_folder, original_image_name)
    cropped_image_path = os.path.join(cropped_images_folder, cropped_image_name)

    original_image = cv2.imread(original_image_path)
    cropped_image = cv2.imread(cropped_image_path)

    # Check if images are loaded properly
    if original_image is None:
        print(f"Original image not found at {original_image_path}")
        continue
    if cropped_image is None:
        print(f"Cropped image not found at {cropped_image_path}")
        continue

    # Convert images to grayscale for matching
    gray_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    gray_cropped = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # Detect ORB keypoints and descriptors
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(gray_original, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray_cropped, None)

    # Match descriptors using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Check if any matches were found
    if not matches:
        print(f"No matches found between {original_image_name} and {cropped_image_name}.")
        continue

    # Sort matches based on distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)

    # Get the matching keypoints from both images
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

    # Find homography to align the images
    H, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)

    # Check if homography is valid
    if H is None:
        print(f"Homography could not be computed for {original_image_name} and {cropped_image_name}.")
        continue

    # Warp the cropped image to align it with the original image
    aligned_cropped = cv2.warpPerspective(cropped_image, H, (original_image.shape[1], original_image.shape[0]))

    # Create a transparent canvas the size of the original image
    transparent_canvas = np.zeros((original_image.shape[0], original_image.shape[1], 4), dtype=np.uint8)

    # Convert aligned cropped image to BGRA
    aligned_cropped_with_alpha = cv2.cvtColor(aligned_cropped, cv2.COLOR_BGR2BGRA)

    # Create a mask of the aligned cropped image
    mask = cv2.cvtColor(aligned_cropped, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    # Find contours to get the bounding box of the aligned cropped image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Get the bounding box coordinates
        x, y, w, h = cv2.boundingRect(contours[0])
        
        # Overlay the aligned cropped image onto the transparent canvas
        transparent_canvas[y:y+h, x:x+w] = aligned_cropped_with_alpha[y:y+h, x:x+w]

        # Save the resulting image with transparency
        output_path = os.path.join(output_directory, f'aligned_{original_image_name}')
        cv2.imwrite(output_path, transparent_canvas)
        print(f"Saved the resulting image at: {output_path}")

