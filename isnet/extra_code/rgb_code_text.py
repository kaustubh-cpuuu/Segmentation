""" Work on the Single Image """
import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm


if __name__ == "__main__":
	rgb_masks = sorted(glob("Product/white_mask/0112_details18042.png"))[:1]

	RGB = {}
	for path in tqdm(rgb_masks):
		x = cv2.imread(path, cv2.IMREAD_COLOR)
		x = x.reshape(-1, 3)

		for pixel in x:
			pixel = tuple(pixel)

			if pixel in RGB:
				RGB[pixel] += 1
			else:
				RGB[pixel] = 1

	sorted_RGB = sorted(RGB.items(), key=lambda x: x[1])
	sorted_RGB.reverse()

	with open("rgb_freq.txt", "w") as file:
		for code, freq in sorted_RGB:
			file.write(f"RGB Code: {code} - Freq: {freq}\n")













""" Works on the Folder """
#
# import os
# import numpy as np
# import cv2
# from glob import glob
# from tqdm import tqdm
#
# def find_unique_rgb_values(image_paths):
# 	RGB = {}
# 	for path in tqdm(image_paths):
# 		# Read the image in color
# 		x = cv2.imread(path, cv2.IMREAD_COLOR)
# 		# Convert from BGR to RGB
# 		x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
# 		# Reshape the image to a 2D array of RGB pixels
# 		x = x.reshape(-1, 3)
#
# 		# Convert pixels to a set of tuples and update the dictionary
# 		for pixel in set(map(tuple, x)):
# 			if pixel in RGB:
# 				RGB[pixel] += 1
# 			else:
# 				RGB[pixel] = 1
#
# 	return RGB
#
#
# if __name__ == "__main__":
# 	# Use glob pattern to match all png images in the folder
# 	red_green_images = 'folder'
# 	image_paths = sorted(glob(os.path.join(red_green_images, '*.png')))
#
# 	# Find unique RGB values across all images
# 	unique_rgb_values = find_unique_rgb_values(image_paths)
#
# 	# Write the unique RGB values to a file
# 	with open("rgb_unique_freq.txt", "w") as file:
# 		for code, freq in unique_rgb_values.items():
# 			file.write(f"RGB Code: {code} - Freq: {freq}\n")
