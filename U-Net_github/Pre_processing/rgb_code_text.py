import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm

if __name__ == "__main__":
	rgb_masks = sorted(glob("vest_1_bag_1_2 (1).png"))[:1]

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