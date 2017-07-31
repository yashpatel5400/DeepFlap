"""
creating a flappy Bird playing
"""

from PIL import Image
import numpy as np
import pyscreenshot as ImageGrab

import cv2
import scipy.misc

def thresh_img(img):
	img = cv2.medianBlur(img,3)
	img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
		cv2.THRESH_BINARY,11,2)
	return img

def extract_digits(target_img, digits):
	target = thresh_img(cv2.imread(target_img, cv2.CV_8UC1))
	
	# scipy.misc.imsave("threshold.jpg", target)
	target_width, target_height = target.shape 
	THERSHOLD = .95

	digits_contained = {}
	for k, digit in enumerate(digits):
		width, height = digit.shape
		for i in range(target_width - width):
			for j in range(target_height - height):
				subimage = target[i:i+width, j:j+height]
				similarity = (np.sum(subimage == digit)) / float(width * height)
				if similarity > THERSHOLD:
					if j not in digits_contained:
						digits_contained[j] = k
					elif similarity > digits_contained[j]:
						digits_contained[j] = k

	sorted_digits = sorted(list(zip(digits_contained.keys(), digits_contained.values())))
	str_num = "".join(str(digit[1]) for digit in sorted_digits)
	if len(str_num) == 0:
		return 0
	return int(str_num)

def main():
	digits = [thresh_img(cv2.imread("assets/{}.png".format(i), cv2.CV_8UC1)) \
		for i in range(10)]

	while True:
		screen      = ImageGrab.grab(bbox=(60, 45, 360, 500))
		game_pixels = np.array(screen)
		digit = ImageGrab.grab(bbox=(175,100,250,150))
		
		target_fn = "target.png"
		digit.save(target_fn)
		number = extract_digits(target_fn, digits)
		print(number)

if __name__ == "__main__":
	main()