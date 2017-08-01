"""
creating a flappy Bird playing
"""

from PIL import Image
import numpy as np
import pyscreenshot as ImageGrab

import matplotlib.pyplot as plt
import cv2
import scipy.misc

import time
import pyautogui

def thresh_img(img):
	# thresholding image to binary
	img = cv2.medianBlur(img,3)
	img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
		cv2.THRESH_BINARY,11,2)
	return img

def extract_digits(target_img, digits):
	# takes the original bounding box on digit location and returns interpreted digit
	target = thresh_img(cv2.imread(target_img, cv2.CV_8UC1))
	
	scipy.misc.imsave("threshold.jpg", target)
	target_width, target_height = target.shape 
	THERSHOLD = .95

	similarities = {}
	digits_contained = {}
	for k, digit in enumerate(digits):
		width, height = digit.shape
		for i in range(target_width - width):
			for j in range(target_height - height):
				subimage = target[i:i+width, j:j+height]
				similarity = (np.sum(subimage == digit)) / float(width * height)
				if similarity > THERSHOLD:
					if j not in similarities:
						similarities[j] = similarity
						digits_contained[j] = k
					elif similarity > similarities[j]:
						similarities[j] = similarity
						digits_contained[j] = k

	sorted_digits = sorted(list(zip(digits_contained.keys(), digits_contained.values())))
	str_num = "".join(str(digit[1]) for digit in sorted_digits)
	if len(str_num) == 0:
		return 0
	return int(str_num)

def bird_extraction(screen, bird):
	# detect bird in screen image and return center of bird on screen coords
	sift = cv2.xfeatures2d.SIFT_create()

	# find the keypoints and descriptors with SIFT
	kp_bird, des_bird     = sift.detectAndCompute(bird,None)
	kp_screen, des_screen = sift.detectAndCompute(screen,None)
	
	# BFMatcher with default params
	bf      = cv2.BFMatcher()
	matches = bf.knnMatch(des_bird, des_screen, k=2)

	# Apply ratio test
	bird_position = []
	for m,n in matches:
		if m.distance < 0.75 * n.distance:
			bird_position.append([m])

	# cv2.drawMatchesKnn expects list of lists as matches.
	matched = cv2.drawMatchesKnn(bird,kp_bird,
		screen,kp_screen,bird_position,screen,flags=2)
	#plt.imshow(matched),plt.show()
	return kp_screen[bird_position[0][0].trainIdx].pt

def main():
	digits = [thresh_img(cv2.imread("assets/{}.png".format(i), cv2.CV_8UC1)) \
		for i in range(10)]
	should_press = True

	# Initiate SIFT detector
	bird = cv2.imread('assets/bird.png',0) 
	
	for i in range(1):
		screen      = ImageGrab.grab(bbox=(60, 45, 360, 500))
		game_pixels = np.array(screen)
		scipy.misc.imsave("screen.png", game_pixels)
		screen = cv2.imread("screen.png",0)

		bird_x, bird_y = bird_extraction(screen, bird)
		pointed_screen = cv2.circle(screen, (int(bird_x), int(bird_y)), 25, (0,0,255))
		scipy.misc.imsave("point.png", pointed_screen)

		# target_fn = "target.png" 
		# digit = ImageGrab.grab(bbox=(175,100,250,150))
		# digit.save(target_fn)
		# number = extract_digits(target_fn, digits)
		# print(number)

		#if should_press:
		#	pyautogui.press("up")
		
if __name__ == "__main__":
	main()