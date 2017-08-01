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

from skimage import measure
try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters

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
	plt.imshow(matched),plt.show()
	return kp_screen[bird_position[0][0].trainIdx].pt

def update_coords(img, i, j, region, coords):
	if img[i][j] != region:
		return
		
	left, right, top, bottom = coords
	if i < left: left = i
	if i > right: right = i
	if j < top: top = j
	if j > bottom: bottom = j
	return [left, right, top, bottom]

def find_pipes(fn):
	n = 250
	l = 256

	im = cv2.imread(fn, 0)
	im = filters.gaussian_filter(im, sigma= l / (4. * n))
	blobs = im > 0.7 * im.mean()
	blobs_labels = measure.label(blobs, background=0)
	labels = np.unique(blobs_labels)

	pipe_candidates = labels[2:]
	region_size = [(np.sum(blobs_labels == label)) for label in pipe_candidates]
	sorted_regions = sorted(list(zip(region_size, pipe_candidates)))
	pipe_regions = [region[1] for region in sorted_regions][-4:]

	# encoded as : (left, right, top, bottom)
	width, height = blobs_labels.shape
	pipe_coords = [(width+1, -1, height+1, -1) for _ in pipe_regions]
	region_to_coord = dict(zip(pipe_regions, pipe_coords))
	for i in range(width):
		for j in range(height):
			for region in region_to_coord:
				new_coords = update_coords(blobs_labels, i, j, 
					region, region_to_coord[region])
				if new_coords is not None:
					region_to_coord[region] = new_coords

	img_with_pipe = cv2.imread(fn, 1)
	for region in region_to_coord:
		left, right, top, bottom = region_to_coord[region]
		cv2.rectangle(img_with_pipe, (top, left), (bottom, right), (0,0,255))
	scipy.misc.imsave("pipe_labels.png", img_with_pipe)

def main():
	digits = [thresh_img(cv2.imread("assets/{}.png".format(i), cv2.CV_8UC1)) \
		for i in range(10)]
	should_press = True

	# Initiate SIFT detector
	bird = cv2.imread('assets/bird.png',0)
	pipe = cv2.imread('assets/pipe-green.png') 
	
	for i in range(1):
		screen      = ImageGrab.grab(bbox=(60, 45, 360, 500))
		game_pixels = np.array(screen)
		scipy.misc.imsave("screen.png", game_pixels)
		
		screen = cv2.imread("screen.png")
		screen_contours = find_pipes("screen.png")

		# screen = cv2.imread("screen.png")
		# bird_extraction(screen, green_pipe)
		# bird_x, bird_y = bird_extraction(screen, bird)
		
		# target_fn = "target.png" 
		# digit = ImageGrab.grab(bbox=(175,100,250,150))
		# digit.save(target_fn)
		# number = extract_digits(target_fn, digits)
		# print(number)

		#if should_press:
		#	pyautogui.press("up")
		
if __name__ == "__main__":
	main()