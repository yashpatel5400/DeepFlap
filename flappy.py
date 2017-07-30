"""
creating a flappy Bird playing
"""

import numpy as np
import pyscreenshot as ImageGrab

def main():
	while True:
		screen      = ImageGrab.grab(bbox=(60, 45, 360, 500))
		game_pixels = np.array(screen)
		digits = ImageGrab.grab(bbox=(175,100,250,150))
		digit_pixels = np.array(digits)

def get_raw_pixels(fn):
	return np.array(Image.open(fn).convert("1")).astype("float32")

digit_representations = [get_raw_pixels("assets/{}.png".format(i)) for i in range(10)]
for digit in digit_representations:
	for 

img = cv2.imread("12.png")
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0,255,0), 3)


def extract_digit(img):


if __name__ == "__main__":
	main()