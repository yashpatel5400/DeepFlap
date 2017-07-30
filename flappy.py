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

if __name__ == "__main__":
	main()