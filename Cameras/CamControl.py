from __future__ import print_function
import os, sys, time, math
import subprocess
import numpy as np
import cv2
from PySpinCapture import PySpinCapture as psc
from liveDisplay import ptgCamStream, imgShow, patternDisplay

DEFAULT_DIRECTORY_PATH = ""

DISPLAY_HEIGHT = 1080
DISPLAY_WIDTH = 1920
NUM_PATTERN = 7

#DEFLECTOMETRY_FREQ = 0.9
DEFLECTOMETRY_FREQ = 1.0

class CamControl:
	def __init__(self):

		self.sessionDir = None

		# self.NumPatterns = NUM_PATTERN
		self.displayWidth = DISPLAY_WIDTH
		self.displayHeight = DISPLAY_HEIGHT

		self.setDefPattern()
		self._isMonochrome = True
		self._is16bits = True
		self.Cam = psc(0, self._isMonochrome, self._is16bits)
		self.height = self.Cam.height
		self.width = self.Cam.width

	def setDir(self, directory, sessionName):
		self.directory = directory
		self.sessionName = sessionName
		self.sessionDir = os.path.join(self.directory, self.sessionName)
		print(self.sessionDir)
		if not os.path.exists(self.sessionDir):
			os.makedirs(self.sessionDir)
			
	def setExposure(self, exposure_to_set):
		
		self.Cam.setExposure(exposure_to_set)

	def setGain(self, gain_to_set):

		self.Cam.setGain(gain_to_set)

	def captureImage(self, fname):
		if fname:
			path = os.path.join(self.sessionDir, fname + ".png")
			flag, img = self.Cam.grabFrame()
			if not flag:
				print("[ERROR]: Didn't get the image!!!")
			else:
				cv2.imwrite(path, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
		else:
			flag, img = self.Cam.grabFrame()
			print("capture!!!!")
			if not flag:
				print("[ERROR]: Didn't get the image!!!")
				return None
			else:
				return img

	def setPatternScale(self, mode=1):
		patternSize = min(self.displayHeight, self.displayWidth)
		boarderSize = math.floor((max(self.displayHeight, self.displayWidth) - patternSize)/2)

		if mode == 0:
			# Img sequences[7]: {gh, ch, gv, cv, g2h, g2v, black}
			self.NumPatterns = 7
		elif mode == 1:
			# Img sequences[8]: {gh, ch, gv, cv, g2h, g2v, black, full}
			self.NumPatterns = 8
		elif mode == 2:
			# Img sequences[9]: {h1, h2, h3, h4, v1, v2, v3, v4, black}
			self.NumPatterns = 9
		elif mode == 3:
			# Img sequences[9]: {h1, h2, h3, h4, v1, v2, v3, v4, black, full}
			self.NumPatterns = 10
		else:
			self.NumPatterns = 8
			print("[WARNING]: Unrecognizable mode. Using Gradient Pattern!!")
		self.patterns = np.zeros((self.displayHeight, self.displayWidth, self.NumPatterns), dtype=np.float32)

		return patternSize, boarderSize

	def setPattern(self):
		patternSize, boarderSize = self.setPatternScale(1)
		# Create spatial coordinates
		x1 = np.linspace(0,1,patternSize)
		y1 = np.linspace(0,1,patternSize)
		x2 = np.linspace(-1,1,patternSize)
		y2 = np.linspace(-1,1,patternSize)

		linearX = x1
		linearY = y1
		linearXinv = 1-x1
		linearYinv = 1-y1
		quadX = (x2**2+1)/2
		quadY = (y2**2+1)/2

		# x direction is flip because of the projection
		self.patterns[:, boarderSize:-boarderSize, 0]= np.tile(linearXinv,(patternSize, 1))
		self.patterns[:, boarderSize:-boarderSize, 1]= np.tile(linearX,(patternSize, 1))
		self.patterns[:, boarderSize:-boarderSize, 2]= np.tile(linearY,(patternSize, 1)).T
		self.patterns[:, boarderSize:-boarderSize, 3]= np.tile(linearYinv,(patternSize, 1)).T
		self.patterns[:, boarderSize:-boarderSize, 4]= np.tile(quadX,(patternSize, 1))
		self.patterns[:, boarderSize:-boarderSize, 5]= np.tile(quadY,(patternSize, 1)).T
		self.patterns[:, boarderSize:-boarderSize, 7] = np.ones((patternSize, patternSize))

	def sinePattern(self, x, y, nu_x, nu_y):
		# Phase Shifts (first entry is white light modulation)
		theta = [0, np.pi / 2, np.pi, 3 / 2 * np.pi]
		[X, Y, Theta] = np.meshgrid(x, y, theta)

		# Calculate Phase Shifts
		phase = (nu_x * X + nu_y * Y) + Theta
		# Simple formula to create fringes between 0 and 1:
		pattern = (np.sin(phase) + 1) / 2

		return pattern

	def setDefPattern(self):
		#patternSize, boarderSize = self.setPatternScale(2)
		patternSize, boarderSize = self.setPatternScale(2)

		patternSize = max(self.displayHeight, self.displayWidth)
		#patternSize = min(self.displayHeight, self.displayWidth)
		
		# Create spatial coordinates
		x = np.linspace(1, self.displayWidth, self.displayWidth)
		y = np.linspace(1, self.displayHeight, self.displayHeight)

		# Frequencies in x and y direction.
		nu_x =  DEFLECTOMETRY_FREQ * 2 * np.pi / patternSize
		nu_y = DEFLECTOMETRY_FREQ * 2 * np.pi / patternSize

		self.patterns[..., 0:4] = self.sinePattern(x, y, nu_x, 0)
		self.patterns[..., 4:8] = self.sinePattern(x, y, 0, nu_y)
		self.patterns[..., 8] = 127 * np.ones((DISPLAY_HEIGHT,DISPLAY_WIDTH))

	def captureSeqImages(self):
		window_name = 'projector'
		cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)

		# If projector is placed right to main screen (see windows properties in your operating system)
		# if the pattern is displayed at the wrong monitor you need to play around with the coordinates here until the image is displayed at the right screen
		cv2.moveWindow(window_name, self.displayWidth, 0)

		cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
		cv2.imshow(window_name, self.patterns[..., 0].astype(np.float32))

		if self._isMonochrome:
			if self._is16bits:
				imgs = np.zeros((self.NumPatterns, self.height, self.width), dtype=np.uint16)
			else:
				imgs = np.zeros((self.NumPatterns, self.height, self.width), dtype=np.uint8)
		else:
			if self._is16bits:
				imgs = np.zeros((self.NumPatterns, self.height, self.width, 3), dtype=np.uint16)
			else:
				imgs = np.zeros((self.NumPatterns, self.height, self.width, 3), dtype=np.uint8)

		cv2.waitKey(200)
		# Loop through 
		for i in range(self.NumPatterns):
			if not i==0:
				cur_img = self.patterns[... ,i]
				cv2.imshow(window_name, cur_img.astype(np.float32))

			cv2.waitKey(400)  # ms # delay time
			# wait for 1000 ms ( syncrhonize this with your aquisition time)
			imgs[i, ...] = self.captureImage(fname=None)

			cv2.waitKey(400) # ms # delay time

		# Don't forgot to close all windows at the end of aquisition
		cv2.destroyAllWindows()

		for i in range(self.NumPatterns):
			fname = os.path.join(self.sessionDir, str(i) + ".png")
			if self._isMonochrome:
				cv2.imwrite(fname, imgs[i, ...], [cv2.IMWRITE_PNG_COMPRESSION, 0])  # RGB to BGR
			else:
				cv2.imwrite(fname, imgs[i, ...][..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 0]) #RGB to BGR

	def liveView(self):

		print("[INFO] starting live view thread...")
		viewGrabber = ptgCamStream(self.Cam).start()
		viewShower = imgShow(frame=viewGrabber.read()).start()

		time.sleep(0.1)
		while viewGrabber.running():
			if viewGrabber.stopped or viewShower.stopped:
				viewShower.stop()
				viewGrabber.stop()
				break

			frame = viewGrabber.read()
			viewShower.frame = frame

		viewGrabber.stop()
		viewShower.stop()
		viewGrabber.__exit__()
		viewShower.__exit__()

		viewGrabber.thread.join()
		viewShower.thread.join()
		del viewGrabber
		del viewShower
		cv2.destroyWindow("Live View")

	def setupHDR(self):
		window_name = 'projector'
		cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)

		# If projector is placed right to main screen (see windows properties in your operating system)
		# if the pattern is displayed at the wrong monitor you need to play around with the coordinates here until the image is displayed at the right screen
		cv2.moveWindow(window_name, self.displayWidth, 0)
		pattern = np.ones((self.displayHeight, self.displayWidth), dtype=np.float32)
		cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
		cv2.imshow(window_name, pattern.astype(np.float32))
		cv2.waitKey(200)
		self.Cam.setupHDR()
		cv2.destroyAllWindows()

	def captureHDRSequence(self):
		# TODO: hdr capture with different pattern
		window_name = 'projector'
		cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)

		# If projector is placed right to main screen (see windows properties in your operating system)
		# if the pattern is displayed at the wrong monitor you need to play around with the coordinates here until the image is displayed at the right screen
		cv2.moveWindow(window_name, self.displayWidth, 0)

		cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
		cv2.imshow(window_name, self.patterns[..., 0].astype(np.float32))

		if self._isMonochrome:
			imgs = np.zeros((self.NumPatterns, self.height, self.width), dtype=np.uint8)
		else:
			imgs = np.zeros((self.NumPatterns, self.height, self.width, 3), dtype=np.uint8)

		cv2.waitKey(200)
		# Loop through
		for i in range(self.NumPatterns):
			if not i == 0:
				cur_img = self.patterns[..., i]
				cv2.imshow(window_name, cur_img.astype(np.float32))
			cv2.waitKey(4)  # ms
			# wait for 1000 ms ( syncrhonize this with your aquisition time)
			imgs[i, ...] = self.captureImage(fname=None)
		# cv2.waitKey(5) # ms

		# Don't forgot to close all windows at the end of aquisition
		cv2.destroyAllWindows()

		for i in range(self.NumPatterns):
			fname = os.path.join(self.sessionDir, str(i) + ".png")
			if self._isMonochrome:
				cv2.imwrite(fname, imgs[i, ...], [cv2.IMWRITE_PNG_COMPRESSION, 0])  # RGB to BGR
			else:
				cv2.imwrite(fname, imgs[i, ...][..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 0])  # RGB to BGR

	def displayCalib(self):
		# TODO: Need to use threading so that the GUI won't stuck!
		pattern = cv2.imread("./8_24_checker.png", cv2.IMREAD_GRAYSCALE)
		# window_name = 'projector'
		# cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
		#
		# # If projector is placed right to main screen (see windows properties in your operating system)
		# # if the pattern is displayed at the wrong monitor you need to play around with the coordinates here until the image is displayed at the right screen
		# cv2.moveWindow(window_name, self.displayWidth, 0)
		#
		# cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
		# cv2.imshow(window_name, pattern.astype(np.float32))
		# k = cv2.waitKey(0)
		# if k == 27:  # wait for ESC key to exit
		# 	cv2.destroyAllWindows()

		patternShower = patternDisplay(displayHeight=self.displayHeight, displayWidth=self.displayWidth, pattern=pattern).start()

	def __exit__(self):
		self.Cam.__exit__()

