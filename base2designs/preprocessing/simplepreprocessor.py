# Copied from pyImageSearch
# import the necessary packages
import cv2

class SimplePreprocessor:
	def __init__(self, width, height, inter=cv2.INTER_AREA):
		# store the target image width, height, and interpolation
		# method used when resizing
		self.width = width
		self.height = height
		self.inter = inter

	def preprocess(self, image):
		# resize the image to a fixed size, ignoring the aspect
		# ratio
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		return cv2.resize(image, (self.width, self.height),
			interpolation=self.inter)