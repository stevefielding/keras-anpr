
# import the necessary packages
from __future__ import print_function

import numpy as np
from scipy.spatial import distance as dist

from base2designs.utils.licensePlateRecogniseDlib import LicensePlateRecogniseDlib


class FindFrameWithPlate:


	def __init__(self, detectorFileName, samePlateMaxDist=100, searchCropFactorX=1, searchCropFactorY=1):
		# initialize some class vars
		#self.licensePlateRecognise = LicensePlateRecognise(char_classifier, digit_classifier, allchar_classifier)
		self.licensePlateRecogniseDlib = LicensePlateRecogniseDlib(detectorFileName)
		self.consec = None
		self.prevPlateOrigins = None
		self.samePlateMaxDist = samePlateMaxDist
		self.newVideoClip = True
		self.newPlateSeq = True
		self.fileCnt = 1
		self.plateImagesFound = 0
		self.plateFound = False
		self.plateSeqUnTerminated = False
		self.searchCropFactorX = searchCropFactorX
		self.searchCropFactorY = searchCropFactorY

	def startNewVideoClip(self):
		self.consec = None
		self.prevPlateOrigins = None
		self.newVideoClip = True
		self.newPlateSeq = True
		self.fileCnt = 1
		self.plateImagesFound = 0
		self.plateSeqUnTerminated = False
		self.plateFound = False

	def getBestFrame(self):
		return (self.consec)

	# process a video frame.
	# First determine if current frame contains a plate that is closer to the center of the frame than the previous frame.
	# If it does then save the frame along with plate details.
	# Frames with plates close together are considered part of a plateSequence
	def processSingleFrame(self, inputImagePath, frame, removeOverlay=False, detectLicenseText=False):
		#cv2.imshow("Frame", imutils.resize(frame,width=400))
		#cv2.waitKey(0)
		# strip the video file name for use when saving still images
		videoFileName = inputImagePath.split("/")[-1].split("[")[0]
		outputFileName = "{}_{}".format(videoFileName, self.fileCnt)
		self.fileCnt += 1
		# if overlay time etc are present then remove the top and bottom of the image
		if (removeOverlay == True):
			frame = frame[75:-40, 0:-1]
		#if (self.searchCropFactor!= 1):
		cfX = (1.0 - 1.0/self.searchCropFactorX) / 2.0
		cfY = (1.0 - 1.0/self.searchCropFactorY) / 2.0
		frame = frame[int(frame.shape[0]*cfY):int(-frame.shape[0]*cfY)-1, int(frame.shape[1]*cfX):int(-frame.shape[1]*cfX)-1]
		# find license plates in the current frame
		(licensePlateFound, licensePlateList) = self.licensePlateRecogniseDlib.detect(frame, imageDebugEnable=False)

		bestPlateFound = False
		# If license plate(s) found in the frame, then determine if any of the plates in the current frame
		# are the same as plates found in the last frame (We will call this sequence of frames a plateSequence).
		# We do this by checking the distance between plates in the current frame and plates in the previous frame.
		# Within the plateSequence determine if the current frame contains a plate closer to the frame centre than any
		# previous frame. If it does, then this is the best candidate so far, and we save the frame to self.consec
		if (licensePlateFound == True):

			# calculate the minimum distance from plate(s) centroid to the frame centre
			# if this is less than the dist for the previous frame, then update consec
			minDistToFrameCentre = None
			frameCentre = np.array(frame.shape[:2]) / 2
			plateOrigins = [i[1] for i in licensePlateList]
			for plateOrigin in plateOrigins:
				distToFrameCentre = dist.euclidean(plateOrigin, frameCentre)
				if minDistToFrameCentre == None:
					minDistToFrameCentre = distToFrameCentre
				elif distToFrameCentre < minDistToFrameCentre:
					minDistToFrameCentre = distToFrameCentre
			if self.consec is None or self.newPlateSeq == True:
				self.consec = [frame, minDistToFrameCentre, licensePlateList, outputFileName]
				self.newPlateSeq = False
				self.plateSeqUnTerminated = True
				self.plateFound = True
			elif minDistToFrameCentre < self.consec[1]:
				# if the distance is smaller than the current distance, then update the
				# bookkeeping variable
				self.consec = [frame, minDistToFrameCentre, licensePlateList, outputFileName]

			# If this is the first frame of a new video clip, then init some class variables
			if self.newVideoClip == True:
				self.newVideoClip = False
			# else the video clip is continuing, so let's determine if we need to start a new sequence
			else:
				minDistBetweenPlates = None
				# Compare all the current frame plate centroids with all the previous
				# frame plate centroids and find the minimum dist between plates
				for plateOrigin in plateOrigins:
					for prevPlateOrigin in self.prevPlateOrigins:
						distBetweenPlates = dist.euclidean(prevPlateOrigin, plateOrigin)
						if minDistBetweenPlates == None:
							minDistBetweenPlates = distBetweenPlates
						elif distBetweenPlates < minDistBetweenPlates:
							minDistBetweenPlates = distBetweenPlates
				# If the minimum distance between the current frame plate centroids and the previous frames
				# plate centroids is greater than samePlateMaxDist, then this is the end of the sequence
				# The sequence has been terminated and the best plate should be saved by the caller using
				# 'getBestPlate' to retrieve the best plate
				if (minDistBetweenPlates > self.samePlateMaxDist):
					# This is a new sequence. Terminate the previous sequence
					bestPlateFound = True
					self.plateImagesFound += 1
					self.newPlateSeq = True
					self.plateSeqUnTerminated = False
			self.prevPlateOrigins = plateOrigins

		return (bestPlateFound)


