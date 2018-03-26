
# import the necessary packages
from __future__ import print_function
import numpy as np
from base2designs.utils.licensePlateRecogniseDlib import LicensePlateRecogniseDlib

class FindFrameWithPlate:

  def __init__(self, detectorFileName, searchCropFactorX=1, searchCropFactorY=1,
               margin = 0, removeOverlay=False):
    # initialize some class vars
    #self.licensePlateRecognise = LicensePlateRecognise(char_classifier, digit_classifier, allchar_classifier)
    self.licensePlateRecogniseDlib = LicensePlateRecogniseDlib(detectorFileName)
    self.searchCropFactorX = searchCropFactorX
    self.searchCropFactorY = searchCropFactorY
    self.margin = margin
    self.removeOverlay = removeOverlay

  # Crop the image, and use dlib FHOG+LSVM to find plates
  # Create a list containing the plates extracted from the image
  # Return the boolean licensePlateFound, and the list of plate images
  def extractPlate(self, image):
    # if overlay time etc are present then remove the top and bottom of the image
    if (self.removeOverlay == True):
      image = image[75:-40, 0:-1]
    # Crop a portion of the image from top, bottom left and right
    # This helps reduce the processing load
    cfX = (1.0 - 1.0/self.searchCropFactorX) / 2.0
    cfY = (1.0 - 1.0/self.searchCropFactorY) / 2.0
    image = image[int(image.shape[0]*cfY):int(-image.shape[0]*cfY)-1, int(image.shape[1]*cfX):int(-image.shape[1]*cfX)-1]
    # find license plates in the current image
    (licensePlateFound, licensePlateList) = self.licensePlateRecogniseDlib.detect(image, imageDebugEnable=False)

    # if license plates have been found, then get the bounding box for each plate and add to list of plates
    licensePlateFoundAndInBounds = False
    plateImages = []
    plateBoxes = []
    if licensePlateFound == True:
      lpBoxes = [i[2] for i in licensePlateList]
      for lpBoxPts in lpBoxes:
        t = lpBoxPts[0][1]
        b = lpBoxPts[1][1]
        l = lpBoxPts[0][0]
        r = lpBoxPts[1][0]
        # check for out of bounds. Only log plates that are within frame boundary
        if t > 0+self.margin and l > 0+self.margin and b < image.shape[0]-self.margin and r < image.shape[1]-self.margin:
          # add some extra margin. Useful if we want to apply image augmentation during training
          # Crop the plate form the image, and append
          # cropped plate and bounding box to lists
          (top, left) = (np.array([t, l]) - self.margin).clip(0)
          bottom = (b + self.margin).clip(0, image.shape[0])
          right = (r + self.margin).clip(0, image.shape[1])
          plateImage = image[top:bottom, left:right]
          plateImages.append(plateImage)
          plateBoxes.append(lpBoxPts)
          licensePlateFoundAndInBounds = True

    return [licensePlateFoundAndInBounds, plateImages, plateBoxes]

