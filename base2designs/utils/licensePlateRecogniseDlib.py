
# import the necessary packages
from __future__ import print_function
import cv2
import dlib
# from pyimagesearch.descriptors.blockbinarypixelsum import BlockBinaryPixelSum
import numpy as np

class LicensePlateRecogniseDlib:

  def __init__(self, detectorFileName):
    # initialize the dlib license plate detector
    self.detector = dlib.simple_object_detector(detectorFileName)


  def detect(self, image, imageDebugEnable=False, detectLicenseText=False):
    # copy the image because we are going to modify, and want to leave original image intact
    imageCopy = image.copy()
    #imageCopy = image
    # reverse the color order from BGR to RGB format used by dlib
    imageCopy = cv2.cvtColor(imageCopy, cv2.COLOR_BGR2RGB)
    # detect the license plates
    plates = self.detector(imageCopy)
    if imageDebugEnable == True:
      print("Number of plates detected: {}".format(len(plates)))
      for k, d in enumerate(plates):
      	print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
      		k, d.left(), d.top(), d.right(), d.bottom()))
      win = dlib.image_window()
      win.clear_overlay()
      win.set_image(imageCopy)
      win.add_overlay(plates)
      dlib.hit_enter_to_continue()
    licensePlateFound = False
    plateList = []
    for lpBox in plates:


      # compute the center of the license plate bounding box
      cX = lpBox.left() + lpBox.width()/2
      cY = lpBox.top() + lpBox.height()/2
      # Python does not use Rect. Use tuples to define top left and bottom right corners
      # ((x0,y0), (x1,y1))

      lpBoxPtsPython = [(max(0,lpBox.left()), max(0,lpBox.top())),
                        (np.array(lpBox.right()).clip(0,image.shape[1]),
                         np.array(lpBox.bottom()).clip(0,image.shape[0]))]


      licensePlateFound = True
      lpOrigin = (cY, cX)
      text = ""

      plateList.append((text,lpOrigin, lpBoxPtsPython))

    return licensePlateFound, plateList

