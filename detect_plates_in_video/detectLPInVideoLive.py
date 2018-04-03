# USAGE
# python detectLPInVideoLive.py --conf conf/lplates_smallset.json

# For every frame in a video clip: Detects license plate locations
# If a plate is found in a frame, then the cropped image of the plate is passed to a character classifier
# and the plate text is predicted.
# The plate text is added to the history buffer.
# Every ~1 second, the frame buffer is processed and a dictionary of unique plates (plateDictBest) is extracted.
# If any of the plates in plateDictBest has not been seen before then they are logged to file
# If every frame in a 1920x1080 video is processed, then can only process 5 fps
# Can process 20 frames per second if decimating by factor of 4, and still full frame size
# profiling:
# python -m cProfile -s time detectLPInVideoLive.py --conf conf/lplates_smallset.json > profile.txt

# import the necessary packages
from __future__ import print_function

import argparse
import shutil
import time
import os
import sys
import cv2
import datetime
import re
from imutils import paths
from keras.models import load_model
from keras.utils import plot_model

# enable search for base2designs module directory in parent directory
sys.path.append(os.path.split(os.getcwd())[0])

from base2designs.datasets import AnprLabelProcessor
from base2designs.utils.charClassifier import CharClassifier
from base2designs.utils.conf import Conf
from base2designs.utils.findFrameWithPlate import FindFrameWithPlate
from base2designs.utils.plateHistory import PlateHistory
from base2designs.utils.folderControl import FolderControl
from base2designs.preprocessing import ImageToArrayPreprocessor
from base2designs.preprocessing import SimplePreprocessor

# construct the argument parser and parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to the json config file")
args = vars(ap.parse_args())

# Check if config file exists
if os.path.exists(args["conf"]) == False:
  print("[ERROR]: --conf \"{}\" does not exist".format(args["conf"]))
  sys.exit()

# Read the json config and initialize the plate detector
conf = Conf(args["conf"])
folderController = FolderControl()
findFrameWithPlate = FindFrameWithPlate(str(conf["dlib_SVM_detector"]),
                                        searchCropFactorX=(conf["searchCropFactorX"]),
                                        searchCropFactorY=(conf["searchCropFactorY"]),
                                        margin = conf["plateMargin"],
                                        removeOverlay=False)

# initialize the image preprocessors
# sp converts image to gray, and then resizes to 128,64
# iap converts the OpenCV numpy array format to Keras image library format. ie adds an extra dimension to the image, ie 128,64,1
# iap should be applied after any preprocessors that use opencv routines.
sp = SimplePreprocessor(128,64)
iap = ImageToArrayPreprocessor()

# set up the label classes
plateChars = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I',
                                      'J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
plateLens = [1,2,3,4,5,6,7]
PLATE_TEXT_LEN = plateLens[-1]
NUM_CHAR_CLASSES = len(plateChars)

# convert the labels from integers to one-hot vectors
alp = AnprLabelProcessor(plateChars, plateLens)

# open the logFile, create if it does not exist, otherwise open in append mode
logFilePath = "{}/{}" .format(conf["output_image_path"], conf["log_file_name"])
if (os.path.isdir(conf["output_image_path"]) != True):
  os.makedirs(conf["output_image_path"])
if os.path.exists(logFilePath) == False:
  logFile = open(logFilePath, "w")
else:
  logFile = open(logFilePath, "a")

# load the pre-trained char classifier network, init the char classifier utility
# and load the plate history utility
print("[INFO] loading pre-trained network...")
ccModel = load_model(conf["model"])
plot_model(ccModel, to_file="anprCharDet.png", show_shapes=True)
cc = CharClassifier(alp, ccModel, preprocessors=[sp,iap], croppedImagePreprocessors=[sp])
plateHistory = PlateHistory(conf["output_image_path"], conf["output_cropped_image_path"], logFile,
                            saveAnnotatedImage=conf["saveAnnotatedImage"] == "true")

validImages = 0
dtNow = datetime.datetime.now()
plateLogLatency = conf["plateLogLatency"] * conf["videoFrameRate"]
perfUpdateInterval = conf["perfUpdateInterval"] * conf["videoFrameRate"]
print("[INFO] opening video source...")
start_time = time.time()
frameCount = 0
frameCntForPlateLog = 0
oldFrameCount = 0
frameDecCnt = 1
destFolderRootName = "{:02}-{:02}-{:02}".format(dtNow.year, dtNow.month, dtNow.day)
folderController.createDestFolders(destFolderRootName, conf["save_video_path"],
                                   conf["output_image_path"], conf["output_cropped_image_path"])

if conf["target"] == "jetson":
  #vs = cv2.VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720,format=(string)I420, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")
  vs = cv2.VideoCapture("v4l2src device=""/dev/video1"" ! appsink")
else:
  vs = cv2.VideoCapture(0)
  vs.set(cv2.CAP_PROP_FRAME_WIDTH,1280 )
  vs.set(cv2.CAP_PROP_FRAME_HEIGHT,720 )
  vs.set(cv2.CAP_PROP_FPS,conf["videoFrameRate"] )

print ("Camera frame: {} x {} @ {} Hz".format(vs.get(cv2.CAP_PROP_FRAME_WIDTH), vs.get(cv2.CAP_PROP_FRAME_HEIGHT), vs.get(cv2.CAP_PROP_FPS)))

# Check for valid video
if vs is None:
  print("[ERROR] Cannot connect to video source")
  sys.exit()

# Prepare findFrameWithPlate for a new video sequence
plateLogFlag = False
perfUpdateFlag = False
firstPlateFound = False
loggedPlateCount = 0
platesReadyForLog = False
while True:
  dtNow = datetime.datetime.now()
  timeNow = "{:02}.{:02}.{:02}".format(dtNow.hour, dtNow.minute, dtNow.second)
  # read the next frame from the video stream
  ret = vs.grab() # grab frame but do not decode

  #if end of current video file, then clean up and quit
  key = cv2.waitKey(1) & 0xFF
  if (ret==False or key != 255):
    # close any open windows
    cv2.destroyAllWindows()
    logFile.close()
    sys.exit()

  # update some tracking variables
  frameCount += 1
  if firstPlateFound == True:
    frameCntForPlateLog += 1
  if frameCntForPlateLog > plateLogLatency:
    plateLogFlag = True
    frameCntForPlateLog = 0
  if frameCount % perfUpdateInterval == 0:
    perfUpdateFlag = True

  # Decimate the frames
  if (frameDecCnt == 1):
    ret, frame = vs.retrieve() # retrieve the already grabbed frame
    frameCopy = frame.copy()
    # process the frame. Find image with license plate
    (licensePlateFound, plateImages, plateBoxes) = findFrameWithPlate.extractPlate(frame)

    # if license plates have been found, then predict the plate text, and add to the history
    if licensePlateFound == True:
      (plateList, plateImagesProcessed) = cc.predictPlateText(plateImages)
      plateHistory.addPlatesToHistory(plateList, plateImagesProcessed, plateBoxes, frame, timeNow, frameCount)
      validImages += 1
      firstPlateFound = True
      platesReadyForLog = True

    # if sufficient time has passed since the last log, then
    # get a dictionary of the best de-duplicated plates,
    # and remove old plates from history
    if plateLogFlag == True:
      platesReadyForLog = False
      plateLogFlag = False
      plateDictBest = plateHistory.selectTheBestPlates()
      plateHistory.removeOldPlatesFromHistory()
      # generate output files, ie cropped Images, full image and log file
      plateHistory.logToFile(plateDictBest, destFolderRootName)
      loggedPlateCount += len(plateDictBest)

    # show the frame and predicted plate text
    if conf["display_video_enable"] == "true":
      if licensePlateFound == True:
        for (plateBox, plateText) in zip(plateBoxes, plateList):
          cv2.rectangle(frameCopy, plateBox[0], plateBox[1], (0, 255, 0), 2)
          cv2.putText(frameCopy, plateText, plateBox[0], cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
      cv2.imshow("Frame", frameCopy)

  # update frame decimator count
  if (frameDecCnt == conf["frameDecimationFactor"]):
    frameDecCnt = 1
  else:
    frameDecCnt += 1

  # update performance tracking
  if perfUpdateFlag == True:
    perfUpdateFlag = False
    curTime = time.time()
    processingTime = curTime - start_time
    start_time = curTime
    frameCountDelta = frameCount - oldFrameCount
    fps = frameCountDelta / processingTime
    oldFrameCount = frameCount
    print("[INFO] Processed {} frames in {:.2f} seconds. Frame rate: {:.2f} Hz".format(frameCountDelta, processingTime, fps))
    print("[INFO] validImages: {}, frameCount: {}".format(validImages, frameCount))






