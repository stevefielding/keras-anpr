# USAGE
# python detectLPInLiveVideo.py --conf conf/lplates_smallset.json

# For every frame in a video clip: Detects license plate locations
# If a plate is found in a frame, and the plate location is the closest to the centre of the frame
# for that plate sequence, then it is flagged as the best frame and saved.
# A plate sequence is considered to be a sequence of plates when the distance between plate
# locations in successive frames is below a threshold
# Program is way too slow if you use every frame, and full frame size (1920x1080)
# Can process 45 frames per second if decimating by factor of 4, and reduce frame size by 2 in x and y dimension
# Doing actual processing on 45/4 frames per second, so that metric is sort of confusing.
# 3 out of 4 frames are simply discarded.
# profiling:
# python -m cProfile -s time removeLPAliasesFromVideo.py --conf conf/sunba_motion_detect_local_test.json > profile.txt

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

# Read the json config and initialize the frame processor
conf = Conf(args["conf"])
folderController = FolderControl()
findFrameWithPlate = FindFrameWithPlate(str(conf["dlib_SVM_detector"]),
                                        samePlateMaxDist=conf["samePlateMaxDist"],
                                        searchCropFactorX=(conf["searchCropFactorX"]), searchCropFactorY=(conf["searchCropFactorY"]))

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

# open the logFile, create if it does not exist
logFilePath = "{}/{}" .format(conf["output_image_path"], conf["log_file_name"])
if (os.path.isdir(conf["output_image_path"]) != True):
  os.makedirs(conf["output_image_path"])
if os.path.exists(logFilePath) == False:
  logFile = open(logFilePath, "w")
else:
  logFile = open(logFilePath, "a")

# load the pre-trained char classifier network, and init the char classifier utility
print("[INFO] loading pre-trained network...")
ccModel = load_model(conf["model"])
plot_model(ccModel, to_file="anprCharDet.png", show_shapes=True)
cc = CharClassifier(alp, ccModel, conf["output_image_path"], conf["output_cropped_image_path"], logFile,
                    saveAnnotatedImage=conf["saveAnnotatedImage"] == "true",
                    preprocessors=[sp,iap], croppedImagePreprocessors=[sp])

validImages = 0
dtNow = datetime.datetime.now()
print("[INFO] opening video source...")
start_time = time.time()
frameCount = 0
frameDecCnt = 1
destFolderRootName = "{}-{}-{}".format(dtNow.year, dtNow.month, dtNow.day)
folderController.createDestFolders(destFolderRootName, conf["save_video_path"],
                                   conf["output_image_path"], conf["output_cropped_image_path"])
vs = cv2.VideoCapture(0)
# wait for valid video
if vs is None:
  print("[ERROR] Cannot connect to video source")
  sys.exit()

# Prepare findFrameWithPlate for a new video sequence
findFrameWithPlate.startNewVideoClip()
while True:
  dtNow = datetime.datetime.now()
  imagePath = "{}.{}.{}".format(dtNow.hour, dtNow.minute, dtNow.second)
  # read the next frame from the video stream
  ret = vs.grab() # grab frame but do not decode
  #ret, frame = vs.read()
  #if end of current video file, then move video file to saveDir, quit loop and get the next video file
  key = cv2.waitKey(1) & 0xFF
  if (ret==False or key == ord("q")):
    # We have reached the end of the video clip, but the previous sequence was unterminated, and the best plate was not saved
    # This happens when the all the plates in a video clip are sufficiently close to be considered part of
    # a single sequence, and the sequence is not terminated
    if findFrameWithPlate.plateSeqUnTerminated == True:
      (bestImage, minDistToFrameCentre, licensePlateList, outputFileName) = findFrameWithPlate.getBestFrame()
      cc.findCharsInPlate(bestImage, licensePlateList, outputFileName, destFolderRootName, frameCount,
                          imagePath[imagePath.rfind("/") + 1:],
                          margin=conf["plateMargin"], imageDebugEnable=conf["imageDebugEnable"]=="true")
      validImages += 1
    # close any open windows
    cv2.destroyAllWindows()
    logFile.close()
    processingTime = time.time() - start_time
    fps = frameCount / processingTime
    print("[INFO] Processed {} frames in {} seconds. Frame rate: {} Hz".format(frameCount, processingTime, fps))
    print("[INFO] validImages: {}, frameCount: {}".format(validImages, frameCount))
    print("[INFO] video source terminated")
    sys.exit()

  # Decimate the the frames
  frameCount += 1
  if (frameDecCnt == 1):
    ret, frame = vs.retrieve() # retrieve the already grabbed frame
    # process the frame. Find image with license plate
    bestPlateFound = findFrameWithPlate.processSingleFrame(imagePath,
                                 frame,
                                 removeOverlay=(conf["removeOverlay"] =="true"),
                                 detectLicenseText=(conf["detectLicenseText"] == "true"))
    # if there are any valid results, save the cropped image to file
    if bestPlateFound == True:
      (bestImage, minDistToFrameCentre, licensePlateList, outputFileName) = findFrameWithPlate.getBestFrame()
      cc.findCharsInPlate(bestImage, licensePlateList, outputFileName, destFolderRootName, frameCount,
                          imagePath[imagePath.rfind("/") + 1:],
                          margin=conf["plateMargin"], imageDebugEnable=conf["imageDebugEnable"]=="true")
      validImages += 1
    # show the frame and record if the user presses a key
    if conf["display_video_enable"] == "true":
      cv2.imshow("Frame", frame)
  if (frameDecCnt == conf["frameDecimationFactor"]):
    frameDecCnt = 1
  else:
    frameDecCnt += 1






