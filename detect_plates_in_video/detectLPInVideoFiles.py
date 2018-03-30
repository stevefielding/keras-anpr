# USAGE
# python detectLPInVideoFiles.py --conf conf/lplates_smallset.json

# For every frame in a video clip: Detects license plate locations
# If a plate is found in a frame, then the cropped image of the plate is passed to a character classifier
# and the plate text is predicted.
# The plate text is added to the history buffer.
# Every ~1 second, the frame buffer is processed and a dictionary of unique plates (plateDictBest) is extracted.
# If any of the plates in plateDictBest has not been seen before then they are logged to file
# If every frame in a 1920x1080 video is processed, then can only process 5 fps
# Can process 20 frames per second if decimating by factor of 4, and still full frame size
# profiling:
# python -m cProfile -s time detectLPInVideoFiles.py --conf conf/lplates_smallset.json > profile.txt

# import the necessary packages
from __future__ import print_function
import argparse
import shutil
import time
import os
import sys
import cv2
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

quit = False
plateLogLatency = conf["plateLogLatency"]* conf["videoFrameRate"]
platesReadyForLog = False
while True:
  # Get the list of video files
  totalFrameCount = 0
  validImages = 0
  loggedPlateCount = 0
  myPaths = paths.list_files(conf["input_video_path"], validExts=(".h264", "mp4"))
  # loop over all the video files
  for videoPath in sorted(myPaths):
    print("[INFO] reading video file {}...".format(videoPath))
    start_time = time.time()
    frameCount = 0
    frameCntForPlateLog = 0
    frameDecCnt = 1
    m = re.search(r"([0-9]{4}[-_][0-9]{2}[-_][0-9]{2})",videoPath)
    if m:
      destFolderRootName = m.group(1) #assumes video stored in sub-directory
    else:
      destFolderRootName = "YYYY-MM-DD"
    folderController.createDestFolders(destFolderRootName, conf["save_video_path"],
                                       conf["output_image_path"], conf["output_cropped_image_path"])
    vs = cv2.VideoCapture(videoPath)
    # wait for valid video
    while vs is None:
      vs = cv2.VideoCapture(videoPath)
      if vs is None:
        print("[ERROR] video: {} is empty".format(videoPath))
        time.sleep(1)
    # Prepare findFrameWithPlate for a new video sequence
    plateLogFlag = False
    firstPlateFound = False
    while True:
      # read the next frame from the video stream
      ret = vs.grab() # grab frame but do not decode
      #ret, frame = vs.read()
      #if end of current video file, then move video file to saveDir, quit loop and get the next video file
      if (ret==False):
        # We have reached the end of the video clip. Save any residual plates to log
        # Remove all the plate history
        if platesReadyForLog == True:
          plateDictBest = plateHistory.selectTheBestPlates()
          # generate output files, ie cropped Images, full image and log file
          plateHistory.logToFile(plateDictBest, destFolderRootName)
          loggedPlateCount += len(plateDictBest)
        plateHistory.clearHistory()
        firstPlateFound = False

        # copy video clip from input directory to saveVideoDir
        outputPathSaveOriginalImage = conf["save_video_path"] + "/" + destFolderRootName + videoPath[videoPath.rfind("/") + 0:]
        if (conf["move_video_file"] == "true"):
          try:
            # os.rename(videoPath, outputPathSaveOriginalImage) #does not work between two different file systems
            shutil.move(videoPath, outputPathSaveOriginalImage)
          except OSError as e:
            print("OS error({0}): {1}".format(e.errno, e.strerror))
            sys.exit(1)
        break

      # Decimate the the frames
      frameCount += 1
      if firstPlateFound == True:
        frameCntForPlateLog += 1
      if frameCntForPlateLog > plateLogLatency:
        plateLogFlag = True
        frameCntForPlateLog = 0
      if (frameDecCnt == 1):
        ret, frame = vs.retrieve() # retrieve the already grabbed frame
        # process the frame. Find image with license plate
        (licensePlateFound, plateImages, plateBoxes) = findFrameWithPlate.extractPlate(frame)

        # if license plates have been found, then predict the plate text, and add to the history
        if licensePlateFound == True:
          (plateList, plateImagesProcessed) = cc.predictPlateText(plateImages)
          plateHistory.addPlatesToHistory(plateList, plateImagesProcessed, plateBoxes, frame, videoPath, frameCount)
          validImages += 1
          firstPlateFound = True
          platesReadyForLog = True

        # if sufficient time has passed since the last log, then
        # get a dictionary of the best de-duplicated plates,
        # and remove old plates from history, then save images and update the log
        if plateLogFlag == True:
          platesReadyForLog = False
          plateLogFlag = False
          plateDictBest = plateHistory.selectTheBestPlates()
          # generate output files, ie cropped Images, full image and log file
          plateHistory.logToFile(plateDictBest, destFolderRootName)
          plateHistory.removeOldPlatesFromHistory()
          loggedPlateCount += len(plateDictBest)

        # show the frame and record if the user presses a key
        if conf["display_video_enable"] == "true":
          cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key is pressed, break from the loop
        if key == ord("q"):
          quit = True
          break
      if (frameDecCnt == conf["frameDecimationFactor"]):
        frameDecCnt = 1
      else:
        frameDecCnt += 1

    processingTime = time.time() - start_time
    fps = frameCount / processingTime
    print("Processed {} frames in {} seconds. Frame rate: {} Hz".format(frameCount, processingTime, fps ) )
    totalFrameCount += frameCount
  key = cv2.waitKey(1) & 0xFF
  # if the `q` key is pressed, break from the loop
  if key == ord("q") or quit == True or conf["infinite_main_loop"] != "true":
    break
  # wait 3 seconds before checking for new video files
  #time.sleep(3)
# close any open windows
print("totalNumberOfImages: {}, imagesWithPlates: {}, platesLogged: {}".format(totalFrameCount, validImages, loggedPlateCount))
cv2.destroyAllWindows()
logFile.close()

