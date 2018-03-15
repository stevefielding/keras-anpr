# USAGE
# python anpr_char_det_load.py --model ../models/save/model-5999-0.1340_lr_0.003_do_0.7_do_0.5_artificial_aug_no_l2reg.hdf5 --imagePath ../../datasets/lplates/verify

# Load a set of images with license plates. The license plate can be the entire image, in which case no annotation file is required,
# and it is assumed that the plate text is embedded in the filename.
# Otherwise read the plate locations and plate text from the annotation file
# Load the model architecture and weights from file, detect license plate characters.
# Generate and display summary of prediction results and display 5 random plate results

# import the necessary packages
import os
import sys
# enable search for base2designs module directory in parent directory
sys.path.append(os.path.split(os.getcwd())[0])
from base2designs.preprocessing import ImageToArrayPreprocessor
from base2designs.preprocessing import SimplePreprocessor
from base2designs.datasets import AnprLabelProcessor
from base2designs.datasets import AnprDatasetLoader
import numpy as np
import argparse
import cv2
from skimage import img_as_ubyte
from keras.utils import plot_model
from keras.models import load_model
import shutil

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--imagePath", required=True,
	help="path to input dataset")
ap.add_argument("-a", "--annFile", required=False, default=None,
	help="path to annotations file")
ap.add_argument("-m", "--model", required=True,
	help="path to model")
ap.add_argument("-b", "--copy_bad_plates", required=False, default="false",
	help="copy plates with incorrect predictions to a new directory. Useful for debugging bad labels")
args = vars(ap.parse_args())

if os.path.exists(args["imagePath"]) == False:
  print("[ERROR] --imagePath \"{}\", does not exist".format(args["imagePath"]))
  sys.exit()
if os.path.exists(args["model"]) == False:
  print("[ERROR] --model \"{}\", does not exist".format(args["model"]))
  sys.exit()

EPOCHS = 10

# grab the list of images that we'll be describing
print("[INFO] loading images...")

# initialize the image preprocessors
# sp converts image to gray, and then resizes to 128,64
# iap converts the OpenCV numpy array format to Keras image library format. ie adds an extra dimension to the image, ie 128,64,1
# iap should be applied after any preprocessors that use opencv routines.
sp = SimplePreprocessor(128,64)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities
# to the range [0, 1]
adl = AnprDatasetLoader(preprocessors=[sp,iap])
(data, labels, winLocs, fileNames, plateCnt) = adl.loadData(args["imagePath"], annFile=args["annFile"], verbose=30,)
data = data.astype("float") / 255.0

# set up the label classes
plateChars = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I',
                                      'J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
plateLens = [1,2,3,4,5,6,7]
PLATE_TEXT_LEN = plateLens[-1]
NUM_CHAR_CLASSES = len(plateChars)

# convert the labels from integers to one-hot vectors
alp = AnprLabelProcessor(plateChars, plateLens)
plateLabelsOneHot = alp.transform(labels)

# rename the input and output vectors, and reshape the output vector
# to match the output from the model
# TODO: Incorporate reshape into the alp.transform function
(testX, testY) = data, plateLabelsOneHot
testY = testY.reshape(-1,PLATE_TEXT_LEN, NUM_CHAR_CLASSES)

# load the pre-trained network
print("[INFO] loading pre-trained network...")
model = load_model(args["model"])
plot_model(model, to_file="anprCharDet.png", show_shapes=True)

# evaluate the network
# get the predictions, and create clean one-hot predictions
print("[INFO] display some results...")
preds = model.predict(testX, batch_size=32)
argMax = preds.argmax(axis=-1)
predsClean = np.zeros_like(preds,dtype=np.int)
for i in np.arange(len(argMax)):
  for j in np.arange(len(argMax[i])):
    predsClean[i,j,argMax[i,j]] = 1

# get the ground truth and predicted plate text
# by converting from one-hot vectors to text
gtPlateText = alp.inverse_transform(testY)
predPlateText = alp.inverse_transform(predsClean)

# Look for duplicate plates
plateDict = {}
dupeCnt = 0
for i in np.arange(len(gtPlateText)):
  plateText = ''.join(gtPlateText[i])
  if plateText in plateDict:
    plateDict[plateText] += 1
    dupeCnt += 1
  else:
    plateDict[plateText] = 1
print ("{} plates in the dataset. {} duplicate plates found".format(len(gtPlateText), dupeCnt))

# Generate some statistics
numCorrChars = 0
totalNumChars = PLATE_TEXT_LEN * len(predPlateText)
numCorrPlates = 0
badPlateIndices = []
for i in np.arange(len(predPlateText)):
  charCorr = 0
  for j in np.arange(PLATE_TEXT_LEN):
    if predPlateText[i][j] == gtPlateText[i][j]:
      numCorrChars += 1
      charCorr += 1
  if charCorr == PLATE_TEXT_LEN:
    numCorrPlates += 1
  else:
    badPlateIndices.append(i)
print("[INFO] Number of correct chars: {:2.02f}%".format(100. * numCorrChars/totalNumChars))
print("[INFO] Assuming one char error per plate, minimum number of correct plates: {:2.02f}%"
      .format(100 * (1.0 - ((1.0 - (numCorrChars/totalNumChars)) / 0.1429))))
print("[INFO] Number of correct plates: {:2.02f}%".format(100. * numCorrPlates / len(predPlateText)))

if (args["copy_bad_plates"] == "true"):
  for i in badPlateIndices:
    print("[INFO] Plate: {}. Predicted plate: {}".format(''.join(gtPlateText[i]), ''.join(predPlateText[i])))
    try:
      shutil.copy(fileNames[i], "badPlates/"+fileNames[i].split('/')[-1])
    except OSError as e:
      print("OS error({0}): {1}".format(e.errno, e.strerror))
      sys.exit(1)



# Display some random results
SCALE = 4
for i in np.random.randint(0, high=len(testY), size=5):
  image = (testX[i] * 255).astype("int")
  image.shape = (image.shape[0], image.shape[1])
  cv_image = cv2.resize(img_as_ubyte(image), (image.shape[1]*SCALE, image.shape[0]*SCALE), interpolation=cv2.INTER_AREA)
  cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
  #print("[INFO] Plate: {}, len: {}. Predicted plate: {}, pred len: {}".format( gtPlateText[i], gtPlateLen[i], predPlateText[i], predPlateLen[i]))
  print("[INFO] Plate: {}. Predicted plate: {}".format( ''.join(gtPlateText[i]), ''.join(predPlateText[i])))
  cv2.imshow("Validation image[{}]".format(i),cv_image)
cv2.waitKey(0)

