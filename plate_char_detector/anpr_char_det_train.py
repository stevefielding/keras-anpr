# USAGE
# python anpr_char_det_train.py --modelPath models --imagePath ./../datasets/lplates/train
#  You can either pass the annFile (xml annotations file), or if you don't then annotations are loaded from the file name

# labelbinarizer
# Fit to full set of 10 numeric and 26 alphas
# lb.fit(['0','1', ... ,'9','a','b', ... ,'z'])
# Then we can transform all the test and training license plates.
# But how do we transform license plates less than 7 characters?
# Appears that we can use an input which was not presented during "fit" operation, eg
# lb.transform(['0','1','2','c','a',"blank"])
# "blank", returns an all zero vector, [0, 0, 0, ... ,0, 0], whereas the other targets return one-hot vectors
# Matt Earl does not have this problem, because he assumes that all plates contain 7 characters
# Check Google Street View paper, and see what they do. StreetView paper, does not backprop when a digit is absent.
# Not sure how that would work in Keras? Can we simply find the output and reflect this back as the target?
# Sounds like a non-standard feature
# If we use the zero vector to represent blanks, does this effectively disable
# back propagation? I don't think so.
# Could just add blank to the list when "fitting". No, what is a blank target?
# OK, so we need an 8th char to represent the number of characters. I do not know how to add a different classifier from
# the other seven, so let's just make it the same length (ie 36), and encode the length as
# '1', '2', ... ,'7'
# TODO: Need to add plate text length. For now only 7 char plate text is allowed

# import the necessary packages
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras import losses
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
from skimage import img_as_ubyte
from keras.utils import plot_model
import os
import sys
from keras.callbacks import ModelCheckpoint
from keras import regularizers
# enable search for base2designs module directory in parent directory
sys.path.append(os.path.split(os.getcwd())[0])
from base2designs.preprocessing import ImageToArrayPreprocessor
from base2designs.preprocessing import SimplePreprocessor
from base2designs.datasets import AnprLabelProcessor
from base2designs.datasets import AnprDatasetLoader
from base2designs.nn.conv import AnprCharDet

def plot(H, epochs, filename):
  # plot the training loss and accuracy
  plt.style.use("ggplot")
  plt.figure()
  plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
  plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
  plt.title("Training Loss")
  plt.xlabel("Epoch #")
  plt.ylabel("Loss")
  plt.legend()
  plt.savefig(filename)
  #plt.show()

def evaluate(model, testX, testY):
  # evaluate the network
  # get the predictions, and create clean one-hot predictions
  print("[INFO] display some results...")
  preds = model.predict(testX, batch_size=32)
  argMax = preds.argmax(axis=-1)
  predsClean = np.zeros_like(preds, dtype=np.int)
  for i in np.arange(len(argMax)):
    for j in np.arange(len(argMax[i])):
      predsClean[i, j, argMax[i, j]] = 1

  # get the ground truth and predicted plate text
  gtPlateText = alp.inverse_transform(testY)
  predPlateText = alp.inverse_transform(predsClean)

  # Generate some statistics
  numCorrChars = 0
  totalNumChars = PLATE_TEXT_LEN * len(predPlateText)
  numCorrPlates = 0
  for i in np.arange(len(predPlateText)):
    charCorr = 0
    for j in np.arange(PLATE_TEXT_LEN):
      if predPlateText[i][j] == gtPlateText[i][j]:
        numCorrChars += 1
        charCorr += 1
    if charCorr == PLATE_TEXT_LEN:
      numCorrPlates += 1
  numCorrPlates = 100. * numCorrPlates / len(predPlateText)
  numCorrChars = 100. * numCorrChars / totalNumChars
  print("[INFO] Number of correct plates: {:2.02f}%".format(numCorrPlates))
  print("[INFO] Number of correct chars: {:2.02f}%".format(numCorrChars))
  return numCorrPlates, numCorrChars

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--imagePath", required=True,
	help="path to input dataset")
ap.add_argument("-a", "--annFile", required=False, default=None,
	help="path to annotations file")
ap.add_argument("-m", "--modelPath", required=True,
	help="path to output model")
args = vars(ap.parse_args())

# Check the arguments before proceeding
if os.path.exists(args["imagePath"]) == False:
  print("[ERROR] --imagePath \"{}\", does not exist.".format(args["imagePath"]))
  sys.exit()
if os.path.exists(args["modelPath"]) == False:
  print("[ERROR] --modelPath \"{}\", does not exist.".format(args["modelPath"]))
  sys.exit()
if args["annFile"] != None:
  if os.path.exists(args["annFile"]) == False:
    print("[ERROR] --annFile \"{}\", does not exist.".format(args["annFile"]))
    sys.exit()

# Some constants
EPOCHS = 2000      # Number of epochs of training
INPUT_WIDTH = 128  # Network input width
INPUT_HEIGHT = 64  # Network input height
LEARN_RATE=0.001   # Network learning rate
augEnabled = True  # Image augmentation. Helps reduce over-fitting

# construct the image generator for data augmentation
# If values are too large, then the plate characters can be moved outside the image boundaries
# Use deep-learning/pb_code/chapter02-data_augmentation to view the augmented images
# 2/26/18 Reduced the magnitude of the variations. This just about keeps the image inside the boundaries
# of the frame
aug = ImageDataGenerator(rotation_range=4, width_shift_range=0.05,
  height_shift_range=0.05, shear_range=0.1, zoom_range=0.1,
  horizontal_flip=False, fill_mode="nearest")

# initialize the image preprocessors
# sp converts image to gray, and then resizes to 128,64
# iap converts the OpenCV numpy array format to Keras image library format. ie adds an extra dimension to the image, ie 128,64,1
# iap should be applied after any preprocessors that use opencv routines.
sp = SimplePreprocessor(INPUT_WIDTH,INPUT_HEIGHT)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities
# to the range [0, 1]
print("[INFO] loading images...")
adl = AnprDatasetLoader(preprocessors=[sp,iap])
(data, labels, winLocs, fileNames, plateCnt) = adl.loadData(args["imagePath"], annFile=args["annFile"], verbose=30,)
if len(data) == 0:
  print("[ERROR] No image files found in \"{}\"".format(args["imagePath"]))
  sys.exit()
data = data.astype("float") / 255.0

# set up the label classes
plateChars = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I',
                                      'J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
plateLens = [1,2,3,4,5,6,7]
PLATE_TEXT_LEN = plateLens[-1]
NUM_CHAR_CLASSES = len(plateChars)

alp = AnprLabelProcessor(plateChars, plateLens)
# convert the labels from integers to one-hot vectors
plateLabelsOneHot = alp.transform(labels)

# partition the data into training and testing splits using 85% of
# the data for training and the remaining 15% for testing
(trainX, testX, trainY, testY) = train_test_split(data, plateLabelsOneHot,
	test_size=0.15, random_state=42)

# Reshape the output vectors to match outputs expected by the model
trainY = trainY.reshape(-1,PLATE_TEXT_LEN, NUM_CHAR_CLASSES)
testY = testY.reshape(-1,PLATE_TEXT_LEN, NUM_CHAR_CLASSES)

# initialize the optimizer and model
print("[INFO] compiling model...")
#opt = SGD(lr=LEARN_RATE, decay=LEARN_RATE/EPOCHS)
#opt = RMSprop(lr=LEARN_RATE, decay=LEARN_RATE/EPOCHS)
opt = RMSprop(lr=LEARN_RATE)
#opt = Adam(lr=LEARN_RATE)
model = AnprCharDet.build(width=INPUT_WIDTH, height=INPUT_HEIGHT, depth=1, textLen=PLATE_TEXT_LEN, numCharClasses=NUM_CHAR_CLASSES)
model.compile(loss='categorical_crossentropy', optimizer=opt)

# Add L2 regularizers to every layer
#for layer in model.layers:
#  layer.kernel_regularizer = regularizers.l2(0.01)

plot_model(model, to_file="anprCharDet.png", show_shapes=True)

# construct the callback to save only the *best* model to disk
# based on the validation loss
fname = os.path.sep.join([args["modelPath"],
	"model-{epoch:03d}-{val_loss:.4f}.hdf5"])
checkpoint = ModelCheckpoint(fname, monitor="val_loss", mode="min",
	save_best_only=True, period=50, verbose=1)
callbacks = [checkpoint]

# train the network
print("[INFO] training network...")
if augEnabled == True:
  H= model.fit_generator(aug.flow(trainX, trainY, batch_size=32),
    validation_data=(testX, testY), epochs=EPOCHS,
    steps_per_epoch=len(trainX) // 32, callbacks=callbacks, verbose=0)
else:
  H = model.fit(trainX, trainY, validation_data=(testX, testY),
	  batch_size=32, epochs=EPOCHS, callbacks=callbacks, verbose=1)
plot(H, EPOCHS, "anpr_char_det_train_plot.png")


# evaluate the network after training
print("[INFO] display some results after training...")
trainError = evaluate(model, testX, testY)

# save the network to disk
#print("[INFO] serializing network...")
#model.save(args["model"])




