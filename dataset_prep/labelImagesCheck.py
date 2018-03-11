# -------------------- labelImagesCheck.py -------------------------
# python labelImagesCheck.py --labelFileIn ../../datasets/lplates_smallset/hand_labelled/images_labelled/labels.txt  \
# --labelFileOut ../../datasets/lplates_smallset/hand_labelled/images_labelled/labelsNew.txt \
# --imagePath ../../datasets/lplates_smallset/hand_labelled/images_labelled/img

# Read a label.txt file and extract the contents to a dictionary where the keys are the filenames
# and the values are the plate text.
# Now display the images with an embedded text box containing the plate text that corresponds with the
# filename key. The user has the opportunity to view the plate text in the image and compare to the plate text
# retrieved from the lables.txt file. The text can be left unmodified or edited. Either way the plate text and
# filename are written to the new output label file.

from imutils import paths
from PIL import Image
import tkinter
from PIL import ImageTk
import argparse
import numpy as np
import os
import sys

# construct the argument parser and parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--labelFileIn", required=True,
	help="path to labels input file")
ap.add_argument("-o", "--labelFileOut", required=True,
	help="path to labels output file")
ap.add_argument("-i", "--imagePath", required=True,
	help="path to image files")
args = vars(ap.parse_args())

# check arguements
if os.path.exists(args["labelFileIn"]) == False:
  print("[ERROR]: --labelFileIn \"{}\" does not exist".format(args["labelFileIn"]))
  sys.exit()
if os.path.exists(args["imagePath"]) == False:
  print("[ERROR]: --imagePath \"{}\" does not exist".format(args["imagePath"]))
  sys.exit()
if os.path.exists(args["labelFileOut"]) == True:
  print("[ERROR]: --labelFileOut \"{}\" already exists. Delete first".format(args["labelFileOut"]))
  sys.exit()

# read the labels file and copy to dictionary
labelFile = open(args["labelFileIn"], "r")
labels = labelFile.read()
labelFile.close()
labelsSplit = [s.strip().split(",") for s in labels.splitlines()]
labelsSplit = np.array(labelsSplit)
labelsDict = dict()
keys = labelsSplit[:,0]
values = labelsSplit[:,1]
for i in np.arange(len(keys)):
  labelsDict[keys[i]] = values[i].upper()
numberOfLabels = len(keys)
print("[INFO] Number of plates in the dataset: {}".format(numberOfLabels))

# get list of all input image files,
# and open the labels output file in write mode.
myPaths = paths.list_files(args["imagePath"], validExts=(".jpg"))
labelOutputFile = open(args["labelFileOut"], "w")

# return key event handler
def return_key_exit_mainloop (event):
	event.widget.quit() # this will cause mainloop to unblock.

# configure the main window, and bind the return key
root = tkinter.Tk()
root.geometry('+%d+%d' % (100,100))
root.bind('<Return>', return_key_exit_mainloop)


# For every file, add a new line of text to the comma delimited label file
# Each line will contain the old fileName, new file name, and license plate characters
for imagePath in sorted(myPaths):

	# Read the image and resize
	image = Image.open(imagePath)
	basewidth = 1000
	wpercent = (basewidth / float(image.size[0]))
	hsize = int((float(image.size[1]) * float(wpercent)))
	image = image.resize((basewidth, hsize), Image.ANTIALIAS)

	# Add image, prompt text, and text Entry box to window, default text is from labels file
	root.geometry('%dx%d' % (image.size[0],image.size[1]))
	tkpi = ImageTk.PhotoImage(image)
	label_image = tkinter.Label(root, image=tkpi)
	label_image.place(x=0,y=0,width=image.size[0],height=image.size[1])
	label = tkinter.Label(root, text="Enter license plate characters")
	label.pack()
	e = tkinter.Entry(root)
	fileName = imagePath.split('/')[-1]
	if fileName in labelsDict:
	  e.insert(0, labelsDict[fileName].upper())
	e.pack()
	e.focus_set()
	root.title(imagePath)
	root.mainloop() # wait until user presses 'return'

	# get the license plate text
	licensePlate = e.get().upper()

	# Finished with the window, destroy
	label_image.destroy()
	e.destroy()
	label.destroy()
	print(licensePlate)

	# extract the file name from the path, create new file name
	oldFileName = imagePath.split("/")[-1]

	# if the number of license plate chars is greater than zero, then 
	# set the output path to the "labeled" directory, and add the
	# file name, ",", and the license plate characters to
	# the label.txt file
	if len(licensePlate) > 0:
		fileNameText = oldFileName + "," + licensePlate + "\n"
		labelOutputFile.write(fileNameText)
	# else if no license plate chars then set the output path to the "discard" directory
	else:
		fileNameText = oldFileName + "," + "NOPLATE" + "\n"
		labelOutputFile.write(fileNameText)

labelOutputFile.close()


