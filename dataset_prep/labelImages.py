# -------------------- labelImages.py -------------------------
# USAGE
# python labelImages.py --input_image_path ../../datasets/lplates_smallset/hand_labelled/images \
# --output_image_path ../../datasets/lplates_smallset/hand_labelled/images_labelled --label_filename labels.txt

# Installing tkinter for Python 3 was problematic
# https://www.raspberrypi.org/forums/viewtopic.php?t=130808
# I think this might be the key: sudo apt-get install python3-pil.imagetk

# Display the image along with a text entry box for entering the license plate characters
# If the license plate characters are visible, then enter into the box, and press return.
# The image will be copied to a new location, and an entry added to the labels.txt file
# If the license plate characters are not visible, then simply press enter, and the image file
# will be copied to a discard directory, and NO entry will be made in labels.txt
# The next image will be displayed. Repeat until all the images are read.
# You can quit the application at any time using Ctrl-C, and then restart at a later time.
# None of the information that you entered will be lost. 

import argparse
from imutils import paths
import shutil
import cv2
from PIL import Image
import tkinter
from PIL import ImageTk
import os
import sys

# construct the argument parser and parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_image_path", required=True, help="path to input image files")
ap.add_argument("-o", "--output_image_path", required=True, help="path to input image files")
ap.add_argument("-l", "--label_filename", required=True, help="label filename. Saved to output_image_path")
ap.add_argument("-r", "--rename", required=False, default=False, help="rename the file")
args = vars(ap.parse_args())

# set the paths for the output files
output_labeled_image_path = args["output_image_path"] + "/img"
output_discard_image_path = args["output_image_path"] + "/discard"
if os.path.exists(output_labeled_image_path) == False:
  os.mkdir(output_labeled_image_path)
if os.path.exists(output_discard_image_path) == False:
  os.mkdir(output_discard_image_path)

# read the config file, get list of all input image files,
# and open the labels output file in append mode.
# ie do not overwrite previous contents
myPaths = paths.list_images(args["input_image_path"])
if os.path.exists(args["output_image_path"]) == False:
  os.mkdir(args["output_image_path"])
labelOutputFile = open(args["output_image_path"] + "/" + args["label_filename"], "a")

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

  # Add image, prompt text, and text Entry box to window
  root.geometry('%dx%d' % (image.size[0],image.size[1]))
  tkpi = ImageTk.PhotoImage(image)
  label_image = tkinter.Label(root, image=tkpi)
  label_image.place(x=0,y=0,width=image.size[0],height=image.size[1])
  label = tkinter.Label(root, text="Enter license plate characters")
  label.pack()
  e = tkinter.Entry(root)
  e.pack()
  e.focus_set()
  root.title(imagePath)
  root.mainloop() # wait until user presses 'return'

  # get the license plate text
  licensePlate = e.get()

  # Finished with the window, destroy
  label_image.destroy()
  e.destroy()
  label.destroy()
  print(licensePlate)

  # extract the file name from the path, create new file name
  oldFileName = imagePath.split("/")[-1]
  #newFileName = str(imageCnt) + imagePath[imagePath.rfind("."):]
  # prepend the image directory, which should contain the date
  # need to do this, because the file name only contains the time of day
  # and may not be unique
  if args["rename"] == True:
    newFileName = imagePath.split("/")[-2] + "_" + oldFileName
  else:
    newFileName = oldFileName

  # if the number of license plate chars is greater than zero, then
  # set the output path to the "labeled" directory, and add the old
  # file name, the new file name, and the license plate characters to
  # the label.txt file
  if len(licensePlate) > 0:
    imagePathOut = output_labeled_image_path + "/" + newFileName
    fileNameText = newFileName + "," + licensePlate.upper() + "\n"
    labelOutputFile.write(fileNameText)
    labelOutputFile.flush()
  # else if no license plate chars then set the output path to the "discard" directory
  else:
    imagePathOut = output_discard_image_path + "/" + newFileName

  # save image file to new location
  try:
    # os.rename(imagePath, imagePathOut) #does not work between two different file systems
    shutil.move(imagePath, imagePathOut)
  except OSError as e:
    print("OS error({0}): {1}".format(e.errno, e.strerror))
    sys.exit(1)



labelOutputFile.close()


