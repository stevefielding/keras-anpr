# Copied from PyImageSearch
# USAGE
# python train_detector.py --detector plateDetector_toy.svm \
# --xml ../../datasets/lplates_smallset/hand_labelled/images_labelled/plateAnnotationsWithLabels.xml

# import the necessary packages
from __future__ import print_function
import argparse
import dlib

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-x", "--xml", required=True, help="path to input XML file")
ap.add_argument("-d", "--detector", required=True, help="Path to the object detector")
args = vars(ap.parse_args())

# grab the default training options for the HOG + Linear SVM detector, then
# train the detector -- in practice, the `C` parameter should be cross-validated
print("[INFO] training detector...")
options = dlib.simple_object_detector_training_options()
options.C = 1.0
options.num_threads = 4
options.be_verbose = True
options.upsample_limit = 1
dlib.train_simple_object_detector(args["xml"], args["detector"], options)

# show the training accuracy
print("[INFO] training accuracy: {}".format(
	dlib.test_simple_object_detector(args["xml"], args["detector"])))

# load the detector and visualize the HOG filter
detector = dlib.simple_object_detector(args["detector"])
win = dlib.image_window()
win.set_image(detector)
dlib.hit_enter_to_continue()
