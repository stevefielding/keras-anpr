# Copied from PyImageSearch
# USAGE
# python test_detector2.py --detector plateDetector_toy.svm \
# --xml  ../../datasets/lplates_smallset/hand_labelled/images_labelled/plateAnnotationsWithLabels.xml
# import the necessary packages
from __future__ import print_function
import argparse
import dlib

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-x", "--xml", required=True, help="path to input XML file")
ap.add_argument("-d", "--detector", required=True, help="Path to trained object detector")
args = vars(ap.parse_args())

# show the training accuracy
print("[INFO] Recall accuracy: {}".format(
	dlib.test_simple_object_detector(args["xml"], args["detector"])))

