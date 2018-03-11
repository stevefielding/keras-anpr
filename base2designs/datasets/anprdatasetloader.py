# import the necessary packages
import numpy as np
import cv2
import re
from collections import namedtuple
from base2designs.utils import paths
import random

plateInfo = namedtuple("plateInfo", ["plateText", "ignore", "top", "left", "width", "height"])


class AnprDatasetLoader:
  def __init__(self, preprocessors=None, detector=None):
    # store the image preprocessor
    self.preprocessors = preprocessors

    # if the preprocessors are None, initialize them as an
    # empty list
    if self.preprocessors is None:
      self.preprocessors = []

    # Copy the FHOG plus linear SVM detector
    self.detector = detector

  def processImage(self, image):
    # check to see if our preprocessors are not None
    if self.preprocessors is not None:
      # loop over the preprocessors and apply each to
      # the image
      for p in self.preprocessors:
        image = p.preprocess(image)

    #image = image[:, :, 0]
    #image.shape = (image.shape[0], image.shape[1], 1)
    return image


  def bb_intersection_over_union(self, boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

  def findNearestPlate(self, image, gtBox, verbose=False):
    # get the predicted boxes
    boxes = self.detector(image)

    # Find the box that best matches the ground truth box
    croppedImage = None
    plateFound = False
    iouMax = 0
    margin = 0
    for (i,b) in enumerate(boxes):
      # Add margin to the bounding box, and clip to image boundaries
      # dlib rectangle bottom right corner is inclusive, but numpy slicing is exclusive.
      # Add 1 to bottom right indices
      (top, left) = (np.array([b.top(), b.left()]) - margin).clip(0)
      (bottom, right) = (np.array([b.bottom(), b.right()]) + 1 + margin*2).clip(0, image.shape[1])

      predBox = [top, left, bottom, right]
      iou = self.bb_intersection_over_union(gtBox,predBox)
      if iou > iouMax:
        iouMax = iou
        (bestT,bestL,bestB,bestR) = (top,left,bottom,right)

    # crop plate from image if the IOU > 0.5
    if iouMax > 0.5:
      #print("top: {}, bottom: {}, left: {}, right: {}".format(bestT,bestL,bestB,bestR))
      croppedImage = image[bestT:bestB, bestL:bestR]
      plateFound = True
      #print("image shape: {}".format(croppedImage.shape))


    # Debug
    if verbose == True:
      if plateFound == True:
        cv2.imshow("findNearestPlate result", croppedImage)
        cv2.waitKey(0)
      else:
        print("No plate detected by FHOG plus lSVMM")

    return plateFound, croppedImage

  def loadAnnotations(self, fileName):
    xmlFile = open(fileName, "r")
    xmlText = xmlFile.read()

    # parse the xml file, and create a list of "image file" data
    allImages = re.findall(r".*<images>(.*)</images>", xmlText, re.DOTALL)
    regex = r"<image file='(.*?)'>\n(.*?)</image>"
    matches = re.finditer(regex, allImages[0], re.DOTALL)

    # Create a list of namedTuples. Each namedTuple consists of plateText, ignore flag, and
    # LP bounding box co-ordinates
    # Add the namedtuple to the dictionary. Use filename as the key
    imageDict = {}
    for (i, match) in enumerate(matches):
      image = match.group(1)
      imageFileName = image.split('/')[-1]
      imageDict[imageFileName] = []
      boxes = re.finditer(r"<box(.*)/>", match.group(2))
      for box in boxes:
        boxGroup1 = box.group(1)
        boxDef = re.findall(r"top='(\d*).*left='(\d*)'.*width='(\d*).*height='(\d*).*desc='(.*?)'", boxGroup1)
        ignore = re.search(r"ignore='(.*?)'", boxGroup1)
        if ignore == None:
          ignore = 0
        else:
          ignore = int(ignore.group(1))
        plateText = boxDef[0][4]
        length = len(plateText)
        for i in range(length,7):
          plateText += '*'
        plateText += str(length)
        myDetail = plateInfo(plateText=plateText, top=int(boxDef[0][0]), left=int(boxDef[0][1]), width=int(boxDef[0][2]),
                             height=int(boxDef[0][3]), ignore=ignore)
        imageDict[imageFileName].append(myDetail)

    return imageDict

  def loadData(self, imagePath, annFile=None, verbose=-1, winH=64, winW=128, margin=0):
    xs = []
    ys = []
    winLocs = []
    fnames = []
    plateGTCnt = 0
    platePredCnt = 0

    # Read the image filenames, and randomly shuffle. Random shuffle actually not required since sklearn.train_test_split will perform
    # random shuffle prior to training.
    imagePaths = np.array(list(paths.list_images(imagePath)))
    random.shuffle(imagePaths)
    print("Found {} image files".format(len(imagePaths)))

    # if no annotation file, then get the plate text from the filename
    # For each image file, pre-process the image and append to list
    # Append the filename and plate text to separate lists
    if annFile == None:
      plateGTCnt = len(imagePaths)
      platePredCnt = len(imagePaths)
      for (i, fileName) in enumerate(imagePaths):
        image = cv2.imread(fileName)
        image = self.processImage(image)
        fnameStripped = fileName.split('/')[-1]
        plateText = fnameStripped.split('_')[-2]
        if len(plateText) == 7:
          fnames.append(fileName)
          xs.append(image)
          ys.append(plateText)
        if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
          print("[INFO] processed {}/{}".format(i + 1, len(imagePaths)))

    # if annotation file, then load the annotations into a dictionary
    # For each image file, check to see if it is in the dictionary. If not, issue a warning
    # Otherwise use the annotated bounding box to crop the plates from the image
    # For every cropped plate, pre-process the plate image and append to list
    # Append the filename and plate text and bounding boxes to seperate lists
    else:
      annDict = self.loadAnnotations(annFile)
      # for every file path
      for (i, fileName) in enumerate(imagePaths):
        image = cv2.imread(fileName)
        fnameStripped = fileName.split('/')[-1]
        if fnameStripped in annDict:
          anns = annDict[fnameStripped]
          for ann in anns:
            plateGTCnt += 1
            winLocs.append([ann.top, ann.left, ann.width, ann.height])
            (top,left) = (np.array([ann.top, ann.left]) - margin).clip(0)
            bottom = (top + ann.height + margin*2).clip(0,image.shape[0])
            right = (left + ann.width + margin*2).clip(0,image.shape[1])
            if self.detector == None:
              imageCropped = image[top:bottom, left:right]
              plateFound = True
            else:
              gtBox = [top,left,bottom,right]
              plateFound, imageCropped = self.findNearestPlate(image, gtBox)
            if plateFound == True:
              imageCropped = self.processImage(imageCropped)
              platePredCnt += 1
              xs.append(imageCropped)
              fnames.append(fileName)
              ys.append(ann.plateText)
        else:
          print("[WARNING] {} not found in labels dict".format(fileName))

        # show an update every `verbose` images
        if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
          print("[INFO] processed {}/{}".format(i + 1, len(imagePaths)))

    print ("Ground truth plates cnt: {}, Predicted plate cnt: {}".format(plateGTCnt, platePredCnt))
    return np.array(xs), np.array(ys), np.array(winLocs), np.array(fnames), np.array([plateGTCnt, platePredCnt])


