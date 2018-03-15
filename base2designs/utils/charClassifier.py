
import numpy as np
import cv2
import re

class CharClassifier:
  def __init__(self, labelProcessor, charClassifierModel, output_image_path,
               output_cropped_image_path, logFile, saveAnnotatedImage=False, preprocessors=None, croppedImagePreprocessors=None):
    # store the labelProcessor
    self.labelProcessor = labelProcessor

    # store the charClassifierModel
    self.charClassifierModel = charClassifierModel

    self.output_image_path = output_image_path
    self.output_cropped_image_path = output_cropped_image_path
    self.logFile = logFile
    self.saveAnnotatedImage = saveAnnotatedImage

    # store the image preprocessors
    self.preprocessors = preprocessors
    self.croppedImagePreprocessors = croppedImagePreprocessors

    # if the preprocessors are None, initialize them as
    # empty lists
    if self.preprocessors is None:
      self.preprocessors = []
    if self.croppedImagePreprocessors is None:
      self.croppedImagePreprocessors = []

  # Apply optional processor to the image
  def processImage(self, image, preprocessors):
    # check to see if our preprocessors are not None
    if preprocessors is not None:
      # loop over the preprocessors and apply each to
      # the image
      for p in preprocessors:
        image = p.preprocess(image)

    return image


  # Find all the bounding boxes for the plates in the input image
  # For every bounding box
  #   Crop the plate from the image
  #   Apply image processing to cropped plate
  #   Apply charClassifierModel to predict plate text
  #   Append the plate text to plateList
  #   Optionally annotate the original image with bounding box and plate text
  #   print the plate text
  #   save the image
  # Return plateList
  def findCharsInPlate(self, bestImage, licensePlateList, outputFileName, destFolderRootName, frameCount, videoFileName, margin=0, imageDebugEnable=False):
    outputFullImageFileName = outputFileName
    plateList = []
    lpBoxes = [i[2] for i in licensePlateList]
    for lpBoxPts in lpBoxes:

      # add some extra margin. Useful if we want to apply image augmentation during training
      t = lpBoxPts[0][1]
      b = lpBoxPts[1][1]
      l = lpBoxPts[0][0]
      r = lpBoxPts[1][0]
      (top, left) = (np.array([t, l]) - margin).clip(0)
      bottom = (b + margin).clip(0, bestImage.shape[0])
      right = (r + margin).clip(0, bestImage.shape[1])
      plateImage = bestImage[top:bottom, left:right]
      plateImageProcessed = self.processImage(plateImage, self.preprocessors)

      # get the predictions, and create clean one-hot predictions
      print("[INFO] display some results...")
      preds = self.charClassifierModel.predict(np.array([plateImageProcessed]), batch_size=32)
      argMax = preds.argmax(axis=-1)
      predsClean = np.zeros_like(preds, dtype=np.int)
      for i in np.arange(len(argMax)):
        for j in np.arange(len(argMax[i])):
          predsClean[i, j, argMax[i, j]] = 1

      # get the predicted plate text
      # by converting from one-hot vectors to text
      # and append to list
      predPlateText = self.labelProcessor.inverse_transform(predsClean)
      predPlateText = ''.join(predPlateText[0])
      plateList.append((predPlateText))
      outputFullImageFileName += '_' + predPlateText

      # write cropped plate image to file, and append plate text to the file name
      # Mimic the file name format used by Supervisely (almost)
      outputCroppedPath = "{}/{}/{}.png".format(self.output_cropped_image_path, destFolderRootName, outputFileName + '_' + predPlateText + '_1')
      cv2.imwrite(outputCroppedPath, self.processImage(plateImage, self.croppedImagePreprocessors))

      # optionally annotate the image
      if self.saveAnnotatedImage == True:
        cv2.rectangle(bestImage, lpBoxPts[0], lpBoxPts[1], (0, 255, 0), 2)
        cv2.putText(bestImage, predPlateText, lpBoxPts[0], cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
      # print the plate text
      print("  plate: {}".format(predPlateText))

    # save the image
    print("[INFO] logging image to file: {}".format(outputFullImageFileName))
    outputPath = "{}/{}/{}.jpg".format(self.output_image_path, destFolderRootName, outputFullImageFileName)
    cv2.imwrite(outputPath, bestImage)

    # update the log file
    # videoFileName, imageFileName, date, time, frameNumber, plateText
    # lplate_toy_video4.mp4,2018_01_01/lplate_toy_video.mp4_51.jpg,2018_01_10,5:05,271,5HUY634,5JHY768
    imageFileName = "{}/{}.jpg".format(destFolderRootName, outputFullImageFileName)
    date = destFolderRootName
    m = re.search(r"^([0-9]{2}[.:][0-9]{2}[.:][0-9]{2})",videoFileName)
    if m:
      time = m.group(1)
    else:
      time = "HH.MM.SS"
    self.logFile.write("{},{},{},{},{},{}\n".format(videoFileName, imageFileName, date, time, frameCount, ','.join(plateList)))
    self.logFile.flush()

    return plateList

