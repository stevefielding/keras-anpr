
import numpy as np
import cv2
import re

class CharClassifier:
  def __init__(self, labelProcessor, charClassifierModel, preprocessors=None, croppedImagePreprocessors=None):
    # store the labelProcessor
    self.labelProcessor = labelProcessor

    # store the charClassifierModel
    self.charClassifierModel = charClassifierModel

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


  # pre-process all the plateImages, and predict the plate text
  def predictPlateText(self, plateImages):
    plateList = []
    plateImagesPreProcessed = []
    for plateImage in plateImages:
      # pre-process the plate image
      plateImageProcessed = self.processImage(plateImage, self.preprocessors)

      # get the predictions, and create clean one-hot predictions
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
      # print("  plate: {}".format(predPlateText))

      # convert image to grey scale and resize to 128x64
      plateImagesPreProcessed.append(self.processImage(plateImage, self.croppedImagePreprocessors))

    return (plateList, plateImagesPreProcessed)
