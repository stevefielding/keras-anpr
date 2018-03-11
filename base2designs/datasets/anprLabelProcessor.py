# import the necessary packages
import numpy as np
from sklearn.preprocessing import LabelBinarizer

class AnprLabelProcessor:
  # init the label binarizers. Maps classes to a set of one-hot vectors
  def __init__(self, plateChars, plateLens):
    # convert the labels from integers to vectors
    self.plate_lb = LabelBinarizer().fit(plateChars)
    self.charCnt_lb = LabelBinarizer().fit(plateLens)
    self.numClassesPerChar = len(plateChars)
    self.maxPlateLen = plateLens[-1]

  # Generate one-hot vectors for every plate
  def transform(self, labels):
    # Create a list of chars for each plate
    plateLabel = np.empty((len(labels), self.maxPlateLen), dtype=np.unicode_)
    for (i, label) in enumerate(labels):
      for j in range(0, self.maxPlateLen):
        plateLabel[i, j] = label[j]

    # Create a list of plate lengths for each plate
    #plateLenLabel = np.zeros((len(labels), 1), dtype=int)
    #for (i, label) in enumerate(labels):
    #  plateLenLabel[i, 0] = label[7]

    # Create the one hot labels for each plate
    #plateLabelsOneHot = np.zeros((len(labels), (37 * 7) + 7), dtype=int)
    plateLabelsOneHot = np.zeros((len(labels), (self.numClassesPerChar * self.maxPlateLen)), dtype=int)
    for i in range(len(labels)):
      oneHotText = self.plate_lb.transform(plateLabel[i])
      #oneHotCharCnt = self.charCnt_lb.transform(plateLenLabel[i])
      #plateLabelsOneHot[i] = np.concatenate((oneHotText.flatten(), oneHotCharCnt.flatten()))
      plateLabelsOneHot[i] = oneHotText.flatten()

    return plateLabelsOneHot

  # for every plate generate license plate chars, and license plate length
  def inverse_transform(self,oneHotLabels):
    plates = []
    plateLens = []
    oneHotLenDemuxed = []
    for i in range(len(oneHotLabels)):
      oneHotDemuxed = []
      for j in range(self.maxPlateLen):
        onehotDemux = np.array(oneHotLabels[i,j])
        oneHotDemuxed.append(onehotDemux)
      oneHotDemuxed = np.array(oneHotDemuxed)
      plate = self.plate_lb.inverse_transform(oneHotDemuxed)
      plates.append(plate)
      #oneHotLenDemux = np.array(oneHotLabels[i, 37 * 7:])
      #oneHotLenDemuxed.append(oneHotLenDemux)
    #oneHotLenDemuxed = np.array(oneHotLenDemuxed)
    #plateLens = (self.charCnt_lb.inverse_transform(oneHotLenDemuxed))

    #return plates, plateLens
    return plates