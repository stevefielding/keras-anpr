# -------------------- analyseLogFile.py -------------------------
# USAGE
# python --logFile lplateLogExample.txt -- reportFile lplateReport.txt

# Process the log file
# Group license plates that match by at least 5 chars
# Discard duplicate license plates that are in the same video clip, and within 1000 frames

import copy
import argparse
import numpy as np
import os
import sys

MIN_FRAME_GAP_BETWEEN_UNIQUE_PLATES = 1000

# construct the argument parser and parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--logFile", required=True,
	help="path to log file")
ap.add_argument("-r", "--reportFile", required=True,
	help="path to report file")
args = vars(ap.parse_args())

# check arguements
if os.path.exists(args["logFile"]) == False:
  print("[ERROR]: --logFile \"{}\" does not exist".format(args["logFile"]))
  sys.exit()
#if os.path.exists(args["reportFile"]) == True:
#  print("[ERROR]: --reportFile \"{}\" already exists. Delete first".format(args["logFile"]))
#  sys.exit()


# read the log file
logFile = open(args["logFile"], "r")
logs = logFile.read()
logFile.close()
logsSplit = [s.strip().split(",") for s in logs.splitlines()]
logsSplit = np.array(logsSplit)
plateDict = dict()

# Create a dictionary (plateDict) with the plateText as the keys
for logLine in logsSplit:
  videoFileName = logLine[0]
  imageFileName = logLine[1]
  date = logLine[2]
  time = logLine[3]
  frameNum = logLine[4]
  plateTexts = logLine[5:]
  print(logLine)
  for plateText in plateTexts:
    if plateText in plateDict:
      # plateText, videoFileName, imageFileName, date, time, frameNumber
      plateDict[plateText].append([plateText, logLine[0], logLine[1], logLine[2], logLine[3], int(logLine[4])])
    else:
      plateDict[plateText] = [[plateText, logLine[0], logLine[1], logLine[2], logLine[3], int(logLine[4])]]

# combine dictionary keys that match by at least 5 chars
plateDictDeDuped = copy.deepcopy(plateDict)
plateDict2 = copy.deepcopy(plateDict)
for plateText1 in plateDict.keys():
  del plateDict2[plateText1]
  for plateText2 in plateDict2.keys():
    matchCnt = 0
    for i in np.arange(len(plateText1)):
      if plateText1[i] == plateText2[i]:
        matchCnt += 1
    if matchCnt >= 5:
      entryAdds = plateDict[plateText2]
      for entryAdd in entryAdds:
        plateDictDeDuped[plateText1].append(entryAdd)
      del plateDictDeDuped[plateText2]

# Compare plate text from similar plates and select the most popular
plateDictPred = {}
for plateText in plateDictDeDuped.keys():
  if len(plateDictDeDuped[plateText]) == 1:
    # if there is only one plate, then copy to predicted plates
    plateDictPred[plateText] = plateDictDeDuped[plateText]
  else:
    # For each char position build a histogram of character frequencies
    charDicts = [{}, {}, {}, {}, {}, {}, {}]
    for plateEntry in plateDictDeDuped[plateText]:
      for (i,char) in enumerate(plateEntry[0]):
        if char in charDicts[i].keys():
          charDicts[i][char] += 1
        else:
          charDicts[i][char] = 1

    # for each histogram, sort and then select the most frequently occurring characters
    predPlateText =[]
    for (i,charDict) in enumerate(charDicts):
      plateTuples = charDict.items()
      plateChars = sorted(plateTuples, key=lambda x:x[1] )
      predPlateText.append(plateChars[-1][0])
    predPlateText = ''.join(predPlateText)

    # Copy dictionary entries from plateDictDeDuped, but use the new predicted plate text as the key
    plateDictPred[predPlateText] = plateDictDeDuped[plateText]

# for each plateText group, create videoFileName sub-groups
plateDictDeDuped2 = {}
plateDictPredCopy = copy.deepcopy(plateDictPred)
for plateText in plateDictPredCopy.keys():
  plateDictSubGroup = {}
  if len(plateDictPredCopy[plateText]) == 1:
    # only one entry for this plateText, so simply copy
    plateDictDeDuped2[plateText] = plateDictPredCopy[plateText]
  else:
    # more than one entry, so create filename sub-groups
    for plateEntry in plateDictPredCopy[plateText]:
      if plateEntry[1] in plateDictSubGroup.keys():
        plateDictSubGroup[plateEntry[1]].append(plateEntry)
      else:
        plateDictSubGroup[plateEntry[1]] = [plateEntry]

    # process the filename sub-groups
    for imageFileName in plateDictSubGroup:
      plateEntries = plateDictSubGroup[imageFileName]
      if len(plateEntries) == 1:
        # There is only one entry, so need to sort, just add to the dictionary
        if plateText in plateDictDeDuped2.keys():
          plateDictDeDuped2[plateText].append(plateEntries[0])
        else:
          plateDictDeDuped2[plateText] = plateEntries
      else:

        # sort the multiple entries by frame number
        # Add the entry with the smallest frame number to the dict
        # and add subsequent entries only if they are separated by a
        # sufficient number of frames
        plateEntriesSorted = sorted(plateEntries, key=lambda x: x[5])
        for (i,plateEntry) in enumerate(plateEntriesSorted):
          if i == 0:
            frameNumBase = plateEntry[5]
            # add the first entry
            if plateText in plateDictDeDuped2.keys():
              plateDictDeDuped2[plateText].append(plateEntry)
            else:
              plateDictDeDuped2[plateText] = [plateEntry]
          elif plateEntry[5] > frameNumBase + MIN_FRAME_GAP_BETWEEN_UNIQUE_PLATES:
            frameNumBase = plateEntry[5]
            plateDictDeDuped2[plateText].append(plateEntry)


# generate the report file
reportFile = open(args["reportFile"], "w")
for plateText in plateDictDeDuped2.keys():
  reportFile.write("{}\n".format(plateText))
  plateEntries = plateDictDeDuped2[plateText]
  plateEntries = sorted (plateEntries, key=lambda x:x[3])
  for plateEntry in plateEntries:
    reportFile.write("  {} {} {} {} {} {}\n".format(plateEntry[3], plateEntry[4], plateEntry[0],
                                        plateEntry[1], plateEntry[2], plateEntry[5] ))
reportFile.close()
print("Finished")





