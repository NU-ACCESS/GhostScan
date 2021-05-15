import os, sys, glob
import math
import argparse
import numpy as np
import cv2
from cv2 import aruco
import scipy.spatial
import math
import matplotlib.pyplot as plt

IMGPATTERN = "*.png"
PRECISION_IMG=np.dtype(np.float32)
numCam = 8

def readFileList(imgFolder):
    print(os.path.join(imgFolder, IMGPATTERN))
    imgFileList = glob.glob(os.path.join(imgFolder, IMGPATTERN))
    # self.imgFileList = os.listdir(self.imgFolder)
    # self.imgFileList.remove('.DS_Store') # remove system database log
    imgFileList.sort()

    return imgFileList

def processDeflectometry(imgPath):

    imgFileList = readFileList(imgPath)
    resultPath = os.path.join(imgPath, "results")
    if not os.path.exists(resultPath):
        os.makedirs(resultPath)

    assert (len(imgFileList) == 8), ("Images in the directory should be 8. Now is ",len(imgFileList))

    img = cv2.imread(os.path.join(imgPath, imgFileList[0]), cv2.IMREAD_COLOR)
    imgSetDeflectometry = np.zeros((8, img.shape[0], img.shape[1], img.shape[2]), dtype=np.float32)
    imgSetDeflectometry[0, ...] = img

    for i in range(len(imgFileList) - 1):

        img = cv2.imread(os.path.join(imgPath, imgFileList[i + 1]), cv2.IMREAD_COLOR)

        # if i == 0:
        #     imgSetDeflectometry = np.zeros((8, img.shape[0], img.shape[1], img.shape[2]), dtype=np.float32)

        imgSetDeflectometry[i + 1, ...] = img
    return imgSetDeflectometry

def setImgSet(imgSet):

    assert imgSet.shape[0] == numCam, "Deflectometry require 8 different patterns!"

    [height, width, nColor] = imgSet[0, ...].shape
    imgSet = (imgSet / np.max(imgSet)).astype(PRECISION_IMG)
    imgVSet = np.zeros((int(numCam / 2), height, width), dtype=PRECISION_IMG)
    imgHSet = np.zeros((int(numCam / 2), height, width), dtype=PRECISION_IMG)

    for n in range(numCam):
        if n < int(numCam / 2):
            img = cv2.cvtColor(imgSet[n, ...], cv2.COLOR_BGR2GRAY)
            imgVSet[n, ...] = img
        elif n >= int(numCam / 2):
            img = cv2.cvtColor(imgSet[n, ...], cv2.COLOR_BGR2GRAY)
            imgHSet[n - int(numCam / 2), ...] = img
    return imgHSet, imgVSet

def setImgSetVar(imgSet):

    assert imgSet.shape[0] == numCam, "Deflectometry require 8 different patterns!"

    [height, width, nColor] = imgSet[0, ...].shape
    #imgSet = (imgSet / np.max(imgSet)).astype(PRECISION_IMG)
    imgVSet = np.zeros((int(numCam / 2), height, width), dtype=PRECISION_IMG)
    imgHSet = np.zeros((int(numCam / 2), height, width), dtype=PRECISION_IMG)

    for n in range(numCam):
        if n < int(numCam / 2):
            img = cv2.cvtColor(imgSet[n, ...], cv2.COLOR_BGR2GRAY)
            imgVSet[n, ...] = img
        elif n >= int(numCam / 2):
            img = cv2.cvtColor(imgSet[n, ...], cv2.COLOR_BGR2GRAY)
            imgHSet[n - int(numCam / 2), ...] = img
    return imgHSet, imgVSet

def calPhase(imgPath, correctPhaseWrap= 3.8):
    imgset = processDeflectometry(imgPath)
    imgHSet, imgVSet = setImgSet(imgSet= imgset)
    for i in range(2):
        if i == 0:
            imgSet = imgVSet
        else:
            imgSet = imgHSet
        PhaseMap = -1 * np.arctan2((imgSet[0, ...] - imgSet[2, ...]), (imgSet[1, ...] - imgSet[3, ...]))
        if correctPhaseWrap != 0:
            PhaseMap = np.angle(np.exp(1j * (PhaseMap - correctPhaseWrap)))
        if i == 0:
            PhaseMap0 = PhaseMap
        else:
            PhaseMap1 = PhaseMap
    print(imgHSet.shape)

    return PhaseMap0, PhaseMap1   


"""
imgPath = "D:/Deflectometry/pyspin0316/mirror0623"
imgset = processDeflectometry(imgPath= imgPath)
print(imgset.shape)
imghset, imgvset = setImgSet(imgSet= imgset)
print(imghset.shape, imgvset.shape)
p0, p1 = calPhase(imgPath)
plt.imshow(p0)
plt.show()
plt.imshow(p1)
plt.show()
"""


