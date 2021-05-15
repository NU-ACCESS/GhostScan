import sys, glob, os
import math
import argparse
import numpy as np
import cv2
from cv2 import aruco
from Calibrations.Camera2ScreenCalib import calib
# from Camera2ScreenCalib import calib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def geoCalibII(geoCalibImgUndistortFile, displayintersect, displayScaleFactor):
    displayintersectX = displayintersect[0,:,:]
    displayintersectY = displayintersect[1,:,:]
    displayintersectX = -displayintersectX
    displayintersectX = displayintersectX / displayScaleFactor
    displayintersectY = displayintersectY / displayScaleFactor
    rawImg = cv2.imread(geoCalibImgUndistortFile)
    rawImgGray = cv2.cvtColor(rawImg, cv2.COLOR_BGR2GRAY)
    dispImg = cv2.imread("8_24_checker.png")
    dispImgGray = cv2.cvtColor(dispImg, cv2.COLOR_BGR2GRAY)
    dispImgGray = cv2.flip(dispImgGray, 1)
    fig = plt.figure(figsize=(15,10))
    plt.imshow(rawImgGray)
    imgSavePath = os.path.join(os.path.join(os.path.join(os.getcwd(), 'CalibrationImages'), 'Geometric'), 'geoCalibResults')
    fig.savefig(imgSavePath + '/original' + '.png')
    plt.show()
    minX = -dispImgGray.shape[1] / 2
    maxX = dispImgGray.shape[1] / 2
    minY = -dispImgGray.shape[0] / 2
    maxY = dispImgGray.shape[0] / 2
    dispPattern = dispImgGray

    retrieveImage = np.zeros((displayintersectX.shape[0], displayintersectX.shape[1]))
    print(retrieveImage.shape)
    for i in range(retrieveImage.shape[0]):
        for j in range(retrieveImage.shape[1]):
            roundCor = [np.round(displayintersectY[i,j]), np.round(displayintersectX[i,j])]
            if minY < roundCor[0] and roundCor[0] < maxY and minX < roundCor[1] and roundCor[1] < maxX:
                #print(int(roundCor[0] + maxY), int(roundCor[1] + maxX))
                retrieveImage[i,j] = dispPattern[int(roundCor[0] + maxY), int(roundCor[1] + maxX)]
    compareImg = retrieveImage * 0.5 + rawImgGray + 0.5
    fig = plt.figure(figsize=(15,10))
    plt.imshow(retrieveImage)
    fig.savefig(imgSavePath + '/reprojected' + '.png')
    plt.show()
    fig = plt.figure(figsize=(15,10))
    plt.imshow(compareImg)
    fig.savefig(imgSavePath + '/comparation' + '.png')
    plt.show()

    #return display_intersect_mat, cam_mirror_intersect_mat, rcamera_dis, tcamera_dis