import os, sys, glob
import math
import argparse
import numpy as np
import cv2
from cv2 import aruco
import scipy.spatial
import math
import matplotlib
import matplotlib.pyplot as plt
import Reconstructions.generatephase as gp
from Calibrations.Camera2ScreenCalib import calib
from Reconstructions.normalInWorld import normalInWorld, normalCompare, calAngFeature


def normal2RGB(normalImage, normal_mat_origin, normalmap_world):
    # Now let's visualize the normal map!
    resultFolder = os.path.join(os.path.join(os.path.join(os.getcwd(), 'CapturedImages'), 'sequenceImages'), 'results')
    fig = plt.figure(figsize=(15,10))
    plt.imshow(normalImage)
    plt.title("Normal Map Encoded in RGB Image")
    fig.savefig(resultFolder + '/Normal Map Encoded in RGB Image' + '.png')
    plt.show()
    # visualize normal map in camera perspective
    norm = matplotlib.colors.Normalize(vmin= -1, vmax = 1)
    norm2 = matplotlib.colors.Normalize(vmin= -1, vmax = 1)
    fig = plt.figure(figsize=(15,10))
    plt.imshow(normal_mat_origin[:,:,0], norm= norm)
    plt.title("X Component")
    plt.colorbar()
    fig.savefig(resultFolder + '/X Component' + '.png')
    plt.show()
    fig = plt.figure(figsize=(15,10))
    plt.imshow(normal_mat_origin[:,:,1], norm= norm)
    plt.title("Y Component")
    plt.colorbar()
    fig.savefig(resultFolder + '/Y Component' + '.png')
    plt.show()
    fig = plt.figure(figsize=(15,10))
    plt.imshow(normal_mat_origin[:,:,2], norm= norm2)
    plt.title("Z Component")
    plt.colorbar()
    fig.savefig(resultFolder + '/Z Component' + '.png')
    plt.show()
    # visualize normal map in world perspective
    fig = plt.figure(figsize=(25,10))
    for i in range(3):
        #plt.figure()
        if i == 2:
            plt.subplot(1,3,i+1)
            plt.imshow(normalmap_world[:,:,i], norm= norm2)
            plt.title("Actual Z")
            plt.tick_params(labelsize=16)
        else:
            plt.subplot(1,3,i+1)
            plt.imshow(normalmap_world[:,:,i], norm= norm)
            if i == 1:
                plt.title("Actual Y")
            else:
                plt.title("Actual X")
            plt.tick_params(labelsize=16)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize= 16)
    plt.show()
    fig.savefig(resultFolder + '/normal2RGB' + '.png')
        

        
