import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from Cameras import MachineVision

def checkImageWithThreshold(exposure=120000, lower_bound=20, upper_bound=255):
    blue   = [0,0,255]
    green = [0,255,0]
    red  = [255,0,0]
    white = [255,255,255]
    black = [0,0,0]
   
    low = lower_bound
    up = upper_bound
    
    lower = [low, low, low]
    upper = [up, up, up]
    
    cam = MachineVision.Flir()
    cam.setExposure(exposure)
    cam.setGain(0)
    cam.getImage(name = str(exposure))
    
    im = cv2.imread('CapturedImages/' + str(exposure) + '.png')
    im[np.all(im <= lower, axis=-1)] = blue
    im[np.all(im >= upper, axis=-1)] = red
    plt.imshow(im)
