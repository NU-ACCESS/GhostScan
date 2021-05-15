from abc import ABC
from CalibrationSession import CalibrationSession
from Calibrations import RadiometricCalibration
import cv2
import os, sys, time, math
import numpy as np
import matplotlib.pyplot as plt

DISPLAY_HEIGHT = 1080
DISPLAY_WIDTH = 1920

class RadiometricCalibDisplaySession(CalibrationSession, ABC):
    def __init__(self, camera, radio_calib, path='CalibrationImages/DisplayRadiometric', exposure=13000, numPhaseShift = 4):
        # Set camera, destination path
        self.camera = camera
        self.path = path
        # self.g = None
        self.radio_calib = radio_calib
        self.g = self.radio_calib.load_calibration_data()
        self.exposure = exposure
        self.camera.setExposure(exposure)
        self.NumPatterns = 256
        self.displayWidth = DISPLAY_WIDTH
        self.displayHeight = DISPLAY_HEIGHT
        self.setDefPattern()

        self.sequenceNum = 2 * numPhaseShift

        

    def calculateDisplayCalibration(self):
        
        pixelValues = self.avgImagePixel()

        DisplayToRadianceData = []

        for pixelValue in pixelValues:
            num = math.exp(self.g[round(pixelValue)] - math.log(self.exposure))
            DisplayToRadianceData.append(num)

        imgSave = 'CapturedImages/sequenceImages/undistortRadioCalib/radioCalibResults'
        fig = plt.figure()
        plt.title("Display to Relative Radiance Curve")
        plt.xlabel('Display pixel value')
        plt.ylabel('Relative Radiance')
        plt.plot(pixelValues, DisplayToRadianceData, 'k')
        # plt.show()
        fig.savefig(imgSave + '/Display to Relative Radiance Curve' + '.png')
        # fig.savefig('Display to Relative Radiance Curve.png')
        return DisplayToRadianceData

    def calUpValue(self, imgPath):
        img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
        img = np.array(img, dtype='uint8')
        row, col = img.shape[0], img.shape[1]
        arr = []
        for i in range(row):
            cols = []
            for j in range(col):
                val = math.exp(self.g[img[i][j]] - math.log(self.exposure))
                cols.append(val)
            arr.append(cols)
        return np.array(arr)


    def calculateReflectivity(self):
        imgPath = 'CapturedImages/sequenceImages/8.png'
        arr = self.calUpValue(imgPath)

        # img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
        # img = np.array(img, dtype='uint8')
        # row, col = img.shape[0], img.shape[1]

        DisplayToRadianceData = self.calculateDisplayCalibration()
        DisplayToRadianceValue = DisplayToRadianceData[255]
        
        # arr = []
        # for i in range(row):
        #     cols = []
        #     for j in range(col):
        #         val = math.exp(self.g[img[i][j]] - math.log(self.exposure)) / DisplayToRadianceValue
        #         cols.append(val)
        #     arr.append(cols)
        # arr = np.array(arr)
        # print(arr)
        
        arr = [[j/DisplayToRadianceValue for j in i] for i in arr]
        imgSave = 'CapturedImages/sequenceImages/undistortRadioCalib/radioCalibResults'
        fig = plt.figure()
        plt.imshow(arr, cmap='viridis')
        plt.colorbar()
        # plt.show()
        fig.savefig(imgSave + '/Reflectivity Map' + '.png')
        # plt.savefig('Reflectivity Map.png')

        Reflectivity = np.array(arr)
        return Reflectivity

    def radiometricCalibUndistortImages(self):
        Reflectivity = self.calculateReflectivity()
        imgSeqPath = 'CapturedImages/sequenceImages/'
        # sequenceNum = 8
        for i in range(self.sequenceNum):
            upValue = self.calUpValue(imgSeqPath + str(i) + '.png')
            # print(upValue)
            radiance = np.divide(upValue, Reflectivity)
            # print(radiance)
            # print(radiance.shape[0], radiance.shape[1])
            correctedImage = self.imageCorrection(radiance)
            # print(correctedImage)
            cv2.imwrite(imgSeqPath + 'undistortRadioCalib/' + str(i) + '.png', correctedImage)

    def imageCorrection(self, radiance):
        DisplayToRadianceData = self.calculateDisplayCalibration()
        row, col = radiance.shape[0], radiance.shape[1]
        imgCorr = []
        for i in range(row):
            cols = []
            for j in range(col):
                t = np.array(abs(DisplayToRadianceData - radiance[i][j]))
                t_min = np.min(t)
                tt = self.findPixelValue(t_min, t)
                cols.append(tt)
            imgCorr.append(cols)
        return np.array(imgCorr)

    def findPixelValue(self, value, arr):
        for i in range(len(arr)):
            if arr[i] == value:
                return i


    def setPatternScale(self):
        patternSize = min(self.displayHeight, self.displayWidth)
        boarderSize = math.floor((max(self.displayHeight, self.displayWidth) - patternSize)/2)

        self.patterns = np.zeros((self.displayHeight, self.displayWidth, self.NumPatterns), dtype=np.float32)
        
        return patternSize, boarderSize

    def setDefPattern(self):
        #patternSize, boarderSize = self.setPatternScale(2)
        patternSize, boarderSize = self.setPatternScale()

        patternSize = max(self.displayHeight, self.displayWidth)
        #patternSize = min(self.displayHeight, self.displayWidth)
        
        for i in range(self.NumPatterns):
            self.patterns[..., i] = i * np.ones((DISPLAY_HEIGHT,DISPLAY_WIDTH))


    def captureImages(self):
        window_name = 'projector'
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)

        # If projector is placed right to main screen (see windows properties in your operating system)
        # if the pattern is displayed at the wrong monitor you need to play around with the coordinates here until the image is displayed at the right screen
        cv2.moveWindow(window_name, self.displayWidth, 0)

        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        cv2.imshow(window_name, self.patterns[..., 0].astype(np.uint8))

        # test for projector
        #imgs = np.zeros((self.NumPatterns, self.camera.height, self.camera.width), dtype=np.uint16)
        # for i in range(self.NumPatterns):
        #     print(i)
        #     if not i==0:
        #         cur_img = (i * np.ones((DISPLAY_HEIGHT,DISPLAY_WIDTH))).astype(np.uint8)
        #         print(cur_img)
        #         cv2.imshow(window_name, cur_img)
        #         #cv2.imshow(window_name, cur_img.astype(np.float32))
        #     else:
        #         cur_img = np.zeros((DISPLAY_HEIGHT,DISPLAY_WIDTH))
        #         cv2.imshow(window_name, cur_img.astype(np.uint8))
        #     cv2.waitKey(40)
        cv2.waitKey(20)
        # Loop through 
        for i in range(self.NumPatterns):
            # if not i==0:
            #     cur_img = self.patterns[... ,i]
            #     cv2.imshow(window_name, cur_img.astype(np.float32))
            cv2.waitKey(40)

            cur_img = self.patterns[... ,i]
            #print(cur_img.shape)
            # print(self.patterns[... ,i])
            cv2.imshow(window_name, cur_img.astype(np.uint8))
            cv2.waitKey(400)  # delay time
            # wait for 1000 ms ( syncrhonize this with your aquisition time)
            # imgs[i, ...] = self.camera.captureImage(fname=None)
            self.camera.getImage(name = 'DisplayRadiometric/'+str(i), calibration=True)

            cv2.waitKey(40) # ms # delay time

        # Don't forgot to close all windows at the end of aquisition
        cv2.destroyAllWindows()

    def avgImagePixel(self):
        pixelValues = []
        files = []
        # Load images
        for file in os.listdir(self.path):
            # Only use .raw files(Balser)
            #if file.endswith(".raw") or file.endswith(".Raw") or file.endswith(".RAW"):
            #    files.append(file)
            if file.endswith(".png") or file.endswith(".Png") or file.endswith(".PNG"):
                files.append(file)
        # Sort files depending on their exposure time from lowest to highest        
        files.sort(key=lambda x: int(x[:-4]))
        # We used exposure time as filenames

        for filename in files:
            #image = np.fromfile(self.path + '/' + filename, dtype=np.uint8)
            # image = cv2.imread(self.path + '/' + filename, cv2.IMREAD_UNCHANGED);
            image = cv2.imread(self.path + '/' + filename, 0)
            average = image.mean(axis=0).mean(axis=0) 
            pixelValues.append(average)

        x = list(range(256))
        imgSave = 'CapturedImages/sequenceImages/undistortRadioCalib/radioCalibResults'
        fig = plt.figure()
        plt.title("Display Value to Average Pixel Value Curve")
        plt.xlabel('Display pixel value')
        plt.ylabel('Average camera pixel value')
        plt.plot(x, pixelValues)
        # plt.show()
        fig.savefig(imgSave + '/Display Value to Average Pixel Value Curve' + '.png')
        # plt.savefig('Display Value to Average Pixel Value Curve.png')

        return pixelValues


    # def calibrate_HDR(self, smoothness=1000):
    #     # Call radiometric calibration
    #     # This computes and returns the camera response function and calculates a HDR image saved in path as PNG and .np
    #     self.g, le = self.radio_calib.get_camera_response(smoothness)
    #     self.radio_calib.plotCurve("Camera response")
    #     return self.g, le

    def load_calibration(self):
        self.radio_calib.load_calibration_data()

    def calibrate_image(self, exposure, path='CalibrationImages/Distorted'):
        # Calibrate radiometric calibrated images from a single exposure
        # Returns set of undistorted images and corresponding g function
        imgs, g = self.radio_calib.calibrate_image(exposure, path)
        return imgs, g


