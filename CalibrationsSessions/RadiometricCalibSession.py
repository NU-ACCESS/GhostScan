from abc import ABC
from CalibrationSession import CalibrationSession
from Calibrations import RadiometricCalibration
import numpy as np

class RadiometricCalibSession(CalibrationSession, ABC):
    def __init__(self, camera, radio_calib, path='CalibrationImages/Radiometric', exposures=0, mean_value = 3500000):
        # Set camera, destination path
        self.camera = camera
        self.path = path
        self.g = None
        self.radio_calib = radio_calib
        if exposures == 0:
            mean_value = mean_value 
            std_value = mean_value * 50
            ndis_no = np.random.normal( mean_value, std_value, size = 2000)
            ndis_no = ndis_no[ (ndis_no >= 5000) & (ndis_no < 30000000)]
            ndis_no = (np.ceil(ndis_no)).astype(int)
            ndis_no = np.sort(ndis_no)
            self.exposures = ndis_no
            # Dark Inside:
            # self.exposures = [5000, 7500, 10000, 15000, 20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000, 180000, 200000, 1640000]
            # self.exposures = [5000,60000, 7000, 8000, 9000, 10000, 12000, 14000, 15000, 16000, 18000, 20000, 22000, 25000, 27000, 30000, 32000, 35000, 37000, 40000, 43000, 45000, 47000, 50000, 52000, 55000, 57000, 60000, 62000, 65000, 67000, 70000, 75000, 80000, 85000, 90000, 95000, 100000, 110000, 120000, 130000, 140000, 150000, 160000, 170000, 180000, 190000, 200000, 230000, 250000, 280000, 320000, 330000, 350000,370000, 400000, 450000, 500000, 550000, 580000,600000, 640000, 700000, 800000, 900000,1000000, 1280000,1500000, 1700000, 1800000, 2000000, 2560000, 3000000, 4000000, 5120000]
        else:
            self.exposures = exposures

    def capture(self):
        # For radiometric calibration a series of differently exposed images of the same object is required
        for exp in self.exposures:
            self.camera.setExposure(exp)
            self.camera.getImage(name='Radiometric/'+str(exp), calibration=True)

    def calibrate_HDR(self, smoothness=1000):
        # Call radiometric calibration
        # This computes and returns the camera response function and calculates a HDR image saved in path as PNG and .np
        self.g, le = self.radio_calib.get_camera_response(smoothness)
        self.radio_calib.plotCurve("Camera response")
        return self.g, le

    def load_calibration(self):
        self.radio_calib.load_calibration_data()
        self.radio_calib.plotCurve("Camera response")

    def calibrate_image(self, exposure, path='CalibrationImages/Distorted'):
        # Calibrate radiometric calibrated images from a single exposure
        # Returns set of undistorted images and corresponding g function
        imgs, g = self.radio_calib.calibrate_image(exposure, path)
        return imgs, g


