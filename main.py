# from Cameras import Webcam, MachineVision, Polarization
from Cameras import MachineVision
from Projections import MainScreen
from CaptureSessions import DeflectometryCapture
from Calibrations import RadiometricCalibration
from CalibrationsSessions import RadiometricCalibSession, RadiometricCalibDisplaySession
#from Reconstructions import DeflectometryReconstruction

#from Visualization import Visualization
nph = 4
# Set-up your camera
# cam = Raspberry.RaspberryCam()
# cam = MachineVision.Basler()
# cam = MachineVision.Flir(exposure=100)
# exp = 6500000
exp = 5000000
cam = MachineVision.Flir()
cam.setExposure(exp)
cam.setGain(0)
# cam.getImage(name = 'test')
# cam.captureSeqImages()
# cam = Webcam.Internal()
# cam.displayCalib()
# cam.getImage(calibration = True, calibrationName = 'Geometric')
# img = cam.getImage(name = 'test')
# print(img.shape)
# cam.getImage(calibration = True, calibrationName = 'Intrinsic', name = '1')
# cam.displayCalib()
# intr_calib = IntrinsicCalibration.IntrinsicCalibration()
# intr_calib_session = IntrinsicCalibSession.IntrinsicCalibSession(cam, intr_calib)
# intr_calib_session.capture()
# intr_calib_session.calibrate()
# cam.viewCameraStream()
# cam.setAutoExposure()
# cam.viewCameraStream()
# cam.captureImage(fname="9")
# cam.captureSeqImages()
# cam.liveView()
# Set-up your projector
# projection = MainScreen.Screen()
# Set up image processing
# image_processing = DeflectometryReconstruction.DeflectometryReconstruction()
# Set up your Deflectometry Session
# cap = DeflectometryCapture.DeflectometryCapture(cam, projection, image_processing, 4)
# Capture images
# cap.capture()
# Compute results
# cap.compute()
# Visualize results
'''

vis = Visualization(cap.image_processing)
vis.showPhaseMaps()
vis.showReference()
vis.showAllImages()
vis.showNormals()


# RadiometricCalibration
# Set up the calibration object
radio_calib = RadiometricCalibration.RadiometricCalibration(cam.getResolution())
# # Set up the calibration session by passing on the camera and the calibration object
# # To set your own exposures please pass them to the RadiometricCalibSession when creating
# radio_calib_session = RadiometricCalibSession.RadiometricCalibSession(cam, radio_calib)
# # # Capture images at different exposures
# radio_calib_session.capture()


# g, g_n = radio_calib_session.calibrate_HDR(smoothness=500)

# radio_calib.load_calibration_data()
# radio_calib.plotCurve('Grayscale')


radio_calib_display = RadiometricCalibDisplaySession.RadiometricCalibDisplaySession(cam, radio_calib, exposure=exp)
# radio_calib_display.captureImages()
# radio_calib_display.avgImagePixel()
# radio_calib_display.calculateDisplayCalibration()
# radio_calib_display.calculateReflectivity()
# after we have got 8 pictures
radio_calib_display.radiometricCalibUndistortImages()