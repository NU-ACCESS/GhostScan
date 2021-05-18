from pypylon import pylon
from Camera import Camera
from abc import ABC
import numpy as np
import cv2
import os, time, math
from skimage import io
from io import BytesIO
from IPython.display import clear_output, Image, display, update_display
import PIL
from Cameras.liveDisplay import ptgCamStream, imgShow, patternDisplay
from Cameras.PySpinCapture import PySpinCapture as psc

class Basler(Camera, ABC):

    def __init__(self, exposure=0.01, white_balance=0, auto_focus=False, grayscale=True):
        #  TODO: pylon.FeaturePersistence.Save("test.txt", camera.GetNodeMap())
        # Setting and initializing the Basler camera

        self.cap = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.cap.Open()
        if self.cap is None:
            print('Warning: unable to open external Basler camera')
        # Get framerate and resolution of camera
        fps = self.getFPS()
        resolution = self.getResolution()
        # Init base class
        super().__init__(exposure, white_balance, auto_focus, fps, resolution, grayscale)
        self.hdr_exposures = None

    def setDir(self, directory, sessionName):
        self.directory = directory
        self.sessionName = sessionName
        self.sessionDir = os.path.join(self.directory, self.sessionName)
        print(self.sessionDir)
        if not os.path.exists(self.sessionDir):
            os.makedirs(self.sessionDir)

    def getAutoExposure(self):
        # Returns if auto exposure is enabled
        return self.cap.ExposureAuto.GetValue()

    def setAutoExposure(self):
        # Turn on auto exposure
        self.cap.ExposureAuto.SetValue("Continuous")

    def getFPS(self):
        # Returns the frame rate
        return self.cap.AcquisitionFrameRate.GetValue()

    def setFPS(self, fps):
        # Sets frame rate
        self.cap.AcquisitionFrameRate.SetValue(fps)
        self.fps = fps

    def setAutoGain(self):
        # Set auto gain
        self.cap.GainAuto.SetValue("Once")

    def getGain(self):
        # Returns the set gain value
        return self.cap.Gain.GetValue()

    def setGain(self, gain):
        # Turn off auto gain
        self.cap.GainAuto.SetValue("Off")
        # Set gain value
        self.cap.Gain.SetValue(gain)

    def getResolution(self):
        # Returns a tuple resolution (width, height)
        resolution = (self.cap.Width.GetValue(), self.cap.Height.GetValue())
        return resolution

    def setResolution(self, resolution):
        # Sets the image resolution
        self.cap.Width.SetValue(resolution[0])
        self.cap.Height.SetValue(resolution[1])
        self.resolution = resolution

    def setSingleFrameCapture(self):
        # Set single frame acquisition mode
        self.cap.AcquisitionMode.SetValue('SingleFrame')

    def setHDRExposureValues(self, exposures):
        self.hdr_exposures = exposures

    def setExposure(self, exposure):
        # Set auto exposure off
        self.cap.ExposureAuto.SetValue("Off")
        # Set exposure value in microseconds
        self.cap.ExposureTime.SetValue(exposure)
        self.exposure = exposure

    def getExposure(self):
        # Returns exposure value in microseconds
        return self.cap.ExposureTime.GetValue()

    def getHDRImage(self, name='test', saveImage=True, saveNumpy=True, timeout=5000):
        if self.calibration is None:
            print("Initialize calibration object of camera class first")
        self.cap.StartGrabbingMax(1)
        img = pylon.PylonImage()
        frames = []
        for e in self.hdr_exposures:
            self.setExposure(e)
            while self.cap.IsGrabbing():
                # Grabs photo from camera
                grabResult = self.cap.RetrieveResult(timeout, pylon.TimeoutHandling_ThrowException)
                if grabResult.GrabSucceeded():
                    # Access the image data.
                    frame = grabResult.Array
                    img.AttachGrabResultBuffer(grabResult)
                grabResult.Release()
            frames.append(frame)
        hdr_frame = self.calibration.radio_calib.get_HDR_image(frames, self.hdr_exposures)
        if saveNumpy:
            np.save('CapturedNumpyData/' + name, hdr_frame)
        if saveImage:
            png_frame = (hdr_frame - np.min(hdr_frame)) / (np.max(hdr_frame) - np.min(hdr_frame))
            png_frame *= 255.0
            io.imsave('CapturedImages/' + name + '.PNG', png_frame.astype(np.uint8))
        return hdr_frame

    def getImage(self, name='test', saveImage=True, saveNumpy=True, calibration=False, timeout=5000):
        try:
            # Take and return current camera frame
            self.cap.StartGrabbingMax(1)
            img = pylon.PylonImage()
            while self.cap.IsGrabbing():
                # Grabs photo from camera
                grabResult = self.cap.RetrieveResult(timeout, pylon.TimeoutHandling_ThrowException)
                if grabResult.GrabSucceeded():
                    # Access the image data.
                    frame = grabResult.Array
                    img.AttachGrabResultBuffer(grabResult)
                grabResult.Release()
            # Save if desired
            if saveImage:
                if calibration:
                    filename = 'CalibrationImages/' + name + '.raw'
                    filenamePNG = 'CalibrationImages/' + name + '.PNG'
                    img.Save(pylon.ImageFileFormat_Raw, filename)
                    img.Save(pylon.ImageFileFormat_Png, filenamePNG)
                else:
                    filename = 'CapturedImages/' + name + '.PNG'
                    img.Save(pylon.ImageFileFormat_Png, filename)
            if saveNumpy:
                if calibration:
                    np.save('CalibrationNumpyData/' + name, frame)
                else:
                    np.save('CapturedNumpyData/' + name, frame)
            img.Release()
            self.cap.StopGrabbing()
            return frame
        except SystemError:
            self.quit_and_open()
            return None

    def viewCameraStream(self):
        # Display live view
        while True:
            cv2.namedWindow('Basler Machine Vision Stream', cv2.WINDOW_NORMAL)
            img = self.getImage(saveImage=False, saveNumpy=False)
            print("Max: ", np.max(img))
            print("Min: ", np.min(img))
            cv2.imshow('Basler Machine Vision Stream', img)
            c = cv2.waitKey(1)
            if c != -1:
                # When everything done, release the capture
                cv2.destroyAllWindows()
                break

    def viewCameraStreamSnapshots(self):
        # Display live view
        while True:
            cv2.namedWindow('Basler Machine Vision Stream', cv2.WINDOW_NORMAL)
            img = self.getImage(saveImage=False, saveNumpy=False)
            cv2.imshow('Basler Machine Vision Stream', img)
            c = cv2.waitKey(1)
            if c != -1:
                # When everything done, release the capture
                cv2.destroyAllWindows()
                self.quit_and_open()
                break

    def viewCameraStreamJupyter(self):
        # Live view in a Jupyter Notebook
        try:
            start = self.getImage(saveImage=False, saveNumpy=False)
            g = BytesIO()
            PIL.Image.fromarray(start).save(g, 'jpeg')
            obj = Image(data=g.getvalue())
            dis = display(obj, display_id=True)
            while True:
                img = self.getImage(saveImage=False, saveNumpy=False)
                if img is None:
                    break
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                f = BytesIO()
                PIL.Image.fromarray(img).save(f, 'jpeg')
                obj = Image(data=f.getvalue())
                update_display(obj, display_id=dis.display_id)
                clear_output(wait=True)
        except KeyboardInterrupt:
            self.quit_and_open()

    def quit_and_close(self):
        # Close camera
        self.cap.Close()

    def quit_and_open(self):
        # Close camera
        self.cap.Close()
        # Create new capture
        self.cap = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.cap.Open()

    def getStatus(self):
        pylon.FeaturePersistence.Save("Basler_Specs.txt", self.cap.GetNodeMap())


try:

    from Cameras.PySpinCapture import PySpinCapture as psc

    print('1')

except ImportError:

    PySpinCapture = None
    

DISPLAY_HEIGHT = 1080
DISPLAY_WIDTH = 1920
NUM_PATTERN = 7
DEFLECTOMETRY_FREQ = 0.9

class Flir(Camera, ABC):

    def __init__(self, exposure=0.01, white_balance=1, auto_focus=False, grayscale=False):
        self.sessionDir = None

        self._isMonochrome = True

        self._is16bits = True

        # self.NumPatterns = NUM_PATTERN
        self.displayWidth = DISPLAY_WIDTH
        self.displayHeight = DISPLAY_HEIGHT

        self.setDefPattern()

        self.Cam = psc(0, self._isMonochrome, self._is16bits)
        self.height = self.Cam.height
        self.width = self.Cam.width

        fps = self.getFPS()

        resolution = self.getResolution()
      
        super().__init__(exposure, white_balance, auto_focus, fps, resolution, grayscale)

        self.hdr_exposures = None
    

    def getImage(self, name='test', saveImage=True, saveNumpy=True, calibration=False, timeout=5000, calibrationName = None):

        try:

            # Take and return current camera frame

            success, img = self.Cam.grabFrame()
            #print(img.shape)

            # Save if desired

            if saveImage:

                if calibration:
                    #filename = 'CalibrationImages/' + name + '.raw'
                    if calibrationName is None:

                        filenamePNG = 'CalibrationImages/' + name + '.PNG'

                        cv2.imwrite(filenamePNG, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    
                    else:
                        filenamePNG = os.path.join('CalibrationImages/' + calibrationName,  name + '.PNG')

                        cv2.imwrite(filenamePNG, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    
                else:

                    filename = 'CapturedImages/' + name + '.PNG'

                    cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            if saveNumpy:

                if calibration:

                    np.save('CalibrationNumpyData/' + name, img)

                else:

                    np.save('CapturedNumpyData/' + name, img)

            # self.Cam.release()

            return img

        except SystemError:

            self.quit_and_open()

            return None



    def setExposure(self, exposure):

        self.Cam.setExposure(exposure)

        

    def getExposure(self):

        return self.Cam.getExposure()



    def getFPS(self):

        return self.Cam.getFPS()



    def setFPS(self, fps):

        self.Cam.setFPS(fps)



    def setAutoGain(self):

        self.Cam.setCamAutoProperty()



    def getGain(self):

        return self.Cam.getGain()



    def setGain(self, gain):

        self.Cam.setGain(gain)
    
    def captureImage(self, fname):
        if fname:
            path = 'CapturedImages/' + fname + '.png'
            #path = os.path.join(self.sessionDir, fname + ".png")
            flag, img = self.Cam.grabFrame()
            if not flag:
                print("[ERROR]: Didn't get the image!!!")
            else:
                cv2.imwrite(path, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        else:
            flag, img = self.Cam.grabFrame()
            print("capture!!!!")
            if not flag:
                print("[ERROR]: Didn't get the image!!!")
                return None
            else:
                return img


    def sinePattern(self, x, y, nu_x, nu_y):
        # Phase Shifts (first entry is white light modulation)
        theta = [0, np.pi / 2, np.pi, 3 / 2 * np.pi]
        [X, Y, Theta] = np.meshgrid(x, y, theta)

        # Calculate Phase Shifts
        phase = (nu_x * X + nu_y * Y) + Theta
        # Simple formula to create fringes between 0 and 1:
        pattern = (np.sin(phase) + 1) / 2

        return pattern
    
    def setPatternScale(self, mode=1):
        patternSize = min(self.displayHeight, self.displayWidth)
        boarderSize = math.floor((max(self.displayHeight, self.displayWidth) - patternSize)/2)

        if mode == 0:
            # Img sequences[7]: {gh, ch, gv, cv, g2h, g2v, black}
            self.NumPatterns = 7
        elif mode == 1:
            # Img sequences[8]: {gh, ch, gv, cv, g2h, g2v, black, full}
            self.NumPatterns = 8
        elif mode == 2:
            # Img sequences[9]: {h1, h2, h3, h4, v1, v2, v3, v4, black}
            self.NumPatterns = 9
        elif mode == 3:
            # Img sequences[9]: {h1, h2, h3, h4, v1, v2, v3, v4, black, full}
            self.NumPatterns = 10
        else:
            self.NumPatterns = 8
            print("[WARNING]: Unrecognizable mode. Using Gradient Pattern!!")
        self.patterns = np.zeros((self.displayHeight, self.displayWidth, self.NumPatterns), dtype=np.float32)

        return patternSize, boarderSize


    def setDefPattern(self):
        #patternSize, boarderSize = self.setPatternScale(2)
        patternSize, boarderSize = self.setPatternScale(2)

        patternSize = max(self.displayHeight, self.displayWidth)
        #patternSize = min(self.displayHeight, self.displayWidth)
        
        # Create spatial coordinates
        x = np.linspace(1, self.displayWidth, self.displayWidth)
        y = np.linspace(1, self.displayHeight, self.displayHeight)

        # Frequencies in x and y direction.
        nu_x =  DEFLECTOMETRY_FREQ * 2 * np.pi / patternSize
        nu_y = DEFLECTOMETRY_FREQ * 2 * np.pi / patternSize

        self.patterns[..., 0:4] = self.sinePattern(x, y, nu_x, 0)
        self.patterns[..., 4:8] = self.sinePattern(x, y, 0, nu_y)
        # self.patterns[..., 8] = 127 * np.ones((DISPLAY_HEIGHT,DISPLAY_WIDTH))

    def captureSeqImages(self):
        window_name = 'projector'
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)

        # If projector is placed right to main screen (see windows properties in your operating system)
        # if the pattern is displayed at the wrong monitor you need to play around with the coordinates here until the image is displayed at the right screen
        cv2.moveWindow(window_name, self.displayWidth, 0)

        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        cv2.imshow(window_name, self.patterns[..., 0].astype(np.float32))

        if self._isMonochrome:
            if self._is16bits:
                imgs = np.zeros((self.NumPatterns, self.height, self.width), dtype=np.uint16)
            else:
                imgs = np.zeros((self.NumPatterns, self.height, self.width), dtype=np.uint8)
        else:
            if self._is16bits:
                imgs = np.zeros((self.NumPatterns, self.height, self.width, 3), dtype=np.uint16)
            else:
                imgs = np.zeros((self.NumPatterns, self.height, self.width, 3), dtype=np.uint8)

        cv2.waitKey(200)
        # Loop through 
        for i in range(self.NumPatterns):
            if not i==0:
                if i == self.NumPatterns - 1:
                    cur_img = 127 * np.ones((DISPLAY_HEIGHT,DISPLAY_WIDTH))
                    cv2.imshow(window_name, cur_img.astype(np.uint8))
                else:
                    cur_img = self.patterns[... ,i]
                    cv2.imshow(window_name, cur_img.astype(np.float32))

            cv2.waitKey(400)  # ms # delay time
            # wait for 1000 ms ( syncrhonize this with your aquisition time)
            imgs[i, ...] = self.captureImage(fname=None)

            cv2.waitKey(400) # ms # delay time

        # Don't forgot to close all windows at the end of aquisition
        cv2.destroyAllWindows()


        for i in range(self.NumPatterns):
            fname = 'CapturedImages/sequenceImages/' + str(i) + ".png"
            # fname = fpath + '\\' + str(i) + '.png'
            if self._isMonochrome:
                cv2.imwrite(fname, imgs[i, ...], [cv2.IMWRITE_PNG_COMPRESSION, 0])  # RGB to BGR
            else:
                cv2.imwrite(fname, imgs[i, ...][..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 0]) #RGB to BGR


    def getResolution(self):

        size = self.Cam.getResolution()

        

        return size



    def setResolution(self, resolution):

        self.Cam.setWidth(resolution[0])

        self.Cam.setHeight(resolution[1])

        

    def getHDRImage(self, name='test', saveImage=True, saveNumpy=True):

        self.Cam.setupHDR()

        imgs = self.Cam.captureHDR()

        if saveNumpy:

            np.save('CapturedNumpyData/' + name, imgs)

        if saveImage:

            png_frame = (imgs - np.min(imgs)) / (np.max(imgs) - np.min(imgs))

            png_frame *= 255.0

            io.imsave('CapturedImages/' + name + '.PNG', png_frame.astype(np.uint8))


    def liveView(self):

        print("[INFO] starting live view thread...")
        viewGrabber = ptgCamStream(self.Cam).start()
        viewShower = imgShow(frame=viewGrabber.read()).start()

        time.sleep(0.1)
        while viewGrabber.running():
            if viewGrabber.stopped or viewShower.stopped:
                viewShower.stop()
                viewGrabber.stop()
                break

            frame = viewGrabber.read()
            viewShower.frame = frame

        viewGrabber.stop()
        viewShower.stop()
        viewGrabber.__exit__()
        viewShower.__exit__()

        viewGrabber.thread.join()
        viewShower.thread.join()
        del viewGrabber
        del viewShower
        cv2.destroyWindow("Live View")

    def viewCameraStream(self):

        img = self.getImage(saveImage=False, saveNumpy=False)

        while True:

            _,img = self.Cam.grabFrameCont()

            cv2.imshow('FLIR camera image', img)

            c = cv2.waitKey(1)

            if c != -1:

                # When everything done, release the capture

                self.Cam._camera.EndAcquisition()

                cv2.destroyAllWindows()

                self.quit_and_open()

                break

            

    def viewCameraStreamJupyter(self):

        # Live view in a Jupyter Notebook

        try:

            start = self.getImage(saveImage=False, saveNumpy=False)

            g = BytesIO()

            PIL.Image.fromarray(start).save(g, 'jpeg')

            obj = Image(data=g.getvalue())

            dis = display(obj, display_id=True)

            while True:

                img = self.getImage(saveImage=False, saveNumpy=False)

                if img is None:

                    break

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                f = BytesIO()

                PIL.Image.fromarray(img).save(f, 'jpeg')

                obj = Image(data=f.getvalue())

                update_display(obj, display_id=dis.display_id)

                clear_output(wait=True)

        except KeyboardInterrupt:

            self.quit_and_open()

            

    def viewCameraStreamJupyterWindows(self):

        # Live view in a Jupyter Notebook

        try:

            start = self.getImage(saveImage=False, saveNumpy=False)

            g = BytesIO()

            PIL.Image.fromarray(start).save(g, 'jpeg')

            obj = Image(data=g.getvalue())

            dis = display(obj, display_id=True)

            while True:

                img = self.getImage(saveImage=False, saveNumpy=False)

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                f = BytesIO()

                PIL.Image.fromarray(img).save(f, 'jpeg')

                obj = Image(data=f.getvalue())

                update_display(obj, display_id=dis.display_id)

        except KeyboardInterrupt:

            self.quit_and_open()

    def displayCalib(self):
        # TODO: Need to use threading so that the GUI won't stuck!
        pattern = cv2.imread("Projections/8_24_checker.png", cv2.IMREAD_GRAYSCALE)
        window_name = 'projector'
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        #
        # # If projector is placed right to main screen (see windows properties in your operating system)
        # # if the pattern is displayed at the wrong monitor you need to play around with the coordinates here until the image is displayed at the right screen
        cv2.moveWindow(window_name, self.displayWidth, 0)
        #
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(window_name, pattern.astype(np.float32))
        #
        # capture checkerboard
        img = self.captureImage(fname=None)
        fname = 'CalibrationImages/Geometric/' + 'test' + ".png"
        cv2.imwrite(fname, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        #
        # exit
        k = cv2.waitKey(0)
        if k == 27: # wait for ESC key to exit
            cv2.destroyAllWindows()
         
        

        # patternShower = patternDisplay(displayHeight=self.displayHeight, displayWidth=self.displayWidth, pattern=pattern).start()


    def quit_and_close(self):

        self.Cam.release()



    def quit_and_open(self):

        self.Cam.release()

        self.Cam = psc(1, self._isMonochrome, self._is16bits)



    def getStatus(self):

        raise NotImplementedError
