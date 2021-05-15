from Camera import Camera

from abc import ABC

import numpy as np

import cv2

from skimage import io

from io import BytesIO

from IPython.display import clear_output, Image, display, update_display



import PIL





try:

    from Cameras.PySpinCapture import PySpinCapture as psc

    print('1')

except ImportError:

    

    PySpinCapture = None



class Flir(Camera, ABC):

    def __init__(self, exposure=0.01, white_balance=1, auto_focus=False, grayscale=False):

        self._isMonochrome = True

        self._is16bits = True

        self.Cam = psc(0, self._isMonochrome, self._is16bits)

        fps = self.getFPS()

        resolution = self.getResolution()

        super().__init__(exposure, white_balance, auto_focus, fps, resolution, grayscale)

        self.hdr_exposures = None



    def getImage(self, name='test', saveImage=True, saveNumpy=True, calibration=False, timeout=5000):

        try:

            # Take and return current camera frame

            success, img = self.Cam.grabFrame()

            # Save if desired

            if saveImage:

                if calibration:

                    filenamePNG = 'CalibrationImages/' + name + '.PNG'

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



    def quit_and_close(self):

        self.Cam.release()



    def quit_and_open(self):

        self.Cam.release()

        self.Cam = psc(1, self._isMonochrome, self._is16bits)



    def getStatus(self):

        raise NotImplementedError

