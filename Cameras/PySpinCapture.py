import os, math, time
import numpy as np
import cv2
import PySpin

class PySpinCapture:
    def __init__(self, index=0, isMonochrome=False, is16bits=False):
        # [Current Support] Single camera usage(select by index)
        self._system = PySpin.System.GetInstance()
        # Get current library version
        self._version = self._system.GetLibraryVersion()
        print('Library version: {}.{}.{}.{}\n'.format(self._version.major, self._version.minor, self._version.type,
                                self._version.build))
        self.index = index
        self.getNumCams()
        self._camera = self._cameraList.GetByIndex(index)
        self._camera.Init()
        self._isMonochrome = isMonochrome
        self._is16bits = is16bits

        self._nodemap = self._camera.GetNodeMap()
        self.getNode()

        self.setAcquisitMode(1)
        self.setPixel()
        self.setSize()
        self.setCamAutoProperty(False)
        #self._camera.BeginAcquisition()

    def __enter__(self):
        return self

    def reset(self):
        self.__init__()

    def getNumCams(self):
        self._cameraList = self._system.GetCameras()

    def print_retrieve_node_failure(self, node, name):

        print('Unable to get {} ({} {} retrieval failed.)'.format(node, name, node))
        print('The {} may not be available on all camera models...'.format(node))
        print('Please try a Blackfly S camera.')

    def getNode(self):
        # Acquisition Mode Node
        self.nodeAcquisitionMode = PySpin.CEnumerationPtr(self._nodemap.GetNode('AcquisitionMode'))
        if not PySpin.IsAvailable(self.nodeAcquisitionMode) or not PySpin.IsWritable(self.nodeAcquisitionMode):
            print("Unable to set acquisition mode to continuous (enum retrieval). Aborting...")
            return False

        # Retrieve entry node from enumeration node
        nodeContinuousAcquisition = self.nodeAcquisitionMode.GetEntryByName("Continuous")
        if not PySpin.IsAvailable(nodeContinuousAcquisition) or not PySpin.IsReadable(
                nodeContinuousAcquisition):
            print("Unable to set acquisition mode to continuous (entry retrieval). Aborting...")
            return False

        self.nodeAcquisitionContinuous = nodeContinuousAcquisition.GetValue()

        # Retrieve entry node from enumeration node
        nodeSingleAcquisition = self.nodeAcquisitionMode.GetEntryByName("SingleFrame")
        if not PySpin.IsAvailable(nodeSingleAcquisition) or not PySpin.IsReadable(
                nodeSingleAcquisition):
            print("Unable to set acquisition mode to Single Frame (entry retrieval). Aborting...")
            return False
        self.nodeAcquisitionSingle = nodeSingleAcquisition.GetValue()

        # Pixel Format Node
        self.nodePixelFormat = PySpin.CEnumerationPtr(self._nodemap.GetNode('PixelFormat'))

        nodePixelFormatMono8 = PySpin.CEnumEntryPtr(
            self.nodePixelFormat.GetEntryByName('Mono8'))
        self.pixelFormatMono8 = nodePixelFormatMono8.GetValue()

        nodePixelFormatMono16 = PySpin.CEnumEntryPtr(
            self.nodePixelFormat.GetEntryByName('Mono16'))
        self.pixelFormatMono16 = nodePixelFormatMono16.GetValue()

        # nodePixelFormatRGB8 = PySpin.CEnumEntryPtr(
        #     self.nodePixelFormat.GetEntryByName('BayerRG8'))
        # self.pixelFormatRGB8 = nodePixelFormatRGB8.GetValue()
        #
        # nodePixelFormatRGB16 = PySpin.CEnumEntryPtr(
        #     self.nodePixelFormat.GetEntryByName('BayerRG16'))
        # self.pixelFormatRGB16 = nodePixelFormatRGB16.GetValue()

        # Image Size Node
        self.nodeWidth = PySpin.CIntegerPtr(self._nodemap.GetNode('Width'))
        self.nodeHeight = PySpin.CIntegerPtr(self._nodemap.GetNode('Height'))

        # Exposure Node
        self._nodeExposureAuto = PySpin.CEnumerationPtr(self._nodemap.GetNode('ExposureAuto'))
        if not PySpin.IsAvailable(self._nodeExposureAuto) or not PySpin.IsWritable(self._nodeExposureAuto):
            self.print_retrieve_node_failure('node', 'ExposureAuto')
            return False

        self._exposureAutoOff = self._nodeExposureAuto.GetEntryByName('Off')
        if not PySpin.IsAvailable(self._exposureAutoOff) or not PySpin.IsReadable(self._exposureAutoOff):
            self.print_retrieve_node_failure('entry', 'ExposureAuto Off')
            return False

        if PySpin.IsAvailable(self._nodeExposureAuto) and PySpin.IsWritable(self._nodeExposureAuto):
            self._exposureAutoOn = self._nodeExposureAuto.GetEntryByName('Continuous')

        self._nodeExposure = PySpin.CFloatPtr(self._nodemap.GetNode('ExposureTime'))
        self._exposureMin = self._nodeExposure.GetMin()
        self._exposureMax = self._nodeExposure.GetMax()

        # Gain Node
        self._nodeGainAuto = PySpin.CEnumerationPtr(self._nodemap.GetNode('GainAuto'))
        if not PySpin.IsAvailable(self._nodeGainAuto) or not PySpin.IsWritable(self._nodeGainAuto):
            self.print_retrieve_node_failure('node', 'GainAuto')
            return False

        self._gainAutoOff = self._nodeGainAuto.GetEntryByName('Off')
        if not PySpin.IsAvailable(self._gainAutoOff) or not PySpin.IsReadable(self._gainAutoOff):
            self.print_retrieve_node_failure('entry', 'GainAuto Off')
            return False

        if PySpin.IsAvailable(self._nodeGainAuto) and PySpin.IsWritable(self._nodeGainAuto):
            self._gainAutoOn = self._nodeGainAuto.GetEntryByName('Continuous')

        self._nodeGain = PySpin.CFloatPtr(self._nodemap.GetNode('Gain'))
        self._gainMin = self._nodeGain.GetMin()
        self._gainMax = self._nodeGain.GetMax()

        self.nodeGammaEnable = PySpin.CBooleanPtr(self._nodemap.GetNode('GammaEnable'))
        if not PySpin.IsAvailable(self.nodeGammaEnable) or not PySpin.IsWritable(self.nodeGammaEnable):
            self.print_retrieve_node_failure('node', 'GammaEnable')
            return False

        nodeAcquisitionFrameRate = PySpin.CFloatPtr(self._nodemap.GetNode('AcquisitionFrameRate'))
        if not PySpin.IsAvailable(nodeAcquisitionFrameRate) or not PySpin.IsWritable(nodeAcquisitionFrameRate):
            print("Unable to set acquisition Frame Rate (enum retrieval). Aborting...")
            return False
        else:
            nodeAcquisitionFrameRate.SetValue(5)

            
    def setAcquisitMode(self, mode=0):
        if mode==0:
            #Single Frame mode
            self.nodeAcquisitionMode.SetIntValue(self.nodeAcquisitionSingle)
        elif mode==1:
            # Continuous mode
            self.nodeAcquisitionMode.SetIntValue(self.nodeAcquisitionContinuous)
            #self.nodeAcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)

        print(self.nodeAcquisitionMode.GetIntValue())
    
    
    def setFPS(self, fps):

        nodeAcquisitionFrameRate = PySpin.CFloatPtr(self._nodemap.GetNode('AcquisitionFrameRate'))

        if not PySpin.IsAvailable(nodeAcquisitionFrameRate) or not PySpin.IsWritable(nodeAcquisitionFrameRate):

            print("Unable to set acquisition Frame Rate (enum retrieval). Aborting...")

            return False

        else:

            nodeAcquisitionFrameRate.SetValue(fps)
    

    def getFPS(self):

        nodeAcquisitionFrameRate = PySpin.CFloatPtr(self._nodemap.GetNode('AcquisitionFrameRate'))

        return nodeAcquisitionFrameRate.GetValue()


    def setPixel(self):
        # Set the pixel format.
        if self._isMonochrome:
            # Enable Mono8 mode.
            if self._is16bits:
                self.nodePixelFormat.SetIntValue(self.pixelFormatMono16)
            else:
                self.nodePixelFormat.SetIntValue(self.pixelFormatMono8)
        # else:
        #     # Enable RGB8 mode.
        #     if self._is16bits:
        #         self.nodePixelFormat.SetIntValue(self.pixelFormatRGB16)
        #     else:
        #         self.nodePixelFormat.SetIntValue(self.pixelFormatRGB8)

    def setSize(self):
        self.width = self.nodeWidth.GetMax()
        self.nodeWidth.SetValue(self.width)

        self.height = self.nodeHeight.GetMax()
        self.nodeHeight.SetValue(self.height)

    def setCamAutoProperty(self, switch=True):
        # [Current Support] Gain, Exposure time
        # In order to manual set value, turn off auto first
        if switch:
            if PySpin.IsAvailable(self._exposureAutoOn) and PySpin.IsReadable(self._exposureAutoOn):
                self._nodeExposureAuto.SetIntValue(self._exposureAutoOn.GetValue())
                print('Turning automatic exposure back on...')

            if PySpin.IsAvailable(self._gainAutoOn) and PySpin.IsReadable(self._gainAutoOn):
                self._nodeGainAuto.SetIntValue(self._gainAutoOn.GetValue())
                print('Turning automatic gain mode back on...\n')

            self.nodeGammaEnable.SetValue(True)
        else:
            self._nodeExposureAuto.SetIntValue(self._exposureAutoOff.GetValue())
            print('Automatic exposure disabled...')

            self._nodeGainAuto.SetIntValue(self._gainAutoOff.GetValue())
            print('Automatic gain disabled...')

            #self.nodeGammaEnable.SetValue(False)
            #print('Gamma disabled...')
    
    def getSize(self):
        width = self.width
        height = self.height
        return width, height

    def getResolution(self):
        width,height = self.getSize()
        return (width,height)

    def setGain(self, gain_to_set):

        if float(gain_to_set) < self._gainMin or float(gain_to_set) > self._gainMax:
            print("[WARNING]: Gain value should be within {} to {}.(Input:{}) Set to half."
                  .format(self._gainMin, self._gainMax, float(gain_to_set)))
            self._nodeGain.SetValue(math.floor(self._gainMax + self._gainMin))
            print("Gain: {}".format(self._nodeGain.GetValue()))
        else:
            self._nodeGain.SetValue(float(gain_to_set))
            print("Gain: {}".format(self._nodeGain.GetValue()))

    def setExposure(self, exposure_to_set):

        exposure_to_set = float(exposure_to_set)
        if exposure_to_set < self._exposureMin or exposure_to_set > self._exposureMax:
            print("[WARNING]: Gain value should be within {} to {}.(Input:{}) Set to half."
                  .format(self._exposureMin, self._exposureMax, exposure_to_set))
            self._nodeExposure.SetValue(math.floor(self._exposureMax + self._exposureMin))
            print("Exposure: {}".format(self._nodeExposure.GetValue()))
        else:
            self._nodeExposure.SetValue(exposure_to_set)
            print("Exposure: {}".format(self._nodeExposure.GetValue()))

    def grabFrame(self):
        self.setAcquisitMode(0)
        self._camera.BeginAcquisition()
        cameraBuff = self._camera.GetNextImage()
        if cameraBuff.IsIncomplete():
            return False, None

        cameraImg = cameraBuff.GetData().reshape(self.height, self.width)
        image = cameraImg.copy()
        cameraBuff.Release()
        self._camera.EndAcquisition()

        if self._isMonochrome:
            return True, image
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BayerRG2BGR)
            return True, image_rgb

    def grabFrameCont(self):
        cameraBuff = self._camera.GetNextImage()
        if cameraBuff.IsIncomplete():
            return False, None

        cameraImg = cameraBuff.GetData().reshape(self.height, self.width)
        image = cameraImg.copy()
        cameraBuff.Release()
        #self._camera.EndAcquisition()

        if self._isMonochrome:
            return True, image
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BayerRG2BGR)
            return True, image_rgb

    def beginAcquisit(self, switch=True):
        if switch:
            self._camera.BeginAcquisition()
        else:
            self._camera.EndAcquisition()

    def getNextImage(self):
        return self._camera.GetNextImage()

    def setupHDR(self):

        if PySpin.IsAvailable(self._exposureAutoOn) and PySpin.IsReadable(self._exposureAutoOn):
            self._nodeExposureAuto.SetIntValue(self._exposureAutoOn.GetValue())
            print('Turning automatic exposure back on...')

        time.sleep(0.5)
        series = [2 ** (-2), 2 ** (-1), 1, 2 ** 1, 2 ** 2]
        midExposure = self._nodeExposure.GetValue()
        print("midExposure: ", midExposure)
        self.exposureHDRList = [midExposure * x for x in series]
        if self.exposureHDRList[0] < self._exposureMin:
            self.exposureHDRList[0] = self._exposureMin
        if self.exposureHDRList[-1] > self._exposureMax:
            self.exposureHDRList[-1] = self._exposureMax

        print("HDR Exposure List: ", self.exposureHDRList)

        self._nodeExposureAuto.SetIntValue(self._exposureAutoOff.GetValue())
        print('Automatic exposure disabled...')

    def captureHDR(self):
        if not hasattr(self, 'exposureHDRList'):
            print("[ERROR]: Need to setup HDR Exposure list first!!!")
            return 0
        imgs = np.zeros((len(self.exposureHDRList), self.height, self.width))
        for index, x in enumerate(self.exposureHDRList):
            self.setExposure(x)
            flag, tmp = self.grabFrame()
            if flag:
                imgs[index, ...] = tmp
            else:
                print("[WARNING]: Invalid Capture!!!")

        return imgs

    def release(self):

        # Turn auto gain and exposure back on in order to return the camera to tis default state
        self.setCamAutoProperty(True)
        if self._camera.IsStreaming():
            self._camera.EndAcquisition()
        self._camera.DeInit()
        #del self._camera
        # self._cameraList.Clear()
        # self._system.ReleaseInstance()

    def __exit__(self):
        self.release()
