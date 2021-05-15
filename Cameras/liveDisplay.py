import time
import threading
from queue import Queue
import cv2
import numpy as np

class imgShow:
    """
    Class that continuously shows a frame using a dedicated thread.
    """
    def __init__(self, displayHeight=1024, displayWidth=768, frame=None):
        self.frame = frame
        self.stopped = False
        self.displayHeight = displayHeight
        self.displayWidth = displayWidth
        self.thread = threading.Thread(target=self.show, args=())
        self.thread.daemon = True
        self.paused = True  # Start out paused.
        self.state = threading.Condition()

    # def __enter__(self):
    #     return self

    def start(self):
        self.thread.start()
        self.paused = False
        return self

    def show(self):

        while not self.stopped:
            with self.state:
                if self.paused:
                    self.state.wait()
            image_display = cv2.resize(self.frame, (self.displayHeight, self.displayWidth))
            cv2.imshow("Live View", image_display)
            if cv2.waitKey(1) == 27:
                self.paused = True
                self.stop()

    def resume(self):
        with self.state:
            self.paused = False
            self.state.notify()  # Unblock self if waiting.

    def pause(self):
        with self.state:
            self.paused = False
            self.state.notify()

    def stop(self):
        self.stopped = True
        # wait until stream resources are released (producer thread might be still grabbing frame)
        #self.thread.join()

    def __exit__(self):
        self.stop()


class ptgCamStream:
    def __init__(self, camera, transform=None, queue_size=128):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = camera
        self.stopped = False
        self.transform = transform

        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queue_size)
        # intialize thread
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.paused = True  # Start out paused.
        self.state = threading.Condition()
    # def __enter__(self):
    #     return self

    def start(self):
        # start a thread to read frames from the file video stream
        self.stream.setAcquisitMode(1)  # set to continuous acquisition
        self.stream.beginAcquisit(True)
        if self.stream._camera.IsStreaming():
            print("Camera Streaming")
        self.paused = False
        self.thread.start()
        return self

    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if cv2.waitKey(1) == 27:
                self.paused = True
                self.stop()

            if self.stopped:
                break

            with self.state:
                if self.paused:
                    self.state.wait()

            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                #(grabbed, frame) = self.stream.read()

                # Single frame
                #grabbed, frame = self.stream.grabFrame()

                #Continuous
                grabbed, frame = self.stream.grabFrameCont()


                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.stopped = True

                # if there are transforms to be done, might as well
                # do them on producer thread before handing back to
                # consumer thread. ie. Usually the producer is so far
                # ahead of consumer that we have time to spare.
                #
                # Python is not parallel but the transform operations
                # are usually OpenCV native so release the GIL.
                #
                # Really just trying to avoid spinning up additional
                # native threads and overheads of additional
                # producer/consumer queues since this one was generally
                # idle grabbing frames.
                if self.transform:
                    frame = self.transform(frame)

                # add the frame to the queue
                self.Q.put(frame)
            else:
                time.sleep(0.1)  # Rest for 10ms, we have a full queue

        # Continuous
        self.stream.beginAcquisit(False)
        #self.stream.release()

    def resume(self):
        with self.state:
            self.paused = False
            self.state.notify()  # Unblock self if waiting.

    def pause(self):
        with self.state:
            self.paused = False
            self.state.notify()

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    # Insufficient to have consumer use while(more()) which does
    # not take into account if the producer has reached end of
    # file stream.
    def running(self):
        return self.more() or not self.stopped

    def more(self):
        # return True if there are still frames in the queue. If stream is not stopped, try to wait a moment
        tries = 0
        while self.Q.qsize() == 0 and not self.stopped and tries < 5:
            time.sleep(0.1)
            tries += 1

        return self.Q.qsize() > 0

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        # wait until stream resources are released (producer thread might be still grabbing frame)
        #self.thread.join()

    def __exit__(self):
        self.stop()

class patternDisplay:
    """
    Class that continuously shows a frame using a dedicated thread.
    """
    def __init__(self, displayHeight=1080, displayWidth=1920, pattern=None):
        self.pattern = pattern
        self.stopped = False
        self.displayHeight = displayHeight
        self.displayWidth = displayWidth

        self.thread = threading.Thread(target=self.show, args=())
        self.thread.daemon = True

        print("0.3")
    # def __enter__(self):
    #     return self

    def start(self):
        self.thread.start()
        return self

    def show(self):
        print("0.4")
        while not self.stopped:
            window_name = 'projector'
            cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)

            # If projector is placed right to main screen (see windows properties in your operating system)
            # if the pattern is displayed at the wrong monitor you need to play around with the coordinates here until the image is displayed at the right screen
            cv2.moveWindow(window_name, self.displayWidth, 0)

            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            cv2.imshow(window_name, self.pattern.astype(np.float32))
            k = cv2.waitKey(0)
            if k == 27:  # wait for ESC key to exit
                self.stop()
                cv2.destroyAllWindows()

    def stop(self):
        self.stopped = True
        # wait until stream resources are released (producer thread might be still grabbing frame)
        #self.thread.join()

    def __exit__(self):
        self.stop()