import time
from datetime import datetime

import cv2
import numpy as np


class MotionDetectorAdaptative():
    def onChange(self, val):  # callback when the user change the detection threshold
        self.detectionThreshold = val

    def __init__(self, detectionThreshold=25,
                 runningAvgAlpha=0.01,
                 ignoreThresholdBiggerThan=80,
                 doRecord=False,
                 showWindows=True,
                 onDetectCallback=None,
                 captureURL=0,
                 activationThreshold=50,
                 resolutionDivider=1,
                 dilateIter=15,
                 erodeIter=5):
        self.writer = None
        self.font = None
        self.doRecord = doRecord  # Either or not record the moving object
        self.show = showWindows  # Either or not show the 2 windows
        self.onDetectCallback = onDetectCallback  # On detect callback
        self.frame = None

        self.gray_frame = None
        self.average_frame = None
        self.absdiff_frame = None
        self.previous_frame = None

        self.resolutionDivider = resolutionDivider
        self.dilateIter = dilateIter
        self.erodeIter = erodeIter
        self.currentsurface = 0
        self.currentcontours = None
        self.detectionThreshold = detectionThreshold
        self.activationThreshold = activationThreshold
        self.runningAvgAlpha = runningAvgAlpha
        self.ignoreThresholdBiggerThan = ignoreThresholdBiggerThan
        self.isRecording = False
        self.trigger_time = 0  # Hold timestamp of the last detection

        self.cap = cv2.VideoCapture(captureURL)
        self.width = self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
        self.surface = (self.width / resolutionDivider) * (self.height / resolutionDivider)

        if doRecord:
            self.initRecorder()

        if showWindows:
            cv2.namedWindow("Image")
            cv2.createTrackbar("Detection treshold: ", "Image", self.detectionThreshold, 100, self.onChange)

    def initRecorder(self):  # Create the recorder
        codec = cv2.cv.CV_FOURCC('M', 'J', 'P', 'G')
        self.writer = cv2.VideoWriter(datetime.now().strftime("%b-%d_%H_%M_%S") + ".wmv", codec, 5,
                                      cv2.cv.GetSize(self.frame), 1)
        # FPS set to 5 because it seems to be the fps of my cam but should be ajusted to your needs
        self.font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 2, 8)  # Creates a font

    def run(self):
        started = time.time()
        adjustingTime = 0
        while True:
            ret, currentframe = self.cap.read()
            instant = time.time()  # Get timestamp o the frame

            currentframe = cv2.resize(currentframe, (int(self.width / self.resolutionDivider),
                                                     int(self.height / self.resolutionDivider)))
            self.processImage(currentframe)  # Process the image

            contourSurface = self.calculateContourSurface()
            # Adopt to light flashes and big changes
            if contourSurface > self.ignoreThresholdBiggerThan:
                adjustingTime = time.time() + 1
                started = time.time() - 4

            if instant < adjustingTime:
                print 'Adjusting'
                cv2.accumulateWeighted(currentframe, self.average_frame, 0.6)

            if not self.isRecording:
                if self.somethingHasMoved(contourSurface):
                    self.trigger_time = instant  # Update the trigger_time
                    if instant > started + 5:  # Wait 5 second after the webcam start for luminosity adjusting etc..
                        # print "Something is moving !"

                        if self.onDetectCallback is not None:
                            self.onDetectCallback()  # Do callback

                        if self.doRecord:  # set isRecording=True only if we record a video
                            self.isRecording = True
                cv2.drawContours(currentframe, self.currentcontours, -1, (0, 0, 255), 1)
            else:
                if instant >= self.trigger_time + 10:  # Record during 10 seconds
                    print "Stop recording"
                    self.isRecording = False
                else:
                    cv2.putText(currentframe, datetime.now().strftime("%b %d, %H:%M:%S"), (25, 30), self.font, 1, 0)
                    self.writer.write(currentframe)  # Write the frame

            if self.show:
                cv2.imshow("Image", currentframe)

            c = cv2.waitKey(1) % 0x100
            if c == 27 or c == 10:  # Break if user enters 'Esc'.
                break

    def processImage(self, curframe):
        if self.absdiff_frame is None or self.absdiff_frame.all():  # For the first time put values in difference, temp and moving_average
            self.absdiff_frame = curframe
            self.previous_frame = curframe
            self.average_frame = np.float32(curframe)  # Should convert because after runningavg take 32F pictures
        else:
            cv2.accumulateWeighted(curframe, self.average_frame, self.runningAvgAlpha)  # Compute the average

        cv2.convertScaleAbs(self.average_frame, self.previous_frame)  # moving_average - curframe

        if self.show:
            cv2.imshow("AVG", self.average_frame)

        cv2.absdiff(curframe, self.previous_frame, self.absdiff_frame)  # moving_average - curframe

        if self.show:
            cv2.imshow("AbsDiff", self.absdiff_frame)

        self.gray_frame = cv2.cvtColor(self.absdiff_frame,
                                       cv2.COLOR_RGB2GRAY)  # Convert to gray otherwise can't do threshold

        if self.show:
            cv2.imshow("gray_frame", self.gray_frame)

        cv2.threshold(self.gray_frame, self.activationThreshold, 255, cv2.THRESH_BINARY, self.gray_frame)

        if self.show:
            cv2.imshow("Threshold", self.gray_frame)

        self.gray_frame = cv2.dilate(self.gray_frame, None, iterations=5)  # to get object blobs

        if self.show:
            cv2.imshow("Dilate", self.gray_frame)

        self.gray_frame = cv2.erode(self.gray_frame, None, iterations=3)

        if self.show:
            cv2.imshow("Erode", self.gray_frame)

    def somethingHasMoved(self, contourSurface):
        if self.detectionThreshold < contourSurface < self.ignoreThresholdBiggerThan:
            return True
        else:
            return False

    def calculateContourSurface(self):
        # Find contours
        contours, wtf = cv2.findContours(self.gray_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.currentcontours = contours  # Save contours
        wholeContourSurface = 0

        # while contours:  # For all contours compute the area
        for contour in contours:  # For all contours compute the area
            contourArea = cv2.contourArea(contour)
            avg = (contourArea * 100) / self.surface
            wholeContourSurface += contourArea
            if avg > self.detectionThreshold:
                self.currentsurface += contourArea

        avg = (self.currentsurface * 100) / self.surface  # Calculate the average of contour area on the total size
        wholeAvg = (wholeContourSurface * 100) / self.surface
        self.currentsurface = 0  # Put back the current surface to 0

        print avg, "\t", wholeAvg

        return avg


if __name__ == "__main__":
    detect = MotionDetectorAdaptative()
    detect.run()
