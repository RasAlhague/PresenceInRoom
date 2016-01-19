from datetime import datetime
from threading import Thread

from PIL import Image

from Constants import *


class OpenCVRoutine(Thread):
    def __init__(self, frame_callback):
        super(OpenCVRoutine, self).__init__()

        self.frame_callback = frame_callback
        self.setDaemon(True)
        self.start()

    def run(self):
        self.opencv_routine()

    def opencv_routine(self):
        cap = cv2.VideoCapture(capture_url)
        while True:
            # Capture frame-by-frame
            ret, bgr_frame = cap.read()

            # Choose color space
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            gray_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)

            self.frame_callback(gray_frame)

            # Display the resulting frame
            if show_preview:
                cv2.imshow('frame', gray_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            if key == ord('2') or record_mode == 2:
                Image.fromarray(rgb_frame).save(
                        learning_set_path + absence_prefix + "_" + str(datetime.now().time()) + ".jpg",
                        "JPEG")

            if key == ord('1') or record_mode == 1:
                Image.fromarray(rgb_frame).save(
                        learning_set_path + presence_prefix + "_" + str(datetime.now().time()) + ".jpg",
                        "JPEG")

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
