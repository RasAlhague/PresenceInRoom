from threading import Timer

import requests

from MotionDetectorContours import MotionDetectorAdaptative
from SendPostAsync import SendPostAsync

relay_address = "http://192.168.1.50"
# captureURL = 0
# captureURL = "http://192.168.1.100:8080/video"
captureURL = "rtsp://192.168.1.51:554/user=admin&password=&channel=1&stream=0.sdp?real_stream--rtp-caching=100"
timer_delay = 2
timer = None


def send_post(to):
    SendPostAsync(target=requests.post, args=(to,)).start()


def set_low(_relay_address):
    send_post(_relay_address + "/gpio/0/low")


def set_high(_relay_address):
    send_post(_relay_address + "/gpio/0/high")


def on_detect():
    global timer
    if timer is None or not timer.isAlive():
        set_high(relay_address)
    else:
        timer.cancel()

    timer = Timer(timer_delay, set_low, [relay_address])
    timer.start()


detect = MotionDetectorAdaptative(contourThreshold=10, onDetectCallback=on_detect, captureURL=captureURL,
                                  activationThreshold=25)
detect.run()
