from threading import Timer

import requests

from MotionDetectorContours import MotionDetectorAdaptative
from SendPostAsync import SendPostAsync

relay_address = "http://192.168.1.52"

# 1 = low; 0 = high
quality = 1
captureURL = "rtsp://192.168.1.51:554/user=admin&password=&channel=1&stream=" + \
             str(quality) + \
             ".sdp?real_stream--rtp-caching=100"

gpio_number = 6

timer = None
timer_delay = 3

after_low_timer = None
after_low_timer_delay = 3


def pass_f():
    print '\t\tReady to High!'


def send_post(to):
    SendPostAsync(target=requests.post, args=(to,)).start()


def set_low(_relay_address):
    send_post(_relay_address + "/gpio/" + str(gpio_number) + "/low")

    global after_low_timer
    after_low_timer = Timer(after_low_timer_delay, pass_f, ())
    after_low_timer.start()


def set_high(_relay_address):
    send_post(_relay_address + "/gpio/" + str(gpio_number) + "/high")


def on_detect():
    global timer
    if (after_low_timer is None) or (not after_low_timer.isAlive()):
        if timer is None or not timer.isAlive():
            set_high(relay_address)
        else:
            timer.cancel()

        timer = Timer(timer_delay, set_low, [relay_address])
        timer.start()


detect = MotionDetectorAdaptative(detectionThreshold=6,
                                  runningAvgAlpha=0.01,
                                  ignoreThresholdBiggerThan=60,
                                  onDetectCallback=on_detect,
                                  captureURL=captureURL,
                                  activationThreshold=50,
                                  showWindows=False)
detect.run()
