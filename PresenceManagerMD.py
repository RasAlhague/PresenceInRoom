from threading import Timer

import requests

from Constants import relay_address, gpio_to_switch, platform_architecture, after_low_timer_delay, \
    timer_delay, after_low_timer, capture_url
from MotionDetectorContours import MotionDetectorAdaptative
from SendPostAsync import SendPostAsync

timer = None

running_avg_alpha = 0.01
slow_running_avg_alpha = 0


def pass_f():
    print '\t\tReady to High!'


def send_post(to):
    SendPostAsync(target=requests.post, args=(to,)).start()


def set_low(_relay_address):
    send_post(_relay_address + "/gpio/" + str(gpio_to_switch) + "/low")

    after_low_timer = Timer(after_low_timer_delay, pass_f, ())
    after_low_timer.start()

    detect.runningAvgAlpha = running_avg_alpha


def set_high(_relay_address):
    send_post(_relay_address + "/gpio/" + str(gpio_to_switch) + "/high")


def on_detect():
    global timer
    if (after_low_timer is None) or (not after_low_timer.isAlive()):
        if timer is None or not timer.isAlive():
            set_high(relay_address)
            detect.runningAvgAlpha = slow_running_avg_alpha  # slowdown to better static human position processing
        else:
            timer.cancel()

        timer = Timer(timer_delay, set_low, [relay_address])
        timer.start()


# platform_architecture = "armv7l"
detect = MotionDetectorAdaptative(detectionThreshold=6,
                                  runningAvgAlpha=running_avg_alpha,
                                  ignoreThresholdBiggerThan=60,
                                  onDetectCallback=on_detect,
                                  captureURL=capture_url,
                                  # captureURL=0,
                                  activationThreshold=30,
                                  showWindows=False if platform_architecture == "armv7l" else True,
                                  resolutionDivider=2 if platform_architecture == "armv7l" else 1,
                                  dilateIter=5 if platform_architecture == "armv7l" else 15,
                                  erodeIter=1 if platform_architecture == "armv7l" else 3)
detect.run()
