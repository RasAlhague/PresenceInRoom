from threading import Timer, Thread

import requests

from MotionDetectorContours import MotionDetectorAdaptative


relay_address = "http://192.168.1.50"
timer_delay = 2
timer = None


def send_post(to):
    try:
        Thread(target=requests.post, args=(to, )).start()
    except:
        print "Oops! ConnectionError"


def set_low(_relay_address):
    send_post(_relay_address + "/gpio0/low")
    print(_relay_address + "/gpio0/low")


def set_high(_relay_address):
    send_post(_relay_address + "/gpio0/high")
    print(_relay_address + "/gpio0/high")


def on_detect():
    global timer
    if timer is None or not timer.isAlive():
        set_high(relay_address)
    else:
        timer.cancel()

    timer = Timer(timer_delay, set_low, [relay_address])
    timer.start()


detect = MotionDetectorAdaptative(threshold=10, onDetectCallback=on_detect)
detect.run()


