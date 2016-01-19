import platform

import cv2

nn_model_file_name = 'nn_model_architecture.json'
nn_model_weights = 'nn_model_weights.h5'

learning_set_path = "Learning Set/"
absence_prefix = "Abs"
presence_prefix = "Pre"
relay_address = "http://192.168.1.52"

timer = None
timer_delay = 1

after_low_timer = None
after_low_timer_delay = 0

font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 2, 8)

# 1 = low; 0 = high
stream_quality = 1

capture_url = "rtsp://192.168.1.51:554/user=admin&password=&channel=1&stream=" + \
              str(stream_quality) + \
              ".sdp?real_stream--rtp-caching=100"

gpio_to_switch = 6

platform_architecture = platform.uname()[4]
show_preview = False if platform_architecture == "armv7l" else True

real_image_size = 352, 288
image_size_divider = 8
image_size = tuple(size / image_size_divider for size in real_image_size)


def divide_image_size(divider):
    return tuple(size / divider for size in real_image_size)


record_mode = 0
