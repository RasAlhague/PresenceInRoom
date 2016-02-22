import platform

import cv2

nn_model_file_name = 'nn_model_architecture.json'
nn_model_weights = 'nn_model_weights.h5'

learning_set_path = "Learning Set/"
absence_prefix = "Abs"
presence_prefix = "Pre"
relay_address = "http://192.168.1.52"

timer = None
timer_delay = 3

after_low_timer = None
after_low_timer_delay = 0

font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 2, 8)

# 1 = low; 0 = high
stream_quality = 1

capture_url = "rtsp://192.168.1.51:554/user=admin&password=&channel=1&stream=" + \
              str(stream_quality) + \
              ".sdp?real_stream--rtp-caching=0"

gpio_to_switch = 6

platform_architecture = platform.uname()[4]
show_preview = False if platform_architecture == "armv7l" else True

real_image_size = 352, 288
image_scale_factor = (0.16, 0.16)
nn_image_scale_factor = (0.06, 0.06)
image_size = (int(round(real_image_size[0] * image_scale_factor[0])),
              int(round(real_image_size[1] * image_scale_factor[1])))
nn_image_size = (int(round(real_image_size[0] * nn_image_scale_factor[0])),
                 int(round(real_image_size[1] * nn_image_scale_factor[1])))

schedule_for_opencv_routine = ("0 0 17 1/1 * ? *", "0 0 7 1/1 * ? *")
# schedule_for_opencv_routine = ("0 43 22 1/1 * ? *", "10 43 22 1/1 * ? *")
