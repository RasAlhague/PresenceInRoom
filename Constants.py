import platform

import cv2

# nn_model_file_name = 'nn_model_architecture.json'
# nn_model_weights = 'nn_model_weights.h5'
nn_path = '16.02.27 10948 lstm(1, 357)/'
nn_model_file_name = nn_path + 'nn_model_architecture.json'
nn_model_weights = nn_path + 'nn_model_weights.h5'
nn_x_path = nn_path + 'X_train'
nn_y_path = nn_path + 'Y_train'

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
nn_image_vector_size = nn_image_size[0] * nn_image_size[1]

# opencv_routine_schedule = {
#     'format': '%H %M %S',
#     'start_opencv_routine_from': '16 00 00',
#     'stop_opencv_routine_at': '09 00 00',
# }

opencv_routine_schedule = {
    'format': '%H %M %S',
    'start_opencv_routine_from': '02 00 00',
    'stop_opencv_routine_at': '01 59 00',
}
