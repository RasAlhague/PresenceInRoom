import sys
import time
from datetime import datetime
from threading import Timer, Thread

import cv2
import requests
from PIL import Image
from flask_restful import Resource
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.svm import SVC

from Constants import *
from Constants import relay_address, gpio_to_switch
from DimensionalityReduction import DimensionalityReduction
from SendPostAsync import SendPostAsync
from WebServer import WebServer
from images_to_ndim_vector import image_to_ndim_vector, create_dataset

timer = None
timer_delay = 3

after_low_timer = None
after_low_timer_delay = 0

font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 2, 8)


def pass_f():
    print '\t\tReady to High!'


def send_post(to):
    SendPostAsync(target=requests.post, args=(to,)).start()


def set_low(_relay_address):
    send_post(_relay_address + "/gpio/" + str(gpio_to_switch) + "/low")

    global after_low_timer
    after_low_timer = Timer(after_low_timer_delay, pass_f, ())
    after_low_timer.start()


def set_high(_relay_address):
    send_post(_relay_address + "/gpio/" + str(gpio_to_switch) + "/high")


def on_detect():
    global timer
    if (after_low_timer is None) or (not after_low_timer.isAlive()):
        if timer is None or not timer.isAlive():
            set_high(relay_address)
        else:
            timer.cancel()

        timer = Timer(timer_delay, set_low, [relay_address])
        timer.start()


def on_record_mode_change(rm):
    global record_mode
    record_mode = rm


def opencv_routine():
    model = None

    # sys.argv.append("--model-from-file")
    if "--model-from-file" in sys.argv:
        model = joblib.load('model.pkl')
    else:
        dataset = create_dataset(learning_set_path, {absence_prefix: 0, presence_prefix: 1}, image_size, img_layers=1)

        X = dataset[:, 0:-1]
        y = dataset[:, -1]

        if show_preview:
            DimensionalityReduction(X, y)

        # model = LogisticRegression(max_iter=100, n_jobs=-1, verbose=1)  # Best match
        # model = SVC(kernel="rbf", C=1, coef0=5, verbose=1, probability=True, decision_function_shape='ovr')
        model = SVC(kernel="linear", C=1, verbose=True, probability=True, decision_function_shape='ovr')  # Best match

        t0 = time.time()
        model.fit(X, y)
        t1 = time.time()

        total = t1 - t0
        print('Fitting time: ' + str(total))
        print(model)

        # save model
        joblib.dump(model, 'model.pkl', compress=3)

        # make predictions
        expected = y
        predicted = model.predict(X)

        # summarize the fit of the model
        print(metrics.classification_report(expected, predicted))
        print(metrics.confusion_matrix(expected, predicted))

    cap = cv2.VideoCapture(capture_url)
    while True:
        # Capture frame-by-frame
        ret, bgr_frame = cap.read()

        # Choose color space
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        # Convert to row vector
        ndim_vector = image_to_ndim_vector(gray_frame, image_size)

        predicted = model.predict(ndim_vector)
        _predict_proba = model._predict_proba(ndim_vector)
        print predicted, "\t", _predict_proba
        if predicted[0] == 0:
            cv2.putText(gray_frame, "0", (20, 30), cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 0)
        if predicted[0] == 1:
            cv2.putText(gray_frame, "1", (20, 30), cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 0)
            on_detect()

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


class Handler(Resource):
    def post(self, rm):
        global record_mode
        record_mode = rm
        return {'record_mode': rm}


opencv_routine = Thread(target=opencv_routine)
opencv_routine.setDaemon(True)
opencv_routine.start()

WebServer().map_post({Handler: "/mode/set/<int:rm>"}).run()
