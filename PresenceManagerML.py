import sys
import time
from threading import Timer

import requests

import WebServer
from Constants import *
from Constants import relay_address, gpio_to_switch
from DimensionalityReduction import DimensionalityReduction
from KerasNNModel import train
from OpenCVRoutine import OpenCVRoutine
from SendPostAsync import SendPostAsync
from images_to_ndim_vector import create_dataset, prepare_image_for_nn


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


def init_model():
    # sys.argv.append("--model-from-file")
    if "-svm" in sys.argv:
        from sklearn.externals import joblib
        from sklearn.svm import SVC
        from sklearn import metrics

        if "--model-from-file" in sys.argv:
            model = joblib.load('model.pkl')
            # return Model(predict=model.predict)
            return model
        else:
            dataset = create_dataset(learning_set_path, {absence_prefix: 0, presence_prefix: 1}, image_size,
                                     img_layers=1)

            X = dataset[:, 0:-1]
            y = dataset[:, -1]

            if show_preview:
                DimensionalityReduction(X, y)

            # model = LogisticRegression(n_jobs=-1, verbose=1)  # Best match
            model = SVC(kernel="linear", C=1, verbose=True, probability=True,
                        decision_function_shape='ovr')  # Best match

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
            return model
            # return Model(predict=model.predict)
    elif "-nn" in sys.argv:
        if "--model-from-file" in sys.argv:
            from keras.models import model_from_json
            model = model_from_json(open(nn_model_file_name).read())
            model.load_weights(nn_model_weights)
            return model
            # return Model(predict=model.predict_classes)
        elif "--model-from-file" not in sys.argv:
            return train()[0]


def frame_handler(gray_frame):
    predicted = None
    _predict_proba = None

    if "-nn" in sys.argv:
        # NN predict time: 0.00577807426453
        ndim_vector = prepare_image_for_nn(gray_frame)
        # ndim_vector = ndim_vector.reshape(1, 1, 22, 18)

        predicted = model.predict_classes(ndim_vector)
        _predict_proba = model.predict(ndim_vector)
    elif "-svm" in sys.argv:
        # SVM predict time: 0.00129914283752
        ndim_vector = cv2.resize(gray_frame, (image_size[0], image_size[1])).reshape(1, -1)
        predicted = model.predict(ndim_vector)
        if model.get_params()['probability']:
            _predict_proba = model._predict_proba(ndim_vector)

    print predicted, "\t", _predict_proba

    if predicted[0] == 0:
        cv2.putText(gray_frame, "0", (20, 30), cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 0)
    if predicted[0] == 1:
        cv2.putText(gray_frame, "1", (20, 30), cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 0)
        on_detect()

model = init_model()

OpenCVRoutine(frame_callback=frame_handler)

WebServer.run()
