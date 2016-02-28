import datetime
import sys
import time
from multiprocessing import Queue
from threading import Timer, Thread

import numpy
import requests

import WebServer
from Constants import *
from Constants import relay_address, gpio_to_switch
from DimensionalityReduction import DimensionalityReduction
from KerasNNModel import lstm_model
from OpenCVRoutineManager import OpenCVRoutineManager
from SendPostAsync import SendPostAsync
from images_to_ndim_vector import create_dataset, prepare_image_for_nn


def current_milli_time():
    return int(round(time.time() * 1000))


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


def load_svm_model():
    from sklearn.externals import joblib
    model = joblib.load('model.pkl')
    return model


def load_nn_model():
    # model = model_from_json(open(nn_model_file_name).read())
    model = lstm_model()
    model.load_weights(nn_model_weights)
    return model


def predict_svm(gray_frame):
    # SVM predict time: 0.00129914283752

    temp_model = model
    if model.__class__ is dict:
        temp_model = model['svm']

    ndim_vector = cv2.resize(gray_frame, (image_size[0], image_size[1])).reshape(1, -1)
    predicted = temp_model.predict(ndim_vector)
    if temp_model.get_params()['probability']:
        _predict_proba = temp_model._predict_proba(ndim_vector)
    else:
        _predict_proba = None

    return predicted, _predict_proba


def predict_nn(ndim_vector):
    # NN predict time: 0.00577807426453

    temp_model = model
    if model.__class__ is dict:
        temp_model = model['nn']

    predicted = temp_model.predict_classes(ndim_vector)
    _predict_proba = temp_model.predict(ndim_vector)

    return predicted, _predict_proba


def init_model():
    if "-svm" in sys.argv:
        from sklearn.externals import joblib
        from sklearn.svm import SVC
        from sklearn import metrics

        if "--model-from-file" in sys.argv:
            return load_svm_model()
        else:
            dataset = create_dataset(learning_set_path, {absence_prefix: 0, presence_prefix: 1}, image_size,
                                     img_layers=1)

            numpy.random.shuffle(dataset)

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

    elif "-nn" in sys.argv:
        from KerasNNModel import train
        if "--model-from-file" in sys.argv:
            return load_nn_model()
        elif "--model-from-file" not in sys.argv:
            return train(lstm_model())[0]

    elif "--comparison" in sys.argv:
        if "--model-from-file" in sys.argv:
            global xy_queue, comparison
            xy_queue = Queue()
            from ComparisonPlot import ComparisonPlot
            comparison = ComparisonPlot(xy_queue)
            return dict(svm=load_svm_model(), nn=load_nn_model())


def frame_handler(frame_queue):
    predicted = None
    _predict_proba = None
    last_prediction = None

    # X_train, Y_train = create_dataset_nn(learning_set_path, {absence_prefix: 0, presence_prefix: 1}, nn_image_size,
    #                                      img_layers=1)
    # X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_train = numpy.load(open(nn_x_path))
    Y_train = numpy.load(open(nn_y_path))
    X_train_l = X_train.shape[0]
    buffer_size = 100

    while True:
        gray_frame = frame_queue.get()

        if "-nn" in sys.argv:
            ndim_vector = prepare_image_for_nn(gray_frame).reshape(1, 1, nn_image_vector_size)
            predicted, _predict_proba = predict_nn(ndim_vector)

            if WebServer.record_mode != 0:
                X_train = numpy.append(X_train, ndim_vector, axis=0)
            if WebServer.record_mode == 1:
                Y_train = numpy.append(Y_train, [[0, 1]], axis=0)
            if WebServer.record_mode == 2:
                Y_train = numpy.append(Y_train, [[1, 0]], axis=0)

            if X_train.shape[0] == X_train_l + buffer_size:
                model.train_on_batch(X_train, Y_train)
                print model.evaluate(X_train, Y_train)
                X_train_l = X_train.shape[0]

        elif "-svm" in sys.argv:
            predicted, _predict_proba = predict_svm(gray_frame)

        elif "--comparison" in sys.argv:
            predicted_nn, _predict_proba_nn = predict_nn(gray_frame)
            print 'NN: \t', predicted_nn, "\t", _predict_proba_nn
            predicted, _predict_proba = predict_svm(gray_frame)
            print 'SVM:\t', predicted, "\t", _predict_proba

            xy_queue.put(([current_milli_time(), current_milli_time()],
                          [_predict_proba_nn[0][1], _predict_proba[0][1]]))

        if "--verbose" in sys.argv:
            print str(datetime.datetime.now().time()), "\t", predicted, "\t", _predict_proba
        elif last_prediction != predicted:
            print str(datetime.datetime.now().time()), "\t", predicted, "\t", _predict_proba

        if predicted[0] == 0:
            cv2.putText(gray_frame, "0", (20, 30), cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 0)
        if predicted[0] == 1:
            cv2.putText(gray_frame, "1", (20, 30), cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 0)
            on_detect()

        last_prediction = predicted


if __name__ == '__main__':
    model = init_model()
    frame_queue = Queue()

    if model:
        OpenCVRoutineManager(opencv_routine_schedule, frame_queue)

        frame_handler_thread = Thread(target=frame_handler, args=(frame_queue,))
        frame_handler_thread.setDaemon(True)
        frame_handler_thread.start()

        WebServer.run()
