import sys
import time
from datetime import datetime
from threading import Timer

import cv2
import requests
from PIL import Image
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.svm import SVC

from Constants import capture_url, show_preview, real_image_size, image_size_divider
from Constants import relay_address, gpio_to_switch
from SendPostAsync import SendPostAsync
from images_to_ndim_vector import image_to_ndim_vector, create_dataset

timer = None
timer_delay = 3

after_low_timer = None
after_low_timer_delay = 3

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


learning_set_path = "Learning Set/"
absence_prefix = "Abs"
presence_prefix = "Pre"

image_size = tuple(size / image_size_divider for size in real_image_size)

model = None

if "--model-from-file" in sys.argv:
    model = joblib.load('model.pkl')
else:
    dataset = create_dataset(learning_set_path, {absence_prefix: 0, presence_prefix: 1}, image_size)

    X = dataset[:, 0:-1]
    y = dataset[:, -1]

    # model = LogisticRegression(max_iter=100, n_jobs=-1, verbose=1)  # Best match
    # model = SVC(kernel="rbf", verbose=1, probability=True)
    model = SVC(kernel="linear", verbose=1, probability=False)  # Best match
    # model = SGDClassifier(n_iter=2000, verbose=2, n_jobs=-1, warm_start=True)  # nice too

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

    # Our operations on the frame come here
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    gray_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    ndim_vector = image_to_ndim_vector(rgb_frame, image_size)

    predicted = model.predict(ndim_vector)
    # predicted = model._predict_proba(ndim_vector)
    print predicted
    if predicted[0] == 0:
        cv2.putText(bgr_frame, "0", (20, 30), cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 0)
    if predicted[0] == 1:
        cv2.putText(bgr_frame, "1", (20, 30), cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 0)
        on_detect()

    # Display the resulting frame
    if show_preview:
        cv2.imshow('frame', bgr_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    if key == ord('2'):
        # model.partial_fit(ndim_vector, [0], classes=unique_y)
        Image.fromarray(rgb_frame).save(learning_set_path + absence_prefix + "_" + str(datetime.now().time()) + ".jpg",
                                        "JPEG")

    if key == ord('1'):
        # model.partial_fit(ndim_vector, [1], classes=unique_y)
        Image.fromarray(rgb_frame).save(learning_set_path + presence_prefix + "_" + str(datetime.now().time()) + ".jpg",
                                        "JPEG")

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
