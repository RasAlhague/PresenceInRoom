from datetime import datetime

import cv2
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression

from images_to_ndim_vector import image_to_ndim_vector
from images_to_ndim_vector import images_to_ndim_vector

absencePath = "Learning Set/Absence/"
presencePath = "Learning Set/Presence/"

imageSize = 160, 88

absenceImages = images_to_ndim_vector(absencePath, 0, imageSize)
presenceImages = images_to_ndim_vector(presencePath, 1, imageSize)

dataset = np.concatenate((absenceImages, presenceImages))
# dataset = np.load("dataset.npy")

X = dataset[:, 0:-1]
y = dataset[:, -1]

unique_y = np.unique(y)

model = LogisticRegression(max_iter=100, n_jobs=-1, verbose=1)  # Best match
# model = SVC(kernel="rbf", max_iter=-1, verbose=1)
# model = SVC(kernel="poly", max_iter=-1, verbose=1, degree=2)
# model = SVC(kernel="linear", verbose=1)  # Best match

# -- Epoch 2000
# Norm: 4787.94, NNZs: 42240, Bias: -24.217423, T: 1384000, Avg. loss: 367011.945610
# model = SGDClassifier(n_iter=2000, verbose=2, n_jobs=-1, warm_start=True)  # nice too


t0 = time.time()
model.fit(X, y)
t1 = time.time()

total = t1-t0
print('Fitting time: ' + str(total))
print(model)

# make predictions
expected = y
predicted = model.predict(X)

# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

# from keras.models import Sequential
# from keras.layers.core import Dense, Dropout, Activation
# from keras.optimizers import RMSprop
# from keras.utils import np_utils
#
# batch_size = 128
# nb_classes = 2
# nb_epoch = 3
#
# # convert class vectors to binary class matrices
# Y_train = np_utils.to_categorical(y, nb_classes)
#
# model = Sequential()
# model.add(Dense(512, input_shape=(10560,)))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(512))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(2))
# model.add(Activation('softmax'))
#
# rms = RMSprop()
# model.compile(loss='categorical_crossentropy', optimizer=rms)
#
# model.fit(X, Y_train,
#           batch_size=batch_size, nb_epoch=nb_epoch,
#           show_accuracy=True, verbose=2,
#           validation_data=(X, Y_train))
# score = model.evaluate(X, Y_train, show_accuracy=True, verbose=0)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])



# cap = cv2.VideoCapture('http://127.0.0.1:8080/web.mjpg')
cap = cv2.VideoCapture(0)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 864)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)

last_coef = []
while True:
    # Capture frame-by-frame
    ret, bgr_frame = cap.read()

    # Our operations on the frame come here
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    gray_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    ndim_vector = image_to_ndim_vector(rgb_frame, imageSize)

    # predicted = model.predict_classes(ndim_vector, batch_size=64)
    predicted = model.predict(ndim_vector)
    if predicted[0] == 0:
        cv2.putText(bgr_frame, "0", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
    if predicted[0] == 1:
        cv2.putText(bgr_frame, "1", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)

    # Display the resulting frame
    cv2.imshow('frame', bgr_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    if key == ord('2'):
        # model.partial_fit(ndim_vector, [0], classes=unique_y)
        Image.fromarray(rgb_frame).save(absencePath + str(datetime.now().time()) + ".jpg", "JPEG")

    if key == ord('1'):
        # model.partial_fit(ndim_vector, [1], classes=unique_y)
        Image.fromarray(rgb_frame).save(presencePath + str(datetime.now().time()) + ".jpg", "JPEG")

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
