import os

import cv2
import numpy as np
from PIL import Image
from keras.utils import np_utils

from Constants import nn_image_scale_factor


def create_dataset(images_folder, prefix_to_label, image_size, img_layers=3):
    i = 0
    images_list = os.listdir(images_folder)
    image_l = image_size[0] * image_size[1] * img_layers + 1
    ndim_vector = np.zeros((images_list.__len__(), image_l), dtype=np.ubyte)

    for file in images_list:
        file_prefix = file.partition("_")[0]
        if file_prefix in prefix_to_label:
            im = np.array(Image.open(images_folder + file))
            if img_layers == 1:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im = cv2.resize(im, (image_size[0], image_size[1]))
            im = cv2.equalizeHist(im)
            ndim_vector[i] = np.append(im, [prefix_to_label[file_prefix]]).reshape(1, -1)
            i += 1
    return ndim_vector


def create_dataset_nn(images_folder, prefix_to_label, image_size, img_layers=3):
    dataset = create_dataset(images_folder, prefix_to_label, image_size, img_layers)

    X_train = dataset[:, 0:-1]
    Y_train = dataset[:, -1]

    X_train = X_train.astype('float32')
    X_train /= 255

    # convert class vectors to binary class matrices
    from KerasNNModel import n_classes
    Y_train = np_utils.to_categorical(Y_train, n_classes)

    return X_train, Y_train


def image_path_to_ndim_vector(image_path, image_size):
    im = Image.open(image_path)
    im.thumbnail(image_size, Image.ANTIALIAS)
    im = cv2.cvtColor(np.array(im), cv2.COLOR_BGR2GRAY)
    cv2.equalizeHist(im, im)

    return np.array(im).reshape(1, -1)


def prepare_image_for_nn(ndim_array, image_path=None, histogram_eq=True):
    if image_path:
        ndim_array = cv2.cvtColor(np.array(Image.open(image_path)), cv2.COLOR_BGR2GRAY)
    im = cv2.resize(ndim_array, (0, 0), fx=nn_image_scale_factor[0], fy=nn_image_scale_factor[1])

    # equalization
    if histogram_eq:
        im = cv2.equalizeHist(im)

    # normalization
    im = im.astype('float32')
    im /= 255

    return im.reshape(1, -1)
