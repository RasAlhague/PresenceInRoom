import os

import numpy as np
from PIL import Image


def create_dataset(images_folder, prefix_to_label, image_size):
    i = 0
    images_list = os.listdir(images_folder)
    image_l = image_size[0] * image_size[1] * 3 + 1
    ndim_vector = np.zeros((images_list.__len__(), image_l), dtype=np.ubyte)

    for file in images_list:
        file_prefix = file.partition("_")[0]
        if file_prefix in prefix_to_label:
            im = Image.open(images_folder + file)
            im.thumbnail(image_size, Image.ANTIALIAS)
            ndim_vector[i] = np.append(np.array(im), [prefix_to_label[file_prefix]]).reshape(1, -1)
            i += 1
    return ndim_vector


def images_to_ndim_vector(images_folder, label, image_size):
    i = 0
    images_list = os.listdir(images_folder)
    image_l = image_size[0] * image_size[1] * 3 + 1
    ndim_vector = np.zeros((images_list.__len__(), image_l), dtype=np.ubyte)

    for file in images_list:
        im = Image.open(images_folder + file)
        im.thumbnail(image_size, Image.ANTIALIAS)
        ndim_vector[i] = np.append(np.array(im), [label]).reshape(1, -1)
        i += 1
    return ndim_vector


def image_path_to_ndim_vector(image_path, image_size):
    im = Image.open(image_path)
    im.thumbnail(image_size, Image.ANTIALIAS)

    return np.array(im).reshape(1, -1)


def image_to_ndim_vector(im, image_size):
    im = Image.fromarray(im)
    im.thumbnail(image_size, Image.ANTIALIAS)

    return np.array(im).reshape(1, -1)

