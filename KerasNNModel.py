from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils import np_utils

from Constants import *
from images_to_ndim_vector import create_dataset


def conv_model():
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3

    model = Sequential()
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(1, image_size[0], image_size[1])
                            ))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms)

    return model


def mlp_model():
    model = Sequential()

    model.add(Dense(n_hidden_layer_neurons, input_shape=(n_features,)))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    for layer in range(0, nb_hidden_layers):
        model.add(Dense(n_hidden_layer_neurons))
        model.add(Activation('relu'))
        model.add(Dropout(dropout))
    model.add(Dense(output_dim=n_classes))
    model.add(Activation("softmax"))

    # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms)

    return model


def train():
    dataset = create_dataset(learning_set_path, {absence_prefix: 0, presence_prefix: 1}, image_size, img_layers=1)

    X_train = dataset[:, 0:-1]
    Y_train = dataset[:, -1]

    X_train = X_train.astype('float32')
    X_train /= 255

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(Y_train, n_classes)

    model = conv_model()
    inpush_shape = (X_train.shape[0], 1, image_size[0], image_size[1])
    model.fit(X_train.reshape(inpush_shape),
              Y_train,
              nb_epoch=nb_epoch,
              batch_size=64,
              show_accuracy=True,
              validation_data=(X_train.reshape(inpush_shape), Y_train),
              verbose=2)
    print model.evaluate(X_train.reshape(inpush_shape), Y_train, batch_size=64)

    # for i in range(0, X_train.shape[0]):
    #     print model.train_on_batch(X_train[i].reshape(1, 1, 22, 18), [Y_train[i]], accuracy=True)

    # model = mlp_model()
    # model.fit(X_train, Y_train, nb_epoch=nb_epoch, batch_size=64, show_accuracy=True,
    #           validation_data=(X_train, Y_train))
    # print model.evaluate(X_train, Y_train, batch_size=64)

    json_string = model.to_json()
    open(nn_model_file_name, 'w').write(json_string)
    model.save_weights(nn_model_weights, overwrite=True)

    return model


image_size = divide_image_size(16)
n_features = image_size[0] * image_size[1]
n_hidden_layer_neurons = n_features * 2  # 512
n_classes = 2
nb_epoch = 20
nb_hidden_layers = 0
dropout = 0.9

if __name__ == '__main__':
    train()
