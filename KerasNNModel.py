import cPickle as pickle
from multiprocessing import Process

import numpy
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.optimizers import RMSprop

from Constants import *
from images_to_ndim_vector import prepare_image_for_nn, create_dataset_nn


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
                            input_shape=(1, nn_image_size[0], nn_image_size[1])
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

    model.add(Dense(n_hidden_layer_neurons, input_shape=(n_features,), init=init_method))
    model.add(Activation(activation_f))
    model.add(Dropout(dropout))
    for layer in range(0, nb_hidden_layers):
        model.add(Dense(n_hidden_layer_neurons, init=init_method))
        model.add(Activation(activation_f))
        model.add(Dropout(dropout))
    model.add(Dense(output_dim=n_classes))
    model.add(Activation("softmax"))

    # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms)

    return model


def lstm_model():
    model = Sequential()
    model.add(LSTM(output_dim=n_hidden_layer_neurons, return_sequences=True, input_shape=(1, nn_image_vector_size)))
    model.add(Dropout(0.2))
    model.add(LSTM(output_dim=n_hidden_layer_neurons, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model


def train(model):
    X_train, Y_train = create_dataset_nn(learning_set_path, {absence_prefix: 0, presence_prefix: 1}, nn_image_size,
                                         img_layers=1)

    # model = conv_model()
    # inpush_shape = (X_train.shape[0], 1, nn_image_size[0], nn_image_size[1])
    # model.fit(X_train.reshape(inpush_shape),
    #           Y_train,
    #           nb_epoch=nb_epoch,
    #           batch_size=64,
    #           show_accuracy=True,
    #           validation_data=(X_train.reshape(inpush_shape), Y_train),
    #           verbose=2)
    # print model.evaluate(X_train.reshape(inpush_shape), Y_train, batch_size=64)

    # for i in range(0, X_train.shape[0]):
    #     print model.train_on_batch(X_train[i].reshape(1, 1, 22, 18), [Y_train[i]], accuracy=True)

    # X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_train = numpy.load(open(nn_x_path))
    Y_train = numpy.load(open(nn_y_path))
    # X_train = X_train[0:9850].reshape((394, 25, nn_image_vector_size))
    # Y_train = Y_train[0:9850]
    model.fit(X_train,
              Y_train,
              nb_epoch=nb_epoch,
              batch_size=batch_size,
              show_accuracy=True,
              validation_data=(X_train, Y_train))
    model_evaluation_score = model.evaluate(X_train, Y_train, batch_size=batch_size)
    print model_evaluation_score

    json_string = model.to_json()
    open(nn_model_file_name, 'w').write(json_string)
    model.save_weights(nn_model_weights, overwrite=True)

    return model, model_evaluation_score


# activation_f = 'relu'
# activation_f = 'sigmoid'
activation_f = 'tanh'
init_method = 'he_normal'
n_features = nn_image_size[0] * nn_image_size[1]
n_hidden_layer_neurons = n_features * 3
n_classes = 2
nb_epoch = 30
nb_hidden_layers = 0
dropout = 0.2
batch_size = 128

if __name__ == '__main__':
    def plot_result():
        import matplotlib.pyplot as plt
        result_file = open('result.dump')
        if result_file:
            result = pickle.load(result_file)

            result_chunks = [result[i:i + 19] for i in xrange(0, len(result), 19)]

            for chunk in result_chunks:
                plt.figure()
                plt.title(chunk[0]['n_hidden_layer_neurons'])

                i_dropout = [i['dropout'] for i in chunk]
                i_model_evaluation_score = [i['model_evaluation_score'] for i in chunk]
                i_test_im_prediction = [i['test_im_prediction'][0][1] for i in chunk]
                i_test_im_prediction2 = [i['test_im_prediction2'][0][1] for i in chunk]

                plt.plot(i_dropout, i_model_evaluation_score)
                plt.plot(i_dropout, i_test_im_prediction)
                plt.plot(i_dropout, i_test_im_prediction2)
            plt.xlabel('dropout')
            plt.ylabel('model_evaluation_score')
            plt.show()


    Process(target=plot_result).start()

    if False:
        test_img = prepare_image_for_nn(None, 'test.jpg')
        test_img2 = prepare_image_for_nn(None, 'Abs_22:20:39.317447.jpg')
        result = []
        for n in range(1, 7):
            n_hidden_layer_neurons = n_features * n
            for i in range(1, 20):
                dropout = i / 20.
                model, model_evaluation_score = train()
                test_im_prediction = model.predict(test_img)
                test_im_prediction2 = model.predict(test_img2)

                result.append(dict(n_hidden_layer_neurons=n_hidden_layer_neurons,
                                   nb_epoch=nb_epoch,
                                   n_features=n_features,
                                   nn_image_size=nn_image_size,
                                   nb_hidden_layers=nb_hidden_layers,
                                   dropout=dropout,
                                   test_im_prediction=test_im_prediction,
                                   test_im_prediction2=test_im_prediction2,
                                   model_evaluation_score=model_evaluation_score))
                pickle.dump(result, open('result.dump', 'wb'))
