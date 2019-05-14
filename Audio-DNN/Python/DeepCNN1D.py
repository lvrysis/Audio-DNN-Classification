import time
import numpy as np
import keras
from keras import backend, metrics, callbacks
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D


def train(x_train, y_train, x_test, y_test, num_classes, epochs):

    # set validation data
    val_size = int(0.1 * x_train.shape[0])
    r = np.random.randint(0, x_train.shape[0], size=val_size)
    x_val = x_train[r, :]
    y_val = y_train[r]
    x_train = np.delete(x_train, r, axis=0)
    y_train = np.delete(y_train, r, axis=0)

    # set x & y shapes
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_val = y_val.reshape(y_val.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)

    print('x_train shape:', x_train.shape)
    print('x_val shape:', x_val.shape)
    print('x_test shape:', x_test.shape)
    print('y_train shape:', y_train.shape)
    print('y_val shape:', y_val.shape)
    print('y_test shape:', y_test.shape)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # define the model
    model = Sequential()

    model.add(Conv1D(32, kernel_size=7, strides=5, activation='tanh', input_shape=(x_train.shape[1], 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Conv1D(64, kernel_size=7, strides=5, activation='elu'))
    # model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Conv1D(64, kernel_size=7, strides=5, activation='elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Conv1D(128, kernel_size=7, strides=5, activation='elu'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(64, activation='elu'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))

    # print the model
    model.summary()
    '''f = open('Models\\model_cnn1d.txt', 'w')
    for i in range(0, len(model.layers)):
        if i == 0:
            print(' ')
        print('{}. Layer {} with input / output shapes: {} / {}'.format(i, model.layers[i].name, model.layers[i].input_shape, model.layers[i].output_shape))
        f.write('{}. Layer {} with input / output shapes: {} / {} \n'.format(i, model.layers[i].name, model.layers[i].input_shape, model.layers[i].output_shape))
        if i == len(model.layers) - 1:
            print(' ')
    f.close()
    keras.utils.plot_model(model, to_file='Models\\model_cnn1d.png')'''

    # compile, fit, evaluate
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=32, epochs=epochs, verbose=2, validation_data=(x_val, y_val), callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])
    time.sleep(5)
    score = model.evaluate(x_test, y_test, verbose=2)
    time.sleep(5)

    # save model
    '''model_json = model.to_json()
    with open("Models\\model_cnn1d.json", "w") as json_file:
        #
        json_file.write(model_json)
    model.save_weights("Models\\model_cnn1d.h5")'''

    # return results
    # print('Test loss: {0:.2f} and accuracy: {1:.2f}'.format(score[0], score[1]))
    print(' ')

    return score[1]