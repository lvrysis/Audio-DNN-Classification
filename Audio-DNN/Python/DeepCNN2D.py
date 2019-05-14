import os, time, keras
from keras import backend, metrics, callbacks, regularizers
from keras.models import Sequential, model_from_json
from keras.layers import Dropout, Flatten, BatchNormalization
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.utils import plot_model, multi_gpu_model
import numpy as np
from sklearn.metrics import classification_report


def train(x_train, y_train, x_test, y_test, num_classes, epochs):

    # set validation data
    val_size = int(0.1 * x_train.shape[0])
    r = np.random.randint(0, x_train.shape[0], size=val_size)
    x_val = x_train[r, :, :]
    y_val = y_train[r]
    x_train = np.delete(x_train, r, axis=0)
    y_train = np.delete(y_train, r, axis=0)

    # set x & y shapes
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
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
    regularizer=0.001
    dropout=0.25

    model = Sequential()

    model.add(Conv2D(16, kernel_size=(3, 3), strides=(2, 1), activation='tanh', kernel_regularizer=regularizers.l2(regularizer), input_shape=(x_train.shape[1], x_train.shape[2], 1)))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Dropout(dropout))

    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='elu', kernel_regularizer=regularizers.l2(regularizer)))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))

    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='elu', kernel_regularizer=regularizers.l2(regularizer)))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))

    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='elu', kernel_regularizer=regularizers.l2(regularizer)))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))
    
    model.add(Flatten())

    model.add(Dense(64, activation='elu', kernel_regularizer=regularizers.l2(regularizer)))
    model.add(Dropout(dropout))

    model.add(Dense(num_classes, activation='softmax'))

    # summarize model
    # plot_model(model, to_file='Models\\model_cnn2d.png')
    f = open('Models\\model_cnn2d.txt', 'w')
    for i in range(0, len(model.layers)):
        if i == 0:
            print(' ')
        print('{}. Layer {} with input / output shapes: {} / {}'.format(i, model.layers[i].name, model.layers[i].input_shape, model.layers[i].output_shape))
        f.write('{}. Layer {} with input / output shapes: {} / {} \n'.format(i, model.layers[i].name, model.layers[i].input_shape, model.layers[i].output_shape))
        if i == len(model.layers) - 1:
            print(' ')
    f.close()
    model.summary()

    # compile, fit evaluate
    try:    model = multi_gpu_model(model, gpus=2, cpu_merge=False)
    except: model = model
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics = ['accuracy'])
    model.fit(x_train, y_train, batch_size=192, epochs=epochs, verbose=2, validation_data=(x_val, y_val), callbacks=[callbacks.EarlyStopping(monitor='val_acc', patience=20, restore_best_weights=True)])
    time.sleep(5)
    score = model.evaluate(x_test, y_test, verbose=2)
    time.sleep(5)

    # save model
    '''model_json = model.to_json()
    with open("Models\\model_cnn2d.json", "w") as json_file:
        # remark
        json_file.write(model_json)
    model.save_weights("Models\\model_cnn2d.h5")'''

    # results
    # print('Test loss: {0:.3f} and accuracy: {1:.3f}'.format(score[0], score[1]))
    print(' ')
    return score[1]


def predict(model, x_test):
    # load
    json_file = open('model_cnn2d.json', 'r')
    model = model_from_json(json_file.read())
    json_file.close()
    model.load_weights("model_cnn2d.h5")

    prediction = model.predict(x_test, batch_size=32, verbose=0)
    print(prediction)