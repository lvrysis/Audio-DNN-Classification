import keras
from keras import utils, callbacks, regularizers
from keras.models import Sequential, model_from_json
from keras.layers import Activation, BatchNormalization, Dropout, Dense
from sklearn.preprocessing import LabelEncoder
import numpy as np
import time

def train(x_train, y_train, x_test, y_test, num_classes, epochs):

    # set validation data
    val_size = int(0.1 * x_train.shape[0])
    r = np.random.randint(0, x_train.shape[0], size=val_size)
    x_val = x_train[r, :]
    y_val = y_train[r]
    x_train = np.delete(x_train, r, axis=0)
    y_train = np.delete(y_train, r, axis=0)

    # set x & y shapes
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
    regularizer = 0.001

    model = Sequential()

    model.add(Dense(units=int((x_train.shape[1]+num_classes)*0.50), activation='tanh', kernel_regularizer=regularizers.l2(regularizer), input_shape=(x_train.shape[1],)))
    model.add(Dropout(0.4))

    model.add(Dense(units=int((x_train.shape[1]+num_classes)*0.50), activation='elu', kernel_regularizer=regularizers.l2(regularizer)))
    model.add(Dropout(0.2))

    model.add(Dense(units=num_classes, activation='softmax'))

    # summarize model
    model.summary()
    '''f = open('Models\\model_ann.txt', 'w')
    for i in range(0, len(model.layers)):
        if i == 0:
            print(' ')
        print('{}. Layer {} with input / output shapes: {} / {}'.format(i, model.layers[i].name, model.layers[i].input_shape, model.layers[i].output_shape))
        f.write('{}. Layer {} with input / output shapes: {} / {} \n'.format(i, model.layers[i].name, model.layers[i].input_shape, model.layers[i].output_shape))
        if i == len(model.layers) - 1:
            print(' ')
    f.close()'''

    # compile, fit, evaluate
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=256, epochs=epochs, verbose=2, validation_data=(x_val, y_val), callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)])
    time.sleep(5)
    score = model.evaluate(x_test, y_test, batch_size=128)

    # save model
    '''model_json = model.to_json()
    with open("Models\\model_ann.json", "w") as json_file:
        # remark
        json_file.write(model_json)
    model.save_weights("Models\\model_ann.h5")'''

    # return results
    # print(' ')
    # print('Best accuracy: {0:.1f}'.format(100*score[1]))
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
    # x_train = np.genfromtxt('Input/data.csv', delimiter=',', usecols=(1,2,3,4,5,6,7,8,9,10))
    # y_train = np.genfromtxt('Input/data.csv', delimiter=',', usecols=(0), dtype='str')
    #
    #
    # encoder = LabelEncoder()
    # encoder.fit(y_train)
    # y_train = encoder.transform(y_train)
    # print(y_train[:-10])
    # y_train = np.reshape(y_train, (y_train.shape[0], 1))
    # print(y_train.shape)
    # # y_train = np.random.randint(3, size=(1000, 1))
    # # print y_train.shape
    #
    # # y_train = np.random.randint(3, size=(7752, 1))
    # # print(y_train.shape)
    # # print(y_train)
    # one_hot_labels = keras.utils.to_categorical(y_train, num_classes=3)
    #
    #
    # # x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
    # model.fit(x_train, one_hot_labels, epochs=100, batch_size=32)
    # score = model.evaluate(x_train, one_hot_labels, batch_size=128)
    #
    # # print("Score Sample: {}".format(score))