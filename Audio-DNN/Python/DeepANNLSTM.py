import keras
import keras.utils
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout
from keras.layers import TimeDistributed, LSTM

def train(x_train, y_train, x_test, y_test, num_classes, epochs):

    # examine x & y shapes
    print(' ')
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    print('y_train shape:', y_train.shape)
    print('y_test shape:', y_test.shape)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # define the model
    ann = Sequential()
    ann.add(Dense(units=32, activation='relu', input_dim=x_train.shape[2]))

    lstm = Sequential()
    lstm.add(TimeDistributed(ann, input_shape=(x_train.shape[1], x_train.shape[2])))
    lstm.add(LSTM(64, dropout=0.05, recurrent_dropout=0.05))
    lstm.add(Dense(num_classes, activation='softmax'))

    # print the model
    f = open('Models\\model_annlstm.txt', 'w')
    for i in range(0, len(ann.layers)):
        if i == 0:
            # print(cnn.summary())
            print(' ')
        print('{}. Layer {} with input / output shapes: {} / {}'.format(i, ann.layers[i].name, ann.layers[i].input_shape, ann.layers[i].output_shape))
        f.write('{}. Layer {} with input / output shapes: {} / {} \n'.format(i, ann.layers[i].name, ann.layers[i].input_shape, ann.layers[i].output_shape))
        if i == len(ann.layers) - 1:
            print(' ')
    for i in range(0, len(lstm.layers)):
        if i == 0:
            # print(lstm.summary())
            # print(' ')
            f.write('\n')
        print('{}. Layer {} with input / output shapes: {} / {}'.format(i, lstm.layers[i].name, lstm.layers[i].input_shape, lstm.layers[i].output_shape))
        f.write('{}. Layer {} with input / output shapes: {} / {} \n'.format(i, lstm.layers[i].name, lstm.layers[i].input_shape, lstm.layers[i].output_shape))
        if i == len(lstm.layers) - 1:
            print(' ')
    f.close()

    # compile, fit, evaluate
    lstm.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adamax(), metrics=['accuracy'])
    lstm.fit(x_train, y_train, batch_size=512, epochs=epochs, verbose=2, validation_data=(x_test, y_test))
    score = lstm.evaluate(x_test, y_test, batch_size=128)

    return score[1]