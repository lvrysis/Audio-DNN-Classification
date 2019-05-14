import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import TimeDistributed, LSTM


def train(x_train, y_train, x_test, y_test, num_classes, epochs):

    # reshape x & y
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)

    print(' ')
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    print('y_train shape:', y_train.shape)
    print('y_test shape:', y_test.shape)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    # define the model
    cnn = Sequential()
    cnn.add(Conv1D(32, kernel_size=7, strides=3, activation='relu', input_shape=(x_train.shape[2], x_train.shape[3])))
    cnn.add(MaxPooling1D(pool_size=2))
    cnn.add(Dropout(0.25))
    cnn.add(Conv1D(64, kernel_size=7, strides=3, activation='relu'))
    cnn.add(MaxPooling1D(pool_size=2))
    cnn.add(Dropout(0.25))
    cnn.add(Conv1D(64, kernel_size=7, strides=3, activation='relu'))
    cnn.add(MaxPooling1D(pool_size=2))
    cnn.add(Dropout(0.25))
    cnn.add(Flatten())

    lstm = Sequential()
    lstm.add(TimeDistributed(cnn, input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3])))
    lstm.add(LSTM(64, dropout=0.25, recurrent_dropout=0.25))
    # lstm.add(Dense(64, activation='relu'))
    lstm.add(Dropout(0.25))
    lstm.add(Dense(num_classes, activation='softmax'))

    # print the model
    keras.utils.print_summary(cnn, print_fn=print)
    keras.utils.print_summary(lstm, print_fn=print)
    # keras.utils.plot_model(cnn, to_file='model.png')

    # compile, fit, evaluate
    lstm.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adamax(), metrics=['accuracy'])
    lstm.fit(x_train, y_train, batch_size=64, epochs=epochs, verbose=2, validation_data=(x_test, y_test))
    score = lstm.evaluate(x_test, y_test, verbose=2)

    # save model
    model_json = lstm.to_json()
    with open("Models\\model_cnn1dlstm.json", "w") as json_file:
        json_file.write(model_json)
    lstm.save_weights("Models\\model_cnn1dlstm.h5")

    # return results
    print(' ')
    print('Test loss: ', score[0], 'and accuracy: ', score[1])

    return score[1]