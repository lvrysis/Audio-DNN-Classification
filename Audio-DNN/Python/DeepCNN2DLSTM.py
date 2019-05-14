import keras
from keras import metrics, callbacks
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import TimeDistributed, LSTM, ConvLSTM2D

def train(x_train, y_train, x_test, y_test, num_classes, epochs):

    # Customize and print x & y shapes
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3], 1)
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)
    print(' ')
    print('Shapes: x_train {}, x_test {}, y_train {}, y_test {}'.format(x_train.shape, x_test.shape, y_train.shape, y_test.shape))

    # Convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    stage = 0
    if stage == 0:
        lstm = Sequential()
        lstm.add(ConvLSTM2D(8, kernel_size=(3, 3), strides=(1, 1), activation='elu', input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3], x_train.shape[4])))
        lstm.add(MaxPooling2D(pool_size=(2, 3)))
        lstm.add(Dropout(0.25))
        lstm.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='elu'))
        lstm.add(MaxPooling2D(pool_size=(2, 2)))
        lstm.add(Dropout(0.25))
        lstm.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='elu'))
        lstm.add(MaxPooling2D(pool_size=(2, 2)))
        lstm.add(Dropout(0.25))
        lstm.add(Flatten())
        lstm.add(Dense(32, activation='elu'))
        lstm.add(Dense(num_classes, activation='softmax'))
    elif stage == 1:
        cnn = Sequential()
        cnn.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=(x_train.shape[2], x_train.shape[3], x_train.shape[4])))
        cnn.add(MaxPooling2D(pool_size=(1, 1)))
        cnn.add(Dropout(0.25))
        lstm = Sequential()
        lstm.add(TimeDistributed(cnn, input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3], x_train.shape[4])))
        lstm.add(ConvLSTM2D(16, kernel_size=(3, 3), strides=(1, 1), dropout=0.25, recurrent_dropout=0.25))
        lstm.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
        lstm.add(MaxPooling2D(pool_size=(1, 2)))
        lstm.add(Dropout(0.25))
        lstm.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
        lstm.add(MaxPooling2D(pool_size=(2, 2)))
        lstm.add(Dropout(0.25))
        lstm.add(Flatten())
        lstm.add(Dense(num_classes, activation='softmax'))
    else:
        cnn = Sequential()
        cnn.add(Conv2D(8, kernel_size=(3, 3), strides=(1, 1), activation='elu', input_shape=(x_train.shape[2], x_train.shape[3], x_train.shape[4])))
        cnn.add(BatchNormalization())
        cnn.add(MaxPooling2D(pool_size=(2, 3)))
        cnn.add(Dropout(0.25))
        cnn.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation='elu'))
        cnn.add(BatchNormalization())
        cnn.add(MaxPooling2D(pool_size=(2, 2)))
        cnn.add(Dropout(0.25))
        cnn.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='elu'))
        cnn.add(BatchNormalization())
        cnn.add(MaxPooling2D(pool_size=(2, 2)))
        cnn.add(Dropout(0.25))
        cnn.add(Flatten())

        lstm = Sequential()
        lstm.add(TimeDistributed(cnn, input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3], x_train.shape[4])))
        lstm.add(LSTM(16, dropout=0.0, recurrent_dropout=0.0))
        lstm.add(Dense(num_classes, activation='softmax'))

    # summarize model
    try:        cnn.summary()
    except:     print(" ")
    finally:    lstm.summary()
    '''f = open('Models\\model_cnn2dlstm.txt', 'w')
    for i in range(0, len(cnn.layers)):
        if i == 0:
            # print(cnn.summary())
            print(' ')
        print('{}. Layer {} with input / output shapes: {} / {}'.format(i, cnn.layers[i].name, cnn.layers[i].input_shape, cnn.layers[i].output_shape))
        f.write('{}. Layer {} with input / output shapes: {} / {} \n'.format(i, cnn.layers[i].name, cnn.layers[i].input_shape, cnn.layers[i].output_shape))
        if i == len(cnn.layers) - 1:
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
    keras.utils.plot_model(cnn, to_file='Models\\model_cnn2dlstm-a.png')
    keras.utils.plot_model(lstm, to_file='Models\\model_cnn2dlstm-b.png')'''

    # compile, fit, evaluate
    lstm.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics = ['accuracy'])
    lstm.fit(x_train, y_train, batch_size=32, epochs=epochs, verbose=2, validation_data=(x_test, y_test), callbacks=[callbacks.EarlyStopping(monitor='val_acc', patience=epochs*0.2, restore_best_weights=True)])
    score = lstm.evaluate(x_test, y_test, verbose=2)

    # save model
    '''model_json = lstm.to_json()
    with open("Models\\model_cnn2dlstm.json", "w") as json_file:
        #
        json_file.write(model_json)
    lstm.save_weights("Models\\model_cnn2dlstm.h5")'''

    print(' ')

    return score[1]