import os, glob
import math
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.image as img
from PIL import Image as img
from scipy.io import wavfile
from sklearn import preprocessing
from keras import backend


def load_features_csv(path, category):

    # timing configuration
    sample_rate = 22050
    fft_size = 256*128
    if category is 0:
        #
        print('Total Window Length: {0:.3f}s'.format(fft_size/sample_rate))

    # load data
    raw = np.genfromtxt(path, delimiter=',', skip_header=1, dtype=float)

    # clean data
    raw = np.where(np.isnan(raw), backend.epsilon(), raw)
    raw = np.where(np.isinf(raw), backend.epsilon(), raw)

    # fill output
    data = raw
    truth = np.full(int(raw.shape[0]), category)

    return data, truth

def load_features_csv_ts(path, category):

    ts_length = 64
    ts_step = 64

    print('Total Window Length: {0:.3f}s'.format(0.0116*ts_step))

    raw = np.genfromtxt(path, delimiter=',', skip_header=1, dtype=float)
    raw[:, 4] = np.log(raw[:, 4])
    raw[:, 3] = np.log(raw[:, 3])
    raw[:, 2] = np.log(raw[:, 2])
    raw = np.delete(raw, [0, 1, 2, 8, 10], axis=1)
    data = preprocessing.normalize(raw[0::2, :], norm='l2')

    ts_data = np.zeros((int(data.shape[0]/ts_step), ts_length, data.shape[1]))
    ts_truth = np.full(int(data.shape[0]/ts_step), category)
    for i in range(0, data.shape[0]-ts_length, ts_step):
        ts_data[int(i/ts_step), :, :] = data[i:i+ts_length, :]

    return ts_data, ts_truth


def load_audio(path, category):

    # configuration
    dBFSThreshold = -54
    length = 8192
    step = int(length * 1.00)
    normalize = 1 # 0 None, 1 Patch
    log = False

    # load data
    rate, raw = wavfile.read(path)
    for i in range(0, raw.shape[0]-max(length,step), step):
        if i == 0: count = 0
        column = raw[i:i + length].astype('float32')/32768
        dbFS = 10*np.log10(np.square(column).mean()) if np.square(column).mean() > 0 else -math.inf
        if dbFS > dBFSThreshold:
            count = count + 1
    data = np.zeros((count, length))
    truth = np.full(count, category)
    for i in range(0, raw.shape[0]-max(length,step), step):
        if i == 0: count = 0
        column = raw[i:i + length].astype('float32')/32768
        rms = math.sqrt(np.square(column).mean())
        dbFS = 10 * np.log10(np.square(column).mean()) if np.square(column).mean() > 0 else -math.inf
        if dbFS > dBFSThreshold:

            if normalize == 0:
                if log:
                    sign = np.sign(column)
                    column = np.log10(np.abs(column) + 1)
                    column = np.multiply(column, sign)

            if normalize == 1:
                column = column[:] / rms #np.amax(np.abs(column))
                if log:
                    sign = np.sign(column)
                    column = np.log10(np.abs(column) + 1)
                    column = np.multiply(column, sign)

            data[count, :] = column
            count = count + 1
            # plt.plot(column[500:600])
            # plt.show()

    print('Class: {}, Full Size: {}, Filtered Size: {}, Window Length: {:.2f}s'.format(category, int(raw.shape[0]/step), count, length/rate))

    return data, truth

def load_audio_ts(path, category):

    length = 2048
    step = 2048
    ts_length = 10
    ts_step = 10

    # texture formation
    rate, raw = wavfile.read(path)
    data = np.zeros((int(raw.shape[0]/step), length))
    for i in range(0, raw.shape[0]-length, step):
        column = raw[i:i + length].astype('float32')/32768
        if np.amax(column) > 0:
            column = column[:] / np.amax(column)
        data[int(i/step), :] = column

    # embedding formation
    ts_data = np.zeros((int(raw.shape[0] / step / ts_step), ts_length, length))
    ts_truth = np.full(int(raw.shape[0] / step / ts_step), category)
    for i in range(0, int(raw.shape[0]/step)-ts_length, ts_step):
        ts_data[int(i/ts_step), :, :] = data[i:i+ts_length, :]

    print('Total Window Length: {0:.1f}s'.format(ts_length * step / rate))

    return ts_data, ts_truth


def load_spectrum_csv(path, category):

    # configuration
    sample_rate = 22050
    fft_step = 256
    length = 112 #112
    step = int(length*1.0)  # 0.25 1.25
    normalize = 0 # 0 None, 1 Patch, 2 Column

    # print info
    if category is 0:
        #
        print('Texture Length: {0:.2f}s'.format(length*fft_step/sample_rate))
    print("Loading class {}".format(category))

    # texture formation
    raw = np.genfromtxt(path, delimiter=',')
    data = np.zeros((int(raw.shape[0]/step), length, raw.shape[1]))
    truth = np.full(int(raw.shape[0]/step), category)
    for i in range(0, raw.shape[0]-max(length,step), step):

        if i == 0:
            #
            raw = np.log10(raw[:, :])

        if normalize == 0:
            #
            column = raw[i:i + length, :]

        if normalize == 1:
            column = raw[i:i + length, :]
            column = (column[:, :] - np.mean(column)) #/ np.std(column)
            # column = (column[:, :] - np.mean(column)) / (np.amax(column) - np.amin(column) + backend.epsilon())
            # column = np.clip(column[:, :], 0.0, 100) + np.random.random(column.shape)/0.001

        if normalize == 2:
            if i == 0: raw = preprocessing.StandardScaler().fit_transform(raw)
            column = raw[i:i + length, :]

        data[int(i / step), :, :] = column

    return data, truth

def load_spectrum_csv_ts(path, category):

    # timing configuration
    sample_rate = 44100
    fft_step = 512
    em_length = 16
    em_step = em_length * 0.50
    tx_length = 8
    tx_step = tx_length * 1.00
    logscale = True
    normalize = True

    if category is 0:
        print('Texture Length: {0:.2f}s'.format(tx_length*fft_step/sample_rate))
        print('Sequence Length: {0:.2f}s'.format(em_length*tx_step*fft_step/sample_rate))

    # texture formation
    raw = np.genfromtxt(path, delimiter=',')
    textures = np.zeros((int(raw.shape[0]/tx_step), tx_length, raw.shape[1]))
    for i in range(0, raw.shape[0]-tx_length, tx_step):

        if logscale:
            texture = np.log10(raw[i:i + tx_length, :])
        else:
            texture = raw[i:i + tx_length, :]

        if normalize:
            texture = (texture[:, :] - np.mean(texture)) / (np.std(texture))

        textures[int(i/tx_step), :, :] = texture

    # embedding formation
    data = np.zeros((int(raw.shape[0]/tx_step/em_step), em_length, tx_length, raw.shape[1]))
    truth = np.full(int(raw.shape[0]/tx_step/em_step), category)
    for i in range(0, int(raw.shape[0]/tx_step)-em_length, em_length):
        #
        data[int(i/em_step), :, :, :] = textures[i:i+em_length, :, :]

    return data, truth


def load_img_ts(path, category):

    # timing configuration
    em_length = 10
    em_step = 10

    files = os.listdir(path)
    files = sorted(files)

    # texture formation
    raw = np.zeros((len(files), 30, 50))
    for i in range(0, len(files), 1):
        image = img.open(path+'\\'+files[i])
        image = image.resize((50, 30), img.ANTIALIAS)
        # plt.imshow(image)
        # plt.show()
        # print(path+'\\'+files[i])
        raw[i, :, :] = np.asarray(image)
        raw[i, :, :] = raw[i, :, :]/255

    # embedding formation
    data = np.zeros((int(raw.shape[0]/em_step), em_length, raw.shape[1], raw.shape[2]))
    truth = np.full(int(raw.shape[0]/em_step), category)
    for i in range(0, raw.shape[0]-em_length-1, em_step):
        data[int(i/em_step), :, :, :] = raw[i:i+em_length, :, :]
        # print("{} -> {}".format(int(i/em_step), np.mean(data[int(i/em_step), :, :, :])))
        # print(truth[i])

    print('Category {} with {} items'.format(category, data.shape[0]))

    return data, truth