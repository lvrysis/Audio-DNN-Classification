def get_available_gpus():
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"] = "2"
    import tensorflow as tf
    from tensorflow.python.client import device_lib
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.75
    set_session(tf.Session(config=config))
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

get_available_gpus()


import time
import math
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_classif, f_classif
from sklearn.model_selection import train_test_split
import DeepANN as ann, DeepANNLSTM as annlstm
import DeepCNN1D as cnn1d, DeepCNN2D as cnn2d
import DeepCNN1DLSTM as cnn1dlstm, DeepCNN2DLSTM as cnn2dlstm
import HandleInput as hi
import Visualize1D as vis1d
import Visualize2D as vis2d


# Utility
def get_array_part(array, start, stop):
    length = array.shape[0]
    return array[int(start*length):int(stop*length)]


# ESR
def ann_esr(mode, epochs, fold, folds):

    path = 'E:\\Desktop\\PhD\\Experiments\\2018 MTAP (Temporal Envirnoment)\\UrbanSound8k\ETi\\'

    s1 = round((folds - fold) * 1.0 / folds, 1)
    s2 = round(s1 + 1.0 / folds, 1)
    print(' ')
    print('Fold {:d} with split points {:1.2f} and {:1.2f}'.format(fold, s1, s2))

    # load data
    x_0, y_0 = hi.load_features_csv(path + '_air_conditioner.csv', 0)
    x_1, y_1 = hi.load_features_csv(path + '_car_horn.csv', 1)
    x_2, y_2 = hi.load_features_csv(path + '_children_playing.csv', 2)
    x_3, y_3 = hi.load_features_csv(path + '_dog_bark.csv', 3)
    x_4, y_4 = hi.load_features_csv(path + '_drilling.csv', 4)
    x_5, y_5 = hi.load_features_csv(path + '_engine_idling.csv', 5)
    x_6, y_6 = hi.load_features_csv(path + '_gun_shot.csv', 6)
    x_7, y_7 = hi.load_features_csv(path + '_jackhammer.csv', 7)
    x_8, y_8 = hi.load_features_csv(path + '_siren.csv', 8)
    x_9, y_9 = hi.load_features_csv(path + '_street_music.csv', 9)

    x_train = np.concatenate((get_array_part(x_0, 0.0, s2),
                              get_array_part(x_1, 0.0, s1),
                              get_array_part(x_2, 0.0, s1),
                              get_array_part(x_3, 0.0, s1),
                              get_array_part(x_4, 0.0, s1),
                              get_array_part(x_5, 0.0, s1),
                              get_array_part(x_6, 0.0, s1),
                              get_array_part(x_7, 0.0, s1),
                              get_array_part(x_8, 0.0, s1),
                              get_array_part(x_9, 0.0, s1),
                              get_array_part(x_0, s2, 1.0),
                              get_array_part(x_1, s2, 1.0),
                              get_array_part(x_2, s2, 1.0),
                              get_array_part(x_3, s2, 1.0),
                              get_array_part(x_4, s2, 1.0),
                              get_array_part(x_5, s2, 1.0),
                              get_array_part(x_6, s2, 1.0),
                              get_array_part(x_7, s2, 1.0),
                              get_array_part(x_8, s2, 1.0),
                              get_array_part(x_9, s2, 1.0)), axis=0)
    y_train = np.concatenate((get_array_part(y_0, 0.0, s2),
                              get_array_part(y_1, 0.0, s1),
                              get_array_part(y_2, 0.0, s1),
                              get_array_part(y_3, 0.0, s1),
                              get_array_part(y_4, 0.0, s1),
                              get_array_part(y_5, 0.0, s1),
                              get_array_part(y_6, 0.0, s1),
                              get_array_part(y_7, 0.0, s1),
                              get_array_part(y_8, 0.0, s1),
                              get_array_part(y_9, 0.0, s1),
                              get_array_part(y_0, s2, 1.0),
                              get_array_part(y_1, s2, 1.0),
                              get_array_part(y_2, s2, 1.0),
                              get_array_part(y_3, s2, 1.0),
                              get_array_part(y_4, s2, 1.0),
                              get_array_part(y_5, s2, 1.0),
                              get_array_part(y_6, s2, 1.0),
                              get_array_part(y_7, s2, 1.0),
                              get_array_part(y_8, s2, 1.0),
                              get_array_part(y_9, s2, 1.0)), axis=0)
    x_test = np.concatenate((get_array_part(x_0, s1, s2),
                              get_array_part(x_1, s1, s2),
                              get_array_part(x_2, s1, s2),
                              get_array_part(x_3, s1, s2),
                              get_array_part(x_4, s1, s2),
                              get_array_part(x_5, s1, s2),
                              get_array_part(x_6, s1, s2),
                              get_array_part(x_7, s1, s2),
                              get_array_part(x_8, s1, s2),
                              get_array_part(x_9, s1, s2)), axis=0)
    y_test = np.concatenate((get_array_part(y_0, s1, s2),
                              get_array_part(y_1, s1, s2),
                              get_array_part(y_2, s1, s2),
                              get_array_part(y_3, s1, s2),
                              get_array_part(y_4, s1, s2),
                              get_array_part(y_5, s1, s2),
                              get_array_part(y_6, s1, s2),
                              get_array_part(y_7, s1, s2),
                              get_array_part(y_8, s1, s2),
                              get_array_part(y_9, s1, s2)), axis=0)

    # remove unwanted features
    x_train = np.delete(x_train, [0,1,2,3,4,5,6,7], axis=1)
    x_test = np.delete(x_test, [0,1,2,3,4,5,6,7], axis=1)
    for i in range(x_train.shape[1], 0, -1):
        if (i%8 == 4 or i%8 == 5 or i%8 == 6 or i%8 == 7) and mode == 'sti':
            x_train = np.delete(x_train, i, axis=1)
            x_test = np.delete(x_test, i, axis=1)

    # standardize
    scale = preprocessing.StandardScaler().fit(x_train)
    x_train = scale.transform(x_train)
    x_test = scale.transform(x_test)

    #  feature selection
    print('Number of features: {}'.format(x_train.shape[1]))
    clf = DecisionTreeClassifier()
    # clf = ExtraTreesClassifier(n_estimators=50)
    # clf = LogisticRegression(multi_class='auto', solver='lbfgs')
    # clf = LinearSVC()
    trans = SelectFromModel(clf, threshold=-np.inf, max_features=75)#int(x_train.shape[1]*0.75))
    x_train = trans.fit_transform(x_train, y_train)
    x_test = trans.transform(x_test)

    '''selector = SelectKBest(mutual_info_classif, k=int(x_train.shape[1]*0.75))
    selector.fit(x_train, y_train)
    x_train = selector.transform(x_train)
    x_test = selector.transform(x_test)'''
    print('Number of selected features: {}'.format(x_train.shape[1]))

    score = ann.train(x_train, y_train, x_test, y_test, 10, epochs)
    print('Fold {} with accuracy: {}'.format(fold, round(100*score,1)))

def cnn_1d_esr(epochs, fold, folds):

    path = 'E:\\Desktop\\PhD\\\Datasets\\UrbanSound8K\\audio\\'

    # folding information
    s1 = round((folds - fold) * 1.0 / folds, 2)
    s2 = round(s1 + 1.0 / folds, 2)
    print(' ')
    print('Fold {:d} with split points {:1.2f} and {:1.2f}'.format(fold, s1, s2))

    # load and form data
    x_0, y_0 = hi.load_audio(path + '_air_conditioner.wav', 0)
    x_1, y_1 = hi.load_audio(path + '_car_horn.wav', 1)
    x_2, y_2 = hi.load_audio(path + '_children_playing.wav', 2)
    x_3, y_3 = hi.load_audio(path + '_dog_bark.wav', 3)
    x_4, y_4 = hi.load_audio(path + '_drilling.wav', 4)
    x_5, y_5 = hi.load_audio(path + '_engine_idling.wav', 5)
    x_6, y_6 = hi.load_audio(path + '_gun_shot.wav', 6)
    x_7, y_7 = hi.load_audio(path + '_jackhammer.wav', 7)
    x_8, y_8 = hi.load_audio(path + '_siren.wav', 8)
    x_9, y_9 = hi.load_audio(path + '_street_music.wav', 9)

    x_train = np.concatenate((get_array_part(x_0, 0.0, s2),
                              get_array_part(x_1, 0.0, s1),
                              get_array_part(x_2, 0.0, s1),
                              get_array_part(x_3, 0.0, s1),
                              get_array_part(x_4, 0.0, s1),
                              get_array_part(x_5, 0.0, s1),
                              get_array_part(x_6, 0.0, s1),
                              get_array_part(x_7, 0.0, s1),
                              get_array_part(x_8, 0.0, s1),
                              get_array_part(x_9, 0.0, s1),
                              get_array_part(x_0, s2, 1.0),
                              get_array_part(x_1, s2, 1.0),
                              get_array_part(x_2, s2, 1.0),
                              get_array_part(x_3, s2, 1.0),
                              get_array_part(x_4, s2, 1.0),
                              get_array_part(x_5, s2, 1.0),
                              get_array_part(x_6, s2, 1.0),
                              get_array_part(x_7, s2, 1.0),
                              get_array_part(x_8, s2, 1.0),
                              get_array_part(x_9, s2, 1.0)), axis=0)
    y_train = np.concatenate((get_array_part(y_0, 0.0, s2),
                              get_array_part(y_1, 0.0, s1),
                              get_array_part(y_2, 0.0, s1),
                              get_array_part(y_3, 0.0, s1),
                              get_array_part(y_4, 0.0, s1),
                              get_array_part(y_5, 0.0, s1),
                              get_array_part(y_6, 0.0, s1),
                              get_array_part(y_7, 0.0, s1),
                              get_array_part(y_8, 0.0, s1),
                              get_array_part(y_9, 0.0, s1),
                              get_array_part(y_0, s2, 1.0),
                              get_array_part(y_1, s2, 1.0),
                              get_array_part(y_2, s2, 1.0),
                              get_array_part(y_3, s2, 1.0),
                              get_array_part(y_4, s2, 1.0),
                              get_array_part(y_5, s2, 1.0),
                              get_array_part(y_6, s2, 1.0),
                              get_array_part(y_7, s2, 1.0),
                              get_array_part(y_8, s2, 1.0),
                              get_array_part(y_9, s2, 1.0)), axis=0)
    x_test = np.concatenate((get_array_part(x_0, s1, s2),
                              get_array_part(x_1, s1, s2),
                              get_array_part(x_2, s1, s2),
                              get_array_part(x_3, s1, s2),
                              get_array_part(x_4, s1, s2),
                              get_array_part(x_5, s1, s2),
                              get_array_part(x_6, s1, s2),
                              get_array_part(x_7, s1, s2),
                              get_array_part(x_8, s1, s2),
                              get_array_part(x_9, s1, s2)), axis=0)
    y_test = np.concatenate((get_array_part(y_0, s1, s2),
                              get_array_part(y_1, s1, s2),
                              get_array_part(y_2, s1, s2),
                              get_array_part(y_3, s1, s2),
                              get_array_part(y_4, s1, s2),
                              get_array_part(y_5, s1, s2),
                              get_array_part(y_6, s1, s2),
                              get_array_part(y_7, s1, s2),
                              get_array_part(y_8, s1, s2),
                              get_array_part(y_9, s1, s2)), axis=0)

    score = cnn1d.train(x_train, y_train, x_test, y_test, 10, epochs)
    print('Fold {} with accuracy: {:.1f}'.format(fold, 100*score))

def cnn_2d_esr(epochs, fold, folds):

    path = 'E:\\Desktop\\PhD\\Experiments\\2018 MTAP (Temporal Envirnoment)\\UrbanSound8k\\2D\\'

    # folding information
    s1 = round((folds - fold) * 1.0 / folds, 2)
    s2 = round(s1 + 1.0 / folds, 2)
    print(' ')
    print('Fold {:d} with split points {:1.2f} and {:1.2f}'.format(fold, s1, s2))

    # load and form data
    x_0, y_0 = hi.load_spectrum_csv(path + '_air_conditioner.csv', 0)
    x_1, y_1 = hi.load_spectrum_csv(path + '_car_horn.csv', 1)
    x_2, y_2 = hi.load_spectrum_csv(path + '_children_playing.csv', 2)
    x_3, y_3 = hi.load_spectrum_csv(path + '_dog_bark.csv', 3)
    x_4, y_4 = hi.load_spectrum_csv(path + '_drilling.csv', 4)
    x_5, y_5 = hi.load_spectrum_csv(path + '_engine_idling.csv', 5)
    x_6, y_6 = hi.load_spectrum_csv(path + '_gun_shot.csv', 6)
    x_7, y_7 = hi.load_spectrum_csv(path + '_jackhammer.csv', 7)
    x_8, y_8 = hi.load_spectrum_csv(path + '_siren.csv', 8)
    x_9, y_9 = hi.load_spectrum_csv(path + '_street_music.csv', 9)

    x_train = np.concatenate((get_array_part(x_0, 0.0, s2),
                              get_array_part(x_1, 0.0, s1),
                              get_array_part(x_2, 0.0, s1),
                              get_array_part(x_3, 0.0, s1),
                              get_array_part(x_4, 0.0, s1),
                              get_array_part(x_5, 0.0, s1),
                              get_array_part(x_6, 0.0, s1),
                              get_array_part(x_7, 0.0, s1),
                              get_array_part(x_8, 0.0, s1),
                              get_array_part(x_9, 0.0, s1),
                              get_array_part(x_0, s2, 1.0),
                              get_array_part(x_1, s2, 1.0),
                              get_array_part(x_2, s2, 1.0),
                              get_array_part(x_3, s2, 1.0),
                              get_array_part(x_4, s2, 1.0),
                              get_array_part(x_5, s2, 1.0),
                              get_array_part(x_6, s2, 1.0),
                              get_array_part(x_7, s2, 1.0),
                              get_array_part(x_8, s2, 1.0),
                              get_array_part(x_9, s2, 1.0)), axis=0)
    y_train = np.concatenate((get_array_part(y_0, 0.0, s2),
                              get_array_part(y_1, 0.0, s1),
                              get_array_part(y_2, 0.0, s1),
                              get_array_part(y_3, 0.0, s1),
                              get_array_part(y_4, 0.0, s1),
                              get_array_part(y_5, 0.0, s1),
                              get_array_part(y_6, 0.0, s1),
                              get_array_part(y_7, 0.0, s1),
                              get_array_part(y_8, 0.0, s1),
                              get_array_part(y_9, 0.0, s1),
                              get_array_part(y_0, s2, 1.0),
                              get_array_part(y_1, s2, 1.0),
                              get_array_part(y_2, s2, 1.0),
                              get_array_part(y_3, s2, 1.0),
                              get_array_part(y_4, s2, 1.0),
                              get_array_part(y_5, s2, 1.0),
                              get_array_part(y_6, s2, 1.0),
                              get_array_part(y_7, s2, 1.0),
                              get_array_part(y_8, s2, 1.0),
                              get_array_part(y_9, s2, 1.0)), axis=0)
    x_test = np.concatenate((get_array_part(x_0, s1, s2),
                              get_array_part(x_1, s1, s2),
                              get_array_part(x_2, s1, s2),
                              get_array_part(x_3, s1, s2),
                              get_array_part(x_4, s1, s2),
                              get_array_part(x_5, s1, s2),
                              get_array_part(x_6, s1, s2),
                              get_array_part(x_7, s1, s2),
                              get_array_part(x_8, s1, s2),
                              get_array_part(x_9, s1, s2)), axis=0)
    y_test = np.concatenate((get_array_part(y_0, s1, s2),
                              get_array_part(y_1, s1, s2),
                              get_array_part(y_2, s1, s2),
                              get_array_part(y_3, s1, s2),
                              get_array_part(y_4, s1, s2),
                              get_array_part(y_5, s1, s2),
                              get_array_part(y_6, s1, s2),
                              get_array_part(y_7, s1, s2),
                              get_array_part(y_8, s1, s2),
                              get_array_part(y_9, s1, s2)), axis=0)

    score = cnn2d.train(x_train, y_train, x_test, y_test, 10, epochs)
    print('Fold {} with accuracy: {:.1f}'.format(fold, 100*score))

for i in range(11, 11):
    # ann_esr('eti', 500, i, 10)
    # cnn_1d_esr(200, i, 10)
    cnn_2d_esr(200, i, 10)


# SMO
def ann_smo(lstm, epochs, folded):

    # load data
    if(lstm):
        x_music, y_music = hi.load_features_csv_ts('E:\\Desktop\\PhD\\Experiments\\2018 JAES (Stable Integration)\\Short-M.csv', 0)
        x_speech, y_speech = hi.load_features_csv_ts('E:\\Desktop\\PhD\\Experiments\\2018 JAES (Stable Integration)\\Short-S.csv', 1)
        x_other, y_other = hi.load_features_csv_ts('E:\\Desktop\\PhD\\Experiments\\2018 JAES (Stable Integration)\\Short-O.csv', 2)
    else:
        x_music, y_music = hi.load_features_csv('E:\\Desktop\\PhD\\Experiments\\2018 JAES (Stable Integration)\\Short-M.csv', 0)
        x_speech, y_speech = hi.load_features_csv('E:\\Desktop\\PhD\\Experiments\\2018 JAES (Stable Integration)\\Short-S.csv', 1)
        x_other, y_other = hi.load_features_csv('E:\\Desktop\\PhD\\Experiments\\2018 JAES (Stable Integration)\\Short-O.csv', 2)

    # fold 1
    x_train = np.concatenate((get_array_part(x_music, 0.33, 1.00),
                              get_array_part(x_speech, 0.33, 1.00),
                              get_array_part(x_other, 0.33, 1.00)), axis=0)
    y_train = np.concatenate((get_array_part(y_music, 0.33, 1.00),
                              get_array_part(y_speech, 0.33, 1.00),
                              get_array_part(y_other, 0.33, 1.00)), axis=0)
    x_test = np.concatenate((get_array_part(x_music, 0.00, 0.33),
                             get_array_part(x_speech, 0.00, 0.33),
                             get_array_part(x_other, 0.00, 0.33)), axis=0)
    y_test = np.concatenate((get_array_part(y_music, 0.00, 0.33),
                             get_array_part(y_speech, 0.00, 0.33),
                             get_array_part(y_other, 0.00, 0.33)), axis=0)

    if(lstm):
        score1 = annlstm.train(x_train, y_train, x_test, y_test, 3, epochs)
    else:
        score1 = ann.train(x_train, y_train, x_test, y_test, 3, epochs)

    if not folded:
        return

    # fold 2
    x_train = np.concatenate((get_array_part(x_music, 0.00, 0.66),
                              get_array_part(x_speech, 0.00, 0.66),
                              get_array_part(x_other, 0.00, 0.66)), axis=0)
    y_train = np.concatenate((get_array_part(y_music, 0.00, 0.66),
                              get_array_part(y_speech, 0.00, 0.66), get_array_part(y_other, 0.00, 0.66)), axis=0)
    x_test = np.concatenate((get_array_part(x_music, 0.66, 1.00),
                             get_array_part(x_speech, 0.66, 1.00),
                             get_array_part(x_other, 0.66, 1.00)), axis=0)
    y_test = np.concatenate((get_array_part(y_music, 0.66, 1.00),
                             get_array_part(y_speech, 0.66, 1.00),
                             get_array_part(y_other, 0.66, 1.00)), axis=0)

    if(lstm):
        score2 = annlstm.train(x_train, y_train, x_test, y_test, 3, epochs)
    else:
        score2 = cnn1d.train(x_train, y_train, x_test, y_test, 3, epochs)


    # fold 3
    x_train = np.concatenate((get_array_part(x_music, 0.00, 0.33),
                              get_array_part(x_speech, 0.00, 0.33),
                              get_array_part(x_other, 0.00, 0.33),
                              get_array_part(x_music, 0.66, 1.00),
                              get_array_part(x_speech, 0.66, 1.00),
                              get_array_part(x_other, 0.66, 1.00)),
                             axis=0)
    y_train = np.concatenate((get_array_part(y_music, 0.00, 0.33),
                              get_array_part(y_speech, 0.00, 0.33),
                              get_array_part(y_other, 0.00, 0.33),
                              get_array_part(y_music, 0.66, 1.00),
                              get_array_part(y_speech, 0.66, 1.00),
                              get_array_part(y_other, 0.66, 1.00)),
                             axis=0)
    x_test = np.concatenate((get_array_part(x_music, 0.33, 0.66),
                             get_array_part(x_speech, 0.33, 0.66),
                             get_array_part(x_other, 0.33, 0.66)),
                            axis=0)
    y_test = np.concatenate((get_array_part(y_music, 0.33, 0.66),
                             get_array_part(y_speech, 0.33, 0.66),
                             get_array_part(y_other, 0.33, 0.66)),
                            axis=0)

    if(lstm):
        score3 = annlstm.train(x_train, y_train, x_test, y_test, 3, epochs)
    else:
        score3 = cnn1d.train(x_train, y_train, x_test, y_test, 3, epochs)

    # print scores
    print('{:1.2f} {:1.2f} {:1.2f} {:1.2f}'.format(score1, score2, score3, score1/3+score2/3+score3/3))

def cnn_1d_smo(lstm, epochs, fold, folds):

    path = 'E:\\Desktop\\PhD\\Experiments\\2018 MTAP (Audio Deep Learning)\\LVLib-SMO-v1\\'

    # folding information
    s1 = round((folds - fold) * 1.0 / folds, 2)
    s2 = round(s1 + 1.0 / folds, 2)
    print(' ')
    print('Fold {:d} with split points {:1.2f} and {:1.2f}'.format(fold, s1, s2))

    # load data
    if(lstm):
        x_music, y_music = hi.load_audio_ts(path + 'Music.wav', 0)
        x_speech, y_speech = hi.load_audio_ts(path + 'Speech.wav', 1)
        x_other, y_other = hi.load_audio_ts(path + 'Others.wav', 2)
    else:
        x_music, y_music = hi.load_audio(path + 'Music.wav', 0)
        x_speech, y_speech = hi.load_audio(path + 'Speech.wav', 1)
        x_other, y_other = hi.load_audio(path + 'Others.wav', 2)

    # make folds
    x_train = np.concatenate((get_array_part(x_music, 0, s1),
                              get_array_part(x_speech, 0, s1),
                              get_array_part(x_other, 0, s1),
                              get_array_part(x_music, s2, 1),
                              get_array_part(x_speech, s2, 1),
                              get_array_part(x_other, s2, 1)), axis=0)
    y_train = np.concatenate((get_array_part(y_music, 0, s1),
                              get_array_part(y_speech, 0, s1),
                              get_array_part(y_other, 0, s1),
                              get_array_part(y_music, s2, 1),
                              get_array_part(y_speech, s2, 1),
                              get_array_part(y_other, s2, 1)), axis=0)
    x_test = np.concatenate((get_array_part(x_music, s1, s2),
                             get_array_part(x_speech, s1, s2),
                             get_array_part(x_other, s1, s2)), axis=0)
    y_test = np.concatenate((get_array_part(y_music, s1, s2),
                             get_array_part(y_speech, s1, s2),
                             get_array_part(y_other, s1, s2)), axis=0)

    # global normalization
    rms = math.sqrt(np.square(x_train).mean())
    # x_train = x_train[:, :] / rms
    # x_test = x_test[:, :] / rms

    # train
    if(lstm):
        #
        score = cnn1dlstm.train(x_train, y_train, x_test, y_test, 3, epochs)
    else:
        #
        score = cnn1d.train(x_train, y_train, x_test, y_test, 3, epochs)

    # print scores
    print('Fold {} with accuracy: {:.1f}'.format(fold, 100 * score))

def cnn_2d_smo(lstm, epochs, fold, folds):

    path = 'E:\\Desktop\\PhD\\Experiments\\2018 MTAP (Audio Deep Learning)\\LVLib-SMO-v3\\'

    # folding information
    s1 = round((folds - fold) * 1.0 / folds, 2)
    s2 = round(s1 + 1.0 / folds, 2)
    print(' ')
    print('Fold {:d} with split points {:1.2f} and {:1.2f}'.format(fold, s1, s2))

    # load data
    if(lstm):
        x_music, y_music = hi.load_spectrum_csv_ts(path + 'Music.csv', 0)
        x_speech, y_speech = hi.load_spectrum_csv_ts(path + 'Speech.csv', 1)
        x_other, y_other = hi.load_spectrum_csv_ts(path + 'Others.csv', 2)
    else:
        x_music, y_music = hi.load_spectrum_csv(path + 'Music.csv', 0)
        x_speech, y_speech = hi.load_spectrum_csv(path + 'Speech.csv', 1)
        x_other, y_other = hi.load_spectrum_csv(path + 'Others.csv', 2)

    #  make folds
    x_train = np.concatenate((get_array_part(x_music, 0, s1),
                              get_array_part(x_speech, 0, s1),
                              get_array_part(x_other, 0, s1),
                              get_array_part(x_music, s2, 1),
                              get_array_part(x_speech, s2, 1),
                              get_array_part(x_other, s2, 1)), axis=0)
    y_train = np.concatenate((get_array_part(y_music, 0, s1),
                              get_array_part(y_speech, 0, s1),
                              get_array_part(y_other, 0, s1),
                              get_array_part(y_music, s2, 1),
                              get_array_part(y_speech, s2, 1),
                              get_array_part(y_other, s2, 1)), axis=0)
    x_test = np.concatenate((get_array_part(x_music, s1, s2),
                             get_array_part(x_speech, s1, s2),
                             get_array_part(x_other, s1, s2)), axis=0)
    y_test = np.concatenate((get_array_part(y_music, s1, s2),
                             get_array_part(y_speech, s1, s2),
                             get_array_part(y_other, s1, s2)), axis=0)

    # global normalization
    x_train = (x_train[:, :, :] - np.mean(x_train)) #/ np.std(x_train)
    x_test = (x_test[:, :, :] - np.mean(x_train)) #/ np.std(x_train)

    # train
    if lstm:
        #
        score = cnn2dlstm.train(x_train, y_train, x_test, y_test, 3, epochs)
    else:
        #
        score = cnn2d.train(x_train, y_train, x_test, y_test, 3, epochs)

    print('Fold {} with accuracy: {:.1f}'.format(fold, 100*score))

for i in range(1, 4):
    # ann_smo(True, 100, True)
    cnn_1d_smo(False, 100, i, 3)
    # cnn_2d_smo(False, 200, i, 3)


# SER
def cnn_1d_emotion(epochs):

    x1, y1 = hi.load_audio_ts('E:\\Desktop\\PhD\\Experiments\\2018 MTAP (Speech Emotion)\\anger.wav', 0)
    x2, y2 = hi.load_audio_ts('E:\\Desktop\\PhD\\Experiments\\2018 MTAP (Speech Emotion)\\disgust.wav', 1)
    x3, y3 = hi.load_audio_ts('E:\\Desktop\\PhD\\Experiments\\2018 MTAP (Speech Emotion)\\fear.wav', 2)
    x4, y4 = hi.load_audio_ts('E:\\Desktop\\PhD\\Experiments\\2018 MTAP (Speech Emotion)\\happiness.wav', 3)
    x5, y5 = hi.load_audio_ts('E:\\Desktop\\PhD\\Experiments\\2018 MTAP (Speech Emotion)\\sadness.wav', 4)

    x = np.concatenate((x1, x2, x3, x4, x5), axis=0)
    y = np.concatenate((y1, y2, y3, y4, y5), axis=0)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

    score = cnn1dlstm.execute(x_train, y_train, x_test, y_test, 5, epochs)

def cnn_2d_emotion(lstm, epochs, fold, folds):

    path = 'E:\\Desktop\\PhD\\Experiments\\2018 MTAP (Speech Emotion)\\AUG-22050-512-256-56\\'

    s1 = round((folds - fold) * 1.0 / folds, 2)
    s2 = round(s1 + 1.0 / folds, 2)
    print(' ')
    print('Fold {}/{} with split at {:1.2f} and {:1.2f}'.format(fold, folds, s1, s2))

    if lstm:
        x1, y1 = hi.load_spectrum_csv_ts(path + 'anger.csv', 0)
        x2, y2 = hi.load_spectrum_csv_ts(path + 'disgust.csv', 1)
        x3, y3 = hi.load_spectrum_csv_ts(path + 'fear.csv', 2)
        x4, y4 = hi.load_spectrum_csv_ts(path + 'happiness.csv', 3)
        x5, y5 = hi.load_spectrum_csv_ts(path + 'sadness.csv', 4)
    else:
        x1, y1 = hi.load_spectrum_csv(path + 'anger.csv', 0)
        x2, y2 = hi.load_spectrum_csv(path + 'disgust.csv', 1)
        x3, y3 = hi.load_spectrum_csv(path + 'fear.csv', 2)
        x4, y4 = hi.load_spectrum_csv(path + 'happiness.csv', 3)
        x5, y5 = hi.load_spectrum_csv(path + 'sadness.csv', 4)

    # x_train, x_test, y_train, y_test = train_test_split(np.concatenate((x1, x2, x3, x4, x5), axis=0), np.concatenate((y1, y2, y3, y4, y5), axis=0), test_size=0.33)
    x_train = np.concatenate((get_array_part(x1, 0, s1),
                              get_array_part(x2, 0, s1),
                              get_array_part(x3, 0, s1),
                              get_array_part(x4, 0, s1),
                              get_array_part(x5, 0, s1),
                              get_array_part(x1, s2, 1),
                              get_array_part(x2, s2, 1),
                              get_array_part(x3, s2, 1),
                              get_array_part(x4, s2, 1),
                              get_array_part(x5, s2, 1),), axis=0)
    y_train = np.concatenate((get_array_part(y1, 0, s1),
                              get_array_part(y2, 0, s1),
                              get_array_part(y3, 0, s1),
                              get_array_part(y4, 0, s1),
                              get_array_part(y5, 0, s1),
                              get_array_part(y1, s2, 1),
                              get_array_part(y2, s2, 1),
                              get_array_part(y3, s2, 1),
                              get_array_part(y4, s2, 1),
                              get_array_part(y5, s2, 1),), axis=0)
    x_test = np.concatenate((get_array_part(x1, s1, s2),
                             get_array_part(x2, s1, s2),
                             get_array_part(x3, s1, s2),
                             get_array_part(x4, s1, s2),
                             get_array_part(x5, s1, s2)), axis=0)
    y_test = np.concatenate((get_array_part(y1, s1, s2),
                             get_array_part(y2, s1, s2),
                             get_array_part(y3, s1, s2),
                             get_array_part(y4, s1, s2),
                             get_array_part(y5, s1, s2)), axis=0)

    if lstm:
        #
        score = cnn2dlstm.train(x_train, y_train, x_test, y_test, 5, epochs)
    else:
        #
        score = cnn2d.train(x_train, y_train, x_test, y_test, 5, epochs)

    print('Fold {}/{}: {}'.format(fold, folds, round(score,2)))

for i in range(4, 4):
    # cnn_1d_emotion(epochs)
    cnn_2d_emotion(False, 200, i, 3)

# VISUAL VAD
def cnn_2d_mouth(epochs, fold, folds):

    path = 'E:\\Desktop\\PhD\\Datasets\\M3C Speakers Localization v3\\Mouths (Diff)\\'

    s1 = round((folds - fold) * 1.0 / folds, 2)
    s2 = round(s1 + 1.0 / folds, 2)
    print(' ')
    print('Fold {}/{} with split at {:1.2f} and {:1.2f}'.format(fold, folds, s1, s2))
    print(' ')
    x1, y1 = hi.load_img_ts(path + '0', 0)
    x2, y2 = hi.load_img_ts(path + '1', 1)

    indices = np.random.choice(x1.shape[0], x2.shape[0], replace=False)
    x1 = x1[indices, :, :, :]
    y1 = y1[indices]

    x_train = np.concatenate((get_array_part(x1, 0, s1),
                              get_array_part(x2, 0, s1),
                              get_array_part(x1, s2, 1),
                              get_array_part(x2, s2, 1),), axis=0)
    y_train = np.concatenate((get_array_part(y1, 0, s1),
                              get_array_part(y2, 0, s1),
                              get_array_part(y1, s2, 1),
                              get_array_part(y2, s2, 1),), axis=0)
    x_test = np.concatenate((get_array_part(x1, s1, s2),
                             get_array_part(x2, s1, s2),), axis=0)
    y_test = np.concatenate((get_array_part(y1, s1, s2),
                             get_array_part(y2, s1, s2),), axis=0)

    # x_train, x_test, y_train, y_test = train_test_split(np.concatenate((x1, x2), axis=0), np.concatenate((y1, y2), axis=0), test_size=0.33)

    # CNN LSTM evaluation
    score = cnn2dlstm.train(x_train, y_train, x_test, y_test, 2, epochs)
    print('Fold {}/{}: {:1.3f}'.format(fold, folds, round(score, 2)))

    # Simple method evaluation
    print(' ')
    x1m = np.mean(x1)
    x2m = np.mean(x2)
    # print('Full mean x1 is {:1.4f} and x2 is {:1.4f}'.format(x1m, x2m))
    correct = 0
    for i in range(0, x1.shape[0], 1):
        if np.mean(x1[i, :, :, :]) < (x1m + x2m) / 2:
            correct = correct + 1
    for i in range(0, x2.shape[0], 1):
        if np.mean(x2[i, :, :, :]) > (x1m + x2m) / 2:
            correct = correct + 1
    # print('Full accuracy is {:1.3f}'.format(correct / (x1.shape[0] + x2.shape[0])))
    x1m = 0
    x2m = 0
    c1 = 0
    c2 = 0
    for i in range(0, x_train.shape[0], 1):
        if y_train[i] == 0:
            x1m = x1m + np.mean(x_train[i, :, :, :])
            c1 = c1 + 1
        else:
            x2m = x2m + np.mean(x_train[i, :, :, :])
            c2 = c2 + 1
    x1m = x1m / c1
    x2m = x2m / c2
    # print('Test mean x1 is {:1.4f} and x2 is {:1.4f}'.format(x1m, x2m))
    correct = 0
    for i in range(0, x_test.shape[0], 1):
        if np.mean(x_test[i, :, :, :]) < (x1m + x2m) / 2 and y_test[i] == 0:
            correct = correct + 1
        elif np.mean(x_test[i, :, :, :]) > (x1m + x2m) / 2 and y_test[i] == 1:
            correct = correct + 1
    print('Test accuracy is {:1.3f}'.format(correct / (x_test.shape[0])))
    print(' ')

for i in range(4, 4):
    cnn_2d_mouth(50, i, 3)


# VISUALIZE
# vis1d.visualize('model_cnn1d', 'conv1d_3', 32)
# vis2d.visualize('model_cnn2d', 'conv2d_2', 16)