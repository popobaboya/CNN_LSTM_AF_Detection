"""
af_blind_fold_evaluation.py
blindfold evaluating of model performance on the completely unseen test
data set
author:     alex shenfield
date:       27/04/2018
"""
import tensorflow as tf

# file handling functionality
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "1"
# useful utilities
import time
import pickle

# let's do datascience ...
import numpy as np

from keras.models import Sequential
from keras import layers

# from keras.optimizers import RMSprop


from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input

from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.models import Model
from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt

# fix random seed for reproduciblity
# 랜덤성을 제어하기 위함, 난수의 생성 패턴을 동일하게 관리
seed = 1337
np.random.seed(seed)




# # import keras deep learning functionality
# from keras.models import Model
# from keras.layers import Input
# from keras.layers import LSTM
# from keras.layers import Bidirectional
# from keras.layers import GlobalMaxPool1D
# from keras.layers import Dense

# scikit learn metrics for evaluation ...
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# my visualisation utilities for plotting some pretty graphs of classifier 
# performance
# import visualisation_utils as my_vis
# from object_detection.utils import ops as my_vis
# from utils import visualization_utils as my_vis



#
# get the data
#

# load the npz file
data_path = './data/test_data.npz'
af_data   = np.load(data_path)


# print("GPU를 사용한 학습")

# with tf.device("/gpu:1"):
# extract the inputs and labels
x_test  = af_data['x_data']
y_test  = af_data['y_data']

# reshape the data to be in the format that the lstm wants
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

#
# create the model

# set the model parameters
n_timesteps = x_test.shape[1]
mode = 'concat'
batch_size = 1024

# create a bidirectional lstm model (based around the model in:
# https://www.kaggle.com/jhoward/improved-lstm-baseline-glove-dropout
# )
# inp = Input(shape=(n_timesteps,1,))
# x = Bidirectional(LSTM(200, 
#                        return_sequences=True))(inp)
# x = GlobalMaxPool1D()(x)
# x = Dense(50, activation="relu")(x)
# x = Dense(1, activation='sigmoid')(x)
# model = Model(inputs=inp, outputs=x)

#
# load saved weights
#

inp = Input(shape=(n_timesteps,1,))

inp = (100,1,)

model = Sequential()
# model.add(Input(shape=(100, 1)))
model.add(layers.Conv1D(200, 3, activation='relu',
                        input_shape=inp))
# model.add(MaxPooling1D(10))
model.add(layers.Conv1D(100, 3, activation='relu'))
model.add(layers.LSTM(200, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))
model.add(GlobalMaxPooling1D())
model.add(Dense(50, input_dim=10, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))   
          



# set the weights to load
# weights_path = './model/CNN_LSTM/initial_runs_20180412_1345/af_lstm_weights.51-0.03.hdf5'
weights_path = './model/CNN_LSTM/initial_runs_20220429_1622/CNN_LSTM_af_lstm_weights.66-0.00.hdf5'
# weights_path = './model/initial_runs_20220208_2050/af_lstm_weights.17-0.05.hdf5'

# load those weights and compile the model (which is needed before we can make
# any predictions)
model.load_weights(weights_path)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# note: although we have specified an optimiser in our model compilation, it
# is obviously not used ...

#
# predict the labels using the model and evaluate over a range of metrics
#

# use the model to predict the labels
y_predict = model.predict(x_test, batch_size=batch_size, verbose=0)
np.save('./results/CNN_LSTM/blindfold_predictions.npy', y_predict)

#
# evaluate predictions
#

# accuracy
accuracy = accuracy_score(y_test, np.round(y_predict))
print('blind-fold accuracy is {0:.5f}'.format(accuracy))

# precision
precision = precision_score(y_test, np.round(y_predict))
print('blind-fold precision is {0:.5f}'.format(precision))

# recall
recall = recall_score(y_test, np.round(y_predict))
print('blind-fold recall is {0:.5f}'.format(recall))

# f1 score
f1 = f1_score(y_test, np.round(y_predict))
print('blind-fold f1 score is {0:.5f}'.format(f1))

# classification report
print(classification_report(y_test, np.round(y_predict), 
                            target_names=['normal', 'af']))

# set the names of the classes
classes = ['normal', 'af']

# get the confusion matrix and plot both the un-normalised and normalised
# confusion plots 
# cm = confusion_matrix(y_test, np.round(y_predict))

# plt.figure(figsize=[5,5])
# my_vis.plot_confusion_matrix(cm, 
#                       classes=classes,
#                       title=None)
# plt.savefig('./results/CNN_LSTM/blind_confusion_plot.png',
#             dpi=600, bbox_inches='tight', pad_inches=0.5)

# plt.figure(figsize=[5,5])
# my_vis.plot_confusion_matrix(cm, 
#                       classes=classes,
#                       normalize=True,
#                       title=None)
# plt.savefig('./results/CNN_LSTM/blind_confusion_plot_normalised.png',
#             dpi=600, bbox_inches='tight', pad_inches=0.5)
# plt.show()

# # calculate and plot the roc curve
# plt.figure(figsize=[5,5])
# title = 'Receiver operating characteristic curve showing ' \
#         'AF diagnostic performance'
# my_vis.plot_roc_curve(y_predict, y_test, title=None)        
# plt.savefig('./results//CNN_LSTM/blind_roc_curve.png',
#             dpi=600, bbox_inches='tight', pad_inches=0.5)
# plt.show()