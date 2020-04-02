'''This project evaluates a CRNN (convoltional recurrent neural network) based wireless channel
equalizer. For simplicity, we use QPSK modulation scheme with AWGN channel.
-------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------
Author: Li Yang
email:liyangsh48@gmail.com
'''

from __future__ import print_function
import numpy as np

# load data generating wrappers
from common.load import *
from common.data_generator import *


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, LSTM
from keras.layers.core import Reshape
from keras.utils import np_utils
from keras import optimizers
from keras.callbacks import Callback
from keras import backend as K
import pandas as pd

np.random.seed(1337)  # for reproducibility


# data generating
# dB = [i for i in range(-10,21)]
dB = [-10,-5,0,5,10,15,20]
smoothingLen = 11
chL = 10
mlp_score = []
cnn_score = []
crnn_score = []
nb_classes=4
batch_size = 1024

for db in dB:
	X_train, Y_train, X_test, Y_test= generateData(100000, 90000, db, smoothingLen, chL, 'cnn')
	X_train_mlp = [np.reshape(item, (24,)) for item in X_train]
	X_train_mlp = np.asarray(X_train_mlp)
	Y_train_mlp = np.asarray(Y_train)

	X_test_mlp = [np.reshape(item,(24,)) for item in X_test]
	X_test_mlp = np.asarray(X_test_mlp)
	Y_test_mlp = np.asarray(Y_test)

	input_size = 24
	# MLP implementation
	model = Sequential()
	model.add(Dense(5, activation='relu', input_shape=(24,)))
	model.add(Dense(5, activation='relu'))
	model.add(Dense(4,activation='softmax'))
	optimizer = optimizers.adam(lr=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-08,decay=0.0)
	model.compile(loss='categorical_crossentropy',
				  optimizer=optimizer,
				  metrics=['accuracy'])
	model.fit(X_train_mlp,Y_train_mlp,batch_size=batch_size,nb_epoch=40,verbose=1)
	score = model.evaluate(X_test_mlp, Y_test_mlp)
	mlp_score.append(score[1])


	# CNN implementation
	X_train_cnn = np.asarray(X_train)
	Y_train_cnn = np.asarray(Y_train)

	X_test_cnn = np.asarray(X_test)
	Y_test_cnn = np.asarray(Y_test)

	# we treat the input data as images, since the data consists of In-Phase & Quadrature parts, which is 2D
	# stucture.
	img_rows, img_cols = 2, 12
	# number of convolutional filters to use
	nb_filters = 32
	# size of pooling area for max pooling
	pool_size = (2, 2)
	# convolution kernel size
	kernel_size = (2, 3)

	X_train = np.asarray(X_train)
	Y_train = np.asarray(Y_train)

	X_test = np.asarray(X_test)
	Y_test = np.asarray(Y_test)

	X_train_cnn = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
	X_test_cnn = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
	input_shape = (img_rows, img_cols, 1)
	model = Sequential()
	model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
	                        border_mode='same',
	                        input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(Dropout(0.9))
	model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=pool_size, border_mode='same'))

	#model.add(Reshape((5,32)))
	# model.add(LSTM(128))
	model.add(Flatten())
	model.add(Dense(128))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))
	optimizer = optimizers.adam(lr=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-08,decay=0.0)
	model.compile(loss='categorical_crossentropy',
	              optimizer=optimizer,
	              metrics=['accuracy'])
	model.summary()

	history = model.fit(X_train_cnn, Y_train_cnn, batch_size=batch_size, nb_epoch=40,
	          verbose=1, validation_split=2.0/10)
	score = model.evaluate(X_test_cnn, Y_test_cnn)
	cnn_score.append(score[1])



	# Conv-LSTM implementation
	X_train_cnn = np.asarray(X_train)
	Y_train_cnn = np.asarray(Y_train)

	X_test_cnn = np.asarray(X_test)
	Y_test_cnn = np.asarray(Y_test)

	# we treat the input data as images, since the data consists of In-Phase & Quadrature parts, which is 2D
	# stucture.
	img_rows, img_cols = 2, 12
	# number of convolutional filters to use
	nb_filters = 32
	# size of pooling area for max pooling
	pool_size = (2, 2)
	# convolution kernel size
	kernel_size = (2, 3)

	X_train = np.asarray(X_train)
	Y_train = np.asarray(Y_train)

	X_test = np.asarray(X_test)
	Y_test = np.asarray(Y_test)

	X_train_cnn = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
	X_test_cnn = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
	input_shape = (img_rows, img_cols, 1)
	model = Sequential()
	model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
	                        border_mode='same',
	                        input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=pool_size, border_mode='same'))

	model.add(Reshape((5,32)))
	model.add(LSTM(128))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))
	optimizer = optimizers.adam(lr=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-08,decay=0.0)
	model.compile(loss='categorical_crossentropy',
	              optimizer=optimizer,
	              metrics=['accuracy'])
	model.summary()

	history = model.fit(X_train_cnn, Y_train_cnn, batch_size=batch_size, nb_epoch=40,
	          verbose=1, validation_split=2.0/10)
	score = model.evaluate(X_test_cnn, Y_test_cnn)
	crnn_score.append(score[1])


	print(mlp_score,cnn_score,crnn_score)

df1 = pd.DataFrame(mlp_score)
df1.to_csv('mlp.csv',index=False, header=False)
df2 = pd.DataFrame(cnn_score)
df2.to_csv('cnn.csv',index=False, header=False)
df3 = pd.DataFrame(crnn_score)
df3.to_csv('crnn.csv',index=False, header=False)



# batch_size = 128
# nb_classes = 4
# nb_epoch = [20,30,40,50,60]




#     X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
#     X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
# else:



# print('X_train shape:', X_train.shape)
# print(X_train.shape[0], 'train samples')
# print(X_test.shape[0], 'test samples')

# class ValHistory(Callback):
# 	def on_train_begin(self,logs={}):
# 		self.val = []

# 	def on_batch_end(self, batch, logs={}):
# 		self.val.append(logs.get('val_acc'))

  


# score = model.evaluate(X_test, Y_test, verbose=0)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])
# print(history.val)
# df1 = pd.DataFrame(history.history['acc'])
# df1.to_csv('acc_dropout_zero'+str(i)+'.csv', index=False, header=False)
# df2 = pd.DataFrame(history.history['val_acc'])
# df2.to_csv('val_acc_epoch'+str(ep)+'.csv', index=False, header=False)
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# plt.show()
