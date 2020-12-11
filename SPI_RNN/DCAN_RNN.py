import numpy as np
import time
from PIL import Image
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation,Reshape,Flatten
from keras.layers.recurrent import LSTM,SimpleRNN
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.optimizers import Adam
from keras.initializers import he_normal
from keras.callbacks import EarlyStopping,TensorBoard
from sklearn.utils import shuffle
from sklearn import datasets
from sklearn.model_selection import train_test_split
from keras.datasets import cifar10

import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

import eval_quality

leaky_relu = LeakyReLU()

def weight_variable(shape,name=None):
    return np.random.normal(scale=.01,size=shape)

def binary_regularizer(weight_matrix):
    return 0.00001*K.sum(K.square(1 + weight_matrix)*K.square(1 - weight_matrix))

np.random.seed(0)

image_size=32
t_length=333
input_dim = 9#There's a little difference between 9 and 37

n = 100000
N = n
N_train = 90000


f = 'stl-10/stl10_' + str(image_size) + 'pix_100000.npy'

train_array = np.load(f)

print('load_finished')

indices = np.random.permutation(range(n))[:N]

X = train_array[indices]
X = X / 255.0
X = X.reshape(len(X),image_size,image_size,1)
Y = X.reshape(len(X),image_size*image_size,)


X_train,X_validation,Y_train,Y_validation = \
    train_test_split(X,Y,train_size=N_train)


epochs = 3000
batch_size = 1000

n_hidden = 300
n_out = image_size*image_size



early_stopping = EarlyStopping(monitor='val_loss',patience=10,verbose=1)



model = Sequential()

model.add(Conv2D(t_length,image_size,use_bias = False,input_shape = (image_size,image_size,1),padding = 'valid'))

model.add(Reshape((t_length//input_dim,input_dim)))

model.add(SimpleRNN(n_hidden,kernel_initializer=weight_variable))
model.add(Dense(n_out,kernel_initializer=weight_variable))
model.add(Activation('relu'))

model.add(Reshape((image_size,image_size,1)))

model.add(Conv2D(64,9,padding = 'same'))

model.add(Activation('relu'))

model.add(Conv2D(32,1,padding = 'same'))

model.add(Activation('relu'))

model.add(Conv2D(1,5,padding = 'same'))

model.add(leaky_relu)

model.add(Reshape((image_size*image_size,)))

model.summary()

model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001,beta_1=0.9,beta_2=0.999),
              metrics=['accuracy'])


hist = model.fit(X_train,Y_train,
                 batch_size=batch_size,
                 epochs=epochs,
                 validation_data=(X_validation,Y_validation),
                 callbacks=[early_stopping])



wf = 'param/DRAN_' + str(image_size) + '_hidden' + str(n_hidden) + '.h5'
model.save_weights(wf)

model = Sequential()

model.add(Conv2D(t_length,image_size,use_bias = False,input_shape = (image_size,image_size,1),padding = 'valid',kernel_regularizer = binary_regularizer))

model.add(Reshape((t_length//input_dim,input_dim)))

model.add(SimpleRNN(n_hidden,kernel_initializer=weight_variable))
model.add(Dense(n_out,kernel_initializer=weight_variable))
model.add(Activation('relu'))

model.add(Reshape((image_size,image_size,1)))

model.add(Conv2D(64,9,padding = 'same'))

model.add(Activation('relu'))

model.add(Conv2D(32,1,padding = 'same'))

model.add(Activation('relu'))

model.add(Conv2D(1,5,padding = 'same'))

model.add(leaky_relu)

model.add(Reshape((image_size*image_size,)))

model.summary()

model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001,beta_1=0.9,beta_2=0.999),
              metrics=['accuracy'])

model.load_weights(wf)


hist = model.fit(X_train,Y_train,
                 batch_size=batch_size,
                 epochs=epochs,
                 validation_data=(X_validation,Y_validation),
                 callbacks=[early_stopping])


acc = hist.history['val_acc']
loss = hist.history['val_loss']

plt.rc('font',family='serif')
fig = plt.figure()
plt.plot(range(len(loss)),loss,
         label='loss',color='black')
plt.xlabel('epochs')
plt.show()

wfb = 'param/DRAN_binalize_' + str(image_size) + '_hidden' + str(n_hidden) +  '.h5'
model.save_weights(wfb)