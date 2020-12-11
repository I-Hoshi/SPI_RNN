import numpy as np
import time
from PIL import Image
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

def pattern_addGaussianNoise(src,mean = 0,sigma = 0.3,start=0,end=333):
    noise_shape = src.shape
    print(noise_shape)
    gauss = np.random.normal(mean,sigma,noise_shape)
    gauss = np.reshape(gauss,noise_shape)
    gauss[:,:,:,:start]=0
    gauss[:,:,:,end:]=0
    noisy = src + gauss
    return noisy

leaky_relu = LeakyReLU()
np.random.seed(0)



image_size=32
t_length=333
input_dim = 9#There's a little difference between 9 and 37

n = 5000
N = 5000


f = 'stl-10/stl10_test_' + str(image_size) + 'pix_5000.npy'

train_array = np.load(f)

print('load_finished')

indices = np.random.permutation(range(n))[:N]

X = train_array[indices]
X = X / 255.0
X = X.reshape(len(X),image_size,image_size,1)

batch_size = 100

n_hidden = 200
n_out = image_size*image_size

def weight_variable(shape,name=None):
    return np.random.normal(scale=.01,size=shape)

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


fw = 'param/DRAN_binalize_' + str(image_size) + '_hidden' + str(n_hidden) + '.h5'
model.load_weights(fw)

w1 = model.layers[0].get_weights()[0]

w_noise = pattern_addGaussianNoise(np.round(w1),0,0.0,0,9*37)
model.layers[0].set_weights(w_noise.reshape((1,image_size,image_size,1,t_length)))


y_ = model.predict(X)

fig = plt.figure()
fig1 = fig.add_subplot(2,2,1)
plt.imshow(np.reshape(y_[0],(image_size,image_size)),cmap='gray')

fig2 = fig.add_subplot(2,2,2)
plt.imshow(np.reshape(X[0],(image_size,image_size)),cmap='gray')

fig3 = fig.add_subplot(2,2,3)
plt.imshow(np.reshape(y_[101],(image_size,image_size)),cmap='gray')

fig4 = fig.add_subplot(2,2,4)
plt.imshow(np.reshape(X[101],(image_size,image_size)),cmap='gray')
plt.show()

img = np.reshape(y_[1],(image_size,image_size))
img2 = np.reshape(X[1],(image_size,image_size))

img = 255 * (img - np.min(img)) / (np.max(img) - np.min(img))
img2 = 255 * (img2 - np.min(img2)) / (np.max(img2) - np.min(img2))

img = img.astype(np.uint8)
img2 = img2.astype(np.uint8)

#print(eval_quality.psnr(img2,img))
#print(eval_quality.ssim(img2,img))

img_array = np.reshape(y_[:N],(N,image_size*image_size))
img_array2 = np.reshape(X[:N],(N,image_size*image_size))


img_array = 255 * (img_array - np.min(img_array,axis=-1,keepdims=True)) / (np.max(img_array,axis=-1,keepdims=True) - np.min(img_array,axis=-1,keepdims=True))
img_array2 = 255 * (img_array2 - np.min(img_array2,axis=-1,keepdims=True)) / (np.max(img_array2,axis=-1,keepdims=True) - np.min(img_array2,axis=-1,keepdims=True))

img_array = np.reshape(img_array,(N,image_size,image_size))
img_array2 = np.reshape(img_array2,(N,image_size,image_size))
img_array = img_array.astype(np.uint8)
img_array2 = img_array2.astype(np.uint8)

print(f)
print(eval_quality.psnr_average(img_array2,img_array))
print(eval_quality.ssim_average(img_array2,img_array))
