import numpy as np
import time
import os
import random as rn
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation,Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.recurrent import LSTM,SimpleRNN
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,TensorBoard
from sklearn.utils import shuffle
from sklearn import datasets
from sklearn.model_selection import train_test_split
from keras.datasets import cifar10

import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

import eval_quality

leaky_relu = LeakyReLU()

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(0)
rn.seed(0)

session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1
)

import keras.backend as K
tf.set_random_seed(0)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

K.tensorflow_backend.set_session(tf.Session(config=tf.ConfigProto(device_count = {'GPU': 0})))


def addGaussianNoise(src,mean = 0,sigma = 0.3):
    noise_shape = src.shape
    gauss = np.random.normal(mean,sigma,noise_shape)
    gauss = np.reshape(gauss,noise_shape)
    noisy = src + gauss

    return noisy

t_length=333
devide = 9#64#mnist-16
 
image_size=32*2

n = 5000
N = 5000


f = 'stl-10/stl10_test_' + str(image_size) + 'pix_5000.npy'

train_array = np.load(f)

print('load_finished')

indices = np.random.permutation(range(n))[:N]

X = train_array[indices]
X = X / 255.0
X = X.reshape(len(X),image_size*image_size)
Y = X.reshape(len(X),image_size*image_size)

s = np.zeros((N,t_length,1))

np.random.seed(0)

random_sequence = np.random.randint(0,2,image_size*image_size*t_length)

pettern = np.reshape(random_sequence,(t_length,image_size*image_size))
pettern = addGaussianNoise(pettern,0,0.0)

for i in range(N):
    
    normalized_image=X[i,:]

    normlized_image = np.reshape(normalized_image,(1,image_size*image_size))

    s[i]=np.reshape(np.sum(pettern*normalized_image,axis=1),(t_length,1))

s = s.reshape(N,int(t_length / devide),devide)

n_in = devide
n_time = t_length // n_in
n_hidden = 200#image_size*image_size / 2
n_out = image_size*image_size



model = Sequential()
model.add(SimpleRNN(n_hidden,input_shape=(n_time,n_in)))
model.add(Dense(n_out))
model.add(Activation('relu'))

model.add(Reshape((image_size,image_size,1)))

model.add(Conv2D(64,9,padding = 'same'))

model.add(Activation('relu'))

model.add(Conv2D(32,1,padding = 'same'))

model.add(Activation('relu'))

model.add(Conv2D(1,5,padding = 'same'))

model.add(leaky_relu)

model.add(Reshape((image_size*image_size,)))


model.load_weights('param/RNNGI_' + str(image_size) + '.h5')



y_ = model.predict(s)


fig = plt.figure(figsize=(9,9))
for i in range(2):

    fig1 = fig.add_subplot(6,6, i + 1)
    plt.imshow(np.reshape(y_[0],(image_size,image_size)),cmap='gray')

plt.show()

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




