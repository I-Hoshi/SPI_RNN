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
#初期シード固定
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
#GPU使わせない。ここにおいておくと初期シード固定と重複してくれる？
K.tensorflow_backend.set_session(tf.Session(config=tf.ConfigProto(device_count = {'GPU': 0})))


def addGaussianNoise(src,mean = 0,sigma = 0.3):
    noise_shape = src.shape
    gauss = np.random.normal(mean,sigma,noise_shape)
    gauss = np.reshape(gauss,noise_shape)
    noisy = src + gauss

    return noisy

t_length=333
devide = 9#mnist-16
 
image_size=32


n_in = devide
n_time = t_length // n_in
n_hidden = 200#image_size*image_size / 2
n_out = image_size*image_size

def weight_variable(shape,name=None):
    return np.random.normal(scale=.01,size=shape)

early_stopping = EarlyStopping(monitor='val_loss',patience=10,verbose=1)



model = Sequential()
model.add(SimpleRNN(n_hidden,kernel_initializer=weight_variable,batch_input_shape=(1,n_time,n_in)))
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


model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001,beta_1=0.9,beta_2=0.999),#mnistlr=0.001,cifar-10lr=0.0001
              metrics=['accuracy'])

model.load_weights('param/RNNGI_' + str(image_size) + '.h5')

np.random.seed(0)

random_sequence = np.random.randint(0,2,image_size*image_size*t_length)

pettern = np.reshape(random_sequence,(t_length,image_size*image_size))
#pettern = np.reshape(np.load('learned_pattern_' + str(image_size) + '.npy'),(t_length,image_size*image_size))

#pettern = addGaussianNoise(pettern,0,0.1)

image_name = 'mnist_9'
image_file = 'image/org/' + str(image_size) + '/' + image_name + '.bmp'
image = Image.open(image_file).resize((image_size, image_size))
image_array =np.asarray(image)

normalized_image = np.reshape(image_array,(1,image_size*image_size))/ 255.0
intensity=np.reshape(np.sum(pettern*normalized_image,axis=1),(t_length,1))
input = intensity.reshape(1,t_length // devide,devide)




y_ = model.predict(input,batch_size=1)



fig = plt.figure(figsize=(9,9))
fig1 = fig.add_subplot(6,6,1)
plt.imshow(np.reshape(y_[0],(image_size,image_size)),cmap='gray')
fig1 = fig.add_subplot(6,6,2)
plt.imshow(np.reshape(image_array,(image_size,image_size)),cmap='gray')

plt.show()

img = np.reshape(y_[0],(image_size,image_size))
img2 = np.reshape(normalized_image,(image_size,image_size))

img = 255 * (img - np.min(img)) / (np.max(img) - np.min(img))
img2 = 255 * (img2 - np.min(img2)) / (np.max(img2) - np.min(img2))

img = img.astype(np.uint8)
img2 = img2.astype(np.uint8)

print(image_name)
print(eval_quality.psnr(img2,img))
print(eval_quality.ssim(img2,img))


output_name = 'image/RNNGI/RNNGI_' + str(image_size) + 'pix_' + image_name + '.bmp'
pil_img = Image.fromarray(img)
pil_img = pil_img.convert("L")
pil_img.save(output_name)



