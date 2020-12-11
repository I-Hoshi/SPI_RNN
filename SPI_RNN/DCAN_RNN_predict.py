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

leaky_relu = LeakyReLU()
np.random.seed(0)

image_size=32
t_length=333
input_dim = 9#There's a little difference between 9 and 37


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


fw = 'param/DRAN_binalize_' + str(image_size) + '_hidden' + str(n_hidden) +  '.h5'
model.load_weights(fw)


w1 = model.layers[0].get_weights()[0]
w1_bin = (w1 > 0) * 1
np.save('learned_pattern_' + str(image_size),w1_bin)
print(np.shape(w1[:,:,0,0]))
#w1 = addGaussianNoise(np.round(w1),0,0.1)
#model.layers[0].set_weights(w1.reshape((1,32,32,1,333)))
fig = plt.figure()
for i in range(32):

    fig1 = fig.add_subplot(8,8,2 * i + 1)
    plt.imshow(np.reshape(w1[:,:,0,i*2],(image_size,image_size)),cmap='gray')

    fig1 = fig.add_subplot(8,8,2 * i + 2)
    plt.imshow(np.reshape(w1[:,:,0,i*2 + 1],(image_size,image_size)),cmap='gray')
plt.show()



image_name = 'cameraman'
image_file = 'image/org/' + str(image_size) + '/' + image_name + '.bmp'
image = Image.open(image_file).resize((image_size, image_size))

X =np.reshape(np.asarray(image),(1,image_size,image_size,1))/255
y = model.predict(X)
y = 255 * (y - np.min(y)) / (np.max(y) - np.min(y))
X = 255 * (X - np.min(X)) / (np.max(X) - np.min(X))
y = y.astype(np.uint8)
X = X.astype(np.uint8)

print(image_name)
print(eval_quality.psnr(X,y))
print(eval_quality.ssim(X,y))

output_name = 'image/DRAN/DRAN_' + str(image_size) + 'pix_' + image_name + '.bmp'
pil_img = Image.fromarray(y.reshape(image_size,image_size))
pil_img = pil_img.convert("L")
pil_img.save(output_name)