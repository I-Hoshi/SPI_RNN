import numpy as np
import time
from keras import backend as K
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

def binary_regularizer(weight_matrix):
    return 0.001*K.sum(K.square(1 + weight_matrix)*K.square(1 - weight_matrix))

image_size=32
t_length=333
input_dim = 9

n = 100000
N = n
N_train = 90000
N_validation = 10000

f = 'stl-10/stl10_' + str(image_size) + 'pix_100000.npy'

train_array = np.load(f)

print('load_finished')

indices = np.random.permutation(range(n))[:N]

X = train_array[indices]
X = X / 255.0
X = X.reshape(len(X),image_size,image_size,1)
Y = X.reshape(len(X),image_size*image_size,)


X_train,X_test,Y_train,Y_test = \
    train_test_split(X,Y,train_size=N_train)

X_train,X_validation,Y_train,Y_validation = \
    train_test_split(X_train,Y_train,test_size=N_validation)


epochs = 2000
batch_size = 1000

n_hidden = 200
n_out = image_size*image_size

def weight_variable(shape,name=None):
    return np.random.normal(scale=.01,size=shape)

early_stopping = EarlyStopping(monitor='val_loss',patience=10,verbose=1)



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

wf = 'DRAN_' + str(image_size) + '_hidden' + str(n_hidden) + '.h5'
model.load_weights(wf)

#w1 = model.layers[0].get_weights()[0]
#print(w1[:,:,0,0])

#fig = plt.figure()
#for i in range(32):

#    fig1 = fig.add_subplot(8,8,2 * i + 1)
#    plt.imshow(np.reshape(w1[:,:,0,i*2],(image_size,image_size)),cmap='gray')

#    fig1 = fig.add_subplot(8,8,2 * i + 2)
#    plt.imshow(np.reshape(w1[:,:,0,i*2 + 1],(image_size,image_size)),cmap='gray')
#plt.show()


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


loss_and_metrics = model.evaluate(X_test,Y_test)

print(loss_and_metrics)


y_ = model.predict(X_test)


wfb = 'param/DRAN_binalize_' + str(image_size) + '_hidden' + str(n_hidden) +  '.h5'
model.save_weights(wfb)

fig = plt.figure()
fig1 = fig.add_subplot(2,2,1)
plt.imshow(np.reshape(y_[100],(image_size,image_size)),cmap='gray')

fig2 = fig.add_subplot(2,2,2)
plt.imshow(np.reshape(Y_test[100],(image_size,image_size)),cmap='gray')

fig3 = fig.add_subplot(2,2,3)
plt.imshow(np.reshape(y_[101],(image_size,image_size)),cmap='gray')

fig4 = fig.add_subplot(2,2,4)
plt.imshow(np.reshape(Y_test[101],(image_size,image_size)),cmap='gray')
plt.show()

img = np.reshape(y_[101],(image_size,image_size))
img2 = np.reshape(Y_test[101],(image_size,image_size))

img = 255 * (img - np.min(img)) / (np.max(img) - np.min(img))
img2 = 255 * (img2 - np.min(img2)) / (np.max(img2) - np.min(img2))

print(eval_quality.PSNR(img2,img))
print(eval_quality.SSIM(img2,img))

img_array = np.reshape(y_[:10],(10,image_size,image_size))
img_array2 = np.reshape(Y_test[:10],(10,image_size,image_size))

img_array = 255 * (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array))
img_array2 = 255 * (img_array2 - np.min(img_array2)) / (np.max(img_array2) - np.min(img_array2))


w1 = model.layers[0].get_weights()[0]
print(w1[:,:,0,0])

fig = plt.figure()
for i in range(32):

    fig1 = fig.add_subplot(8,8,2 * i + 1)
    plt.imshow(np.reshape(w1[:,:,0,i*2],(image_size,image_size)),cmap='gray')

    fig1 = fig.add_subplot(8,8,2 * i + 2)
    plt.imshow(np.reshape(w1[:,:,0,i*2 + 1],(image_size,image_size)),cmap='gray')
plt.show()
#print(eval_quality.PSNR_average(img_array2,img_array))
#print(eval_quality.SSIM_average(img_array2,img_array))
#pil_img = Image.fromarray(img)
#pil_img = pil_img.convert("L")
#pil_img.save('CNN.bmp')

