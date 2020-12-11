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
from keras.optimizers import Adam,SGD
from keras.callbacks import EarlyStopping,TensorBoard
from sklearn.utils import shuffle
from sklearn import datasets
from sklearn.model_selection import train_test_split
from keras.datasets import cifar10

import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

import eval_quality

#変分ドロップアウトを試す！！！

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

image_size=32

n = 100000
N = 100000
N_train = 90000

f = 'stl-10/stl10_' + str(image_size) + 'pix_100000.npy'

train_array = np.load(f)

t_length=333

devide = 9#64#mnist-16


indices = np.random.permutation(range(n))[:N]

X = train_array[indices]
X = X / 255.0
X = X.reshape(len(X),image_size*image_size)
Y = X.reshape(len(X),image_size*image_size)



np.random.seed(0)

random_sequence = np.random.randint(0,2,image_size*image_size*t_length)

pettern = np.reshape(random_sequence,(t_length,image_size*image_size))
cashname = 'cash/RNNGI/lightintensity_stl_' + str(image_size) + '_' + str(t_length) + '.npy'
if(os.path.exists(cashname)):
    print('file exists')
    s = np.load(cashname)
    print('loaded')

else:
    print('file not exists')
    s = np.zeros((N,t_length,1))
    for i in range(N):
    
        normalized_image=X[i,:]

        normlized_image = np.reshape(normalized_image,(1,image_size*image_size))

        s[i]=np.reshape(np.sum(pettern*normalized_image,axis=1),(t_length,1))

    s = s.reshape(N,int(t_length / devide),devide)
    np.save(cashname,s)


np.random.seed(0)

X_train,X_validation,Y_train,Y_validation = \
    train_test_split(s,Y,train_size=N_train)


'''
モデル設定
'''
epochs = 140
batch_size = 1000

n_in = devide
n_time = t_length // n_in
n_hidden = 200#image_size*image_size / 2
n_out = image_size*image_size

def weight_variable(shape,name=None):
    return np.random.normal(scale=.01,size=shape)

early_stopping = EarlyStopping(monitor='val_loss',patience=10,verbose=1)

old_session = KTF.get_session()

session = tf.Session('')
KTF.set_session(session)
KTF.set_learning_phase(1)


model = Sequential()
model.add(SimpleRNN(n_hidden,kernel_initializer=weight_variable,input_shape=(n_time,n_in)))
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


model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001,beta_1=0.9,beta_2=0.999,decay=0),#mnistlr=0.001,cifar-10lr=0.0001
              metrics=['accuracy'])

tb_cb = TensorBoard(log_dir='./logs2',histogram_freq=0,
                            batch_size=batch_size,write_graph=True,write_grads=False,
                            write_images=False,embeddings_freq=0,embeddings_layer_names=None,
                            embeddings_metadata=None)
cbks = [tb_cb]

'''
モデル学習
'''




hist = model.fit(X_train,Y_train,
                 batch_size=batch_size,
                 epochs=epochs,
                 validation_data=(X_validation,Y_validation),
                 callbacks=[early_stopping])

'''
学習の進み具合を可視化
'''
acc = hist.history['val_acc']
loss = hist.history['val_loss']

plt.rc('font',family='serif')
fig = plt.figure()
plt.plot(range(len(loss)),loss,
         label='loss',color='black')
plt.xlabel('epochs')
plt.show()


model.save_weights('param/RNNGI_' + str(image_size) + '.h5')



#pil_img = Image.fromarray(img)
#pil_img = pil_img.convert("L")
#pil_img.save('RNNGI.bmp')

#pil_img = Image.fromarray(img)
#pil_img = pil_img.convert("L")
#pil_img.save('hidden_100.bmp')

#KTF.set_session(old_session)

'''
メモ
lossとval_lossがCNNだとだいたい同じくらいなのにRNNだとval_lossがあまり下がらない
分割を荒くすることで若干改善
やっぱりパラメータ数は偉大

cifar10
t_length = 512
devide = 16
hidden = 100
image_size=32

mnist
t_length=256
devide=64
hidden=100
image_size=28
'''


