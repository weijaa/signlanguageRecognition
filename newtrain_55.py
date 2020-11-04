from matplotlib.pyplot import show
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, ConvLSTM2D, Flatten, MaxPooling2D, AveragePooling2D, MaxPooling3D, Dropout,BatchNormalization, Activation, TimeDistributed , Conv2D, Conv3D, LSTM, SpatialDropout2D
from keras.utils import np_utils, to_categorical
from sklearn import preprocessing
import os
from os import listdir
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import random
import keras
from keras import regularizers, optimizers


path = 'D:/realData/matrixData/data55'
files = listdir(path)
max = 0
x = []
y = []
x_train = []
y_train = []


# for j  in range(5):
for file in files:
    y_l = file.split('_')
    y_train.append(y_l[0])
    # print(y_l[0])
    i = np.load(path+'/'+file)
    i = i['sl_arr']
    i = i.reshape(len(i),55, 55,1)
    if len(i) > max:
        max = len(i)
    x_pad = np.pad(array=i, pad_width=((0,(90-len(i))),(0,0),(0,0),(0,0)), mode='constant', constant_values=0) # padding time = max 
    x_train.append(x_pad)



np.random.seed(550)
np.random.shuffle(x_train) 
np.random.seed(550)
np.random.shuffle(y_train)

# c = list(zip(x_train, y_train))
# # print(c[0])
# random.shuffle(c)
# # print(c[0])
# x_train, y_train = zip(*c)

le = preprocessing.LabelEncoder() 
y_label_encoded = le.fit_transform(y_train)
# y_label_encoded = y_label_encoded.reshape(len(y_train),1)
y_train = keras.utils.to_categorical(y_label_encoded,11)
x_train_pad = np.array(x_train)
print(type(y_train))
print(y_train.shape)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
model = Sequential()
model.add(TimeDistributed(Conv2D(64,(3, 3), kernel_regularizer=regularizers.l2(0.001), activation='relu', padding='same'),input_shape=(90,55, 55, 1)))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(AveragePooling2D(pool_size=(2,2))))
# model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(SpatialDropout2D(0.5)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(64, return_sequences=True))
model.add(Flatten())

model.add(Dense(128,kernel_regularizer=regularizers.l2(0.001),  activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,kernel_regularizer=regularizers.l2(0.001), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(11, activation= 'softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',optimizer=optimizers.SGD(lr = 0.01),metrics=['accuracy'])




train_history = model.fit(x_train_pad, y_train,epochs=100, verbose=1, batch_size=1,  shuffle=True, validation_split= 0.1)
model.save('model_1020.h5')

def show_train_history(train_history,train):
    plt.plot(train_history.history[train])
    # plt.plot(train_history.history[validation])
    plt.xlabel('Epoches')
    plt.ylabel(train)
    plt.legend([train],loc='upper left')
    plt.savefig('./' + str(train))
    plt.show()


show_train_history(train_history,'accuracy')
show_train_history(train_history,'loss')
show_train_history(train_history,'val_loss')
show_train_history(train_history,'val_accuracy')

