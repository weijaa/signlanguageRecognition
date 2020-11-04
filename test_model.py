from json import load
from typing import Dict
from matplotlib.pyplot import show
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, ConvLSTM2D, Flatten, MaxPooling2D, Dropout,BatchNormalization, Activation, TimeDistributed , Conv2D, LSTM, SpatialDropout2D
from keras.utils import np_utils, to_categorical
from sklearn import preprocessing
import os
from os import listdir
from keras.optimizers import RMSprop
from tensorflow import keras
import matplotlib.pyplot as plt
import random
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

path = 'D:/realData/testMatrix2'
files = listdir(path)
max = 0
x = []
y = []
x_test = []
y_test = []


# for j  in range(5):
for file in files:
    y_l = file.split('_')
    y_test.append(y_l[0])
    # print(y_l[0])
    i = np.load(path + '/' + file)
    i = i['sl_arr']
    i = i.reshape(len(i),55, 55,1)
    if len(i) > max:
        max = len(i)
        print(y_l[0],max)
    x_pad = np.pad(array=i, pad_width=((0,(90-len(i))),(0,0),(0,0),(0,0)), mode='constant', constant_values=0) # padding time = max 
    x_test.append(x_pad)


label_dict = {0:'eat', 1:'goodbye', 2:'hello', 3:'helpme', 4:'morning', 5:'night', 6:'please', 7:'sleep', 8:'sorry', 9:'thanks', 10:'welcome'}
labels=['eat', 'goodbye', 'hello', 'helpme', 'morning', 'night', 'please', 'sleep', 'sorry', 'thanks', 'welcome']


le = preprocessing.LabelEncoder() 
y_label_encoded = le.fit_transform(y_test)
print(y_label_encoded)
y_test_onehot = keras.utils.to_categorical(y_label_encoded)
x_test = np.array(x_test)
y_test_onehot = np.array(y_test_onehot)
# y_test_OneHot = to_categorical(y_test)
print('x_test:',x_test.shape)
print('y_test:',y_test_onehot)



def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens,#这个地方设置混淆矩阵的颜色主题，这个主题看着就干净~
                          normalize=True):
   
 
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(15, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    #这里这个savefig是保存图片，如果想把图存在什么地方就改一下下面的路径，然后dpi设一下分辨率即可。
    plt.savefig('confusionmatrix32.png')
    plt.show()



model = tf.keras.models.load_model("D:/realData/model_1031.h5")
scores = model.evaluate(x_test,y_test_onehot)
prediction = model.predict_classes(x_test)
# print(label_dict[prediction[0]])
conf_mat = confusion_matrix(y_true=y_label_encoded, y_pred=prediction)
# plt.figure()
plot_confusion_matrix(conf_mat, normalize=False,target_names=labels,title='Confusion Matrix')

