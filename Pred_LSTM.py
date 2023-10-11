from __future__ import print_function

import math
import random
import string
import numpy as np
from os import listdir
from os.path import isfile, join
import numpy
import cv2
from array import array
from numpy import linalg as LA

import numpy as np

np.random.seed(1337)   
import pandas as pd
from PIL import Image
import sys
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt
from keras.layers import LSTM


def Calc_Wt(TRR,TST):
        classes=0;
        try:
                file= open("LSTM.obj",'rb')
                RNN = pickle.load(file)
                file.close()
                classes = RNN.predict_classes(TST, batch_size=batch_size)
                #proba = model.predict_proba(train_set_x[:0], batch_size=32)
                ind=0;
        except:
                if classes>=0:
                        WTRN = TRR
                        R, C = np.shape(WTRN)
                        M = []
                        ERR = []
                        WTST = TST
                        R, C = np.shape(WTRN)
                        for i in range(0, R):
                            RR = WTRN[i]
                            Temp = np.subtract(WTST, RR)
                            ERR = LA.norm(Temp)
                            M.append(ERR)
                        ind = np.argmin(M);
                       # print('ERROR',min(M))
                        if min(M)>=1:
                                ind=60;
                return ind
    

def TRAIN():
        batch_size = 10
        nb_classes = 10
        nb_epoch =4

        def plot_confusion_matrix(cm, classes,
                                  normalize=False,
                                  title='Confusion matrix',
                                  cmap=plt.cm.Reds):
            """
            This function prints and plots the confusion matrix.
            Normalization can be applied by setting `normalize=True`.
            """
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                print("Normalized confusion matrix")
            else:
                print('Confusion matrix, without normalization')

            #print(cm)

            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)

            fmt = '.2f' if normalize else 'd'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

            plt.ylabel('TRUE CLASS')
            plt.xlabel('PREDICTED CLASS')
            plt.tight_layout()

        file= open("LSTM_FPR.obj",'rb')
        cnf_matrix = pickle.load(file)
        file.close()
        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix[0:2,0:2], classes=['NORMAL','ABNORMAL'],title='CONFUSION MATRIX (IMAGES)')
        # Plot normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix[0:2,0:2], classes=['NORMAL','ABNORMAL'], normalize=True,title='CONFUSION MATRIX (SCALED)')
        plt.show()



        # CNN INPUT
        def Database(Path, LableFile=""):
            dataset = []
            gbr = pd.read_csv(LableFile, sep="\t")
            for i in range(1,21):
                image = Image.open(Path + '1 (' + str(i)+').jpg')
                img = Image.open(Path +  '1 (' + str(i)+').jpg').convert('LA') #tograyscale
                pixels = [f[0] for f in list(img.getdata())]
                dataset.append(pixels)
            if len(LableFile) > 0:
                df = pd.read_csv(LableFile)
                return np.array(dataset), gbr["Class"].values
            else:
                return np.array(dataset)

        if __name__ == '__main__':
            Data, y = Database("TEST/","label20.csv")
            print(np.shape(Data))
            y=np.array(np.zeros((20,), dtype=int))
            y1=np.array(np.zeros((5,), dtype=int))
            y2=np.array(np.ones((5,), dtype=int))
            y3=np.array(2*np.ones((5,), dtype=int))
            y4=np.array(3*np.ones((5,), dtype=int))
            y=[y1,y2,y3,y4]
            y= np.reshape(y,(20,1))
            print(type(y))
            print(y)
        ##    y=np.transpose(y);
            #Split the train set and validation set
            train_set_x = Data[:20]
            print('TRAIN SET SIZE',np.shape(train_set_x))
            val_set_x = Data[20:]
            train_set_y = y[:20]
            val_set_y = y[20:]
            (X_train, y_train), (X_test, y_test) = (train_set_x,train_set_y),(val_set_x,val_set_y)
            print(y_train)

            # input image dimensions
            img_rows, img_cols = 640, 480
            # number of convolutional filters to use
            nb_filters = 32
            # size of pooling area for max pooling
            pool_size = (2, 2)
            # convolution kernel size
            kernel_size = (3, 3)

            # Checking if the backend is Theano or Tensorflow
            if K.image_dim_ordering() == 'th':
                X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
                X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
                input_shape = (1, img_rows, img_cols)
            else:
                X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
                X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
                input_shape = (img_rows, img_cols, 1) 

            X_train = X_train.astype('float32')
            X_test = X_test.astype('float32')
            X_train /= 255
            X_test /= 255
            print('X_train shape:', X_train.shape)
            print(X_train.shape[0], 'train samples')
            print(X_test.shape[0], 'test samples')
            look_back=1;
            # convert class vectors to binary class matrices
            Y_train = np_utils.to_categorical(y_train, nb_classes)
            Y_test = np_utils.to_categorical(y_test, nb_classes)
            
            model = Sequential()
            model.add(LSTM(1, input_shape=(1, look_back)))
            model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=input_shape))
            model.add(Activation('relu'))
            model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=pool_size))
            model.add(Dropout(0.25))
            model.add(Flatten())
            model.add(Dense(128))
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
            model.add(Dense(nb_classes))
            model.add(Activation('softmax'))

            model.compile(loss='categorical_crossentropy',
                    optimizer='sgd',
                    metrics=['accuracy'])

            model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=2, validation_data=(X_test, Y_test))
            #model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=2, validation_data=(X_test, Y_test))
            #model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=2, show_accuracy=True, validation_data=(X_test, Y_test))
            print(np.shape(X_test))
            #print(model.predict(X_train[:19]))
            classes = model.predict_classes(X_train, batch_size=batch_size)
            #proba = model.predict_proba(train_set_x[:0], batch_size=32)
            print(classes)
            print(Y_train)
            CNN=model
            """
            model.save('model.h5')
            from keras.models import load_model
            model = load_model('model.h5')
            filehandler = open("LSTM.obj","wb")
            pickle.dump(CNN,filehandler)
            filehandler.close()
            #score = model.evaluate(X_test, Y_test, verbose=0)
        ##    print(score)
        ##    print('Test score:', score[0])
        ##    print('Test accuracy:', score[1])
            file= open("LSTM.obj",'rb')
            CNN = pickle.load(file)
            file.close()
            """

