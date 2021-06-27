from __future__ import print_function

##import tkinter
##import tkinter.filedialog
##from tkinter import messagebox
#root=tkinter.Tk()
from tkinter import filedialog

import os
from os import listdir
from os.path import isfile, join
import time
from PyQt4 import QtCore, QtGui
import cv2


import math
import random
import string
import numpy as np
import numpy
import cv2
from array import array
from numpy import linalg as LA
from Pred_LSTM import Calc_Wt
import pickle

import matplotlib.pyplot as plt

from tkinter import * 
from tkinter import messagebox 
root = Tk() 
root.geometry("300x200")


import itertools
        
def build_filters():
 filters = []
 ksize = 31
 for theta in np.arange(0, np.pi, np.pi / 16):
     kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
     kern /= 1.5*kern.sum()
     filters.append(kern)
 return filters
 
def process(img, filters):
     accum = np.zeros_like(img)
     for kern in filters:
         fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
         np.maximum(accum, fimg, accum)
     return accum
def thresholded(center, pixels):
    out = []
    for a in pixels:
        if a >= center:
            out.append(1)
        else:
            out.append(0)
    return out

def get_pixel_else_0(l, idx, idy, default=0):
    try:
        return l[idx,idy]
    except IndexError:
        return default

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):
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


    
print('******Start*****')
try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow1(object):
    

    def setupUii(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(1200, 800)
        MainWindow.setStyleSheet(_fromUtf8("\n""background-image: url(III3.jpg);\n"""))
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.pushButton = QtGui.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(750, 180, 111, 27))
        self.pushButton.clicked.connect(self.quit)
        self.pushButton.setStyleSheet(_fromUtf8("background-color: rgb(0, 0, 0);\n"
"color: rgb(0, 0, 0);"))
       
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
#################################################################
        

        self.pushButton_2 = QtGui.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(550, 180, 131, 27))
        self.pushButton_2.clicked.connect(self.show1)
        self.pushButton_2.setStyleSheet(_fromUtf8("background-color: rgb(0, 0, 0);\n"
"color: rgb(0, 0, 0);"))
        self.pushButton_2.setObjectName(_fromUtf8("pushButton_2"))
        
        self.pushButton_4 = QtGui.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(550, 220, 131, 27))
        self.pushButton_4.clicked.connect(self.show2)
        self.pushButton_4.setStyleSheet(_fromUtf8("background-color: rgb(0, 0, 0);\n"
"color: rgb(0, 0, 0);"))
        self.pushButton_4.setObjectName(_fromUtf8("pushButton_4"))
        
        

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
       
        

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "LUNG CANCER DETECTION", None))
        self.pushButton_2.setText(_translate("MainWindow", "TEST", None))
        self.pushButton_4.setText(_translate("MainWindow", "TRAIN", None))
        #self.pushButton_5.setText(_translate("MainWindow", "PLOT", None))
        self.pushButton.setText(_translate("MainWindow", "Exit", None))

    def quit(self):
        print ('Process end')
        print ('******End******')
        quit()
         
    def show1(self):
        # -----------------------------------------------
        #               MAIN CODE
        # -----------------------------------------------
        # READ IMAGE
        image_path= filedialog.askopenfilename(filetypes = (("BROWSE CT IMAGE", "*.jpg"), ("All files", "*")))
        img=cv2.imread(image_path)
        #img=cv2.imread('Train/B (5).jpg')
        img = cv2.resize(img,(100,100),3)
        img1 = cv2.GaussianBlur(img,(5,5),0)
        img1 = cv2.medianBlur(img,5)
        gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        GF1=build_filters()
        TST1=process(gray, GF1)
        img = cv2.resize(TST1,(100,100))
        TST=np.reshape(img,(10000))
        TST=np.transpose(TST)
        #TST=np.transpose(TST)
        print('TST FV',np.shape(TST));
        TRR=[];
        #TRR=np.loadtxt('a.txt')
        file= open("GNET.obj",'rb')
        TRR = pickle.load(file)
        file.close()
        SUPLBL=[1,2,3,4,5,6,7,8,9,10]
        IND=Calc_Wt(TRR,TST)
        #print('INITIAL INDEX',IND)
        IND=int(np.ceil(IND/5));
        #print('CLASSWISE INDEX',IND)
        ind=IND
        #print(ind)


        if ind==0:
           print('NORMAL')
           messagebox.showinfo(title='RESULT', message='NORMAL')
        if ind==1:
           print('NORMAL')
           messagebox.showinfo(title='RESULT', message='NORMAL')
        if ind==2:
           print('BENIGN')
           messagebox.showinfo(title='RESULT', message='BENIGN')
        if ind==3:
           print('MALIGNANT')
           messagebox.showinfo(title='RESULT', message='MALIGNANT')

        root.destroy()

         
    def show2(self):
        
        import tensorflow as tf
        import matplotlib.pyplot as plt
        from keras.layers import Input, Dense
        from  keras import regularizers
        from  keras.models import Sequential, Model
        from  keras.layers import Input, Flatten, Dense, Dropout, BatchNormalization
        from  keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
        from keras.layers import Concatenate
        from keras.preprocessing.image import ImageDataGenerator
        from keras.optimizers import Adam, SGD
        import pickle

        # define parameters
        CLASS_NUM = 5
        BATCH_SIZE = 16
        EPOCH_STEPS = int(4323/BATCH_SIZE)
        IMAGE_SHAPE = (512, 512, 3)
        IMAGE_TRAIN = 'Gnet'
        MODEL_NAME = 'Gnet.h5'
        def inception(x, filters):
                path1 = Conv2D(filters=filters[0], kernel_size=(1,1), strides=1, padding='same', activation='relu')(x)
                path2 = Conv2D(filters=filters[1][0], kernel_size=(1,1), strides=1, padding='same', activation='relu')(x)
                path2 = Conv2D(filters=filters[1][1], kernel_size=(3,3), strides=1, padding='same', activation='relu')(path2)
                path3 = Conv2D(filters=filters[2][0], kernel_size=(1,1), strides=1, padding='same', activation='relu')(x)
                path3 = Conv2D(filters=filters[2][1], kernel_size=(5,5), strides=1, padding='same', activation='relu')(path3)
                path4 = MaxPooling2D(pool_size=(3,3), strides=1, padding='same')(x)
                path4 = Conv2D(filters=filters[3], kernel_size=(1,1), strides=1, padding='same', activation='relu')(path4)
                return Concatenate(axis=-1)([path1,path2,path3,path4])
        def auxiliary(x, name=None):
                layer = AveragePooling2D(pool_size=(5,5), strides=3, padding='valid')(x)
                layer = Conv2D(filters=128, kernel_size=(1,1), strides=1, padding='same', activation='relu')(layer)
                layer = Flatten()(layer)
                layer = Dense(units=256, activation='relu')(layer)
                layer = Dropout(0.4)(layer)
                layer = Dense(units=CLASS_NUM, activation='softmax', name=name)(layer)
                return layer
        def googlenet():
                layer_in = Input(shape=IMAGE_SHAPE)
                layer = Conv2D(filters=64, kernel_size=(7,7), strides=2, padding='same', activation='relu')(layer_in)
                layer = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(layer)
                layer = BatchNormalization()(layer)
                layer = Conv2D(filters=64, kernel_size=(1,1), strides=1, padding='same', activation='relu')(layer)
                layer = Conv2D(filters=192, kernel_size=(3,3), strides=1, padding='same', activation='relu')(layer)
                layer = BatchNormalization()(layer)
                layer = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(layer)
                layer = inception(layer, [ 64,  (96,128), (16,32), 32]) #3a
                layer = inception(layer, [128, (128,192), (32,96), 64]) #3b
                layer = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(layer)
                layer = inception(layer, [192,  (96,208),  (16,48),  64]) #4a
                aux1  = auxiliary(layer, name='aux1')
                layer = inception(layer, [160, (112,224),  (24,64),  64]) #4b
                layer = inception(layer, [128, (128,256),  (24,64),  64]) #4c
                layer = inception(layer, [112, (144,288),  (32,64),  64]) #4d
                aux2  = auxiliary(layer, name='aux2')
                layer = inception(layer, [256, (160,320), (32,128), 128]) #4e
                layer = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(layer)
                layer = inception(layer, [256, (160,320), (32,128), 128]) #5a
                layer = inception(layer, [384, (192,384), (48,128), 128]) #5b
                layer = AveragePooling2D(pool_size=(7,7), strides=1, padding='valid')(layer)
                layer = Flatten()(layer)
                layer = Dropout(0.4)(layer)
                layer = Dense(units=256, activation='linear')(layer)
                main = Dense(units=CLASS_NUM, activation='softmax', name='main')(layer)
                model = Model(inputs=layer_in, outputs=[main, aux1, aux2])  
                return model

        model= googlenet()
        model.summary()


        print(model.summary())
        file= open("TRNMDL.obj",'rb')
        cnf_matrix = pickle.load(file)
        file.close()

        plt.figure()
        plot_confusion_matrix(cnf_matrix[0:3,0:3], classes=['NORMAL ','BENIGN','MALIGNANT'], normalize=True,title='Proposed Method')
        plt.show()


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    MainWindow = QtGui.QMainWindow()
    ui = Ui_MainWindow1()
    ui.setupUii(MainWindow)
    MainWindow.move(550, 170)
    MainWindow.show()
    sys.exit(app.exec_())
    

