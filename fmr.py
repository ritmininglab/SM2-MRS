
from __future__ import division
import numpy as np
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras.callbacks import Callback

import os
import copy
import random




def counttotal(xlist):
    total = 0
    for i in range(len(xlist)):
        total += xlist[i].shape[0]
    return total

class CustomCallbackVAE(Callback):
    def on_train_begin(self, logs={}):    
        self.epochs = 0    
    def on_epoch_end(self, batch, logs={}):    
        self.epochs += 1     
        if self.epochs % 20 == 1:     
            print("epoch {}, loss {:3.2f}={:3.2f}+{:3.2f}+{:3.2f}".format(
                self.epochs, logs["loss"], logs["ra_loss"], logs["rb_loss"],
                logs["ry_loss"])
                )


def getlayeridx(model, layerName):
    index = None
    for idx, layer in enumerate(model.layers):
        if layer.name == layerName:
            index = idx
            break
    return index


def getlayerweights(model, layername):
    idx = getlayeridx(model, layername)
    return model.layers[idx].get_weights() 


def checklistminmax(lists):
    minval = np.min(lists[0])
    maxval = np.max(lists[0])
    for i in range(1,len(lists)):
        minval = min(np.min(lists[i]),minval)
        maxval = max(np.max(lists[i]),maxval)
    return [minval, maxval]


def checklayerweights(model, layerlist, idx2):
    for i in range(0, len(layerlist)):
        layername = layerlist[i]
        idx = getlayeridx(model, layername)
        temp = model.layers[idx].get_weights()
        print(layername, checklistminmax([temp[idx2]]))


def myloss(y_true, y_pred):
    temp = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=y_true,logits=y_pred))
    return temp


def mmd(y_true, y_pred):
    lossw = 0.1
    x = y_true[0,:]
    T = x.shape[0]
    rows = tf.tile(tf.expand_dims(x,axis=1), [1,T,1])
    cols = tf.tile(tf.expand_dims(x,axis=0), [T,1,1])
    term1 = tf.reduce_mean(tf.exp(tf.reduce_sum(tf.square(rows-cols), axis=-1)))
    term2 = 1
    term3 = tf.reduce_mean(tf.exp(tf.reduce_sum(tf.square(x),axis=-1)))
    temp = lossw*(term1+term2+term3)
    return temp

adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.99, epsilon=1e-07,) 
sgd = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
verbose = 0



# load data

X1raw = np.loadtxt(open('Data2/gaze3.csv',"rb"),delimiter=",",skiprows=0)
X2raw = np.loadtxt(open('Data2/paths3.csv',"rb"),delimiter=",",skiprows=0)
X3raw = np.loadtxt(open('Data2/tmd3.csv',"rb"),delimiter=",",skiprows=0)
Yraw = np.loadtxt(open('Data2/Cat3b.csv',"rb"),delimiter=",",skiprows=0)




dimcls = 2
nmodal = 3
f1 = 15
f2 = 1
voc = 789
dimh = 32

Tmax = X2raw.shape[-1] -3
nins = int(Yraw.shape[0])

dummy = np.zeros((nins,Tmax,1))

X1 = -8*np.ones((nins,Tmax,f1))
X2 = -8*np.ones((nins,Tmax,f1))
X3 = -8*np.ones((nins,Tmax,f2))
Y = 0*np.ones((nins,Tmax,dimcls))

for i in range(nins):
    Tnow = int(X1raw[i*f1,2])
    clsnow = int(Yraw[i]-1)
    X1[i,0:Tnow,:] = np.copy(np.transpose(X1raw[i*f1:(i+1)*f1,3:3+Tnow]))
    X2[i,0:Tnow,:] = np.copy(np.transpose(X2raw[i*f1:(i+1)*f1,3:3+Tnow]))
    X3[i,0:Tnow,:] = np.copy(np.transpose(X3raw[i*f2:(i+1)*f2,3:3+Tnow]))
    Y[i,0:Tnow,clsnow] = 1

mse = tf.keras.metrics.mean_squared_error
catce = tf.keras.metrics.categorical_crossentropy



# model specification

from modelMGMFMR import maemask
dims = [f1,f1,f2,dimh,dimh,dimh,dimh,dimh,dimh,dimh,dimh]
Nbatch = 1
kldivide = Nbatch*dimh*2
params = [dims,Nbatch, kldivide,dimcls, Tmax]
w1 = 1.
w2 = 1/f1
w3 = 3

x1 = Input(batch_shape=(Nbatch, None, dims[0]), name='inputx1') 
x2 = Input(batch_shape=(Nbatch, None, dims[1]), name='inputx2') 
x3 = Input(batch_shape=(Nbatch, None, dims[2]), name='inputx3') 

m = Model(inputs=[x1,x2,x3], 
          outputs=maemask([x1,x2,x3], params))

m.compile(loss={'ra': mse,'rb': mse,'rc': mse,'ry': mse,
                'z1a':mmd,'z1b':mmd,'z1c':mmd,'z1':mmd,},
          loss_weights={'ra': w1,'rb': w1,'rc': w2,'ry': w3,
                'z1a':w1,'z1b':w1,'z1c':w1,'z1':w1},
          optimizer=adam,
          metrics={'ra':'mse','rb':'mse','rc':'mse','ry':'mse'}) 


# model training

mode = 0
if mode==0:
    m.fit([X1,X2,X3],
          {'ra': X1,'rb': X2,'rc': X3,'ry': Y,
           'z1a':dummy,'z1b':dummy,'z1':dummy,},
          batch_size=Nbatch, 
          epochs=500,
          verbose=verbose,
          callbacks=[CustomCallbackVAE()])
    

m.load_weights('fmr.h5')




# load test data


X1traw = np.loadtxt(open('Data2/gaze2.csv',"rb"),delimiter=",",skiprows=0)
X2traw = np.loadtxt(open('Data2/paths2.csv',"rb"),delimiter=",",skiprows=0)
X3traw = np.loadtxt(open('Data2/tmd3.csv',"rb"),delimiter=",",skiprows=0)
Ytraw = np.loadtxt(open('Data2/Cat2b.csv',"rb"),delimiter=",",skiprows=0)

ntins = int(Ytraw.shape[0])

X1t = -8*np.ones((ntins,Tmax,f1))
X2t = -8*np.ones((ntins,Tmax,f1))
X3t = -8*np.ones((ntins,Tmax,f2))
meta = np.zeros((ntins,))
for i in range(ntins):
    Tnow = int(X1traw[i*f1,2])
    meta[i] = Tnow
    clsnow = int(Ytraw[i]-1)
    X1t[i,0:Tnow,:] = np.copy(np.transpose(X1traw[i*f1:(i+1)*f1,3:3+Tnow]))
    X2t[i,0:Tnow,:] = np.copy(np.transpose(X2traw[i*f1:(i+1)*f1,3:3+Tnow]))
    X3t[i,0:Tnow,:] = np.copy(np.transpose(X3traw[i*f2:(i+1)*f2,3:3+Tnow]))




# inference

match = 0
for i in range(ntins):
    preds = m.predict([X1t[i:i+1,:],X2t[i:i+1,:],X3t[i:i+1,:]])
    Tnow = int(meta[i])
    pred = preds[3][0,0:Tnow,:].argmax(-1)
    predcount = np.sum(to_categorical(pred, dimcls), axis=0)
    predcls = predcount.argmax(-1)
    match += predcls== Ytraw[i]-1
acc = match/ntins


print('Test accuracy =', acc)
