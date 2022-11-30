

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
        if self.epochs % 50 == 1:     
            print("epoch {}, loss {:3.2f}={:3.2f}+{:3.2f}+{:3.2f}+{:3.2f}+{:3.2f}".format(
                self.epochs, logs["loss"], logs["da1_loss"], logs["da2_loss"],
                logs["db1_loss"], logs["db2_loss"], logs["dc_loss"])
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


def passweights(m1, m2):
    ws = m1.get_weights()
    m2.set_weights(ws)
    return m2


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

Nbatch = 5
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
dimh = 64


Tmax = X2raw.shape[-1] -3
nins = int(Yraw.shape[0])

meta = np.zeros((nins,))
pos = 0
for i in range(nins):
    Tnow = int(X1raw[i*f1,2])
    meta[i] = Tnow
nobs = int(np.sum(meta))

X1 = np.zeros((nobs,f1))
X2 = np.zeros((nobs,f1))
X3 = np.zeros((nobs,f2))
Y = np.zeros((nobs,))
Y1hot = np.zeros((nobs,dimcls))
pos = 0
for i in range(nins):
    Tnow = int(meta[i])
    clsnow = int(Yraw[i]-1)
    X1[pos:pos+Tnow,:] = np.copy(np.transpose(X1raw[i*f1:(i+1)*f1,3:3+Tnow]))
    X2[pos:pos+Tnow,:] = np.copy(np.transpose(X2raw[i*f1:(i+1)*f1,3:3+Tnow]))
    X3[pos:pos+Tnow,:] = np.copy(np.transpose(X3raw[i*f2:(i+1)*f2,3:3+Tnow]))
    Y[pos:pos+Tnow] = clsnow
    Y1hot[pos:pos+Tnow, clsnow] = 1
    pos += Tnow


remainder = np.mod(nobs,Nbatch)
nrows = X1.shape[0] - remainder
X1tr = X1[0:nrows,:]
X2tr = X2[0:nrows,:]
X3tr = X3[0:nrows,:]
Y1hot = Y1hot[0:nrows,:]

mse = tf.keras.metrics.mean_squared_error
catce = tf.keras.metrics.categorical_crossentropy




# model specification


from modelMGMFMR import vaelightclassify as vae
dims = [f1,f1,f2,dimh,dimh,dimh,dimh,dimh]
kldivide = Nbatch*dimh*2
params = [dims,Nbatch, kldivide, dimcls]
w1 = 1.
w2 = 1/f1
w3 = 1.

x1 = Input(batch_shape=(Nbatch, dims[0]), name='inputx1') 
x2 = Input(batch_shape=(Nbatch, dims[1]), name='inputx2') 
x3 = Input(batch_shape=(Nbatch, dims[2]), name='inputx3') 


m = Model(inputs=[x1,x2,x3], 
          outputs=vae([x1,x2,x3], params))

m.compile(loss={'da1': mse,'da2': mse, 'db1': mse,'db2': mse,
                'dc1': mse,'dc2': mse, 'dc': catce},
          loss_weights={'da1': w1,'da2': w1, 'db1': w1,'db2': w1,
                'dc1': w2,'dc2': w2,'dc':w3},
          optimizer=adam,
          metrics={'da1':'mse','db1':'mse','dc1':'mse','dc':'mse'}) 



# training


mode = 0
if mode==0:
    
    m.fit([X1tr,X2tr,X3tr],
          {'da1': X1tr,'da2': X1tr, 'db1': X2tr,'db2': X2tr,
           'dc1': X3tr,'dc2': X3tr,'dc': Y1hot},
          batch_size=Nbatch, 
          epochs=1000,
          verbose=verbose,
          callbacks=[CustomCallbackVAE()])

    





m.load_weights('mgm.h5')


x01 = Input(batch_shape=(1, dims[0]), name='inputx1') 
x02 = Input(batch_shape=(1, dims[1]), name='inputx2')
x03 = Input(batch_shape=(1, dims[2]), name='inputx3') 

m0 = Model(inputs=[x01,x02,x03],
          outputs=vae([x01,x02,x03], params))

m0 = passweights(m,m0)



# load test data


X1traw = np.loadtxt(open('Data2/gaze2.csv',"rb"),delimiter=",",skiprows=0)
X2traw = np.loadtxt(open('Data2/paths2.csv',"rb"),delimiter=",",skiprows=0)
X3traw = np.loadtxt(open('Data2/tmd3.csv',"rb"),delimiter=",",skiprows=0)
Ytraw = np.loadtxt(open('Data2/Cat2b.csv',"rb"),delimiter=",",skiprows=0)

ntins = int(Ytraw.shape[0])

metat = np.zeros((ntins,))
pos = 0
for i in range(ntins):
    Tnow = int(X1traw[i*f1,2])
    metat[i] = Tnow
nobst = int(np.sum(metat))

X1t = np.zeros((nobst,f1))
X2t = np.zeros((nobst,f1))
X3t = np.zeros((nobst,f2))
Yt = np.zeros((nobst,))

pos = 0
for i in range(ntins):
    Tnow = int(metat[i])
    clsnow = int(Ytraw[i]-1)
    X1t[pos:pos+Tnow,:] = np.copy(np.transpose(X1traw[i*f1:(i+1)*f1,3:3+Tnow]))
    X2t[pos:pos+Tnow,:] = np.copy(np.transpose(X2traw[i*f1:(i+1)*f1,3:3+Tnow]))
    X3t[pos:pos+Tnow,:] = np.copy(np.transpose(X3traw[i*f2:(i+1)*f2,3:3+Tnow]))
    Yt[pos:pos+Tnow] = clsnow
    
    pos += Tnow


# inference


encodest = m0.predict([X1t,X2t,X3t])
preds = encodest[-1]

match = 0
pos = 0
for i in range(ntins):
    Tnow = int(metat[i])
    pred = preds[pos:pos+Tnow].argmax(-1)
    predcount = np.sum(to_categorical(pred, dimcls), axis=0)
    predcls = predcount.argmax(-1)
    match += predcls== Ytraw[i]-1
    pos+=Tnow
acc = match/ntins


print('Test accuracy =', acc)