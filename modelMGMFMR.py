from __future__ import division

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from skimage.transform import resize as imresize

from tensorflow.keras import layers
from tensorflow.keras.layers import Reshape, MaxPooling2D, Flatten, UpSampling2D
from tensorflow.keras.layers import Concatenate, Activation
from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Lambda, Multiply, Add
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LSTM, Masking
from tensorflow.keras import regularizers



def getlayeridx(model, layerName):
    index = None
    for idx, layer in enumerate(model.layers):
        if layer.name == layerName:
            index = idx
            break
    return index



def klgauss(mu,logvar,var,mu0,var0,nbatch,dim1):
    term0 = -0.5*nbatch*dim1
    term1 = 0.5*tf.reduce_sum(np.log(var0) - logvar)
    term2 = 0.5*tf.reduce_sum((var + (mu - mu0)**2) / var0)
    sumkl = term1 + term2 + term0
    return sumkl


def klgauss01(mu,logvar,var,nbatch,dim1):
    term0 = -0.5*nbatch*dim1
    term1 = 0.5*tf.reduce_sum(- logvar)
    term2 = 0.5*tf.reduce_sum(var + mu**2)
    sumkl = term1 + term2 + term0
    return sumkl


def mcgauss(mu,logvar,nbatch,dim1):
    std = tf.exp(0.5*logvar)
    eps = tf.random.truncated_normal((nbatch, dim1))
    mc = tf.add(mu, tf.multiply(eps, std))
    return mc



class mmdFixT(layers.Layer):
    
    def __init__(self, lossw, name='name'):
        super(mmdFixT, self).__init__(name=name)
        self.lossw = lossw
        
    def build(self, inshape):
        self.dim = inshape[-1]
        self.T = inshape[-2]
        
    def call(self, xs):
        
        x = xs[0,:]
        rows = tf.tile(tf.expand_dims(x,axis=1), [1,self.T,1])
        cols = tf.tile(tf.expand_dims(x,axis=0), [self.T,1,1])
        term1 = tf.reduce_mean(tf.exp(tf.reduce_sum(tf.square(rows-cols), axis=-1)))
        term2 = 1
        term3 = tf.reduce_mean(tf.exp(tf.reduce_sum(tf.square(x),axis=-1)))
        self.add_loss(self.lossw*(term1+term2+term3))
        
        return xs



def maemask(datas, params):
    
    data = datas
    dims = params[0]
    nbatch = params[1]
    kldivide = params[2]
    dimcls = params[3]
    Tmax = params[4]
    reg = regularizers.l2(1e-5)
    
    maskval = -8
    
    maska = Masking(mask_value=maskval,input_shape=(Tmax, dims[0]),name='maska')(data[0])
    
    maskb = Masking(mask_value=maskval,input_shape=(Tmax, dims[1]),name='maskb')(data[1])
    
    maskc = Masking(mask_value=maskval,input_shape=(Tmax, dims[2]),name='maskc')(data[2])
    
    z1a = LSTM(dims[3],return_sequences=True,kernel_regularizer=reg,bias_regularizer=reg,name='z1a')(maska)
    
    z1b = LSTM(dims[4],return_sequences=True,kernel_regularizer=reg,bias_regularizer=reg,name='z1b')(maskb)
    
    z1c = LSTM(dims[5],return_sequences=True,kernel_regularizer=reg,bias_regularizer=reg,name='z1c')(maskc)
    
    cat1 = Concatenate(name='cat1')([z1a,z1b,z1c])
    
    z1 = LSTM(dims[6],return_sequences=True,kernel_regularizer=reg,bias_regularizer=reg,name='z1')(cat1)
    
    fy = LSTM(dims[7],return_sequences=True,kernel_regularizer=reg,bias_regularizer=reg,name='fy')(z1)
    
    cat2a = Concatenate(name='cat2a')([z1a,fy])
    
    cat2b = Concatenate(name='cat2b')([z1b,fy])
    
    cat2c = Concatenate(name='cat2c')([z1c,fy])
    
    f2a = LSTM(dims[8],return_sequences=True,kernel_regularizer=reg,bias_regularizer=reg,name='f2a')(cat2a)
    
    f2b = LSTM(dims[9],return_sequences=True,kernel_regularizer=reg,bias_regularizer=reg,name='f2b')(cat2b)
    
    f2c = LSTM(dims[10],return_sequences=True,kernel_regularizer=reg,bias_regularizer=reg,name='f2c')(cat2c)
    
    ra = Dense(dims[0], kernel_regularizer=reg,bias_regularizer=reg,name='ra')(f2a)
    
    rb = Dense(dims[1], kernel_regularizer=reg,bias_regularizer=reg,name='rb')(f2b)
    
    rc = Dense(dims[2], kernel_regularizer=reg,bias_regularizer=reg,name='rc')(f2c)
    
    ry = Dense(dimcls, activation='softmax', kernel_regularizer=reg,bias_regularizer=reg,name='ry')(fy)
    
    return [ra,rb,rc,ry,z1a,z1b,z1c,z1]


class Sample(layers.Layer):
    
    def __init__(self, nbatch, dim1, kldivide, priormu=0, priorvar=1, name='layername'):
        super(Sample, self).__init__(name=name)
        self.nbatch = nbatch
        self.dim1 = dim1
        self.mu0 = priormu
        self.var0 = priorvar
        self.kldivide = kldivide
        
    def call(self, x):
        mu = x[0]
        logvar = x[1]
        std = tf.exp(0.5*logvar)
        eps = tf.random.truncated_normal((self.nbatch, self.dim1))
        mc = tf.add(mu, tf.multiply(eps, std))
        
        term0 = -0.5*self.dim1
        term1 = 0.5*tf.reduce_sum(np.log(self.var0) - logvar)
        term2 = 0.5*tf.reduce_sum((tf.exp(logvar) + (mu - self.mu0)**2) / self.var0)
        sumkl = term1 + term2 + term0
        self.add_loss(sumkl / self.kldivide)
        
        return mc



class Sample2(layers.Layer):
    
    def __init__(self, nbatch, dim1, kldivide, priormu=0, priorvar=1, name='layername'):
        super(Sample2, self).__init__(name=name)
        self.mu0 = priormu
        self.var0 = priorvar
        self.kldivide = kldivide
        
    def build(self, inshape):
        self.nbatch = inshape[0][0]
        self.dim1 = inshape[0][1]
        
    def call(self, x):
        mu1 = x[0]
        logvar1 = x[1]
        var1 = tf.exp(logvar1)
        mu2 = x[2]
        logvar2 = x[3]
        var2 = tf.exp(logvar2)

        varinv1 = tf.math.reciprocal(var1)
        varinv2 = tf.math.reciprocal(var2)
        var3 = tf.math.reciprocal(varinv1 + varinv2 + 1)
        mu3 = (mu1*varinv1 + mu2*varinv2)*var3
        logvar3 = tf.math.log(var3)

        var1z = tf.math.reciprocal(varinv1 + 1)
        mu1z = (mu1*varinv1)*var1z
        logvar1z = tf.math.log(var1z)
        var2z = tf.math.reciprocal(varinv2 + 1)
        mu2z = (mu2*varinv2)*var2z
        logvar2z = tf.math.log(var2z)

        mc1 = mcgauss(mu1z,logvar1z,self.nbatch,self.dim1)
        mc2 = mcgauss(mu2z,logvar2z,self.nbatch,self.dim1)
        mc3 = mcgauss(mu3,logvar3,self.nbatch,self.dim1)
        
        kl1 = klgauss01(mu1z,logvar1z,var1z,self.nbatch,self.dim1)
        kl2 = klgauss01(mu2z,logvar2z,var2z,self.nbatch,self.dim1)
        kl3 = klgauss01(mu3,logvar3,var3,self.nbatch,self.dim1)
        self.add_loss((kl1+kl2+kl3) / self.kldivide)
        
        return [mc1,mc2,mc3]
    



class Sample3(layers.Layer):
    
    def __init__(self, nbatch, dim1, kldivide, priormu=0, priorvar=1, name='layername'):
        super(Sample3, self).__init__(name=name)
        self.mu0 = priormu
        self.var0 = priorvar
        self.kldivide = kldivide
        
    def build(self, inshape):
        self.nbatch = inshape[0][0]
        self.dim1 = inshape[0][1]
        
    def call(self, x):
        mu1 = x[0]
        logvar1 = x[1]
        var1 = tf.exp(logvar1)
        mu2 = x[2]
        logvar2 = x[3]
        var2 = tf.exp(logvar2)

        varinv1 = tf.math.reciprocal(var1)
        varinv2 = tf.math.reciprocal(var2)
        var3 = tf.math.reciprocal(varinv1 + varinv2 + 1)
        mu3 = (mu1*varinv1 + mu2*varinv2)*var3
        logvar3 = tf.math.log(var3)

        var1z = tf.math.reciprocal(varinv1 + 1)
        mu1z = (mu1*varinv1)*var1z
        logvar1z = tf.math.log(var1z)
        var2z = tf.math.reciprocal(varinv2 + 1)
        mu2z = (mu2*varinv2)*var2z
        logvar2z = tf.math.log(var2z)

        mc1 = mcgauss(mu1z,logvar1z,self.nbatch,self.dim1)
        mc2 = mcgauss(mu2z,logvar2z,self.nbatch,self.dim1)
        mc3 = mcgauss(mu3,logvar3,self.nbatch,self.dim1)
        
        kl1 = klgauss01(mu1z,logvar1z,var1z,self.nbatch,self.dim1)
        kl2 = klgauss01(mu2z,logvar2z,var2z,self.nbatch,self.dim1)
        kl3 = klgauss01(mu3,logvar3,var3,self.nbatch,self.dim1)
        self.add_loss((kl1+kl2+kl3) / self.kldivide)
        
        return [mc1,mc2,mc3,mu3]




class Linear2(layers.Layer):
    
    def __init__(self, nbatch,dim1,l2, name):
        super(Linear2, self).__init__(name=name)
        self.nbatch = nbatch
        self.dim1 = dim1
        self.l2 = l2
        
    def build(self, inshape):
        indim = inshape[0][-1]
        w_init = tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.01)
        b_init = tf.keras.initializers.Constant(0.)
        self.w = self.add_weight(shape=(indim, self.dim1),
            initializer=w_init, trainable=True, name="w")
        self.b = self.add_weight(shape=(self.dim1,), initializer=b_init, trainable=True, name="b")
    
    def call(self, x):
        out1 = tf.matmul(x[0], self.w) + self.b
        out2 = tf.matmul(x[1], self.w) + self.b
        regloss = tf.reduce_sum(tf.square(self.w)) + tf.reduce_sum(tf.square(self.b))
        self.add_loss(self.l2 * regloss)
        return [out1, out2]

    
    
class Linear2relu(layers.Layer):
    
    def __init__(self, nbatch,dim1,l2, name):
        super(Linear2relu, self).__init__(name=name)
        self.nbatch = nbatch
        self.dim1 = dim1
        self.l2 = l2
        
    def build(self, inshape):
        indim = inshape[0][-1]
        w_init = tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.01)
        b_init = tf.keras.initializers.Constant(0.)
        self.w = self.add_weight(shape=(indim, self.dim1),
            initializer=w_init, trainable=True, name="w")
        self.b = self.add_weight(shape=(self.dim1,), initializer=b_init, trainable=True, name="b")
    
    def call(self, x):
        out1 = tf.matmul(x[0], self.w) + self.b
        out2 = tf.matmul(x[1], self.w) + self.b
        regloss = tf.reduce_sum(tf.square(self.w)) + tf.reduce_sum(tf.square(self.b))
        self.add_loss(self.l2 * regloss)
        return [tf.nn.relu(out1), tf.nn.relu(out2)]


class Sample4(layers.Layer):
    
    def __init__(self, nbatch, dim1, kldivide, priormu=0, priorvar=1, name='layername'):
        super(Sample4, self).__init__(name=name)
        self.mu0 = priormu
        self.var0 = priorvar
        self.kldivide = kldivide
        
    def build(self, inshape):
        self.nbatch = inshape[0][0]
        self.dim1 = inshape[0][1]
        
    def call(self, x):
        mu1 = x[0]
        logvar1 = x[1]
        var1 = tf.exp(logvar1)
        mu2 = x[2]
        logvar2 = x[3]
        var2 = tf.exp(logvar2)
        mu4 = x[4]
        logvar4 = x[5]
        var4 = tf.exp(logvar4)
        

        varinv1 = tf.math.reciprocal(var1)
        varinv2 = tf.math.reciprocal(var2)
        varinv4 = tf.math.reciprocal(var4)
        var3 = tf.math.reciprocal(varinv1 + varinv2 + varinv4 + 1)
        mu3 = (mu1*varinv1 + mu2*varinv2 + mu4*varinv4)*var3
        logvar3 = tf.math.log(var3)

        var1z = tf.math.reciprocal(varinv1 + 1)
        mu1z = (mu1*varinv1)*var1z
        logvar1z = tf.math.log(var1z)
        var2z = tf.math.reciprocal(varinv2 + 1)
        mu2z = (mu2*varinv2)*var2z
        logvar2z = tf.math.log(var2z)
        var4z = tf.math.reciprocal(varinv4 + 1)
        mu4z = (mu4*varinv4)*var4z
        logvar4z = tf.math.log(var4z)

        mc1 = mcgauss(mu1z,logvar1z,self.nbatch,self.dim1)
        mc2 = mcgauss(mu2z,logvar2z,self.nbatch,self.dim1)
        mc4 = mcgauss(mu4z,logvar4z,self.nbatch,self.dim1)
        mc3 = mcgauss(mu3,logvar3,self.nbatch,self.dim1)
        
        kl1 = klgauss01(mu1z,logvar1z,var1z,self.nbatch,self.dim1)
        kl2 = klgauss01(mu2z,logvar2z,var2z,self.nbatch,self.dim1)
        kl4 = klgauss01(mu4z,logvar4z,var4z,self.nbatch,self.dim1)
        kl3 = klgauss01(mu3,logvar3,var3,self.nbatch,self.dim1)
        self.add_loss((kl1+kl2+kl3+kl4) / self.kldivide)
        
        return [mc1,mc2,mc4,mc3,mu3]



def vaelightclassify(datas, params):
    data = datas
    dims = params[0]
    nbatch = params[1]
    kldivide = params[2]
    dimcls = params[3]
    reg = regularizers.l2(1e-5)
    mmt = 0.9
    
    b1a = Dense(dims[3], activation='relu',kernel_regularizer=reg,bias_regularizer=reg,name='b1a')(data[0])
    
    b1b = Dense(dims[4], activation='relu',kernel_regularizer=reg,bias_regularizer=reg,name='b1b')(data[1])
    
    b1c = Dense(dims[5], activation='relu',kernel_regularizer=reg,bias_regularizer=reg,name='b1c')(data[2])
    
    b2a = Dense(dims[6], activation='relu',kernel_regularizer=reg,bias_regularizer=reg,name='b2a')(b1a)
    
    b2b = Dense(dims[6], activation='relu',kernel_regularizer=reg,bias_regularizer=reg,name='b2b')(b1b)
    
    b2c = Dense(dims[6], activation='relu',kernel_regularizer=reg,bias_regularizer=reg,name='b2c')(b1c)
    
    mua = Dense(dims[6], activation=None, kernel_regularizer=reg,bias_regularizer=reg,name='mua')(b2a)
    
    vara = Dense(dims[6], activation=None, kernel_regularizer=reg,bias_regularizer=reg,name='vara')(b2a)
    
    mub = Dense(dims[6], activation=None, kernel_regularizer=reg,bias_regularizer=reg,name='mub')(b2b)
    
    varb = Dense(dims[6], activation=None, kernel_regularizer=reg,bias_regularizer=reg,name='varb')(b2b)
    
    muc = Dense(dims[6], activation=None, kernel_regularizer=reg,bias_regularizer=reg,name='muc')(b2c)
    
    varc = Dense(dims[6], activation=None, kernel_regularizer=reg,bias_regularizer=reg,name='varc')(b2c)
    
    sample = Sample4(nbatch, dims[6], kldivide, priormu=0, priorvar=1, name='sample')([mua,vara,mub,varb,muc,varc])

    b3a = Linear2relu(nbatch,dims[3], 1e-5,name='b3a')([sample[0],sample[3]])
    
    b3b = Linear2relu(nbatch,dims[4], 1e-5,name='b3b')([sample[1],sample[3]])
    
    b3c = Linear2relu(nbatch,dims[5], 1e-5,name='b3c')([sample[2],sample[3]])
    
    b4a = Linear2(nbatch,dims[0], 1e-5,name='b4a')(b3a)
    
    b4b = Linear2(nbatch,dims[1], 1e-5,name='b4b')(b3b)
    
    b4c = Linear2(nbatch,dims[2], 1e-5,name='b4c')(b3c)
    
    da1 = Lambda(lambda x: x[0], name='da1') (b4a)
    
    da2 = Lambda(lambda x: x[1], name='da2') (b4a)
    
    db1 = Lambda(lambda x: x[0], name='db1') (b4b)
    
    db2 = Lambda(lambda x: x[1], name='db2') (b4b)
    
    dc1 = Lambda(lambda x: x[0], name='dc1') (b4c)
    
    dc2 = Lambda(lambda x: x[1], name='dc2') (b4c)
    
    dc = Dense(dimcls, activation='softmax',kernel_regularizer=reg,bias_regularizer=reg,name='dc')(sample[4])
    
    return [da1,da2,db1,db2,dc1,dc2, dc]




def resizeTrainData(N,h1,w1,raw):
    images = np.zeros((N, h1,w1, 3), dtype=np.float32)
    for ii in range(N):
        img = raw[ii]
        r_img = imresize(img, (h1,w1))
        images[ii, :] = np.copy(r_img)
    return images


def reduceinitialization(m,divisor):
    ws = m.get_weights()
    wsnew = []
    for i in range(len(ws)):
        temp = ws[i]/divisor
        wsnew.append(temp)
    m.set_weights(wsnew)
    return m


def reducevarencoded(m,reduce):
    idx = getlayeridx(m, 'd2var')
    weight = m.layers[idx].get_weights()
    m.layers[idx].set_weights([weight[0],weight[1]-reduce])
    return m


def getEmbedding(m, img, h1, w1):
    resized = resizeTrainData(1,h1,w1,img)
    checks = m.predict(resized,  batch_size=1)
    muq = checks[-2][0]
    logvarq = checks[-1][0]
    return [muq, logvarq]


def retrieveZ(mulist, logvarlist, muq, varq, threshold):
    minkl = 1000*threshold
    dim1 = muq.shape[-1]
    
    distancelist = np.zeros((len(mulist),))
    
    for i in range(len(mulist)):
        mup = mulist[i]
        varp = logvarlist[i]
        
        term0 = -0.5*dim1
        term1 = 0.5*np.sum(varp - varq)
        term2 = 0.5*np.sum((np.exp(varq) + (muq - mup)**2) / np.exp(varp))
        sumkl = term1 + term2 + term0
        
        distancelist[i] = sumkl
        
        if sumkl<minkl:
            result = i
            minkl = sumkl
    if minkl>threshold:
        result = -1
    return [result, distancelist]
