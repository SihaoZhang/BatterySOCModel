# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 17:05:34 2018

@author: Sihao
"""
from scipy import *  
from scipy.linalg import norm, pinv  
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pylab

def load_data(PATH1):
    x = pd.read_excel(PATH1)
    return x

def data_extract(df, smooth):
    x1 = df[['totalVoltage']].values
    x2 = df[['totalCurrent']].values
    t = df[['maxTemperatureBatteryValue','minTemperatureBatteryValue']].values
    y = df['totalSOC'].values
    y = np.expand_dims(y, axis=1)
    t = (t[:,0] + t[:,1])/2
    t = np.expand_dims(t, axis=1)
    x1 = np.hstack((x1,x2))
    x = np.hstack((x1,t))
    x = pd.rolling_mean(x, smooth)
    return x, y

def data_preprocessing(x, y):
    x_std = np.std(x)
    x_mean = np.mean(x)
    y_std = np.std(y)
    y_mean = np.mean(y)
    x = (x - x_mean)/x_std
    y = (y - y_mean)/y_std
    return x_std, x_mean, y_std, y_mean, x, y

def data_split(x, y, start, end):
    xtrain = x[start:end]
    xtr = np.array(xtrain)
    xtest = x[start:end]
    xts = np.array(xtest)
    ytrain = y[start:end]
    ytr = np.array(ytrain)
    ytest = y[start:end]
    yts = np.array(ytest)
    return xtr, xts, ytr, yts

class RBF:  
       
    def __init__(self, indim, numCenters, outdim):  
        self.indim = indim  
        self.outdim = outdim  
        self.numCenters = numCenters  
        self.centers = [random.uniform(-1, 1, indim) for i in range(numCenters)]  
        self.beta = 10
        self.W = random.random((self.numCenters, self.outdim))  
           
    def _basisfunc(self, c, d):  
        assert len(d) == self.indim  
        return exp(-self.beta * norm(c-d)**2)  
       
    def _calcAct(self, X):  
        # calculate activations of RBFs  
        G = zeros((X.shape[0], self.numCenters), float)  
        for ci, c in enumerate(self.centers):  
            for xi, x in enumerate(X):  
                G[xi,ci] = self._basisfunc(c, x)  
        return G  
       
    def train(self, X, Y):  
        # choose random center vectors from training set  
        rnd_idx = random.permutation(X.shape[0])[:self.numCenters]  
        self.centers = [X[i,:] for i in rnd_idx]  
           
        print("center", self.centers)  
        # calculate activations of RBFs  
        G = self._calcAct(X)  
        print(G)  
           
        # calculate output weights (pseudoinverse)  
        self.W = dot(pinv(G), Y)  
           
    def test(self, X):  
           
        G = self._calcAct(X)  
        Y = dot(G, self.W)  
        return Y  
    
def descale(std, mean, data):
    data = data*std + mean
    return data

def Kalman_Filter(x, z, n_iter, Q, R, x_int, P_int, sz):
    # allocate space for arrays
    xhat=np.zeros(sz)      # a posteri estimate of x
    P=np.zeros(sz)         # a posteri error estimate
    xhatminus=np.zeros(sz) # a priori estimate of x
    Pminus=np.zeros(sz)    # a priori error estimate
    K=np.zeros(sz)         # gain or blending factor
    
    xhat[0] = x_int
    P[0] = P_int
    
    for k in range(1,n_iter):
        # time update
        xhatminus[k] = xhat[k-1]  #X(k|k-1) = AX(k-1|k-1) + BU(k) + W(k),A=1,BU(k) = 0
        Pminus[k] = P[k-1]+Q      #P(k|k-1) = AP(k-1|k-1)A' + Q(k) ,A=1
     
        # measurement update
        K[k] = Pminus[k]/( Pminus[k]+R ) #Kg(k)=P(k|k-1)H'/[HP(k|k-1)H' + R],H=1
        xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k]) #X(k|k) = X(k|k-1) + Kg(k)[Z(k) - HX(k|k-1)], H=1
        P[k] = (1-K[k])*Pminus[k] #P(k|k) = (1 - Kg(k)H)P(k|k-1), H=1
    
    return P, Pminus, K, xhat, xhatminus

def plotfig(z, xhat, x, n_iter, Pminus):
    pylab.figure()
#   pylab.plot(z,'k+',label='noisy measurements')     #测量值
 
    pylab.plot(xhat,'b-',label='a posteri estimate')  #过滤后的值
    pylab.plot(x,color='g',label='truth value')    #系统值
    pylab.legend()
    pylab.xlabel('Iteration')
    pylab.ylabel('State of Capacity')
     
    pylab.figure()
    valid_iter = range(1,n_iter) # Pminus not valid at step 0
    pylab.plot(valid_iter,Pminus[valid_iter],label='a priori error estimate')
    pylab.xlabel('Iteration')
    pylab.ylabel('$(Voltage)^2$')
    pylab.setp(pylab.gca(),'ylim',[0,.01])
    pylab.show()

    
if __name__ == '__main__':
    #load data
    PATH1 = 'bms20171222.xlsx'
    data = load_data(PATH1)
    df = pd.DataFrame(data)
    
    #extract data
    smooth = 500 #smoothing range
    x, y = data_extract(df, smooth)
    
    #set trainset and testset
    start, end = 1000, 6500 #valid data range
    xtr, xts, ytr, yts = data_split(x, y, start, end)

    #preprocessing
    xtr_std, xtr_mean, ytr_std, ytr_mean, xtr, ytr = data_preprocessing(xtr, ytr)
    xts_std, xts_mean, yts_std, yts_mean, xts, yts = data_preprocessing(xts, yts)
      
    # rbf regression  
    rbf = RBF(3, 24, 1)  
    rbf.train(xtr, ytr)  
    z = rbf.test(xts)  
    
    #descale
    yh = descale(yts_std, yts_mean, z)
    ys = descale(yts_std, yts_mean, yts)
    
    #kalman_filter
    n_iter = 5500 #iteration time
    sz = (n_iter,) # size of array
    x = ys[:,0] # truth value (typo in example at top of p. 13 calls this z)
#    z = yh[:,0] # observations (normal about x, sigma=0.1)
    z = np.random.normal(yh[:,0],0.1,size=sz)   
    Q = 1e-5 # process variance
    R = 0.2**2 # estimate of measurement variance, change to see effect
    # intial guesses
    x_int = 800
    P_int= 5.0
    P, Pminus, K, xhat, xhatminus = Kalman_Filter(x, z, n_iter, Q, R, x_int, P_int, sz)
    MSE = np.mean((xhat - x)**2)
    print('Mean Square Error is', MSE)
    #figure plot
    plotfig(z, xhat, x, n_iter, Pminus)      


