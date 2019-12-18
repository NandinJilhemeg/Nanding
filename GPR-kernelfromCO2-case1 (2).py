#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels     import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared
# Import training data
from scipy.io import loadmat
trainingdata = loadmat('trainingset.mat',squeeze_me=True)

# Convert data
def convertdata(trainingdata):
    data_col = ["P_DP", "P_IP", "P_WH", "T_WF", "T_WH", "oil_rate"]
    newdata = np.zeros((6, np.shape(trainingdata["P_DP"])[0]))
    i = 0
    for col in data_col:
        newdata[i,:] = trainingdata[col]
        i += 1
    
    return newdata
Data = convertdata(trainingdata)
# X is feature of training data, X(no.samples,no.features)
X = np.c_[Data[0],Data[1],Data[2],Data[3],Data[4]]
# y is target of training data, y(no.samples,no.output dimensions) 
y = Data[5]
# Kernel 
k1 = 150.0**2 * RBF(length_scale=50.0) # seasonal component# long term smooth rising trend
#k2 = 2**2 * RBF(length_scale=100.0) \
#   * ExpSineSquared(length_scale=10, periodicity=1.0,
#                     periodicity_bounds="fixed")  # seasonal component
# medium term irregularities
#k3 = 0.5**2 * RationalQuadratic(length_scale=1.0, alpha=1.0)
#k4 = 0.1**2 * RBF(length_scale=0.1) \
  # + WhiteKernel(noise_level=0.1**2,
      #            noise_level_bounds=(1e-3, np.inf))  # noise terms
kernel = k1 + k2 
gp = GaussianProcessRegressor(kernel=kernel, alpha=2,optimizer=None, normalize_y=True)
gp.fit(X, y)
y_pred,y_cov=gp.predict(X,return_cov=True)
y_pred,y_std=gp.predict(X,return_std=True)
y_mean=np.mean(y)
# Determination coefficient,decribe the relation between X and y  
TSS,ESS = 0,0
for i in range(0,len(y)):
    TSS,ESS = TSS + (y[i] - y_mean)**2,ESS + (y[i] - y_pred[i])**2
R_sq = 1 - (ESS/TSS)
# Time series, x-axis
x = np.linspace(1,24,24)

plt.figure()
# Plotting actual output
plt.plot(x, y, 'r:', label=r'$f(x) = x\,\sin(x)$')
# Plotting observation points
plt.plot(x, y, 'r.', markersize=10, label='Observations')
# Plotting prediction of output
plt.plot(x, y_pred, 'b-', label='Prediction')

plt.fill_between(x, y_pred - y_std*50, y_pred + y_std*50,
                 alpha=0.4,color='k')
# Import test data
from scipy.io import loadmat
testdata = loadmat('testset.mat',squeeze_me=True)
# Convert test data
def converttestdata(testdata):
    data_col = ["P_DP", "P_IP", "P_WH", "T_WF", "T_WH"]
    newtestdata = np.zeros((5, np.shape(testdata["P_DP"])[0]))
    i = 0
    for col in data_col:
        newtestdata[i,:] = testdata[col]
        i += 1
    
    return newtestdata
Datat = converttestdata(testdata)
Xt = np.c_[Datat[0],Datat[1],Datat[2],Datat[3],Datat[4]]
gp = GaussianProcessRegressor(kernel=kernel, alpha=4,optimizer=None, normalize_y=True)
gp.fit(X, y)
y_pred_test,y_cov_test=gp.predict(Xt,return_cov=True)
#y_pred,y_std=gp.predict(Xt,return_std=True)
y_pred_test,y_std_test=gp.predict(Xt,return_std=True)
# Time series, x-axis
x = np.linspace(1,168,24)
x_= np.linspace(1,168,168)

plt.figure()
# Plotting actual output
plt.plot(x, y, 'r:', label=r'actual oil prodction rate')
# Plotting observation points
plt.plot(x, y, 'r.', markersize=10,)
# Plotting prediction of output
plt.plot(x_, y_pred_test, 'b-', label='Prediction of test dataset')

plt.fill_between(x_, y_pred_test - y_std_test*40, y_pred_test + y_std_test*40,
                 alpha=0.4,color='k')
plt.xlabel("Total number of well tests")
plt.ylabel(r"Oil production rate [m3/d]")
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

