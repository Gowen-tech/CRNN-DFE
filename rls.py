#coding:utf8
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time

# np.random.seed(123)



# #构造数据
# M = 30000 # num of data samples
# T = 20000 # number of training
# dB = 20;
# L = 12
# ch = [0.0410+0.0109j,0.0495+0.0123j,0.0672+0.0170j,0.0919+0.0235j, 0.7920+0.1281j,0.3960+0.0871j,0.2715+0.0498j,0.2291+0.0414j,0.1287+0.0154j, 0.1032+0.0119j]
# ch = ch/np.linalg.norm(ch)
# EqD = int(round((L+10)/2))
# Tx = np.sign(np.random.rand(M) * 2 - 1) + 1j*np.sign(np.random.rand(M) * 2 - 1) #30000
# x = signal.fftconvolve(ch,Tx)[:M]   #信道卷积 x.shape = (30000,)
# n=np.random.randn(1,M)+1j*np.random.randn(1,M);
# n=n/np.linalg.norm(n)*pow(10,(-dB/20))*np.linalg.norm(x);
# x = x + n
# x = x.ravel()

# fig = plt.figure()
# ax1 = fig.add_subplot(3,1,1)
# ax1.scatter(Tx.real, Tx.imag)
# ax2 = fig.add_subplot(3,1,2)
# ax2.scatter(x.real, x.imag)
# #plt.show()
def RLS(X, Tx, x, L, chL, T):
    EqD = int(round((L+chL)/2))
    print EqD
    c = np.zeros( (1,L+1) )
    R_inverse = 100*np.eye(L+1)


    for k in range(T-10):
        #print k+10+L-1-EqD,k,T-10
        e = Tx[k+10] - c.dot(X[k+10,:]);
        filtered_infrmn_vect = R_inverse.dot(X[k+10,:]);  # (13,1)
        norm_error_power = np.conj(X[k+10,:].T).dot(filtered_infrmn_vect);
        gain_constant = 1 / (1 + norm_error_power);
        norm_filtered_infrmn_vect = gain_constant * np.conj(filtered_infrmn_vect.T);
        c = c + e * norm_filtered_infrmn_vect;
        R_inverse = R_inverse - np.conj(norm_filtered_infrmn_vect.reshape((13,1))).dot(norm_filtered_infrmn_vect.reshape((1,13)));
    # 恢复
    
    #print np.where((sb - Tx)!=0)[0]

# fig = plt.figure()
# ax1 = fig.add_subplot(2,2,1)
# ax1.scatter(Tx.real, Tx.imag)
# ax2 = fig.add_subplot(2,2,2)
# ax2.scatter(x.real, x.imag)
# ax3 = fig.add_subplot(2,2,3)
# ax3.scatter(sb.real, sb.imag)
# plt.show()
# X = np.array(X).T  #(13,2969)

# c = np.zeros( (1,L+1) );
# R_inverse = 100*np.eye(L+1)


# for k in range(1):

#     e = Tx[k+10+L-EqD] - c.dot(X[:,k+10])


#     filtered_infrmn_vect = R_inverse.dot(X[:,k+10])
#     print filtered_infrmn_vect.shape
#     # print  X[:,k+10].shape
#     # break
#     norm_error_power = X[:,k+10].T.dot(filtered_infrmn_vect)
#     print norm_error_power
#     gain_constant = 1 / (1 + norm_error_power)

#     norm_filtered_infrmn_vect = gain_constant * filtered_infrmn_vect.T
#     c = c + e * norm_filtered_infrmn_vect.reshape((1,L+1));

#     R_inverse = R_inverse - norm_filtered_infrmn_vect.reshape((L+1,1)).dot(norm_filtered_infrmn_vect.reshape((1,L+1)))
# # sb = np.dot(c,X)
# # print sb
