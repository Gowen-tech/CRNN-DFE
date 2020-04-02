#coding:utf8
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from scipy import signal
np.random.seed(123)

test = np.loadtxt('../test.txt')     #32768 接收端  qiqi
send = np.loadtxt('../send.txt')     # 2 * 4096  iqiq

# #plot the dataset
# fig = plt.figure()
# ax1 = fig.add_subplot(3,1,1)
# ax1.scatter(send[0],send[1])
# ax2 = fig.add_subplot(3,1,2)
# ax2.scatter(test[1:100:2],test[:100:2])

x = np.array([complex(x,y) for (x,y) in zip(test[1::2], test[::2])])

TxS = np.hstack((send,send,send,send)).T   # 发送端发出的数据 16384 * 2

L = 12
chL = 5
EqD = int(round((L+chL)/2))   #8
X = []
for i in range(16384-L):
    X.append(x[i:i+L+1])
X = np.array(X).T   # (16000, 13)

c = np.zeros( (1,L+1) );
R_inverse = 100*np.eye(L+1)

Tx = np.array([complex(a,b) for (a,b) in zip(TxS[:,0],TxS[:,1])])
for k in range(16000):
    e = Tx[k+10+L-EqD] - c.dot( X[:,k+10]);
    print e
    filtered_infrmn_vect = R_inverse.dot(X[:,k+10]);  # (13,1)
    norm_error_power = np.conj(X[:,k+10].T).dot(filtered_infrmn_vect);
    gain_constant = 1 / (1 + norm_error_power);
    norm_filtered_infrmn_vect = gain_constant * np.conj(filtered_infrmn_vect.T);
    c = c + e * norm_filtered_infrmn_vect;
    R_inverse = R_inverse - np.conj(norm_filtered_infrmn_vect.reshape((13,1))).dot(norm_filtered_infrmn_vect.reshape((1,13)));

sb = np.dot(c,X)


fig = plt.figure()
ax1 = fig.add_subplot(2,2,1,title=u"发送序列")
ax1.scatter(Tx.real, Tx.imag)
ax2 = fig.add_subplot(2,2,2,title=u"接收序列")
ax2.scatter(x.real, x.imag)
ax3 = fig.add_subplot(2,2,3,title=u"恢复序列")
ax3.scatter(sb.real, sb.imag)
plt.show()
