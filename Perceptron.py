# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from matplotlib import pyplot as plt

#随机生成数
np.random.seed(12)
num_observations = 500
x1 = np.random.multivariate_normal([0,0],[[1,.75],[.75,1]],num_observations)
x2 = np.random.multivariate_normal([1,4],[[1,.75],[.75,1]],num_observations)

X= np.vstack((x1,x2)).astype(np.float32)
y= np.hstack((-np.ones(num_observations), np.ones(num_observations)))#y置为-1和1，前500为-1


#感知机函数
def perceptron_sgd_plot(X, Y):     
    
    '''
    train perceptron and plot the total loss in each epoch.          
    :param X: data samples     
    :param Y: data labels     
    :return: weight vector as a numpy array     
    '''
    
    w = np.zeros(len(X[0]))
    
    #调整eta和n参数，结果会有变化，主要就是调整这两个参数     
    eta = 0.7
    n = 30     
    errors = [] 
 
    for t in range(n):         
        total_error = 0         
        for i, x in enumerate(X):             
            if (np.dot(X[i], w)*Y[i]) <= 0:                 
                total_error += (np.dot(X[i], w)*Y[i])                 
                w = w + eta*X[i]*Y[i]
        errors.append(total_error*-1)          
    #plt.plot(errors)     
    plt.xlabel('Epoch')     
    plt.ylabel('Total Loss')          
    return w 
w=perceptron_sgd_plot(X,y)
print(w)


for d, sample in enumerate(X):
# Plot the negative samples
    if d < 500:
        plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2,color='red')
# Plot the positive samples
    else:
        plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2,color='green')


# Print the hyperplane calculated by perceptron_sgd() 画超平面

x2=[w[0],w[1],-w[1],w[0]] 
x3=[w[0],w[1],w[1],-w[0]] 
 
x2x3 =np.array([x2,x3]) 
X,Y,U,V = zip(*x2x3) 
ax = plt.gca() 
ax.quiver(X,Y,U,V,scale=1, color='blue') 

