#!/usr/bin/env python
# coding: utf-8

import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# ### Load intermediate layer output
in_data = np.load('data/dnn_base/in_data.npy')
layer1_output = np.load('data/dnn_base/layer1_output.npy')
layer2_output = np.load('data/dnn_base/layer2_output.npy')
layer3_output = np.load('data/dnn_base/layer3_output.npy')
layer4_output = np.load('data/dnn_base/layer4_output.npy')
layer5_output = np.load('data/dnn_base/layer5_output.npy')
activation1_output = np.load('data/dnn_base/activation1_output.npy')
activation2_output = np.load('data/dnn_base/activation2_output.npy')
activation3_output = np.load('data/dnn_base/activation3_output.npy')
activation4_output = np.load('data/dnn_base/activation4_output.npy')
activation5_output = np.load('data/dnn_base/activation5_output.npy')

print()
print("Model intermediate layer output")
print("==========================")
print()
print("in_data :", in_data.shape)
print("----------------------------------------------")
print("layer1_output :", layer1_output.shape)
print("----------------------------------------------")
print("activation1_output :", activation1_output.shape)
print("----------------------------------------------")
print("layer2_output :", layer2_output.shape)
print("----------------------------------------------")
print("activation2_output :", activation2_output.shape)
print("----------------------------------------------")
print("layer3_output :", layer3_output.shape)
print("----------------------------------------------")
print("activation3_output :", activation3_output.shape)
print("----------------------------------------------")
print("layer4_output :", layer4_output.shape)
print("----------------------------------------------")
print("activation4_output :", activation4_output.shape)
print("----------------------------------------------")
print("layer5_output :", layer5_output.shape)
print("----------------------------------------------")
print("activation5_output :", activation5_output.shape) # <------- softmax func

# ### Kernel Tools

class RBFkernel():
    def __init__(self, sigma=0.5):
        self.sigma = sigma
        
    def __call__(self, x, y):
        numerator = -1 * np.linalg.norm(x - y, ord=2)**2
        denominator = self.sigma ** 2
        return np.exp(numerator / denominator)
    
    def get_params(self):
        return self.sigma
    
    def set_params(self, sigma):
        self.sigma = sigma
                
            
def gram_matrix(kernel, data, m):
    """
    Arguments:
    =========
    - kernel : kernel function 
    - data : data samples, shape=(m, dim(data_i))
    - m : number of samples
    """
    gram_matrix = np.zeros((m, m))
    for i in tqdm(range(m)):
        for j in range(m):
            gram_matrix[i][j] = kernel(data[i], data[j])
            
    return gram_matrix


def hsic(k, l, m, X, Y):
    """
    Arguments:
    =========
    - k : kernel function for X
    - l : kernel function for Y
    - m : number of samples
    - X : data samples, shape=(m, dim(X_i))
    - Y : data samples, shape=(m, dim(Y_i))
    """
    H = np.full((m, m), -(1/m)) + np.eye(m)
    K = gram_matrix(k, X, m)
    print("Gram(X) :", K, "\nGram(X) mean :", K.mean())
    L = gram_matrix(l, Y, m)
    print("Gram(Y) :", L, "\nGram(Y) mean :", L.mean())
    HSIC = np.trace(np.dot(K, np.dot(H, np.dot(L, H)))) / ((m - 1)**2)
    return HSIC


# **sample code**
# 
# ```python
# kernel = RBFkernel(sigma=1)
# kernel(np.array([1,2,3]), np.array([1,2,6]))
# ```
# Output: 0.011108996538242306

# ### Measuring HSIC

# In_data vs layer1_output
# 40 min

def calc_hsic(X, Y, X_name=None, Y_name=None, m=10000, sigma=1):

    print(X_name+".shape :", X.shape)
    print(Y_name+".shape :", Y.shape)

    kernel_x = RBFkernel(sigma=sigma)
    kernel_y = RBFkernel(sigma=sigma * np.sqrt(X.shape[1]/Y.shape[1]))

    hsic_value = hsic(kernel_x, kernel_y, m, X, Y)
    print()
    print("-------------------------------------------------")
    print("HSIC("+X_name+", "+Y_name+") =", hsic_value)
    print("m :", m)
    print("sigma :", sigma)
    print("-------------------------------------------------")
    print()


X=activation5_output
X_name="iact5"

calc_hsic(X, in_data, X_name=X_name, Y_name="in_data")
calc_hsic(X, layer1_output, X_name=X_name, Y_name="lay1")
calc_hsic(X, activation1_output, X_name=X_name, Y_name="act1")
calc_hsic(X, layer2_output, X_name=X_name, Y_name="lay2")
calc_hsic(X, activation2_output, X_name=X_name, Y_name="act2")
calc_hsic(X, layer3_output, X_name=X_name, Y_name="lay3")
calc_hsic(X, activation3_output, X_name=X_name, Y_name="act3")
calc_hsic(X, layer4_output, X_name=X_name, Y_name="lay4")
calc_hsic(X, activation4_output, X_name=X_name, Y_name="act4")
calc_hsic(X, layer5_output, X_name=X_name, Y_name="lay5")
calc_hsic(X, activation5_output, X_name="lay1", Y_name="act5")
