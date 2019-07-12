import numpy as np
from tqdm import tqdm 


# read data
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
print("==============================================")
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
print("activation5_output :", activation5_output.shape)
print()


class RBFkernel():
    def __init__(self, sigma=0.5):
        self.sigma = sigma
        
    def __call__(self, x, y):
        numerator = -1 * np.linalg.norm(x - y, ord=2)**2
        denominator = 2 * (self.sigma**2)
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
    # gram_matrix = np.zeros((m, m))
    for k in tqdm(range(int(m/100))):

        if k != 0:
            np.save('data/calc_hsic/gram_matrix__'+str(k*100-100)+'-'+str(k*100), gram_matrix)
            del gram_matrix
        gram_matrix = np.zeros((100, m))

        for i in tqdm(range(100)):
            for j in tqdm(range(m)):
                gram_matrix[i][j] = kernel(data[i], data[j])
                # gram_matrix[i][j] = kernel(data[i], data[j])

    # print(gram_matrix_i) # debug code
    # return gram_matrix


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
    H = np.diag(np.full(m, 1 - (1/m)))
    K = gram_matrix(k, X, m)
    L = gram_matrix(l, Y, m)
    HSIC = np.trace(np.dot(K, np.dot(H, np.dot(L, H)))) / ((m - 1)**2)
    return HSIC



# m = 1000
m = 60000
X = in_data[:m]
Y = layer3_output[:m]

kernel = RBFkernel(sigma=75)

gram_matrix(kernel, X, m)
gram_matrix(kernel, Y, m)
# hsic = hsic(kernel, kernel, m, X, Y)

print()
print("----------------------------------------------")
print("HSIC(in_data, layer3_output)", hsic)
