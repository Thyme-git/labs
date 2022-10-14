import numpy as np
from tqdm import tqdm
from numba import cuda


# X_test: (28, 28)
# X_train_device (n_trains, 28, 28)
@cuda.jit
def coreFunc(dist_device, X_train_device, n_trains, X_test):
    id = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if id < n_trains:
        d = 0
        for i in range(28):
            for j in range(28):
                d += (X_train_device[id, i, j]-X_test[i, j])**2
        d = d**.5
        dist_device[id] = d

class Knn(object):

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):

        # TODO Predict the label of X by
        # the k nearest neighbors.

        # Input:
        # X: np.array, shape (n_samples, n_features)

        # Output:
        # y: np.array, shape (n_samples,)

        # Hint:
        # 1. Use self.X and self.y to get the training data.
        # 2. Use self.k to get the number of neighbors.
        # 3. Use np.argsort to find the nearest neighbors.

        # YOUR CODE HERE
        # raise NotImplementedError
        ...

        # amount of test set elements
        n_samples = len(X)
        y = np.zeros(n_samples)
        # amount of training set elements
        n_trains = len(self.X)

        X_train_device = cuda.to_device(self.X)
        threads_per_block = 64
        blocks_per_grid = int(n_trains/threads_per_block)+1
        
        print('training start...')
        for i in tqdm(range(n_samples)):
            # L2 distances of sample i to training sets
            # dist = np.zeros(n_trains)
            # for j in range(n_trains):
            #     dist[j] = np.power(np.sum(np.power(X[i]-self.X[j], 2)), .5)
            
            # accelerate with cuda
            dist_device = cuda.device_array(n_trains)
            coreFunc[blocks_per_grid, threads_per_block](dist_device, X_train_device, n_trains, X[i])
            dist = dist_device.copy_to_host()
            retIndices = np.argsort(dist)[:self.k]
            
            dic = {}
            for index in retIndices:
                if self.y[index] not in dic.keys():
                    dic[self.y[index]] = 1
                else:
                    dic[self.y[index]] += 1

            ret = list(dic.keys())[0]
            for k in dic.keys():
                if dic[k] > dic[ret]:
                    ret = k

            y[i] = ret            
            

        return y

        # End of todo
