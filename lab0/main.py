import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from knn import Knn


def load_mnist(root='/home/thyme/homework/dian团队招新/hello-dian.ai/lab0/mnist'):

    # TODO Load the MNIST dataset
    # 1. Download the MNIST dataset from
    #    http://yann.lecun.com/exdb/mnist/
    # 2. Unzip the MNIST dataset into the
    #    mnist directory.
    # 3. Load the MNIST dataset into the
    #    X_train, y_train, X_test, y_test
    #    variables.

    # Input:
    # root: str, the directory of mnist

    # Output:
    # X_train: np.array, shape (6e4, 28, 28)
    # y_train: np.array, shape (6e4,)
    # X_test: np.array, shape (1e4, 28, 28)
    # y_test: np.array, shape (1e4,)

    # Hint:
    # 1. Use np.fromfile to load the MNIST dataset(notice offset).
    # 2. Use np.reshape to reshape the MNIST dataset.

    # YOUR CODE HERE
    # raise NotImplementedError
    ...


    # names of training set and test set
    X_train_name = 'train-images-idx3-ubyte'
    y_train_name = 'train-labels-idx1-ubyte'
    X_test_name = 't10k-images-idx3-ubyte'
    y_test_name = 't10k-labels-idx1-ubyte'

    # read files
    # set offset to skip the begin of images/labels data
    X_train = np.fromfile(root+'/'+X_train_name, dtype=np.uint8, offset=16)
    y_train = np.fromfile(root+'/'+y_train_name, dtype=np.uint8, offset=8)
    X_test = np.fromfile(root+'/'+X_test_name, dtype=np.uint8, offset=16)
    y_test = np.fromfile(root+'/'+y_test_name, dtype=np.uint8, offset=8)

    # reshape
    X_train = np.reshape(X_train, (int(6e4), 28, 28))
    X_test = np.reshape(X_test, (int(1e4), 28, 28))

    return X_train, y_train, X_test, y_test
    # End of todo


def main():
    X_train, y_train, X_test, y_test = load_mnist()

    knn = Knn()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    correct = sum((y_test - y_pred) == 0)

    print('==> correct:', correct)
    print('==> total:', len(X_test))
    print('==> acc:', correct / len(X_test))

    # plot pred samples
    fig, ax = plt.subplots(nrows=4, ncols=5, sharex='all', sharey='all')
    fig.suptitle('Plot predicted samples')
    ax = ax.flatten()
    for i in range(20):
        img = X_test[i]
        ax[i].set_title(y_pred[i])
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
