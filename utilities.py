import pickle
import matplotlib.pyplot as plt
from pca import PCA_
import numpy as np
from sklearn.svm import SVC

from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA


def load_mnist(dataset='assert/mnist.pkl'):
    """

    :param dataset:
    :return:
    """
    with open(dataset, 'rb') as f:
        return pickle.load(f, encoding="latin1")


def main():
    train_set, valid_set, test_set = load_mnist()
    train_x, train_y = train_set
    valid_x, valid_y = valid_set
    test_x, test_y = test_set

    # Visualization
    # pca = PCA_(num_comp=3)
    # pca.fit(train_x)
    # new_x = pca.transform(valid_x)
    #
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # sc = ax.scatter(new_x[:, 0], new_x[:, 1], new_x[:, 2], c=valid_y, cmap=plt.cm.get_cmap('Spectral', 10))
    # plt.colorbar(sc, ticks=np.array(range(10)))
    # plt.show()

    # Classification
    pca = PCA_(num_comp=100)
    pca.fit(train_x)
    train_x_pc = pca.transform(train_x)
    test_x_pc = pca.transform(test_x)

    svc = SVC(kernel='rbf', gamma='auto')
    svc.fit(train_x_pc, train_y)
    pre_y_svc = svc.predict(test_x_pc)

    # Showing result
    samples = test_x[:18]
    pre_y = pre_y_svc[:18]
    plt.figure()
    for i in range(18):
        plt.subplot(3, 6, i+1)
        plt.imshow(samples[i].reshape(28,28), cmap='gray')
        title = 'True:' + str(test_y[i]) + '\nSVC:' + str(pre_y[i])
        plt.title(title)
        plt.axis('off')
    plt.show()



if __name__ == '__main__':
    main()
