# Data pre-processing
import pickle
from scipy.io import loadmat
import cv2
# Packages for plotting
import matplotlib.pyplot as plt
from pca import PCA_
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA


def classify_ATT(pca, X_train, X_test, y_train, y_test, h, w):
    """Classify the AT&T dataset."""
    X_train_pc = pca.transform(X_train)
    X_test_pc = pca.transform(X_test)
    svm = SVC(kernel='rbf', gamma='auto')
    svm.fit(X_train_pc, y_train)
    y_pred = svm.predict(X_test_pc)
    titles = [('True:' + str(y_test[i] - 1) + '\nPredicted:' + str(y_pred[i] - 1)) for i in
              range(y_pred.shape[0])]
    plot_result(X_test, titles, h, w, 2, 4)
    plt.show()


def eigenface(X_train, h, w):
    """Compute PCA of training dataset and return trained PCA"""
    pca = PCA(n_components=20, whiten=True)
    # pca = PCA_(n_components=20, solver='SVD', whiten=True)
    pca.fit(X_train)
    eigenfaces = pca.components_.reshape((20, h, w))
    eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
    plot_result(eigenfaces, eigenface_titles, h, w, 2, 4)
    plt.show()
    return pca


def classify_mydigit(pca, svc):
    """
    Use trained pca and svc to classify my handwriting digits.
    :param pca: trained pca
    :param svc: trained svc
    :return:
    """
    dig = loadmat('./MNIST/dig.mat')
    dig_x = dig['dig']
    dig_x_pc = pca.transform(dig_x)
    pre_y = svc.predict(dig_x_pc)
    plt.figure()
    for i in range(6):
        plt.subplot(1, 6, i+1)
        plt.imshow(dig_x[i].reshape(28, 28), cmap='gray')
        title = 'SVC:' + str(pre_y[i])
        plt.title(title)
        plt.axis('off')
    plt.show()


def classify_mnist(train_x, train_y, test_x, test_y):
    """
    Run this function to classify the MNIST dataset and show part of the
    classification results.
    :param train_x: training dataset
    :param train_y: training label
    :param test_x: test dataset
    :param test_y: test label
    :return:
    """
    # Model fitting
    pca = PCA_()
    pca.fit(train_x)
    train_x_pc = pca.transform(train_x)
    svc = SVC(kernel='rbf', gamma='auto')
    svc.fit(train_x_pc, train_y)

    # # Classify MNIST
    test_x_pc = pca.transform(test_x)
    pre_y_svc = svc.predict(test_x_pc)
    # Showing result
    samples = test_x[:18]
    pre_y = pre_y_svc[:18]
    titles = [('True:' + str(test_y[i]) + '\nSVC:' + str(pre_y[i])) for i in range(18)]
    plot_result(samples, titles, 28, 28, 3, 6)
    plt.show()
    return pca, svc


def load_ATT(path='./faces'):
    """
    Load AT&T dataset.
    :param path: relative path of dataset
    :return X: data
            y: label
            h: height of image
            w: width of image
    """
    all_data_set = list()
    all_data_label = list()
    for i in range(1, 41):
        for j in range(1, 11):
            filename = path + '/s' + str(i) + str(j) + '.pgm'
            img = cv2.imread(filename)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = img_gray.shape
            img_col = img_gray.reshape(h * w)
            all_data_set.append(img_col)
            all_data_label.append(i)
    X = np.array(all_data_set)
    y = np.array(all_data_label)
    return X, y, h, w


def load_mnist(dataset='MNIST/mnist.pkl'):
    """
    Load MNIST dataset
    :param dataset: path of MNIST dataset
    :return: splitted MNIST dataset
    """
    with open(dataset, 'rb') as f:
        return pickle.load(f, encoding="latin1")


def visualization_mnist(train_x, valid_x, valid_y):
    """Visualization the dimension-reduction result"""
    pca = PCA_(n_components=3)
    pca.fit(train_x)
    new_x = pca.transform(valid_x)
    # plot in 3D coordinate
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    sc = ax.scatter(new_x[:, 0], new_x[:, 1], new_x[:, 2], c=valid_y, cmap=plt.cm.get_cmap('Spectral', 10))
    plt.colorbar(sc, ticks=np.array(range(10)))
    plt.show()


def plot_result(images, titles, h, w, n_row=3, n_col=4):
    """Plot result in gallery"""
    plt.figure()
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap='gray')
        plt.title(titles[i])
        plt.axis('off')


