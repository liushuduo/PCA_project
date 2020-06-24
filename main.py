from utilities import *
from sklearn.model_selection import train_test_split


def PCA_MNIST():
    """Run this function for analysis of MNIST dataset."""
    # Load dataset
    train_set, valid_set, test_set = load_mnist()
    train_x, train_y = train_set
    valid_x, valid_y = valid_set
    test_x, test_y = test_set

    # Visualization mnist in 3D coordinate.
    visualization_mnist(train_x, valid_x, valid_y)

    # Classify MNIST and showing classified result
    pca, svc = classify_mnist(train_x, train_y, test_x, test_y)

    # Classify my handwriting digits and show classification result
    classify_mydigit(pca, svc)


def PCA_ATT():
    """Run this function for analysis of """
    # Load dataset
    X, y, h, w = load_ATT()
    # Split data into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Display eigenface, compute PCA.
    pca = eigenface(X_train, h, w)

    # Use computed pca to train SVM and show the result.
    classify_ATT(pca, X_train, X_test, y_train, y_test, h, w)


if __name__ == '__main__':
    # Run PCA for AT&T dataset
    PCA_ATT()
    # Run PCA for MNIST dataset
    # PCA_MNIST()
