from sampling.oversampling.oversampling import oversampling
from sampling.preprocessing import preprocess
from sampling.undersampling.undersampling import undersampling

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = preprocess()

    undersampling(X_train, X_test, y_train, y_test)
    oversampling(X_train, X_test, y_train, y_test)