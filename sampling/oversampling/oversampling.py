from sampling.oversampling.utils import adasyn


def oversampling (X_train, X_test, y_train, y_test):
    res = adasyn(X_train, X_test, y_train, y_test)
    print(f'ADASYN: {res}')

    pass