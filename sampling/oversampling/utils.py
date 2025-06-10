from imblearn.over_sampling import SMOTENC, ADASYN

from sampling.metrics import apply_model, get_metrics


def smotenc (X_train, X_test, y_train, y_test):
    # sm = SMOTENC()
    pass


def adasyn (X_train, X_test, y_train, y_test):
    ada = ADASYN(sampling_strategy='not majority',
                 random_state=72)

    X_train_r, y_train_r = ada.fit_resample(X_train, y_train)

    y_pred = apply_model(X_train_r, X_test, y_train_r)
    return get_metrics(y_pred, y_test)