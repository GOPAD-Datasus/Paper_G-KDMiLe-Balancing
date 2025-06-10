import xgboost
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def apply_model (X_train, X_test, y_train):
    xgb = xgboost.XGBClassifier()
    xgb.fit(X_train, y_train)

    # y_pred
    return xgb.predict(X_test)


def get_metrics (y_pred, y_test):
    return {'acc': accuracy_score(y_test, y_pred),
            'pre': precision_score(y_test, y_pred),
            'rec': recall_score(y_test, y_pred),
            'f1s': f1_score(y_test, y_pred)}