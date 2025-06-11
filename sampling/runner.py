from collections import Counter

import xgboost
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import *

from preprocessing import preprocess


class Runner :
    def __init__(self):
        (self.Xtrain, self.Xtest,
         self.ytrain, self.ytest) = preprocess()


    def _get_base (self):
        print(sorted(Counter(self.ytrain).items()))

        xgb = xgboost.XGBClassifier()
        xgb.fit(self.Xtrain, self.ytrain)

        return self._get_metrics(xgb.predict(self.Xtest))


    def _get_metrics(self, ypred):
        target_names = ['0) Não óbito', '1) Óbito']
        print(classification_report_imbalanced(self.ytest,
                                               ypred,
                                               target_names=target_names))

        return {'acc': accuracy_score(self.ytest, ypred),
                'pre': precision_score(self.ytest, ypred),
                'rec': recall_score(self.ytest, ypred),
                'f1s': f1_score(self.ytest, ypred)}


    def pipeline (self, model):
        Xsample, ysample = model.fit_resample(self.Xtrain,
                                              self.ytrain)

        print(sorted(Counter(ysample).items()))

        xgb = xgboost.XGBClassifier()
        xgb.fit(Xsample, ysample)

        return self._get_metrics(xgb.predict(self.Xtest))

