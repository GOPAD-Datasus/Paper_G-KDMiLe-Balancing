from collections import Counter

import pandas as pd
import xgboost
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import *
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from preprocess import preprocess


class Runner :
    def __init__(self):
        df = preprocess()

        pca = PCA(n_components=1, random_state=72)
        df['mother'] = \
            pca.fit_transform(df[['IDADEMAE', 'ESTCIVMAE',
                                  'ESCMAE2010', 'CODOCUPMAE',
                                  'QTDFILVIVO', 'QTDFILMORT']])
        df['newborn'] = \
            pca.fit_transform(df[['APGAR5', 'RACACOR', 'PESO']])
        df['prenatal'] = \
            pca.fit_transform(df[['CONSULTAS', 'GESTACAO',
                                  'MESPRENAT', 'PARTO']])

        ss = StandardScaler()
        X = pd.DataFrame(ss.fit_transform(df.drop(['OBITO'],
                                                  axis=1)))
        y = df['OBITO']

        (self.Xtrain, self.Xtest,
         self.ytrain, self.ytest) = \
            tts(X, y, test_size=0.7, random_state=72)

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

