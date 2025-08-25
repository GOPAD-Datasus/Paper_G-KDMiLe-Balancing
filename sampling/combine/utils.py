from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import EditedNearestNeighbours as ENN, TomekLinks
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import NearestNeighbors

from sampling.runner import Runner


def smoteenn():
    enn = ENN(sampling_strategy='not minority')

    smote = SMOTE(sampling_strategy='minority',
                  random_state=72,
                  k_neighbors=7)

    model = SMOTEENN(sampling_strategy='all',
                     random_state=72,
                     enn=enn,
                     smote=smote)

    runner = Runner()
    return runner.pipeline(model)


def smotetomek():
    knn = NearestNeighbors(n_neighbors=3,
                           algorithm="ball_tree")

    tomek = TomekLinks(sampling_strategy='majority',
                       n_jobs=-1)

    smote = SMOTE(sampling_strategy='minority',
                  random_state=72,
                  k_neighbors=knn)

    model = SMOTETomek(random_state=72,
                       smote=smote,
                       tomek=tomek)

    runner = Runner()
    return runner.pipeline(model)
