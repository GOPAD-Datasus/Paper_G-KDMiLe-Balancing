from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import SMOTE

from runner import Runner


def smoteenn ():
    enn = EditedNearestNeighbours(sampling_strategy='not minority')

    smote = SMOTE(sampling_strategy='minority',
                  random_state=72,
                  k_neighbors=7)

    model = SMOTEENN(sampling_strategy='all',
                     random_state=72,
                     enn=enn,
                     smote=smote)

    runner = Runner()
    return runner.pipeline(model)