from imblearn.over_sampling import SMOTENC, ADASYN, RandomOverSampler

from runner import Runner


def random_os ():
    model = RandomOverSampler(sampling_strategy='minority',
                              random_state=72,
                              shrinkage=0.1)

    runner = Runner()
    return runner.pipeline(model)


def smotenc ():
    # sm = SMOTENC()
    pass


def adasyn ():
    model = ADASYN(sampling_strategy='not majority',
                   random_state=72)

    runner = Runner()
    return runner.pipeline(model)