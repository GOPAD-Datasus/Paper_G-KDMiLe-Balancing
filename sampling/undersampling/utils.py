from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, EditedNearestNeighbours, OneSidedSelection

from runner import Runner


def random_us ():
    model = RandomUnderSampler(sampling_strategy='not minority',
                               random_state=72)

    runner = Runner()
    return runner.pipeline(model)


def edited_nn ():
    model = EditedNearestNeighbours(sampling_strategy='not minority',
                                    n_jobs=-1)

    runner = Runner()
    return runner.pipeline(model)


def one_sided_selection ():
    model = OneSidedSelection(sampling_strategy='majority',
                              random_state=72,
                              n_seeds_S=10,
                              n_jobs=-1)

    runner = Runner()
    return runner.pipeline(model)