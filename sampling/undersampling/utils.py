from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, EditedNearestNeighbours, OneSidedSelection

from runner import Runner


def random_us ():
    model = RandomUnderSampler(sampling_strategy='not minority',
                               random_state=72)

    runner = Runner()
    return runner.pipeline(model)


def cluster_centroids (X_train, X_test, y_train, y_test):
    # Kmeans : auto voting
    cc = ClusterCentroids(sampling_strategy='not minority',
                          random_state=72)

    X_train_r, y_train_r = cc.fit_resample(X_train, y_train)

    y_pred = apply_model(X_train_r, X_test, y_train_r)
    return get_metrics(y_pred, y_test)


def edited_nn ():
    model = EditedNearestNeighbours(sampling_strategy='not minority')

    runner = Runner()
    return runner.pipeline(model)


def one_sided_selection ():
    model = OneSidedSelection(sampling_strategy='majority',
                              random_state=72,
                              n_seeds_S=10)

    runner = Runner()
    return runner.pipeline(model)