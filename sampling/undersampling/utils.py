from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, EditedNearestNeighbours

from sampling.metrics import apply_model, get_metrics


def random_undersampler (X_train, X_test, y_train, y_test):
    rus = RandomUnderSampler(sampling_strategy='not minority',
                             random_state=72)

    X_train_res, y_train_res = rus.fit_resample(X_train, y_train)

    y_pred = apply_model(X_train_res, X_test, y_train_res)
    return get_metrics(y_pred, y_test)


def cluster_centroids (X_train, X_test, y_train, y_test):
    # Kmeans : auto voting
    cc = ClusterCentroids(sampling_strategy='not minority',
                          random_state=72)

    X_train_r, y_train_r = cc.fit_resample(X_train, y_train)

    y_pred = apply_model(X_train_r, X_test, y_train_r)
    return get_metrics(y_pred, y_test)


def edited_nearest_neighbours (X_train, X_test, y_train, y_test):
    enn = EditedNearestNeighbours(sampling_strategy='not minority')

    X_train_r, y_train_r = enn.fit_resample(X_train, y_train)

    y_pred = apply_model(X_train_r, X_test, y_train_r)
    return get_metrics(y_pred, y_test)