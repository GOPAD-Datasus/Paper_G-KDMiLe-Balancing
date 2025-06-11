from sampling.undersampling.utils import *


def undersampling ():
    print(f'Random Under Sampler: {random_us()}')

    # complexidade alta
    # res = cluster_centroids(X_train, X_test, y_train, y_test)
    # print(f'Cluster Centroid: {res}')

    print(f'Edited Nearest Neighbours: {edited_nn()}')

    print(f'One Sided Selection: {one_sided_selection()}')

    pass