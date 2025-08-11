from sampling.undersampling.utils import random_us, edited_nn, one_sided_selection


def undersampling ():
    print(f'Random Under Sampler: {random_us()}')

    print(f'Edited Nearest Neighbours: {edited_nn()}')

    print(f'One Sided Selection: {one_sided_selection()}')
