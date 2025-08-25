from sampling import combine, oversampling, undersampling, Runner


if __name__ == '__main__':
    runner = Runner()
    print(f'{runner.get_base()}')

    undersampling()
    oversampling()
    combine()
