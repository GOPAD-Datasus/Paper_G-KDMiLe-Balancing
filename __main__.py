from combine.combine import combining
from runner import Runner
from sampling.oversampling.oversampling import oversampling
from sampling.preprocessing import preprocess
from sampling.undersampling.undersampling import undersampling

if __name__ == '__main__':
    #runner = Runner()
    #print(f'{runner._get_base()}')

    undersampling()
    oversampling()
    combining()