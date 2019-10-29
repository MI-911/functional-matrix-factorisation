from data.data_utils import DataUtil
from os.path import join
from fmf import FunctionalMatrixFactorization




if __name__ == "__main__": 
    FMF = FunctionalMatrixFactorization()
    FMF.load_data(from_file=join('data', 'ratings.csv'))
    FMF.fit()