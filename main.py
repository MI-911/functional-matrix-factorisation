from data.data_utils import load_ratings
from os.path import join




if __name__ == "__main__": 
    samples = load_ratings(from_file=join('data', 'ratings.csv'))
    print(f'Loaded {len(samples)} samples...')