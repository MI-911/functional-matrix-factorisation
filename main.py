from data.data_utils import load_ratings
from os.path import join

# Given a rating, the answer should be 
# <= ANSWER_THRESHOLD --> DISLIKE 
#  > ANSWER_THRESHOLD --> LIKE 
# ... otherwise       --> UNKNOWN
ANSWER_THRESHOLD = 3


if __name__ == "__main__": 
    samples = load_ratings(from_file=join('data', 'ratings.csv'))
    print(f'Loaded {len(samples)} samples...')