import pandas as pd 
import json 


def unpack_row(row): 
    row = row[-1]
    return list(map(int, [row['userId'], row['movieId'], row['rating'], row['timestamp']]))


def load_ratings(from_file=None): 
    with open(from_file) as fp: 
        df = pd.read_csv(fp)
        return [unpack_row(row) for row in df.iterrows()]


