import pandas as pd 
import torch as tt 
import json 


ANSWER_THRESHOLD = 3

class AnswerTypes(): 
    LIKE    = 1
    DISLIKE = -1
    UNKNOWN = 0


class DataUtil(): 

    def __init__(self): 
        pass 

    def __unpack_row__(self, row): 
        row = row[-1]
        return list(map(int, [row['userId'], row['movieId'], row['rating'], row['timestamp']]))


    def __load_ratings__(self, from_file=None): 
        with open(from_file) as fp: 
            df = pd.read_csv(fp)
            return [self.__unpack_row__(row) for row in df.iterrows()]

    
    def load_interaction_matrix(self, from_file=None):
        ratings =  self.__load_ratings__(from_file=from_file)
        uid_map, u_count = {}, 0
        mid_map, m_count = {}, 0

        for u, m, r, t in ratings: 
            if u not in uid_map: 
                uid_map[u] = u_count
                u_count += 1
            if m not in mid_map: 
                mid_map[m] = m_count
                m_count += 1

        M = tt.zeros((u_count, m_count))
        for u, m, r, t in ratings: 
            u = uid_map[u]
            m = mid_map[m]
            M[u][m] = -1 if r <= ANSWER_THRESHOLD else 1 

        self.M = M




