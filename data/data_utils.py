import pandas as pd 
import torch as tt 
import json 


ANSWER_THRESHOLD = 3

class AnswerType(): 
    LIKE    = 1
    DISLIKE = 2
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

    
    def load(self, file=None):
        ratings =  self.__load_ratings__(from_file=file)
        uid_map, uidx_map, u_count = {}, {}, 0
        mid_map, midx_map, m_count = {}, {}, 0


        for u, m, r, t in ratings: 
            if u not in uid_map: 
                uid_map[u] = u_count
                uidx_map[u_count] = u
                u_count += 1
            if m not in mid_map: 
                mid_map[m] = m_count
                midx_map[m_count] = m
                m_count += 1

        M = tt.zeros((u_count, m_count))
        for u, m, r, t in ratings: 
            u = uid_map[u]
            m = mid_map[m]
            M[u][m] = r

        self.M = M
        self.answer_map = tt.where(M > 0, tt.ones_like(M), tt.zeros_like(M))
        self.users = uid_map.values()
        self.items = mid_map.values()
        self._uid_map = uid_map
        self._mid_map = mid_map
        self._uidx_map = uidx_map
        self._midx_map = midx_map

    def get_answer(self, u, i): 
        r = self.M[u][i]
        if not r: 
            return AnswerType.UNKNOWN
        else: 
            return self.__answer_from_rating__(r)

    def __answer_from_rating__(self, r): 
        return AnswerType.LIKE if r > ANSWER_THRESHOLD else AnswerType.DISLIKE

    def get_user_group(self, users=None, item=None, answer=None): 
        return [u_idx for u_idx in users if self.get_answer(u_idx, item) == answer]


    def get_rated_items(self, user): 
        return [i_idx for i_idx in self.items if not self.get_answer(user, i_idx) == AnswerType.UNKNOWN]
    
    def get_items_in_user_group(self, users, answer=None): 
        rated_items = []
        for u_idx in users: 
            u_ratings = self.M[u_idx]
            u_rated_items = u_ratings.where(u_ratings > 0, tt.zeros_like(u_ratings)).nonzero()
            rated_items += u_rated_items.tolist()

        return set(rated_items)
        
