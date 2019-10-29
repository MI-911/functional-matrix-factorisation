import torch as tt 
from data.data_utils import DataUtil, AnswerType
from os.path import join
import math


class Tree(): 

    def __init__(self, profile=None): 
        self.item = None 
        self.children = None 
        self.profile = profile  


class FunctionalMatrixFactorization(): 

    def __init__(self, interview_length=None, k=10):
        '''
        Initializes a FunctionalMatrixFactorization object with the following settings: 

        interview_length: int
            The depth of the decision tree used to conduct the interview 

        k: int 
            The number of latent factors in the embeddings for users and items 
        ''' 
        self.D = interview_length
        self.k = k

    
    def load_data(self, from_file=None): 
        self._D = DataUtil()
        self._D.load(file=from_file)

    def __build__(self, n_items, n_users): 
        self.I = tt.rand((n_items, self.k))
        self.U = tt.rand((n_users, self.k))

    def __build_node__(self, users=None, node=None, current_depth=None, max_depth=None): 
        print(f'Building node at depth {current_depth}')
        split_item = None 
        smallest_err = math.inf
        for i in self._D.items: 
            uU, uU_group = self.__get_optimal_profile__(i, users, answer=AnswerType.UNKNOWN)
            uL, uL_group = self.__get_optimal_profile__(i, users, answer=AnswerType.LIKE)
            uD, uD_group = self.__get_optimal_profile__(i, users, answer=AnswerType.DISLIKE)

            # Calculate loss for each group 
            print('Calculating loss')
            i_loss = 0 
            for uX, uX_group, uX_items in [(uL, uL_group), 
                                           (uD, uD_group), 
                                           (uU, uU_group)]: 
                for u in uX_group: 
                    for i in self._D.get_rated_items(u): 
                        i_loss += (self._D.M[u][i] - uL @ self.I[i]) ** 2

                if i_loss < smallest_err: 
                    split_item = i 
                    smallest_err = i_loss

        # We are assigned a node with a profile, now add the item and children
        node.item = split_item 
        uL_node, uD_node, uU_node = (Tree(profile=uX) for uX in [uL, uD, uU])
        node.children = { answer_type : uX_node for answer_type, uX_node in [(AnswerType.LIKE, uL_node), 
                                                                            (AnswerType.DISLIKE, uD_node), 
                                                                            (AnswerType.UNKNOWN, uU_node)]}
        # If the tree isn't deep enough yet, keep finding items to split on                                                                     
        if current_depth <= max_depth: 
            for uX_group, uX, uX_node in [(uL_group, uL_node), 
                                          (uD_group, uD_node), 
                                          (uU_group, uU_node)]: 
                self.__build_node__(users=uX_group, node=uX_node, current_depth=current_depth + 1, max_depth=max_depth)
        
        return node


    def __get_optimal_profile__(self, item, users, answer=None): 
        group = self._D.get_user_group(users=users, item=item, answer=answer)
        n_users = len(group)
        coefficient_matrix = tt.zeros((self.k, self.k))
        coefficient_vector = tt.zeros((1, self.k))
        for i_u, u in enumerate(group): 
            for i in self._D.get_rated_items(u):
                print(f'Coeff matrix: {(i_u / n_users) * 100 : 2.2f}%', end='\r')
                i_vector = self.I[i].reshape((1, self.k))
                coefficient_matrix += i_vector.T @ i_vector
                coefficient_vector += self._D.M[u][i] * self.I[i] 

        print(coefficient_matrix)
        print(coefficient_matrix.shape)
        input()
        coefficient_matrix = tt.inverse(coefficient_matrix)
        return coefficient_matrix @ coefficient_vector, group


    def fit(self): 
        self.__build__(len(self._D.items), len(self._D.users))

        has_converged = False 
        while not has_converged: 
            # 1. Fit a decision tree 
            print(f'Building tree...')
            tree = self.__build_node__(users=self._D.users, node=Tree(profile=None), current_depth=0, max_depth=self.D)
            # 2. Fit item embeddings
            break
          