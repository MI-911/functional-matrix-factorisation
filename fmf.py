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
            print('Finding optimal profile for LIKE')
            uL, uL_group, uL_items = self.__get_optimal_profile__(i, users, answer=AnswerType.LIKE)
            print('Finding optimal profile for DISLIKE')
            uD, uD_group, uD_items = self.__get_optimal_profile__(i, users, answer=AnswerType.DISLIKE)
            print('Finding optimal profile for UNKNOWN')
            uU, uU_group, uU_items = self.__get_optimal_profile__(i, users, answer=AnswerType.UNKNOWN)

            # Calculate loss for each group 
            print('Calculating loss')
            i_loss = 0 
            for uX, uX_group, uX_items in [(uL, uL_group, uL_items), 
                                           (uD, uD_group, uD_items), 
                                           (uU, uU_group, uU_items)]: 
                for u in uX_group: 
                    for i in uX_items: 
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
        items_in_group = self._D.get_items_in_user_group(group, answer=answer)

        coefficient_matrix = tt.zeros((self.k, self.k))
        coefficient_vector = tt.zeros((1, self.k))
        for u in group: 
            for i in items_in_group: 
                coefficient_matrix += self.I[i] @ self.I[i].T 
                coefficient_vector += self._D.M[u][i] * self.I[i] 

        coefficient_matrix = tt.inverse(coefficient_matrix)
        return coefficient_matrix @ coefficient_vector, group, items_in_group


    def fit(self): 
        self.__build__(len(self._D.items), len(self._D.users))

        has_converged = False 
        while not has_converged: 
            # 1. Fit a decision tree 
            print(f'Building tree...')
            tree = self.__build_node__(users=self._D.users, node=Tree(profile=None), current_depth=0, max_depth=self.D)
            # 2. Fit item embeddings
            break
          