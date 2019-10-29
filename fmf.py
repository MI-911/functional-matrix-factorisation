import torch as tt 
from data.data_utils import DataUtil, AnswerType
from os.path import join
from pathos.multiprocessing import ProcessingPool as Pool
import math
from functools import reduce

class SplitStatistics(): 

    def __init__(self, item=None, loss=None, unknown_profile=None, liked_profile=None, disliked_profile=None): 
        self.item = item 
        self.loss = loss 
        self.unknown_profile = unknown_profile
        self.liked_profile = liked_profile
        self.disliked_profile = disliked_profile


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
        self.max_cpu_count = 4

    
    def load_data(self, from_file=None): 
        self._D = DataUtil()
        self._D.load(file=from_file)

    def __build__(self, n_items, n_users): 
        self.I = tt.rand((n_items, self.k))
        self.U = tt.rand((n_users, self.k))

    def __get_chunks__(self, l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def __build_node__(self, users=None, node=None, current_depth=None, max_depth=None): 
        def __get_split_loss__(item, users):
            uU, uU_group, U_loss = self.__get_optimal_profile_v2__(item, users, answer=AnswerType.UNKNOWN)
            uL, uL_group, L_loss = self.__get_optimal_profile_v2__(item, users, answer=AnswerType.LIKE)
            uD, uD_group, D_loss = self.__get_optimal_profile_v2__(item, users, answer=AnswerType.DISLIKE)

            print(f'Calculated split loss for item {item}')

            return SplitStatistics(item=item, loss=U_loss + L_loss + D_loss, unknown_profile=uU, liked_profile=uL, disliked_profile=uD)

        print(f'Building node at depth {current_depth}')

        n_items = len(self.I)
        users = list(users)

        split_inputs = [(item, users) for item in range(n_items)]
        
        with Pool(processes=self.max_cpu_count) as p: 
            items_with_loss = p.map(__get_split_loss__, split_inputs)

        print(f'Processed all items...')
        items_with_loss = sorted(items_with_loss, key=lambda x: x['loss'])
        o = items_with_loss[0]
        split_item, uU, uL, uD, split_loss = o['item'], o['unknown'], o['like'], o['dislike'], o['loss']
        print(f'Best split is on item w. index {split_item} with loss {split_loss}')

        node.item = split_item 
        uL_node, uD_node, uU_node = (Tree(profile=uX) for uX in [uL, uD, uU])
        node.children = { answer_type : uX_node for answer_type, uX_node in [(AnswerType.LIKE, uL_node), 
                                                                            (AnswerType.DISLIKE, uD_node), 
                                                                            (AnswerType.UNKNOWN, uU_node)]}
        # If the tree isn't deep enough yet, keep finding items to split on
        uL_group = self._D.get_user_group(users=users, item=split_item, answer=AnswerType.LIKE) 
        uD_group = self._D.get_user_group(users=users, item=split_item, answer=AnswerType.DISLIKE)   
        uU_group = self._D.get_user_group(users=users, item=split_item, answer=AnswerType.UNKNOWN)          

        if current_depth <= max_depth: 
            for uX_group, uX_node in [(uL_group, uL_node), 
                                          (uD_group, uD_node), 
                                          (uU_group, uU_node)]: 
                self.__build_node__(users=uX_group, node=uX_node, current_depth=current_depth + 1, max_depth=max_depth)
        
        return node

    
    
        


    def __get_optimal_profile_v2__(self, item, users, answer=None): 
        # Get the indeces of users in this user group, then the indeces 
        # of items rated by those users
        user_indeces = self._D.get_user_group(users=users, item=item, answer=answer)

        # Extract the item embeddings for the item indeces and extract
        # the user-item submatrix for our user group and item set
        I = self.I
        R = self._D.M[user_indeces]
        R_map = self._D.answer_map[user_indeces]

        # Calculate the optimal user embedding for every user in the user group
        # and return the summed embedding
        u = (I.T @ I).inverse() @ (I.T @ R.T).sum(dim=1)

        U = tt.ones((len(user_indeces), self.k)) * u
        P = (U @ I.T) * R_map  # Nullify any prediction where there isn't a rating
        loss = (P - R) ** 2
        return u, user_indeces, loss.sum()


    def fit(self): 
        self.__build__(len(self._D.items), len(self._D.users))

        has_converged = False 
        while not has_converged: 
            # 1. Fit a decision tree 
            print(f'Building tree...')
            tree = self.__build_node__(users=range(len(self.U)), node=Tree(profile=None), current_depth=0, max_depth=self.D)
            # 2. Fit item embeddings
            break