import torch as tt 
from data.data_utils import DataUtil, AnswerType
from os.path import join
from pathos.multiprocessing import ProcessingPool as Pool
import math
from functools import reduce
from joblib import Parallel, delayed


def unwrap_self(args, **kwargs):
    # Manually unpack the 'self' argument for parallel execution of class method
    return FunctionalMatrixFactorization.__get_split_loss__(*args, **kwargs) 


class Tree(): 

    def __init__(self, profile=None, backend=None): 
        self.item = None 
        self.children = [] 
        self.backend = backend
        self.profile = profile  

    def conduct(self, user): 
        if not self.children: 
            return self.profile 
        
        next_question = {
            answer_type : child_node
            for answer_type, child_node 
            in zip([AnswerType.LIKE, AnswerType.DISLIKE, AnswerType.UNKNOWN], self.children)
        }[self.backend._D.get_answer(user, self.item)]

        return next_question.conduct(user)

    def size(self): 
        if not self.children: 
            return 1
        return 1 + sum([n.size() for n in self.children])
         


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
        self.max_cpu_count = 6
        self.item_regularisation = 0.0015
        self.user_regularisation = 0.0015

    
    def load_data(self, from_file=None): 
        self._D = DataUtil()
        self._D.load(file=from_file)

    def __build__(self, n_items, n_users): 
        self.I = tt.rand((n_items, self.k))
        self.U = tt.rand((n_users, self.k))

    def __get_chunks__(self, l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def __get_split_loss__(self, item, users):
        uU, uU_group, U_loss = self.__get_optimal_profile_v2__(item, users, answer=AnswerType.UNKNOWN)
        uL, uL_group, L_loss = self.__get_optimal_profile_v2__(item, users, answer=AnswerType.LIKE)
        uD, uD_group, D_loss = self.__get_optimal_profile_v2__(item, users, answer=AnswerType.DISLIKE)

        print(f'Splitting... ({(item / len(self.I)) * 100 : 2.2f}%) ', end='\r')
        return item, (U_loss + L_loss + D_loss), uU, uL, uD
    
    def __parallelized_split__(self, inputs): 
        return Parallel(n_jobs=self.max_cpu_count, backend='threading')(delayed(unwrap_self)(o) for o in inputs)

    def __build_node__(self, users=None, node=None, current_depth=None, max_depth=None): 
        item_indeces = range(len(self.I))
        splits = self.__parallelized_split__((self, item, users) for item in item_indeces)

        splits = sorted(splits, key=lambda o: o[1])
        split_item, split_loss, uU, uL, uD = splits[0]

        node.item = split_item 
        uL_node, uD_node, uU_node = (Tree(profile=uX, backend=self) for uX in [uL, uD, uU])
        node.children = { answer_type : uX_node for answer_type, uX_node in [(AnswerType.LIKE, uL_node), 
                                                                            (AnswerType.DISLIKE, uD_node), 
                                                                            (AnswerType.UNKNOWN, uU_node)]}
        # If the tree isn't deep enough yet, keep finding items to split on
        uL_group = self._D.get_user_group(users=users, item=split_item, answer=AnswerType.LIKE) 
        uD_group = self._D.get_user_group(users=users, item=split_item, answer=AnswerType.DISLIKE)   
        uU_group = self._D.get_user_group(users=users, item=split_item, answer=AnswerType.UNKNOWN)          
        self.c_n_nodes += 1
        print(f'Built node at depth {current_depth} ({self.c_n_nodes} out of {self.n_nodes} nodes constructed)')
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

    
    def __update_user_embeddings__(self): 
        for user in len(self.U): 
            self.U[user] = self.interview.conduct(user)


    def __update_item_embeddings__(self): 
        coefficients = tt.zeros((self.k, self.k))
        coeff_vector = tt.zeros((1, self.k))
        for user, Ta in enumerate(self.U): 
            coefficients += Ta @ Ta.T + (tt.eye((self.k, self.k)) * self.item_regularisation)
            for item in self._D.get_rated_items(user): 
                coeff_vector += Ta * self._D.M[user][item] 

        return coefficients.inverse() @ coeff_vector


    def evaluate(self, i): 
        loss = tt.sqrt((((self.U @ self.I.T) - self._D.M) ** 2).sum())
        print(f'RMSE: {loss} (epoch {i})') 

    def fit(self): 
        self.__build__(len(self._D.items), len(self._D.users))

        has_converged = False 
        self.interview = None
        n = 1
        while not has_converged: 
            # 1. Fit a decision tree 
            self.n_nodes = int((3 ** (self.D + 1) - 1) / 2)
            self.c_n_nodes = 0
            print(f'Building tree with {self.n_nodes} nodes...')
            self.interview = self.__build_node__(users=range(len(self.U)), node=Tree(profile=None, backend=self), current_depth=0, max_depth=self.D)
            print(f'Updating user embeddings...')
            self.__update_user_embeddings__()
            # 2. Fit item embeddings
            print(f'Updating item embeddings...')
            self.__update_item_embeddings__()
            self.evaluate(n)
            n += 1
