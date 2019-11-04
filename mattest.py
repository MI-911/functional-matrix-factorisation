import torch as tt
import random
from functools import reduce

def get_rated_movies(R=None, user_group=None):
    # Returns, for every user, a list of indices for rated movies
    # e.g. [
    #        [1, 2]
    #        [0, 3, 9],
    #        [1, 3]
    #      ]
    group_matrix = R[[user_group]]
    return [
        [i for i, rating in enumerate(user_ratings) if rating > 0]
        for user_ratings in group_matrix
    ]


def get_embedding_rating_pairs(users, R, movie_embeddings):
    embeddings = []
    ratings = []
    for u in users:
        for m in R[u].nonzero():
            embeddings.append(movie_embeddings[m].tolist())
            ratings.append(R[u][m].item())

    embeddings = tt.Tensor(embeddings)
    embeddings = embeddings.reshape(embeddings.shape[0], embeddings.shape[-1])

    ratings = tt.Tensor(ratings).reshape(len(ratings), 1)

    return embeddings, ratings


if __name__ == '__main__':
    n_users = 500
    n_movies = 50
    k = 5
    regularisation = 0.0015

    R = tt.randint(0, 5, (n_users, n_movies)).type_as(tt.rand(1))
    U = tt.rand((n_users, k))
    M = tt.rand((n_movies, k))
    R_map = tt.where(R > 0, tt.ones_like(R), tt.zeros_like(R))

    user_group = random.sample(range(n_users), 150)
    rated_movies = get_rated_movies(R=R, user_group=user_group)



    coeff_matrix = tt.zeros((k, k))
    coeff_vector = tt.zeros((1, k))

    for u, movies in zip(user_group, rated_movies):
        for m in movies:
            v = M[m].reshape((1, k))  # [5] --> [1, 5] so we can transpose to [5, 1]

            coeff_matrix += v.T @ v  # + tt.eye(k) * regularisation
            coeff_vector += v * R[u][m]

    coeff_matrix = coeff_matrix.inverse()

    print(coeff_matrix)
    print(coeff_vector)



    # Matrix attempt
    # 1. Extract one large matrix of all (duplicated) movie embeddings in order of their
    # ratings. Do the same for ratings
    embeddings, ratings = get_embedding_rating_pairs(user_group, R, M)

    coeff_matrix = (embeddings.T @ embeddings).inverse()
    coeff_vector = (embeddings * ratings).sum(dim=0)

    print(coeff_matrix)
    print(coeff_vector)

    print(coeff_matrix @ coeff_vector)

