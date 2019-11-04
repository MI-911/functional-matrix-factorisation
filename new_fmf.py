import torch as tt
import pandas as pd
from os.path import join
import math
import time
from datetime import timedelta


def load_ratings(from_file=None):
    def unpack_row(row):
        row = row[-1]
        return list(map(int, [row['userId'], row['movieId'], row['rating'], row['timestamp']]))

    with open(from_file) as fp:
        df = pd.read_csv(fp)
        return [unpack_row(row) for row in df.iterrows()]


def get_rating_matrix(from_file=None):
    ratings = load_ratings(from_file=from_file)
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

    R = tt.zeros((u_count, m_count))
    for u, m, r, t in ratings:
        u = uid_map[u]
        m = mid_map[m]
        R[u][m] = r

    answer_map = tt.where(R > 0, tt.ones_like(R), tt.zeros_like(R))

    return R, uid_map, mid_map, answer_map


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


def split_users(users=None, movie=None, R=None):
    RL, RD, RU = [], [], []
    for u in users:
        if R[u][movie] == 0:
            RU.append(u)
        elif R[u][movie] > 3:
            RL.append(u)
        else:
            RD.append(u)

    return RL, RD, RU


class Node:

    def __init__(self, users=None, FMF=None, profile=None):
        self.users = users
        self.FMF = FMF
        self.optimal_profile = profile

        self.movie = None

        # Child nodes
        self.like = None
        self.dislike = None
        self.unknown = None

    def interview(self, user_answers):
        # Assuming the user_answers is a dict from movie indices to
        # ratings, we recursively call the child nodes until a leaf node is met,
        # at which point we return the profile.
        if self.is_leaf():
            return self.optimal_profile

        rating = user_answers[self.movie]
        if rating == 3:
            return self.unknown.interview(user_answers)
        elif rating > 3:
            return self.like.interview(user_answers)
        else:
            return self.dislike.interview(user_answers)

    def is_leaf(self):
        return not (self.like or self.dislike or self.unknown)

    def split(self, movies=None):
        # For every movie, split the users and calculate the
        # optimal user profiles for each user group.

        # For every item, remember the loss incurred with these profiles.
        # Choose the item with the lowest loss as the split item, and
        # return the liked, disliked and unknown user groups along with
        # their optimal profiles for the FMF class to build the tree further.

        best_loss = math.inf
        best_movie = None
        best_profiles = None

        for i, movie in enumerate(movies):
            start_time = time.time()

            RL, RD, RU = split_users(users=self.users, movie=movie, R=self.FMF.R)
            (uL, uL_loss), (uD, uD_loss), (uU, uU_loss) = (self.profile(user_group=Rx) for Rx in [RL, RD, RU])
            loss = uL_loss + uD_loss + uU_loss
            if loss < best_loss:
                best_loss = loss
                best_movie = movie
                best_profiles = (RL, uL), (RD, uD), (RU, uU)

            end_time = time.time()
            elapsed_time = end_time - start_time
            time_remaining = (elapsed_time * (len(movies) - i))
            print(f'Splitting... {(i / len(movies)) * 100 : 2.2f}% (ETA in {timedelta(seconds=time_remaining)})')

        self.movie = best_movie
        return best_profiles

    def profile_old(self, user_group=None):
        # NOTE: Deprecated, use profile instead. This is just here for
        # reflection in the paper/report
        rated_movies = get_rated_movies(R=self.FMF.R, user_group=user_group)

        coeff_matrix = tt.zeros((self.FMF.k, self.FMF.k))
        coeff_vector = tt.zeros((1, self.FMF.k))

        for i, (u, movies) in enumerate(zip(user_group, rated_movies)):
            for m in movies:
                v = self.FMF.M[m].reshape((1, self.FMF.k))  # [5] --> [1, 5] so we can transpose to [5, 1]

                coeff_matrix += v.T @ v + tt.eye(self.FMF.k) * self.FMF.regularisation
                coeff_vector += v * self.FMF.R[u][m]

        profile = coeff_matrix.inverse() @ coeff_vector.reshape(self.FMF.k)

        # Use the profile to evaluate how well it works for this user group
        loss = self.loss(profile=profile, user_group=user_group, rated_movies=rated_movies)
        return profile, loss

    def profile(self, user_group):
        def extract_embeddings(user_group):
            # We can save some time by extracting a matrix of
            # (duplicated) movie embeddings in the same order
            # as their user ratings occur in the ratings array.
            # By multiplying the ratings array onto that matrix,
            # we are doing the same as in the for-loop approach
            # except much faster. For the coefficient matrix, we
            # can simply calculate the M.T @ M for the embedding
            # matrix M, which is much faster as well.

            embeddings = []
            ratings = []
            for u in user_group:
                for m in self.FMF.R[u].nonzero():
                    ratings.append(self.FMF.R[u][m].item())
                    embeddings.append(self.FMF.M[m].tolist())

            embeddings = tt.Tensor(embeddings)
            embeddings = embeddings.reshape(embeddings.shape[0], embeddings.shape[-1])

            ratings = tt.Tensor(ratings).reshape(len(ratings), 1)
            return embeddings, ratings

        def loss(profile, movie_embeddings, ratings):
            # Do some Torch magic to calculate the dot product
            # between the profile vector and all movie vectors
            n_ratings = len(ratings)

            profiles = tt.ones((n_ratings, self.FMF.k)) * profile
            ratings = ratings.reshape((n_ratings, 1, 1))
            predictions = (
                tt.bmm(profiles.reshape((n_ratings, 1, self.FMF.k)),
                       movie_embeddings.reshape((n_ratings, self.FMF.k, 1)))
                )

            err = (ratings - predictions) ** 2
            return err.sum()

        M, R = extract_embeddings(user_group)

        if not len(M):
            return None, math.inf  # Don't split on an item the we can't construct a profile for

        coeff_matrix = (M.T @ M).inverse()
        coeff_vector = (M * R).sum(dim=0)

        profile = coeff_matrix @ coeff_vector
        loss = loss(profile, M, R)
        return profile, loss


class FMF:

    def __init__(self, R=None, k=None, max_depth=None, regularisation=None):
        self.R = R
        self.k = k
        self.max_depth = max_depth

        self.n_users = len(R)
        self.n_movies = len(R[0])

        self.U = tt.rand((self.n_users, self.k))
        self.M = tt.rand((self.n_movies, self.k))

        self.regularisation = regularisation

    def build_node(self, users=None, profile=None, depth=None):
        node = Node(users=users, FMF=self, profile=profile)
        if depth >= self.max_depth:
            return node

        (RL, uL), (RD, uD), (RU, uU) = node.split(movies=range(self.n_movies))
        node.like = self.build_node(users=RL, profile=uL, depth=depth + 1)
        node.dislike = self.build_node(users=RD, profile=uD, depth=depth + 1)
        node.unknown = self.build_node(users=RU, profile=uU, depth=depth + 1)

        return node

    def fit(self):
        root = Node(users=range(self.n_users), FMF=self)
        (RL, uL), (RD, uD), (RU, uU) = root.split(movies=range(self.n_movies))

        # Recursively construct child nodes
        root.like = self.build_node(users=RL, profile=uL, depth=1)
        root.dislike = self.build_node(users=RD, profile=uD, depth=1)
        root.unknown = self.build_node(users=RU, profile=uU, depth=1)

        return root


if __name__ == '__main__':
    R, uid_map, mid_map, answer_map = get_rating_matrix(from_file=join('data', 'ratings.csv'))
    FMF = FMF(R=R, k=5, max_depth=3, regularisation=0.0015)

    tree = FMF.fit()
