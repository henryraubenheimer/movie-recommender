"""
Apply Bayesian Personalised Ranking to the constructed implicit data
"""
import math
import pickle

import numpy as np

for K_metric in [10, 20, 100]:
    print('K_metric =', K_metric)
    with open('./pickle_storage/users.pkl', 'rb') as file:
        users = pickle.load(file)
    with open('./pickle_storage/movies.pkl', 'rb') as file:
        movies = pickle.load(file)
    with open('./pickle_storage/BPR_train_user_to_movie.pkl', 'rb') as file:
        train_utom = pickle.load(file)
    with open('./pickle_storage/BPR_train_movie_to_user.pkl', 'rb') as file:
        train_mtou = pickle.load(file)
    with open('./pickle_storage/BPR_test_user_to_movie.pkl', 'rb') as file:
        test_utom = pickle.load(file)
    with open('./pickle_storage/BPR_test_movie_to_user.pkl', 'rb') as file:
        test_mtou = pickle.load(file)

    # Set the hyper-parameters
    K = 50
    num_iterations = 2
    alpha = 0.1
    lam = 0.1
    max_xuij = 20

    # Initialise the parameters
    U = np.random.normal(0, 1/math.sqrt(K), (K, len(users)))
    V = np.random.normal(0, 1/math.sqrt(K), (K, len(movies)))

    # Define the loss function
    def metrics_K(user2movie):

        hits = 0
        num_positives = 0
        for u in range(len(user2movie)):
            scores = np.dot(U[:, u], V)
            top_scores = np.argsort(scores)
            hits += sum(1 for movie in top_scores[-K_metric:]
                        if movie in user2movie[u])
            num_positives += len(user2movie[u])

        precision = hits/(len(user2movie)*K_metric)
        recall = hits/num_positives
        return precision, recall

    # BPR applied to the training set
    for n in range(num_iterations):
        for u in range(len(train_utom)):
            if len(train_utom[u]) > 0:
                i = np.random.choice(train_utom[u])
                j = np.random.choice(list(movies.values()))
                if j not in train_utom[u]:
                    xuij = np.dot(U[:, u], V[:, i])-np.dot(U[:, u], V[:, j])
                    if abs(xuij) > max_xuij:
                        xuij = max_xuij*np.sign(xuij)
                    first_term = math.exp(-xuij)/(1+math.exp(-xuij))

                    U[:, u] = U[:, u]+alpha*(first_term*(V[:, i]-V[:, j])
                                             + lam*np.linalg.norm(U[:, u]))
                    V[:, i] = V[:, i]+alpha*(first_term*U[:, u]
                                             + lam*np.linalg.norm(V[:, i]))
                    V[:, j] = V[:, j]+alpha*(first_term*(-U[:, u])
                                             + lam*np.linalg.norm(V[:, j]))

        precision, recall = metrics_K(train_utom)
        print('Train Precision@'+str(K_metric)+' -> '+str(precision), end='')
        print(', Train Recall@'+str(K_metric)+' -> '+str(recall))
        precision, recall = metrics_K(test_utom)
        print('Test Precision@'+str(K_metric)+' -> '+str(precision), end='')
        print(', Test Recall@'+str(K_metric)+' -> '+str(recall))
