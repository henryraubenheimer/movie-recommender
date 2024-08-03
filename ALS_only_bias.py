"""
Train and test the ALS bias only model
"""
import math
import pickle
import time

import numpy as np

# Read the data structures
with open('./pickle_storage/users.pkl', 'rb') as file:
    users = pickle.load(file)
with open('./pickle_storage/movies.pkl', 'rb') as file:
    movies = pickle.load(file)
with open('./pickle_storage/user_to_movie.pkl', 'rb') as file:
    user_to_movie = pickle.load(file)
with open('./pickle_storage/movie_to_user.pkl', 'rb') as file:
    movie_to_user = pickle.load(file)
with open('./pickle_storage/train_user_to_movie.pkl', 'rb') as file:
    train_utom = pickle.load(file)
with open('./pickle_storage/train_movie_to_user.pkl', 'rb') as file:
    train_mtou = pickle.load(file)
with open('./pickle_storage/test_user_to_movie.pkl', 'rb') as file:
    test_utom = pickle.load(file)
with open('./pickle_storage/test_movie_to_user.pkl', 'rb') as file:
    test_mtou = pickle.load(file)

max_iterations = 20
# Initialise the parameters
user_biases = np.zeros(len(users))
item_biases = np.zeros(len(movies))

# Set the hyper-parameters
lam = 1.0
tau = 1.0


# Define the loss functions
def loss(user2movie):
    s1 = 0
    s2 = 0
    for m in range(len(user2movie)):
        for n, r in user2movie[m]:
            s1 += (r-user_biases[m]-item_biases[n])**2
    for m in range(len(user_biases)):
        s2 += user_biases[m]**2
    for n in range(len(item_biases)):
        s2 += item_biases[n]**2
    return -s1*lam/2-s2*tau/2


def rmse(user2movie):
    s = 0
    N = 0
    for m in range(len(user2movie)):
        for n, r in user2movie[m]:
            s += (r-user_biases[m]-item_biases[n])**2
            N += 1
    return math.sqrt(s/N)


prev_RMSE = rmse(user_to_movie)
cur_RMSE = prev_RMSE
prev_time = time.time()
# Bias only ALS applied to all data
print('All Data Results:')
i = 0
while i < max_iterations and prev_RMSE >= cur_RMSE:
    elapsed_time = round(time.time()-prev_time, 3)
    print(str(i)+': Loss -> '+str(loss(user_to_movie)), end="")
    print(', RMSE -> '+str(cur_RMSE)+' ('+str(elapsed_time)+' sec)')

    prev_time = time.time()
    prev_RMSE = cur_RMSE
    for m in range(len(user_biases)):
        # Update the user biases
        inds = [v[0] for v in user_to_movie[m]]
        bias = lam*sum([v[1] for v in user_to_movie[m]]-item_biases[inds])
        bias = bias / (lam*len(inds)+tau)
        user_biases[m] = bias

    for n in range(len(item_biases)):
        # Update the item biases
        inds = [u[0] for u in movie_to_user[n]]
        bias = lam*sum([u[1] for u in movie_to_user[n]]-user_biases[inds])
        bias = bias / (lam*len(inds)+tau)
        item_biases[n] = bias
    cur_RMSE = rmse(user_to_movie)

    i += 1

# Reinitialise
user_biases = np.zeros(len(users))
item_biases = np.zeros(len(movies))
prev_RMSE = rmse(test_utom)
cur_RMSE = prev_RMSE
prev_time = time.time()
# Bias only ALS applied to just the training set
print('\nTraining and Test Set Results:')
i = 0
while i < max_iterations and prev_RMSE >= cur_RMSE:
    elapsed_time = round(time.time()-prev_time, 3)
    print(str(i)+': Train Loss -> '+str(loss(train_utom)), end='')
    print(', Train RMSE -> '+str(rmse(train_utom)), end='')
    print(', Test Loss -> '+str(cur_RMSE)+' ('+str(elapsed_time)+' sec)')

    prev_time = time.time()
    prev_RMSE = cur_RMSE
    for m in range(len(user_biases)):
        # Update the user biases
        inds = [v[0] for v in train_utom[m]]
        bias = lam*sum([v[1] for v in train_utom[m]]-item_biases[inds])
        bias = bias / (lam*len(inds)+tau)
        user_biases[m] = bias

    for n in range(len(item_biases)):
        # Update the item biases
        inds = [u[0] for u in train_mtou[n]]
        bias = lam*sum([u[1] for u in train_mtou[n]]-user_biases[inds])
        bias = bias / (lam*len(inds)+tau)
        item_biases[n] = bias
    cur_RMSE = rmse(user_to_movie)

    i += 1
