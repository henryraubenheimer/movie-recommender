"""
Train and test the ALS model for the AB test
"""
import math
import pickle
import time

import numpy as np

# Read the data structures
with open('./pickle_storage/AB_users.pkl', 'rb') as file:
    users = pickle.load(file)
with open('./pickle_storage/AB_movies.pkl', 'rb') as file:
    movies = pickle.load(file)
with open('./pickle_storage/AB_user_to_movie.pkl', 'rb') as file:
    user_to_movie = pickle.load(file)
with open('./pickle_storage/AB_movie_to_user.pkl', 'rb') as file:
    movie_to_user = pickle.load(file)

# Set the hyper-parameters
K = 10
lam = 1.0
tau = 1.0

max_iterations = 10
# Initialise the parameters
U = np.random.normal(0, 1/math.sqrt(K), (K, len(users)))
V = np.random.normal(0, 1/math.sqrt(K), (K, len(movies)))
user_biases = np.zeros(len(users))
item_biases = np.zeros(len(movies))


# Define the loss functions
def loss(user2movie):
    s1 = 0
    s2 = 0
    s3 = 0
    for m in range(len(user2movie)):
        for n, r in user2movie[m]:
            s1 += (r-np.dot(U[:, m], V[:, n])-user_biases[m]-item_biases[n])**2
    for m in range(np.shape(U)[1]):
        s2 += np.dot(U[:, m], U[:, m])
    for n in range(np.shape(V)[1]):
        s2 += np.dot(V[:, n], V[:, n])
    for m in range(len(user_biases)):
        s3 += user_biases[m]**2
    for n in range(len(item_biases)):
        s3 += item_biases[n]**2
    return -s1*lam/2-s2*tau/2-s3*tau/2


def rmse(user2movie):
    s = 0
    N = 0
    for m in range(len(user2movie)):
        for n, r in user2movie[m]:
            s += (r-np.dot(U[:, m], V[:, n])-user_biases[m]-item_biases[n])**2
            N += 1
    return math.sqrt(s/N)


prev_RMSE = rmse(user_to_movie)
cur_RMSE = prev_RMSE
prev_time = time.time()
# ALS applied to all data
print('All Data Results:')
i = 0
while i < max_iterations and prev_RMSE >= cur_RMSE:
    elapsed_time = round(time.time()-prev_time, 3)
    print(str(i)+': Loss -> '+str(loss(user_to_movie)), end="")
    print(', RMSE -> '+str(cur_RMSE)+' ('+str(elapsed_time)+' sec)')

    prev_time = time.time()
    prev_RMSE = cur_RMSE
    for m in range(np.shape(U)[1]):
        # Update the user biases
        inds = [v[0] for v in user_to_movie[m]]
        rating = [v[1] for v in user_to_movie[m]]
        bias = lam*sum(rating-np.matmul(U[:, m], V[:, inds])-item_biases[inds])
        bias = bias / (lam*len(inds)+tau)
        user_biases[m] = bias

        s1 = np.zeros((K, K))
        for n, r in user_to_movie[m]:
            s1 += np.outer(V[:, n], V[:, n])

        s2 = np.zeros((K))
        for n, r in user_to_movie[m]:
            s2 += (r-user_biases[m]-item_biases[n])*V[:, n]
        U[:, m] = np.matmul(np.linalg.inv(lam*s1+np.diag([tau]*K)), lam*s2)

    for n in range(np.shape(V)[1]):
        # Update the item biases
        inds = [u[0] for u in movie_to_user[n]]
        rating = [u[1] for u in movie_to_user[n]]
        bias = lam*sum(rating-np.matmul(V[:, n], U[:, inds])-user_biases[inds])
        bias = bias / (lam*len(inds)+tau)
        item_biases[n] = bias

        s1 = np.zeros((K, K))
        for m, r in movie_to_user[n]:
            s1 += np.outer(U[:, m], U[:, m])

        s2 = np.zeros((K))
        for m, r in movie_to_user[n]:
            s2 += (r-user_biases[m]-item_biases[n])*U[:, m]

        V[:, n] = np.matmul(np.linalg.inv(lam*s1+np.diag([tau]*K)), lam*s2)
    cur_RMSE = rmse(user_to_movie)

    i += 1

# Save the parameters
with open('./pickle_storage/AB_U.pkl', 'wb') as file:
    pickle.dump(U, file)
with open('./pickle_storage/AB_V.pkl', 'wb') as file:
    pickle.dump(V, file)
with open('./pickle_storage/AB_user_biases.pkl', 'wb') as file:
    pickle.dump(user_biases, file)
with open('./pickle_storage/AB_item_biases.pkl', 'wb') as file:
    pickle.dump(item_biases, file)
