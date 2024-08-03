"""
Train and test the ALS model
"""
import math
import pickle
import time

import matplotlib.pyplot as plt

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

# Set the hyper-parameters
K = 10
lam = 1
tau = 1

max_iterations = 20
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
        try:
            for n, r in user2movie[m]:
                s += (r-np.dot(U[:, m], V[:, n])-user_biases[m]-item_biases[n])**2
                N += 1
        except TypeError:
            print(user2movie[m])
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

# Variables to store loss for the plot
train_loss = []
train_rmse = []

# Reinitialise
U = np.random.normal(0, 1/math.sqrt(K), (K, len(users)))
V = np.random.normal(0, 1/math.sqrt(K), (K, len(movies)))
user_biases = np.zeros(len(users))
item_biases = np.zeros(len(movies))
prev_RMSE = rmse(test_utom)
cur_RMSE = prev_RMSE
prev_time = time.time()
# ALS applied to just the training set
print('\nTraining and Test Set Results:')
i = 0
while i < max_iterations and prev_RMSE >= cur_RMSE:
    train_loss.append(loss(train_utom))
    train_rmse.append(rmse(train_utom))
    elapsed_time = round(time.time()-prev_time, 3)
    print(str(i)+': Train Loss -> '+str(train_loss[-1]), end='')
    print(', Train RMSE -> '+str(train_rmse[-1]), end='')
    print(', Test RMSE -> '+str(cur_RMSE)+' ('+str(elapsed_time)+' sec)')

    if i == max_iterations-1:
        break

    prev_time = time.time()
    prev_RMSE = cur_RMSE
    for m in range(len(user_biases)):
        # Update the user biases
        inds = [v[0] for v in train_utom[m]]
        rating = [v[1] for v in train_utom[m]]
        bias = lam*sum(rating-np.matmul(U[:, m], V[:, inds])-item_biases[inds])
        bias = bias / (lam*len(inds)+tau)
        user_biases[m] = bias

        s1 = np.zeros((K, K))
        for n, r in train_utom[m]:
            s1 += np.outer(V[:, n], V[:, n])

        s2 = np.zeros((K))
        for n, r in train_utom[m]:
            s2 += (r-user_biases[m]-item_biases[n])*V[:, n]
        U[:, m] = np.matmul(np.linalg.inv(lam*s1+np.diag([tau]*K)), lam*s2)

    for n in range(len(item_biases)):
        # Update the item biases
        inds = [u[0] for u in train_mtou[n]]
        rating = [u[1] for u in train_mtou[n]]
        bias = lam*sum(rating-np.matmul(V[:, n], U[:, inds])-user_biases[inds])
        bias = bias / (lam*len(inds)+tau)
        item_biases[n] = bias

        s1 = np.zeros((K, K))
        for m, r in train_mtou[n]:
            s1 += np.outer(U[:, m], U[:, m])

        s2 = np.zeros((K))
        for m, r in train_mtou[n]:
            s2 += (r-user_biases[m]-item_biases[n])*U[:, m]

        V[:, n] = np.matmul(np.linalg.inv(lam*s1+np.diag([tau]*K)), lam*s2)
    cur_RMSE = rmse(user_to_movie)

    i += 1

# Plot the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.plot(range(1, len(train_loss)+1), train_loss)
ax1.set_title('Training Loss')
ax1.set_xlabel('Iteration')
ax2.plot(range(1, len(train_rmse)+1), train_rmse)
ax2.set_title('Training RMSE')
ax2.set_xlabel('Iteration')
plt.tight_layout()
plt.show()

# Save the parameters
with open('./pickle_storage/U.pkl', 'wb') as file:
    pickle.dump(U, file)
with open('./pickle_storage/V.pkl', 'wb') as file:
    pickle.dump(V, file)
with open('./pickle_storage/user_biases.pkl', 'wb') as file:
    pickle.dump(user_biases, file)
with open('./pickle_storage/item_biases.pkl', 'wb') as file:
    pickle.dump(item_biases, file)
