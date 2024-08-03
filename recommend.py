"""
This code finds recommendations for the dummy user
"""
import pickle

import numpy as np

with open('./pickle_storage/movies.pkl', 'rb') as file:
    movies = pickle.load(file)
with open('./pickle_storage/U.pkl', 'rb') as file:
    U = pickle.load(file)
with open('./pickle_storage/V.pkl', 'rb') as file:
    V = pickle.load(file)
with open('./pickle_storage/user_biases.pkl', 'rb') as file:
    user_biases = pickle.load(file)
with open('./pickle_storage/item_biases.pkl', 'rb') as file:
    item_biases = pickle.load(file)


# a function that finds a movie ID from an index
def find_ID(i):
    for ID, index in movies.items():
        if i == index:
            return ID
    return None


# function that finds a recommendation score
def find_score(m, n):
    return np.dot(U[:, m], V[:, n])+0.15*item_biases[n]


# Initialise parallel lists to store the least and most polarising movies
# along with their values
recs = []
rec_values = []
m = len(user_biases)-1
for i in range(np.shape(V)[1]):
    # dont recommend movies that the user has already seen
    if movies['4993'] == i:
        continue

    if len(recs) < 5:
        recs.append(i)
        rec_values.append(find_score(m, i))
    else:
        j = np.argmin(rec_values)
        if find_score(m, i) > rec_values[j]:
            recs[j] = i
            rec_values[j] = find_score(m, i)

for i in recs:
    print(find_ID(i))
