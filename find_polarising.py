"""
Finds the most and least polarising movies
"""
import pickle

import numpy as np

with open('./pickle_storage/movies.pkl', 'rb') as file:
    movies = pickle.load(file)
with open('./K=10/V.pkl', 'rb') as file:
    V = pickle.load(file)


# a function that finds a movie ID from an index
def find_ID(i):
    for ID, index in movies.items():
        if i == index:
            return ID
    return None


# Initialise parallel lists to store the least and most polarising movies
# along with their values
least = []
least_values = []
most = []
most_values = []

for i in range(np.shape(V)[1]):
    if len(least) < 5:
        least.append(i)
        least_values.append(np.linalg.norm(V[:, i]))
    else:
        j = np.argmax(least_values)
        if np.linalg.norm(V[:, i]) < least_values[j]:
            least[j] = i
            least_values[j] = np.linalg.norm(V[:, i])

    if len(most) < 5:
        most.append(i)
        most_values.append(np.linalg.norm(V[:, i]))
    else:
        j = np.argmin(most_values)
        if np.linalg.norm(V[:, i]) > most_values[j]:
            most[j] = i
            most_values[j] = np.linalg.norm(V[:, i])

print('The least polarising movies:')
for i in least:
    print(find_ID(i))
print('The most polarising movies:')
for i in most:
    print(find_ID(i))
