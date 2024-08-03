"""
Plot the rating distribution
"""
import pickle

import matplotlib.pyplot as plt

# Read the data structures
with open('./pickle_storage/user_to_movie.pkl', 'rb') as file:
    user_to_movie = pickle.load(file)
with open('./pickle_storage/movie_to_user.pkl', 'rb') as file:
    movie_to_user = pickle.load(file)

# Find the user densities
max_density = 0
for i in range(len(user_to_movie)):
    if len(user_to_movie[i]) > max_density:
        max_density = len(user_to_movie[i])
user_density = [0] * (max_density + 1)
for i in range(len(user_to_movie)):
    user_density[len(user_to_movie[i])] += 1

# Find the movie densities
max_density = 0
for i in range(len(movie_to_user)):
    if len(movie_to_user[i]) > max_density:
        max_density = len(movie_to_user[i])
movie_density = [0] * (max_density + 1)
for i in range(len(movie_to_user)):
    movie_density[len(movie_to_user[i])] += 1

# Plot the densities
plt.xscale("log")
plt.yscale("log")
plt.xlabel("degree")
plt.ylabel("frequency")
plt.scatter(range(len(movie_density)), movie_density,
            color='green', marker='o', s=5, label='movie')
plt.scatter(range(len(user_density)), user_density,
            color='blue', marker='o', s=5, label='user')
plt.legend()

plt.savefig('power_laws.svg')
