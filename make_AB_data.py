"""
Read in the data into usable formats
Add simulated dummy users
"""
import csv
import pickle

import numpy as np

path = '../ml-latest-small/ratings.csv'
movie_path = '../ml-latest-small/movies.csv'

users = {}
movies = {}

user_to_movie = []
movie_to_user = []

movie_data = {}

i = -1
new_user = False
j = -1
new_movie = False
with open(path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader, None)
    for row in csv_reader:
        user_id, movie_id, rating, timestamp = row
        rating = float(rating)
        new_user = user_id not in users.keys()
        if new_user:
            i += 1
            users[user_id] = i
        new_movie = movie_id not in movies.keys()
        if new_movie:
            j += 1
            movies[movie_id] = j

        if new_user:
            user_to_movie.append([(movies[movie_id], rating)])
        else:
            user_to_movie[users[user_id]].append((movies[movie_id], rating))
        if new_movie:
            movie_to_user.append([(users[user_id], rating)])
        else:
            movie_to_user[movies[movie_id]].append((users[user_id], rating))

# Create a data structure that maps movie_id to movie title
with open(movie_path, 'r', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader, None)
    for row in csv_reader:
        movie_id, title, genres = row
        movie_data[movie_id] = title

# Create simulated dummy users
for di in range(1, 101):
    i += 1
    users['d'+str(di)] = i
    # Let the dummy user have random ratings for 5 movies
    movie_indexes = np.random.randint(0, len(movies), 40)
    ratings = np.random.randint(1, 11, len(movie_indexes))/2
    # Append relevant data to other data structures
    user_to_movie.append([(movie_indexes[0], ratings[0])])
    for mi in range(1, len(movie_indexes)):
        user_to_movie[i].append((movie_indexes[mi], ratings[mi]))
        movie_to_user[movie_indexes[mi]].append((i, ratings[mi]))

with open('./pickle_storage/AB_users.pkl', 'wb') as file:
    pickle.dump(users, file)
with open('./pickle_storage/AB_movies.pkl', 'wb') as file:
    pickle.dump(movies, file)
with open('./pickle_storage/AB_user_to_movie.pkl', 'wb') as file:
    pickle.dump(user_to_movie, file)
with open('./pickle_storage/AB_movie_to_user.pkl', 'wb') as file:
    pickle.dump(movie_to_user, file)
with open('./pickle_storage/AB_movie_data.pkl', 'wb') as file:
    pickle.dump(movie_data, file)
