"""
Construct implicit data
Split the data into training and testing sets
"""
import csv
import pickle
import random

num_rows = 25000096
train_prop = 0.8
num_train_rows = int(num_rows*train_prop)
num_test_rows = num_rows-num_train_rows
path = '../ml-25m/'

users = {}
movies = {}

BPR_user_to_movie = []
BPR_movie_to_user = []
BPR_train_user_to_movie = []
BPR_train_movie_to_user = []
BPR_test_user_to_movie = []
BPR_test_movie_to_user = []

# Read the data, split the data, and add it to the data structures
i = -1
new_user = False
j = -1
new_movie = False
with open(path+'ratings.csv', 'r') as csv_file:
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
            BPR_user_to_movie.append([])
            BPR_train_user_to_movie.append([])
            BPR_test_user_to_movie.append([])
        if new_movie:
            BPR_movie_to_user.append([])
            BPR_train_movie_to_user.append([])
            BPR_test_movie_to_user.append([])
        BPR_user_to_movie[users[user_id]].append((movies[movie_id]))
        BPR_movie_to_user[movies[movie_id]].append((users[user_id]))

        # only consider data with ratings >= 4.0
        if rating >= 4.0:
            if random.random() < train_prop:
                BPR_train_user_to_movie[users[user_id]]\
                    .append(movies[movie_id])
                BPR_train_movie_to_user[movies[movie_id]]\
                    .append(users[user_id])
                num_train_rows -= 1
            else:
                BPR_test_user_to_movie[users[user_id]].append(movies[movie_id])
                BPR_test_movie_to_user[movies[movie_id]].append(users[user_id])
                num_test_rows -= 1
        if num_train_rows == 0:
            train_prop = 0
        if num_test_rows == 0:
            train_prop = 1

with open('./pickle_storage/users.pkl', 'wb') as file:
    pickle.dump(users, file)
with open('./pickle_storage/movies.pkl', 'wb') as file:
    pickle.dump(movies, file)
with open('./pickle_storage/BPR_user_to_movie.pkl', 'wb') as file:
    pickle.dump(BPR_user_to_movie, file)
with open('./pickle_storage/BPR_movie_to_user.pkl', 'wb') as file:
    pickle.dump(BPR_movie_to_user, file)
with open('./pickle_storage/BPR_train_user_to_movie.pkl', 'wb') as file:
    pickle.dump(BPR_train_user_to_movie, file)
with open('./pickle_storage/BPR_train_movie_to_user.pkl', 'wb') as file:
    pickle.dump(BPR_train_movie_to_user, file)
with open('./pickle_storage/BPR_test_user_to_movie.pkl', 'wb') as file:
    pickle.dump(BPR_test_user_to_movie, file)
with open('./pickle_storage/BPR_test_movie_to_user.pkl', 'wb') as file:
    pickle.dump(BPR_test_movie_to_user, file)
