"""
Split the dataset into a train and test set
Convert the data into usable formats
"""
import csv
import pickle
import random

total_genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance',
                'Sci-Fi', 'Thriller', 'War', 'Western', '(no genres listed)']

# Set variables needed to read and split the data
num_rows = 25000096
train_prop = 0.8
num_train_rows = int(num_rows*train_prop)
num_test_rows = num_rows-num_train_rows
path = '../ml-25m/'

# Initialise the data structures
# The following two structures map from ids to indices
users = {}
movies = {}
# The following four structures map from indices to lists of tuples
user_to_movie = []
movie_to_user = []
train_user_to_movie = []
train_movie_to_user = []
test_user_to_movie = []
test_movie_to_user = []

movie_data = []

# Read the genres and add it to movie_data
i = -1
with open(path+'movies.csv', 'r', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader, None)
    for row in csv_reader:
        movie_id, name, genres = row
        i += 1
        movies[movie_id] = i
        genres = genres.split("|")
        movie_data.append([total_genres.index(g) for g in genres])
movie_to_user = [[] for _ in range(len(movies))]
train_movie_to_user = [[] for _ in range(len(movies))]
test_movie_to_user = [[] for _ in range(len(movies))]

# Read the data, split the data, and add it to the data structures
i = -1
new_user = False
ratings = 0
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

        if new_user:
            user_to_movie.append([])
            train_user_to_movie.append([])
            test_user_to_movie.append([])
        user_to_movie[users[user_id]].append((movies[movie_id], rating))
        movie_to_user[movies[movie_id]].append((users[user_id], rating))
        if random.random() < train_prop:
            train_user_to_movie[users[user_id]]\
                .append((movies[movie_id], rating))
            train_movie_to_user[movies[movie_id]]\
                .append((users[user_id], rating))
            num_train_rows -= 1
        else:
            test_user_to_movie[users[user_id]]\
                .append((movies[movie_id], rating))
            test_movie_to_user[movies[movie_id]]\
                .append((users[user_id], rating))
            num_test_rows -= 1

        if num_train_rows == 0:
            train_prop = 0
        if num_test_rows == 0:
            train_prop = 1
        ratings += 1

# Uncomment this code to add a dummy user who likes LOTR
# users['d'] = len(users)
# user_to_movie.append([(movies['4993'], 5)])
# movie_to_user[movies['4993']].append((users['d'], 5))

# Save the data structures
with open('./pickle_storage/users.pkl', 'wb') as file:
    pickle.dump(users, file)
with open('./pickle_storage/movies.pkl', 'wb') as file:
    pickle.dump(movies, file)
with open('./pickle_storage/movie_data.pkl', 'wb') as file:
    pickle.dump(movie_data, file)
with open('./pickle_storage/user_to_movie.pkl', 'wb') as file:
    pickle.dump(user_to_movie, file)
with open('./pickle_storage/movie_to_user.pkl', 'wb') as file:
    pickle.dump(movie_to_user, file)
with open('./pickle_storage/train_user_to_movie.pkl', 'wb') as file:
    pickle.dump(train_user_to_movie, file)
with open('./pickle_storage/train_movie_to_user.pkl', 'wb') as file:
    pickle.dump(train_movie_to_user, file)
with open('./pickle_storage/test_user_to_movie.pkl', 'wb') as file:
    pickle.dump(test_user_to_movie, file)
with open('./pickle_storage/test_movie_to_user.pkl', 'wb') as file:
    pickle.dump(test_movie_to_user, file)
