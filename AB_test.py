"""
Simulate an AB test
"""
import csv
import pickle

import numpy as np

from scipy.stats import norm

np.set_printoptions(suppress=True, threshold=np.inf)

with open('./pickle_storage/AB_users.pkl', 'rb') as file:
    users = pickle.load(file)
with open('./pickle_storage/AB_movies.pkl', 'rb') as file:
    movies = pickle.load(file)
with open('./pickle_storage/AB_movie_data.pkl', 'rb') as file:
    movie_data = pickle.load(file)
with open('./pickle_storage/AB_U.pkl', 'rb') as file:
    U = pickle.load(file)
with open('./pickle_storage/AB_V.pkl', 'rb') as file:
    V = pickle.load(file)
with open('./pickle_storage/AB_user_biases.pkl', 'rb') as file:
    user_biases = pickle.load(file)
with open('./pickle_storage/AB_item_biases.pkl', 'rb') as file:
    item_biases = pickle.load(file)


# find the key of a dictionary by its value
def find_key_by_value(dictionary, target_value):
    for key, value in dictionary.items():
        if value == target_value:
            return key
    return None


# create recommendations for a user
def createreco(userid, versionid):
    m = users[userid]

    if versionid == 'A':
        bias_effect = 0.15
    else:
        bias_effect = 0

    recos = {}
    for n in range(len(item_biases)):
        score = np.dot(U[:, m], V[:, n])+bias_effect*item_biases[n]
        if len(recos) < 5:
            movie_id = find_key_by_value(movies, n)
            recos[(movie_id, movie_data[movie_id])] = score
        else:
            if score > recos[min(recos, key=recos.get)]:
                recos.pop(min(recos, key=recos.get))
                movie_id = find_key_by_value(movies, n)
                recos[(movie_id, movie_data[movie_id])] = score

    return list(recos.keys())


# simulate feedback
def sim_feedback(userid, movieid):
    m = users[userid]
    n = movies[movieid]
    s = np.array([])
    for r in np.arange(0.5, 5.5, 0.5):
        s = np.append(s, norm.pdf(np.dot(U[:, m], V[:, n]) + item_biases[n]
                                  + user_biases[m], loc=r, scale=1.5))
    s = s/sum(s)
    return np.random.choice(np.arange(0.5, 5.5, 0.5), p=s)


# write the feedback to a csv file
with open('feedback.csv', mode='w', newline='') as file:
    file.truncate(0)
    writer = csv.writer(file)
    writer.writerow(['user_id', 'movie_id', 'feedback', 'version_id'])
    for d in range(1, 101):
        u = 'd'+str(d)
        if np.random.random() < 0.5:
            version = 'A'
        else:
            version = 'B'
        recs = createreco(u, version)
        for movie in recs:
            writer.writerow([u, movie[0], sim_feedback(u, movie[0]), version])
