"""
Plot the 2D item embeddings
"""
import pickle

import matplotlib.pyplot as plt

with open('./pickle_storage/movies.pkl', 'rb') as file:
    movies = pickle.load(file)
with open('./K=2/V.pkl', 'rb') as file:
    V = pickle.load(file)

IDs = ['47', '107', '364', '593', '46062', '119190']
Titles = ['Seven', 'Muppet Treasure Island', 'The Lion King',
          'Silence of the Lambs', 'High School Musical', 'Walking on Sunshine']
for i in range(len(IDs)):
    plt.plot([0, V[0, movies[IDs[i]]]], [0, V[1, movies[IDs[i]]]],
             label=Titles[i])
    plt.legend()
plt.show()
