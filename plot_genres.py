# """
# Plot the 2D genre embeddings
# """
# import pickle

# import matplotlib.pyplot as plt

# import numpy as np

# with open('./K=2/F.pkl', 'rb') as file:
#     F = pickle.load(file)

# total_genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy',
#                 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
#                 'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance',
#                 'Sci-Fi', 'Thriller', 'War', 'Western', '(no genres listed)']

# for n in range(np.shape(F)[1]):
#     plt.plot([0, F[0, n]], [0, F[1, n]], label=total_genres[n],
#              color=plt.cm.tab20(n))

# plt.legend()
# plt.show()

import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

with open('./K=2/F.pkl', 'rb') as file:
    F = pickle.load(file)

total_genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance',
                'Sci-Fi', 'Thriller', 'War', 'Western', '(no genres listed)']

fig = plt.figure(figsize=(10, 6))
gs = GridSpec(1, 2, width_ratios=[1, 3])

ax = fig.add_subplot(gs[1])

for n in range(np.shape(F)[1]):
    ax.plot([0, F[0, n]], [0, F[1, n]], label=total_genres[n], color=plt.cm.tab20(n))

ax.tick_params(axis='both', which='major', labelsize=14)
leg_ax = fig.add_subplot(gs[0])
leg_ax.axis('off')
leg_ax.legend(handles=ax.lines, loc='center', bbox_to_anchor=(0.5, 0.5), frameon=False, fontsize=14)

plt.tight_layout()
plt.show()
