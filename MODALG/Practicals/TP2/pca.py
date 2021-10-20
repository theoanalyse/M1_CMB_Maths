#!/usr/bin/env python3

# Good old imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_centering_functions import normalize_data, get_std, get_mean

# Importing the data
data = pd.read_csv('deca.txt', sep="\t")

# Cleaning a few columns
my_data = data.drop(['Points', 'Rank', 'Competition'], axis=1)

# Store the values in and Matrix
XX = np.array(my_data.values)

# Normalize data
X = normalize_data(XX)

# SVD of the normalized data
U, S, VT = np.linalg.svd(X)

# Compute the two most important directions (associated to the two biggest eigenvalues)
new_directions = VT.T[:, :2]

# Project the cloud of points on the new directions
X_pca = X.dot(new_directions)


# fetches x and y coordinate of each point X_i
xs = [X_pca[i, 0] for i in range(len(X_pca))]
ys = [X_pca[i, 1] for i in range(len(X_pca))]

# display PCA using scatter plot
plt.style.use("ggplot")
plt.title("Manual PCA")
plt.scatter(xs, ys, c='g', marker='*', s=70)
plt.show()
