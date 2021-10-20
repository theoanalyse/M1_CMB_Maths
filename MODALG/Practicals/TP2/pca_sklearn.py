from sklearn import preprocessing
from sklearn import decomposition
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('deca.txt', sep="\t")
my_data = data.drop(['Points', 'Rank', 'Competition'], axis=1)

X = my_data.values
Xc = preprocessing.scale(X, with_std=False)
Xs = preprocessing.scale(X)

pca = decomposition.PCA(n_components=2)
pca.fit(Xs)
X_projected = pca.transform(Xs)

print(X_projected)

plt.scatter(X_projected[:, 0], X_projected[:, 1])
pcs = pca.components_

plt.title("Expected PCA using SkLearn")
plt.show()
