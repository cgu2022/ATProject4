import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as pyplot
from kMeans12 import KMeans

X, y = make_blobs(centers=4, n_samples=500, n_features=2, shuffle=True, random_state=42)
print(X)
print(X.shape)

# print(y)

clusters = 3 #len(np.unique(y))
print(clusters)

k = KMeans(K=clusters, max_iters=150, plot_steps=True)
y_pred, centroid_locations, time = k.predict(X)

k.plot()

#print("y_pred:",y_pred)
#print('centroid_locations:', centroid_locations)