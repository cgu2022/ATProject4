import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(42)

def euclidean_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KMeans:

    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        
        # List of sample indicies for each cluster
        self.clusters = [[] for i in range(self.K)]
        # mean feature vector for each cluster
        self.centroids = []

    def predict(self, X):

        print("\nRunning Predict!")

        self.X = X
        self.n_samples, self.n_features = X.shape
        
        # initalize centroids
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]
        #print("Initial Centroids:", self.centroids)

        beginning_time = time.time()
        start_time = time.time()
        times = []

        # optimization
        for i in range(self.max_iters):
            #update clusters
            self.clusters = self._create_clusters(self.centroids)
            if self.plot_steps:
                    self.plot()

            #update centroids
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            if self.plot_steps:
                    self.plot()

            # check if converged
            if self._is_converged(centroids_old, self.centroids): # check if this is 0
                break

            print("Pass " + str(i) + " :", "--- %s seconds ---" % (time.time() - start_time))
            times.append(time.time() - start_time)
            start_time = time.time()
        
        print("Predict Finished!")
        print("Total Time:", (time.time() - beginning_time),'\n')
        # return cluster labels
        return self._get_cluster_labels(self.clusters), self.centroids, times
    
    # each sample will get the label of the cluster it was assigned to
    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels

    # Assign the samples to the closest centroids to create clusters
    def _create_clusters(self, centroids):
        clusters = [[] for i in range(self.K)] # List of lists: each sublist stores the points that are closest to it
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids) # Get index of closest cluster
            clusters[centroid_idx].append(idx)
        return clusters
    
    # distance of the current sample to each centroid
    def _closest_centroid(self, sample, centroids):
        distances = [euclidean_distance(sample, centroid) for centroid in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    # assign mean value of clusters to centroids
    def _get_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters): # cluster is the list of indices that are in the cluster
            #print("self.X[cluster]:", self.X[cluster])
            if len(self.X[cluster]) == 0:
                cluster_mean = 0.0
            else:
                cluster_mean = np.mean(self.X[cluster], axis=0) #X[cluseter], axis=0 will only return the samples that are in the current cluster
            centroids[cluster_idx] = cluster_mean # mean of all the points in the cluster
        return centroids

    # distances between each old and new centroids, fol all centroids
    def _is_converged(self, centroids_old, centroids):
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0 # if centroids don't move anymore
    
    # Copied and pasted from matplotlib & other sites
    def plot(self):
        fig, ax = plt.subplots(figsize=(12,8))
        
        for i, index in enumerate(self.clusters):
            point = self.X[index].T # print("Point:", *point)
            ax.scatter(*point)
        
        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)
        
        plt.show()