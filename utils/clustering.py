# Skript based on https://github.com/facebookresearch/deepcluster/blob/master/clustering.py
# Facebook, Inc.

import time

import faiss
from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True

__all__ = ['Kmeans']


def run_kmeans(x, nmb_clusters, verbose=False, init_cents=None):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)
    clus.niter = 20
    clus.max_points_per_centroid = 10000000

    if init_cents is not None:
        clus.centroids.resize(init_cents.size)
        faiss.memcpy(clus.centroids.data(), faiss.swig_ptr(init_cents), init_cents.size * 4)

    index = faiss.IndexFlatL2(d)

    # perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)
    losses = faiss.vector_to_array(clus.obj)
    if verbose:
        print('k-means loss evolution: {0}'.format(losses))

    centroids = faiss.vector_to_array(clus.centroids).reshape((nmb_clusters, d))

    return [int(n[0]) for n in I], losses[-1], centroids



class Kmeans:
    def __init__(self, k):
        self.k = k

    def cluster(self, data, verbose=False, init_cents=None):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        xb = data
        # cluster the data
        I, loss, cents = run_kmeans(xb, self.k, verbose, init_cents)
        self.images_lists = [[] for i in range(self.k)]
        for i in range(len(data)):
            self.images_lists[I[i]].append(i)

        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))

        return cents

def make_graph(xb, nnn):
    """Builds a graph of nearest neighbors.
    Args:
        xb (np.array): data
        nnn (int): number of nearest neighbors
    Returns:
        list: for each data the list of ids to its nnn nearest neighbors
        list: for each data the list of distances to its nnn NN
    """
    N, dim = xb.shape

    # L2
    index = faiss.IndexFlatL2(dim)

    index.add(xb)
    D, I = index.search(xb, nnn + 1)
    return I, D
