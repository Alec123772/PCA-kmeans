import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import torch
from Stat import Stat


class kMeans:

    def __init__(self, data, rescl_method='normalize'):
        '''initialize the kmeans class on a set of datapoints'''
        self.stat = Stat(rescl_method)
        if data != None:
            self.loadData(data)
        else:
            pass

    def loadData(self, data):
        '''loads the data into this class'''
        self.raw_data = data # localize the data
        self.data = self.stat.scale(data) # rescaled data
        self.n_samp, self.n_vars = self.data.size() # basic info about the data
        pass

    def compare_ks(self, ks, init='k-means++', n_init=20):
        ''''''
        allSSDs = []
        for k in ks:
            kmeans = KMeans(k, init=init, n_init=n_init).fit(self.data)
            SSDs = 0
            for i in range(self.n_samp):
                label = kmeans.labels_[i]
                mean = torch.tensor(kmeans.cluster_centers_[label])
                dist = torch.norm(self.data[i]-mean)
                SSDs += dist**2
            allSSDs.append(SSDs*k**2)
        plt.bar(list(range(1,len(allSSDs)+1)),allSSDs)
        plt.show()

    def dokMeans(self, n_clusters, init='k-means++', n_init=20):
        '''performs kmeans on the data loaded into the class'''
        # does kmeans
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters, init=init, n_init=n_init) # kmeans model
        self.kmeans.fit(self.data) # fit the data to our kmeans model
        # makes self.clusters (list of indexes in each cluster)
        self.clusters = [[] for _ in range(self.n_clusters)]
        for i in range(self.n_samp):
            self.clusters[self.kmeans.labels_[i]].append(i)
        return self.clusters

    def graphClusters(self):
        ''''''
        if self.n_vars > 3:
            raise AttributeError('too many variables to graph')
            return None
        # get the plot ready
        if self.n_vars == 3:
            fig = plt.figure()
            ax = fig.add_subplot(projection = '3d')
        # mainloop
        colors = ['red','orange','yellow','green','blue','purple','pink','cyan']
        for cluster,color in zip(self.clusters,colors):
            clus_data = self.data[cluster] # data from the cluster
            # parse the data into x/y/z
            xs = clus_data[:,0] # x data
            if self.n_vars == 1:
                ys = [0]*len(xs)
            if self.n_vars >= 2:
                ys = clus_data[:,1]
            if self.n_vars == 3:
                zs = clus_data[:,2]
            # graph the data in the appropriate color
            if self.n_vars != 3:
                plt.scatter(xs,ys,color=color)
            else:
                ax.scatter(xs,ys,zs,marker='o',c=color)
        plt.show()
