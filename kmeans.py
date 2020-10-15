from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class kMeans: 

    def __init__(self, image_data = None): 
        #???
    def doKmeans(init, n_clusters, n_init, dataSet): 
        k_means = KMeans(init, n_clusters, n_init)
        #fit the data to our k_means model
        k_means.fit(dataSet)   
        return k_means

    def labels(k_means): 
        k_means_labels = k_means.labels_ #List of labels of each dataset
        print("The list of labels of the clusters are " + str(np.unique(k_means_labels)))
        G = len(np.unique(k_means_labels)) #Number of labels
        #2D matrix  for an array of indexes of the given label
        cluster_index= [[] for i in range(G)]
        for i, label in enumerate(k_means_labels,0):
            for n in range(G):
                if label == n:
                 cluster_index[n].append(i)
                else:
                    continue 
        return cluster_index

    def visualize(cluster, number, k_means): 
        cluster_index = label(k_means)
        plt.figure(figsize=(20,20))
        clust = cluster #enter label number to visualise
        num = number #num of data to visualize from the cluster
        for i in range(1,num): 
            plt.subplot(10, 10, i) #(Number of rows, Number of column per row, item number)
            plt.imshow(X[cluster_index[clust][i+500]].reshape(X_train.shape[1], X_train.shape[2]), cmap = plt.cm.binary)
            
        plt.show()

        Y_clust = [[] for i in range(G)]
        for n in range(G):
            Y_clust[n] = y[cluster_index[n]] #Y_clust[0] contains array of "correct" category from y_train for the cluster_index[0]
            assert(len(Y_clust[n]) == len(cluster_index[n])) #dimension confirmation
        #counts the number of each category in each cluster
        def counter(cluster):
            unique, counts = np.unique(cluster, return_counts=True)
            label_index = dict(zip(unique, counts))
            return label_index
        label_count= [[] for i in range(G)]
        for n in range(G):
            label_count[n] = counter(Y_clust[n])

        label_count[1] #Number of items of a certain category in cluster 1


        k_means_cluster_centers = k_means.cluster_centers_ #numpy array of cluster centers
        k_means_cluster_centers.shape #comes from 10 clusters and 420 features


    def plot(k_means)
        k_means_labels = k_means.labels_
        layout = go.Layout(
            title='<b>Cluster Visualisation</b>',
            yaxis=dict(title='<i>Y</i>'),
            xaxis=dict(title='<i>X</i>')
        )

        colors = ['red','green' ,'blue','purple','magenta','yellow','cyan','maroon','teal','black']
        trace = [ go.Scatter3d() for _ in range(11)]
        for i in range(0,10):
            my_members = (k_means_labels == i)
            index = [h for h, g in enumerate(my_members) if g]
            trace[i] = go.Scatter3d(
                    x=Clus_dataSet[my_members, 0],
                    y=Clus_dataSet[my_members, 1],
                    z=Clus_dataSet[my_members, 2],
                    mode='markers',
                    marker = dict(size = 2,color = colors[i]),
                    hovertext=index,
                    name='Cluster'+str(i),
        
                    )

        fig = go.Figure(data=[trace[0],trace[1],trace[2],trace[3],trace[4],trace[5],trace[6],trace[7],trace[8],trace[9]], layout=layout)
            
        py.offline.iplot(fig)