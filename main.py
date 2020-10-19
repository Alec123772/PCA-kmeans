#MAIN#
# this is the code i used to generate the pca/kmeans analysis

import torch
from ImPCA import ImPCA
from kMeans import kMeans
import os
from ImageHelper import ImageHelper

# VARIABLES
__DATA_FPATH__ = 'imgs.pt' # file path of the images
__RAW_IMAGE_FPATH__ = 'images.pt' # file path to raw images
__PCA_FPATH__ = 'pca.pt' # file path to save/load pca analysis to/from
__USE_GPU__ = True # use GPU?
__PCA_DIMENSIONS__ = [i for i in range(1,500)] # pca dimensions to use for clustering
__N_CLUSTERS__ = 4 # number of clusters in kMeans
__OUTPUT_FOLDER__ = './Clusters0' # where to output clusters

# MAIN CODE
if __name__ == '__main__':
    '''this section does PCA'''
    if __PCA_FPATH__ in os.listdir('./'): # if it has
        pca = ImPCA() # initialize the class
        pca.load(__PCA_FPATH__) # load the pca analysis
    else: # if it hasn't
        pca = ImPCA(__DATA_FPATH__) # load the data into the pca class
        pca.doPCA(GPU=__USE_GPU__) # do the pca analysis
        pca.save(__PCA_FPATH__) # save it for the future
    # now get the pca data you want for kmeans
    pca_data = pca.getComps(__PCA_DIMENSIONS__)

    '''this section does kmeans'''
    kmeans = kMeans(pca_data)
    kmeans.dokMeans(__N_CLUSTERS__)

    '''this part exports the photos'''
    clusters = kmeans.clusters
    IH = ImageHelper('images.pt')
    IH.exportClusters(clusters, './Clusters2')
