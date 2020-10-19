#
import torch
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

class ImageHelper:
    def __init__(self, raw_images):
        if type(raw_images) == str:
            self.images = torch.load(raw_images)
        elif type(raw_images) == torch.tensor:
            self.images = raw_images

    def exportClusters(self, clusters, folder_path):
        os.mkdir(folder_path)
        for cn in range(len(clusters)):
            clus_path = folder_path+'/Cluster'+str(cn)
            os.mkdir(clus_path) # make the folder
            for i in tqdm(range(len(clusters[cn])),'cluster'+str(cn)+'/'+str(len(clusters))):
                idx = clusters[cn][i]
                img = np.array(self.images[idx])
                img = Image.fromarray(img.astype(np.uint8))
                img.save(clus_path+'/'+str(i)+'.png')
