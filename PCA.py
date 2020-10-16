# imPCA class

# imports
import torch
import matplotlib.pyplot as plt
from warnings import warn
import numpy as np
from time import time
import random


class ImPCA:
    '''class for implementing the PCA algorithm on a set of images'''

    class Stat:
        '''subclass stats handels resizing/rescaling the data so that
        PCA can better interpret it, as well as unscaling the data to
        return it to it's original state. contains tons of useful methods'''
        def __init__(self, rescl_method):
            # a dictionary for easily acessing the rescaling functions with a string
            self.rescl_dict = {
                'normalize': self.normalize,
                'normalize bv': self.normalize_bv,
                'standardize': self.standardize,
                'standardize bv': self.standardize_bv
            }
            self.unscl_dict = {
                'normalize': self.un_normalize,
                'normalize bv': self.un_normalize_bv,
                'standardize': self.un_standardize,
                'standardize bv': self.un_standardize_bv
            }

            # localize rescaling method (and make sure it's valid)
            if rescl_method not in self.rescl_dict.keys():
                raise ValueError('%s is not a valid rescaling method' % str(rescl_method))
            else:
                self.rescl_method = rescl_method

            # assign appropriate functions
            self.scale = self.rescl_dict[self.rescl_method]
            self.unScale = self.unscl_dict[self.rescl_method]

        def loadSelf(self, stat_dict):
            self.__dict__ = stat_dict

        def normalize(self, data):
            '''returns a normalized version of the data where all variables are
            between -1 and 1 and the mean is zero'''
            ndata = data.clone()
            self.mean = torch.mean(ndata) # remember the mean
            ndata = ndata-self.mean # make the mean is zero
            self.absmax = torch.max(abs(ndata)) # remember the max/min
            ndata = ndata/self.absmax # make the largest/smallest variable one/negative one
            return ndata

        def un_normalize(self, data):
            '''this takes in a set of datapoints, and reverses the normalization
            process to produce the original data before it had been rescaled'''
            ndata = data.clone() # clone the data
            ndata = ndata*self.absmax # multiply it by what we divided by before
            ndata = ndata+self.mean # add what we subtracted before
            return ndata

        def normalize_bv(self, data):
            '''normalizes the data by variables, i.e. each individual variable's mean
            is zero and range is [-1,1], insead of that being the mean/range of *all* varables'''
            ndata = data.clone() # clone the data
            self.means = torch.mean(ndata,dim=0) # get the means of variable
            ndata = ndata-self.means # make the mean of each variable zero
            self.absmaxs = torch.max(abs(ndata),dim=0)[0] # get max/min value in each column
            ndata = ndata/self.absmaxs # make max/min per column one/negative one
            return ndata

        def un_normalize_bv(self, data):
            '''this reverses the normilazation by variable, blowing each variable back up to
            it's original range as defined by the original data set'''
            ndata = data.clone() # clone the data
            ndata = ndata*self.absmaxs # multiply by what we divided each variable by
            ndata = ndata+self.means # add what we subtracted each variable by
            return ndata

        def standardize(self, data):
            '''returns a standardized version of the data so that the mean of all
            variables is zero and the standard deviation is one'''
            ndata = data.clone() # clone the data
            self.mean = torch.mean(ndata) # mean of data
            ndata = ndata-self.mean # now the mean is zero
            self.sd = torch.sqrt(torch.sum(ndata**2)/(ndata.numel()-1)) # standard deviation
            ndata = ndata/self.sd # now the standard deviation is one
            return ndata

        def un_standardize(self, data):
            '''undoes the standardization of the data, returning the original data
            given a standardized version of the data'''
            ndata = data.clone() # clone the data
            ndata = ndata*self.sd # multiply by the original standard deviation
            ndata = ndata+self.mean # add back the mean
            return ndata

        def standardize_bv(self, data):
            '''returns a version of the data where the mean and standard deviation
            for *each variable* is zero and one, respectively'''
            ndata = data.clone() # clone the data
            self.means = torch.mean(ndata,dim=0) # mean by variable
            ndata = ndata-self.means # now the mean is zero
            self.sds = torch.sqrt(torch.sum(ndata**2,dim=0)/(ndata.size()[0]-1)) # standard deviation
            ndata = ndata/self.sds # now standard deviation is one
            return ndata

        def un_standardize_bv(self, data):
            '''undoes standardization by variable, taking in a set of standardized samples
            and returning what those variables would look like before standardization'''
            ndata = data.clone() # clone the data
            ndata = data*self.sds # multiply by the original standard deviations
            ndata = ndata+self.means # add back the mean of each variable
            return ndata

    class Timer:
        '''really basic timing class to help with timing things within the PCA class
        to see what might be slowing down your calculation'''
        def __init__(self):
            '''initializes timer and records the time when timer was initialized'''
            self.time0 = time()
            self.show_progress = False # initialize show progress to false
            pass

        def start(self,label):
            '''starts a timer for label'''
            if self.show_progress:
                print(label,' ...')
                self.label = label
                self.tstart = time()

        def end(self):
            '''ends the timer after it's started, label is remembered'''
            if self.show_progress:
                self.tend = time()
                print('done! took ',round(self.tend-self.tstart,3),' sec')

        def readout(self,label):
            '''reads out the label with the time since initialization'''
            if self.show_progress:
                print(label,' t=',round(time()-self.time0,3),' sec')

    def __init__(self,image_data=None):
        '''initialize the PCA class, can be initialized with image data if
        it's provided, or it can be initialized without data if you plan on
        loading in a saved instance or loading in the data later. image_data
        can be a torch tensor of image data or a string file path pointing to the data'''
        self.timer = self.Timer() # initializes a timer class
        if image_data != None:
            # if data is provided, load it into the class
            self.loadData(image_data)
        # initialize done[stuff] to False
        self.donePCA = False
        self.doneComps = False
        self.doneStats = False
        self.suppress_warnings = False

    def __user_in_yn__(self):
        '''for internal use, gets a binary yes/no user input'''
        inp = ''
        while inp not in ['y','n']:
            inp = str(input('(y/n): ')).lower()
            if inp == 'y':
                return True
            elif inp == 'n':
                return False

    def loadData(self,image_data):
        '''loads the image data into the PCA class and gets basic data about it.
        image_data can be a torch tensor with image data or a filepath for the data'''
        # if it's a file path then load the file
        if type(image_data) == str:
            image_data = torch.load(image_data)
        # localize the data
        self.raw_images = image_data

        # get the size of the data
        raw_images_size = self.raw_images.size()

        # figure out what kind of image data it is (RBG?) and get data about it's shape
        if len(raw_images_size) == 3:
            self.RGB = False # not rgb data
            self.n_samp, self.imy, self.imx = raw_images_size # localize info about the data
            self.n_vars = self.imy * self.imx # total number of variables
        elif len(raw_images_size) == 4:
            self.RGB = True # rgb data
            self.n_samp, self.imy, self.imx, _ = raw_images_size # localize info about images
            self.n_vars = self.imy * self.imx * 3 # total number of variables
        else:
            raise AttributeError('image data is not formatted correctly')

        # vectorize the image data
        self.raw_data = self.raw_images.reshape(self.n_samp,self.n_vars)

    def doPCA(self, k='max', rescl_method='normalize', GPU=False, GPU_name='cuda:0', suppress_warnings=False, calc_stats=True, calc_comps=True, show_progress=False):
        '''this function performs the PCA algorithm to the data set, it returns
        None, and all the information it generates is locally stored in the class

        arguments:
        k - number of eigenvectors to generate, a larger k means better accuracy but
            it must be between 1 and n_vars/3. 'max' is also an acceptable value
        rescl_method - the method used to rescale the data, acceptable arguments can
            be found in PCA.stat.rescl_dict.keys()
        GPU - one of the calculations run by this program can be pretty time-costly
            running on the cpu, so setting this to true can speed up the calculations
        GPU_name - the name of the GPU you wish to use if GPU=True (default is cuda:0)
        suppress_warnings - if set to true, no warnings will be generated while running
        calc_stats - if true, statistics on the data will automatically be calculated
        calc_comps - if true, new components of the data will automatically be calculated
        show_progress - if true, the class will print out some messages when it starts/
            finishes various calculations, and how long each of those calculations took'''
        # step -1) do some shit with the timer
        self.timer.show_progress =  show_progress # tell the timer weather or not it's activated
        self.timer.readout('starting PCA') # first timer readout

        # step 0, check if PCA has already been done
        self.suppress_warnings = suppress_warnings # localize this variable through out this class (will b reser)
        if self.donePCA == True and self.suppress_warnings == False:
            warn('PCA has already been completed for this data, are you sure you want to proceed?')
            if self.__user_in_yn__() == False:
                return None # do nothing if user replies no

        # STEP 1) rescale the data

        # initialize a local stat class
        self.stat = self.Stat(rescl_method)

        # now apply the rescaling method to the data
        self.timer.start('rescaling data')
        self.data = self.stat.scale(self.raw_data)
        self.timer.end()

        # STEP 2) calculate the covariance matrix
        self.timer.start('calculating covariance matrix')
        dataT = torch.transpose(self.data,dim0=0,dim1=1) # transposed version of the data
        covmat = torch.matmul(dataT,self.data) # math magic! for quick cov. mat. calculation
        self.timer.end()

        # STEP 3) calculate eigenstuffs!

        # make sure the value of k is aye o-K (haha!)
        if k == 'max':
            self.k = self.n_vars//3-1
        elif k > self.n_vars//3:
            if suppress_warnings == False:
                warn('k=%d is too large, using k=%d (max) instead' % (k,self.n_vars//3-1))
            self.k = self.n_vars//3-1
        elif k <= self.n_vars//3:
            self.k = int(k)
        else:
            raise ValueError('%s is not a valid value for k' % str(k))

        # check if we want to use the GPU for this
        if GPU == True:
            covmat = covmat.view(-1,self.n_vars,self.n_vars)
            covmat = covmat.to(GPU_name)
        else:
            covmat = covmat.view(-1,self.n_vars,self.n_vars)

        # do the calculation
        self.timer.start('calculating eigenstuffs')
        eigvals,eigvecs = torch.lobpcg(covmat,k=self.k)
        self.timer.end()

        # send shit back to the cpu(?)
        if GPU == True:
            eigvals,eigvecs = eigvals.cpu(),eigvecs.cpu()

        # reformat and localize eigenstuffs
        self.eigvals = eigvals[0]
        self.eigvecs = torch.transpose(eigvecs[0],dim0=0,dim1=1)

        # finally, update donePCA
        self.donePCA = True

        if calc_stats == True: # if stats are wanted, calculate those
            self.timer.start('calculating statistics')
            self.calcStats()
            self.timer.end()

        if calc_comps == True: # if components are wanted, calculate those
            self.timer.start('calculating new components')
            self.calcComps(GPU=GPU, GPU_name=GPU_name)
            self.timer.end()

        self.timer.show_progress = False # reset back to false
        self.suppress_warnings = False # reset to false
        pass

    def calcStats(self):
        '''this function generates statistics about the eigenvals/vecs that are
        found by the doPCA function. by default, this function is called with doPCA'''
        if self.doneStats == True and self.suppress_warnings == False: # check if statistics have already been calculated
            warn('stats have already been calculated, are you sure you want to proceed?')
            if self.__user_in_yn__() == False:
                return None
        # SCREE STATS
        self.totVar = torch.sum(self.eigvals) # total variation
        self.screeScores = self.eigvals/self.totVar # variation per dimension
        self.screePcts = self.screeScores*100 # variation per dimension (%)
        self.cumSum = [torch.sum(self.screeScores[:i+1]).view(1,-1) for i in range(self.k)]
        self.cumSum = torch.cat(self.cumSum) # cummulative sum of variation per dim
        self.__scree_warnings__() # generate some warnings about the scree scores
        pass

    def calcComps(self, GPU=False, GPU_name='cuda:0', show_progress = True):
        '''calculates new components of all data points under the new basis of eigenvectors'''
        if self.doneComps == True and self.suppress_warnings == False: # check if statistics have already been calculated
            warn('components have already been calculated, are you sure you want to proceed?')
            if self.__user_in_yn__() == False:
                return None

        # are we using the GPU?
        if GPU: # if yes, send all that shit to the gpu
            self.data.to(GPU_name)
            self.eigvecs.to(GPU_name)

        # clever matrix multiplication to get components sUPA quick
        self.comps = torch.matmul(self.data,torch.transpose(self.eigvecs,dim0=0,dim1=1))

        # did we use the GPU?
        if GPU: # if yes, send all that shit back to the cpu
            self.comps.cpu()
            self.data.cpu()
            self.eigvecs.cpu()

    def save(self,fpath):
        '''save the instance of this class (and all the calculations therin)
        to the file under fpath'''
        self.stat_dict = self.stat.__dict__
        torch.save(self.__dict__,fpath)

    def load(self,fpath):
        '''loads a previously used instance of this class'''
        self.__dict__ = torch.load(fpath)
        self.stat.loadSelf(self.stat_dict)

    def __scree_warnings__(self):
        '''this function is for internal use, and is used to readout some warnings
        incase the data indicates that the scree data may not be good'''
        if torch.sum(self.screeScores[-1-self.k//10:]) > 0.1:
            warn('the smallest 10% of dimensions account for > 10% of the variation,\
                  scree scores may not be accurate')
        if self.k < self.n_vars/10:
            warn('fewer than 10% of all eigen vectors have been calculated,\
                  scree scores may not be accurate')

    def PCAPlot(self,ndim=2):
        '''makes a plot of the converted data to help visualize the clumping. ndim
        specifies the number of dimensions you wish to show on the graph, can be 1, 2, or 3'''
        if ndim not in [1,2,3]:
            raise ValueError('%d is not a valid number of dimensions for the PCA plot' % ndim)
        # get the x/y/z data for the number of dimensions specified
        xs = np.array(self.comps[:,0]).flatten() #xs
        if ndim == 1:
            ys = np.zeros(np.shape(xs)) #ys = 0 if ndim = 1
        if ndim == 2 or ndim == 3:
            ys = np.array(self.comps[:,1]).flatten() # ys
        if ndim == 3:
            zs = np.array(self.comps[:,2]).flatten() # zs
        # get the variation accounted for by axis/total
        scrs = [round(self.screePcts[i].item(),3) for i in range(ndim)]
        totvar = str(round(sum(scrs),3))+'%'
        scrs = [str(s)+'%' for s in scrs]
        # plot that shit!
        if ndim in [1,2]:
            plt.scatter(xs,ys,color='b',marker='o') # plot xs and ys
            plt.title('PCA Plot Representing '+totvar+' of Variation')
            plt.xlabel('PCA0 - '+scrs[0])
            if ndim == 2:
                plt.ylabel('PCA1 - '+scrs[1]) # label the y axis if its a 2d plot
        elif ndim == 3:
            fig = plt.figure()
            ax = fig.add_subplot(projection = '3d')
            ax.scatter(xs,ys,zs,c='b',marker='o')
            ax.set_title('PCA Plot Representing '+totvar+'% of Variation')
            ax.set_xlabel('PCA0 - '+scrs[0])
            ax.set_ylabel('PCA1 - '+scrs[1])
            ax.set_zlabel('PCA2 - '+scrs[2])
        plt.show()

    def screePlot(self, n=None, pctCap=None, graph_type=None):
        '''displays a plot of variation by eigenvector
        args: n - if specified, only the top n eigenvectors will be plotted; pctCap -
        specifies a percentage (between zero and one), so the graph will only display
        scree scores until the total is greater than the percentage here; graph_type -
        can be 'line' or 'bar', but the function will automatically choose this for you if left blank'''
        # figure out what data to display
        if n != None:
            data = self.screePcts[:n] # if n is specified
        elif pctCap != None:
            n = torch.sum(self.cumSum*100<pctCap)
            data = self.screePcts[:n]
        else:
            n = 10 # default number of data points to show
            data = self.screePcts[:n]
        tvar = self.cumSum[n-1].item()*100 # total variation accounted for by data
        tvar = round(tvar,3) # make it pretty...
        data = np.array(data).flatten() # data to numpy array

        # figure out the type of graph to show, if none is specified
        if graph_type == None:
            if n > 50:
                graph_type = 'line'
            else:
                graph_type = 'bar'

        # plot that shit
        if graph_type == 'line':
            plt.plot(data)
        elif graph_type == 'bar':
            xs = list(range(1,n+1))
            plt.bar(xs,data)
        # titles and shit to make it ~~~pretty~~~
        plt.title('Scree Plot Representing '+str(tvar)+'% of Variation')
        plt.xlabel('Eigenvectors by Rank')
        plt.ylabel('Percentage of Variation Accounted For')
        plt.show()

    def showImage(self,img=None, n=None, title=None, subtitle=None):
        '''the end-all be-all function to show a SINGLE image from this class. img argument
        can be set to an integer, in which case it will show the data point at the index;
        a list of integers, in which case it will show the images at all indexes in the list;
        a torch tensor of an image, in which case it will show the image in the array; a set
        of images as a torch tensor, in which case it will show each image; a torch tensor of
        a data point (vector), in which case it will unscale it and show the original image; or
        a torch tensor of data pointS, in which case it will unscale and show each one.
        title and subtitle should be set to a list of titles/subtitles for sets of data.
        alternitavely to img, n can be set to a number and n random images from the data will be shown'''

        if type(img)==int: # single image index is given
            # img is the index of the desired image in the dataset
            imgarr = self.raw_images[img] # load the image
            imgarr = imgarr/torch.max(imgarr) # rescale the values between zero and one for matplotlib
            imgarr = np.array(imgarr.cpu()) # send it to an np array

            if self.RGB: # if it's rgb data just show it
                plt.imshow(imgarr)
            else: # otherwise show it as black/white
                plt.imshow(imgarr,cmap='gray')

            if type(title)==str: # if theres a title, title it that
                plt.title(title)
            if type(subtitle)==str: # if theres a subtitle, use that
                plt.xlabel(subtitle)
            else: # otherwise just
                plt.xlabel('index = '+str(img)) # subtitle it with the index

            plt.show()
            return None # stop right there

        elif type(img) == list: # list of image indecies are given
            # img is a list of indecies of images to display
            for i in range(len(img)):
                index = img[i] # index of the image to display
                try: # try to get the title of the image
                    t = title[i]
                except:
                    t=None
                self.showImage(img=index,title=t) # call this class to show the image
            return None # stop right there
        elif type(img) == type(None) and type(n) == int: # show n random images
            for _ in range(n):
                i = random.randint(0,self.n_samp-1)
                self.showImage(img=i)
            return None # stop right there
        elif type(img) != torch.tensor: # then idk wtf to do
            raise TypeError('%s is not an acceptable type for img' % type(img))

        print(type(img))
        imgsize = img.size() # get the tensor's shape

        if len(imgsize)==1 and imgsize[0] == self.n_vars: # single datapoint!!!
            imgarr = self.stat.unScale(img) # unscale the datapoint
            imgarr = imgarr/torch.max(imgarr) # make the max one for matplotlib
            imgarr = np.array(imgarr) # to nparray

            if self.RGB: # reshape and display the data propperly
                imgarr = imgarr.reshape(self.imy,self.imx,3)
                plt.show(imgarr)
            else:
                imgarr = imgarr.reshape(self.imy,self.imx)
                plt.imshow(imgarr,cmap='gray')

            if type(title)==str: # if theres a title, title it
                plt.title(title)
            if type(subtitle)==str: # if theres a subtitle, use that
                plt.xlabel(subtitle)

            plt.show()
            return None # stop right there

        elif len(imgsize)==2 and imgsize[1] == self.n_vars: # a set of datapoints is given
            for i in range(imgsize[0]):
                imgdata = img[i] # get the image data
                try: # try to get the title
                    t = title[i]
                except:
                    t = None
                try: # try to get the subtitle
                    st = subtitle[i]
                except:
                    st = 'index = '+str(i) # otherwise just use the index
                # call this class to show the image data
                self.showImage(imgdata, title=t, subtitle = st)
            pass
            return None # stop right there
        elif imgsize[0] == self.imy and imgsize[1] == self.imx: # an image is given
            imgarr = img/torch.max(img) # make image data between zero and one
            imgarr = np.array(imgarr)

            if self.RGB: # if it's rgb data just show it
                plt.imshow(imgarr)
            else: # otherwise show it as black/white
                plt.imshow(imgarr,cmap='gray')

            if type(title)==str: # if theres a title, title it
                plt.title(title)
            if type(subtitle)==str: # if theres a subtitle, use that
                plt.xlabel(subtitle)

            plt.show()
            return None # stop right there
        elif imgsize[1] == self.imy and imgsize[2] == self.imx: # a set of images are given
            for i in range(imgsize[0]):
                imgarr = img[i] # get the image data
                try: # try to get the title
                    t = title[i]
                except:
                    t = None
                try: # try to get the subtitle
                    st = subtitle[i]
                except:
                    st = 'index = '+str(i) # otherwise just use the index
                # call this class to display image
                self.showImage(img=imgarr,title=t,subtite=st)
            return None # stop right there
        pass

    def showManyImages(self,image_data,titles=None):
        '''makes a plot to show ALL the images in image_data at once. the image
        data argument must be tensors with shape (_, imy, imx[, 3]) and it must
        be propperly scaled (all values between zero and one)'''
        nimgs = len(image_data) # number of images to show
        # make sure that the titles list and number of images match
        if titles and len(titles) != nimgs:
            raise AttributeError('The length of the titles list and number of images must match')
        # figure out rows / columns
        rows = int(nimgs**0.5) # number of columns of images to display
        columns = nimgs/rows # number of rows to display
        if columns % 1 != 0: # make rows an integer such that rows*columns >= ndim
            columns = int(columns+1)
        else:
            columns = int(columns)
        # setup the plot stuffs
        axes = []
        fig = plt.figure()
        # main loop
        for i in range(nimgs):
            # get the image
            img = image_data[i]
            # now show the image
            axes.append(fig.add_subplot(rows,columns,i+1)) # add the subplot to the axes list
            if titles:
                axes[-1].set_title(titles[i]) # title the subplot with the dimension number
            plt.imshow(img) # show the image on the subplot
        plt.show()
        pass

    def showImageFromData(self,datapoint,title=None, subtitle=None):
        '''converts a datapoint back to the image that it's representing and displays that image'''
        # unscale the datapoint
        datapoint = self.stat.unScale(datapoint)
        # reshape the datapoint
        if self.RGB:
            img = datapoint.reshape(self.imy,self.imx,3)
        else:
            img = datapoint.reshape(self.imy,self.imx)
        # put it in the right range
        img = img - torch.min(img)
        img = img/torch.max(img)
        img = np.array(img) # make it a numpy array
        plt.imshow(img)
        if title: # if theres a title provided, use it
            plt.title(title)
        if subtitle: # if theres a subtitle provided, use it
            plt.xlabel(subtitle)
        plt.show()

    def getCompsData(self,data,ndim=None):
        '''takes in a single datapoint or a set of datapoints, and returns the
        components of the data under the basis of the eigenvectors. if ndim is
        a number, it will only return (ndim) components of each vector'''
        # is the data a point or a list of points?
        if len(data.size()) == 1:
            singlepoint = True
        else:
            singlepoint = False
        # format the data correctly
        data = data.reshape(-1,self.n_vars)
        # ~~~clever matrix multiplication~~~ to get the components
        comps = torch.matmul(data,torch.transpose(self.eigvecs,dim0=0,dim1=1))
        # if its a single point, only return a list of components
        if singlepoint == True:
            comps = comps[0]
        # figure out how many components to return
        if ndim == None:
            return comps
        else:
            return comps[:ndim]

    def visCompression(self,nimgs=1,ndim=10, mindim=0):
        '''a way to visualize how well images are preserved under the basis of
        the eigenvectors.'''
        if type(ndim) == int: # if ndim is an integer, format it as a list
            ndim = [ndim]
        # main loop
        for _ in range(nimgs): # do this once for each image requested
            ri = random.randint(0,self.n_samp-1) # pick a random image
            # make the image data tensor
            if self.RGB: # figure out the size of each image
                imgsize = (self.imy,self.imx,3)
            else:
                imgsize = (self.imy,self.imx)
            image_data = torch.zeros(len(ndim)+1,*imgsize) # image data tensor
            img = self.raw_images[ri] # first image (original)
            img = img-torch.min(img) # make the min zero
            img = img/torch.max(img) # make the max one
            image_data[0] = img # first image (original)
            nextidx = 1 # next image index
            titles = ['Original'] # titles list
            # get the data ready
            comps = self.comps[ri,mindim:] # get components of the datapoint after mindim
            recimg = torch.zeros(self.data[ri].size()) # image recreation (all zeros)
            for i in range(max(ndim)): # loop through 0 to max(ndim)
                recimg = recimg + comps[i]*self.eigvecs[mindim+i] # add scaled basis to the recreation
                if i+1 in ndim: # if we're on one of the requested dimensions
                    img = self.stat.unScale(recimg) # unscale the recreated image
                    img = img-torch.min(img) # make the min zero
                    img = img/torch.max(img) # make the max one
                    img = img.reshape(imgsize) # reshape the image data
                    image_data[nextidx] = img # add the image
                    nextidx += 1 # increment the index
                    titles.append('Dim'+str(mindim)+'--Dim'+str(mindim+i+1)) # add the title
            self.showManyImages(image_data,titles)
        pass

    def visualizePCA(self,ndim=10,mindim=0):
        '''displays the principal components as though they are images to visualize
        what each one might correspond to. shows components mindim through mindim+ndim'''
        # setup the image data
        if self.RGB:
            image_data = torch.zeros(ndim,self.imy,self.imx,3)
        else:
            image_data = torch.zeros(ndim,self.imy,self.imx)
        titles = [] # list of titles
        # main loop!
        for i in range(ndim):
            # start by getting the image
            img = self.eigvecs[i+mindim] # image data from the i+mindim th eigenvector
            img = self.stat.unScale(img) # un-scale the data as though it's an image
            # need to force image data between zero and one
            img = img-torch.min(img)
            img = img/torch.max(img)
            # now resize image data (depending on if it's rgb or not)
            if self.RGB:
                img = img.reshape(self.imy,self.imx,3)
            else:
                img = img.reshape(self.imy,self.imx)
            # and put the image and title in the tensor/list
            image_data[i] = img # add the image to the tensor
            titles.append('Dim'+str(i+mindim)) # add the title to the list
        self.showManyImages(image_data,titles)
        pass

    def visualizeComponents(self,nimgs=1,ncomps=100,mincomp=0):
        '''this visualizes nimgs of random images, then plots below them their component values
        by component as a bar graph. pretty cool, but i aint found a practical use for it yet.'''
        # setup the plot
        rows = 2 # 2 rows
        columns = nimgs # 1 column
        axes = []
        fig = plt.figure()
        for imgn in range(nimgs):
            i = random.randint(0,self.n_samp) # pick a random index
            # get the image data
            img = self.raw_images[i] # get the image
            img = img/torch.max(img) # scale the image right
            # get the component data
            ys = self.comps[i,mincomp:ncomps]
            xs = list(range(mincomp+1,ncomps+1))
            # plot the image!
            axes.append(fig.add_subplot(rows,columns,1+imgn))
            axes[-1].set_title('Image')
            plt.imshow(img)
            # plot the components
            axes.append(fig.add_subplot(rows,columns,1+imgn+nimgs))
            axes[-1].set_xlabel('Components by #')
            axes[-1].set_ylabel('Component Values')
            plt.bar(xs,ys)
        # show that shit
        plt.show()

###########################################
