# stat class for statistics n shit
import torch

class Stat:
    '''stat class handels resizing/rescaling the data so that
    PCA/kmeans can better interpret it, as well as unscaling the data to
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









############
