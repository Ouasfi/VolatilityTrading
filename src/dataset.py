import torch 
from pylab import *
#@title Relative positioning Sampler

class WSampler(torch.utils.data.sampler.Sampler):
    """A class to sample data points according to labels distribution
    
    Arguments:
    ---------
        dataset (Dataset): dataset to sample from
        batch_size (int): batch size
        weights (list): proportion of each target label in  a batch.
    """

    def __init__(self,dataset, batch_size,  weights):
    
        
        self.batch_size = batch_size
        self.dataset = dataset
        self.weights = torch.DoubleTensor(weights)
        self.size = len(self.dataset)
    def __iter__(self):
        """

        Yields:
            tuple: a tuple of a sample indice and a target class
        """        
        num_batches = self.size// self.batch_size
        while num_batches > 0:
            target  = torch.multinomial(
            self.weights, 1, replacement=True) 
            #t = choice(arange(0, self.size, 1))
            yield target
            
            num_batches -=1

    def __len__(self):
        return len(self.dataset)   
class Dataset(torch.utils.data.Dataset):
    """In Memory dataset class from a pandas DataFrame

    """    
    def __init__(self, df): 
        """In Memory dataset class from a pandas DataFrame

        Args:
            df (DataFrame): A DataFrame of features and targets
        """        
        self.df = df
    
    def __getitem__(self, y):
        sample = self.df[self.df['Target'] == y.item()].sample(1).drop(columns = 'Target')
        x = torch.from_numpy(sample.values).float()
        return x, y
    def __len__(self): return len(self.df)