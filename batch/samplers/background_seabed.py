import numpy as np
from utils.np import getGrid, nearest_interpolation
from batch.samplers.sampler import Sampler


class BackgroundSeabed(Sampler):
    def __init__(self, *args, **kwargs):
        """

        :param echograms: A list of all echograms in set
        """
        super().__init__(*args, **kwargs)
        
        self.echograms = kwargs["echograms"]
        self.window_size = kwargs["window_size"]

    def get_label(self):
        return 1
        

    def get_sample(self):
        """

        :return: [(int) y-coordinate, (int) x-coordinate], (Echogram) selected echogram
        """
        #Random echogram
        ei = np.random.randint(len(self.echograms))

        #Random x-loc with y-location of seabed
        x = np.random.randint(self.window_size[1]//2, self.echograms[ei].shape[1] - (self.window_size[1]//2))
        y = self.echograms[ei].get_seabed()[x]

        #Check if there is any fish-labels in crop
        grid = getGrid(self.window_size) + np.expand_dims(np.expand_dims([y,x], 1), 1)
        labels = nearest_interpolation(self.echograms[ei].label_memmap(), grid, boundary_val=0, out_shape=self.window_size)

        if np.any(labels != 0):
            return self.get_sample() #Draw new sample

 
        eg_s = self.echograms[ei].shape
        
        window_offset = [self.window_size[0]//2, self.window_size[1]//2]

        # bound x,y to be within window 
        x = max(min(x, eg_s[1] - window_offset[1]), window_offset[1])
        y = max(min(y, eg_s[0] - window_offset[0]), window_offset[0])

        xmin, xmax = x - (window_offset[1]), x + (window_offset[1])
        ymin, ymax = y - (window_offset[0]), y + (window_offset[0])


        return [[ymin, ymax],[xmin, xmax]], self.echograms[ei]
