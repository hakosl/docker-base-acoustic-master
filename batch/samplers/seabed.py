import numpy as np
from batch.samplers.sampler import Sampler

class Seabed(Sampler):
    def __init__(self, *args, **kwargs):
        """

        :param echograms: A list of all echograms in set
        """
        super().__init__(*args, **kwargs)
        
        self.echograms = kwargs["echograms"]
        self.window_size = kwargs["window_size"]


    def get_sample(self):
        """

        :return: [(int) y-coordinate, (int) x-coordinate], (Echogram) selected echogram
        """
        #Random echogram
        ei = np.random.randint(len(self.echograms))

        #Random x-loc
        x = np.random.randint(self.window_size[1]//2, self.echograms[ei].shape[1] - (self.window_size[1]//2))
        y = self.echograms[ei].get_seabed()[x] + np.random.randint(-self.window_size[0], self.window_size[0])

        # Correct y if window is not inside echogram
        if y < self.window_size[0]//2:
            y = self.window_size[0]//2
        if y > self.echograms[ei].shape[0] - self.window_size[0]//2:
            y = self.echograms[ei].shape[0] - self.window_size[0]//2

        eg_s = self.echograms[ei].shape
        
        window_offset = [self.window_size[0]//2, self.window_size[1]//2]

        # bound x,y to be within window 
        x = max(min(x, eg_s[1] - window_offset[1]), window_offset[1])
        y = max(min(y, eg_s[0] - window_offset[0]), window_offset[0])

        xmin, xmax = x - (window_offset[1]), x + (window_offset[1])
        ymin, ymax = y - (window_offset[0]), y + (window_offset[0])


        return [[ymin, ymax],[xmin, xmax]], self.echograms[ei]

