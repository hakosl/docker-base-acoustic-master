import numpy as np

from batch.samplers.sampler import Sampler
from batch.samplers.shool import Shool

class ShoolSeabed(Shool):
    def __init__(self, *args, **kwargs):
        """

        :param echograms: A list of all echograms in set
        """
        super().__init__(*args, **kwargs)
        
        
        self.max_dist_to_seabed = kwargs["window_size"][0] // 2
        

        #Remove shools that are not close to seabed
        self.shools = \
            [(e, o) for e, o in self.shools if
             np.abs(e.get_seabed()[int((o['bounding_box'][2] + o['bounding_box'][3]) / 2)] - o['bounding_box'][1]) <
             self.max_dist_to_seabed]

