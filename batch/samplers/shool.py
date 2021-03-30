import numpy as np
from batch.samplers.sampler import Sampler
class Shool(Sampler):
    def __init__(self, *args, **kwargs):
        """

        :param echograms: A list of all echograms in set
        """
        super().__init__(*args, **kwargs)
        
        self.echograms = kwargs["echograms"]
        self.window_size = kwargs["window_size"]

        self.random_sample = kwargs["random_sample"]
        self.fish_type = kwargs["fish_type"]

        self.shools = []
        #Remove echograms without fish
        if self.fish_type == 'all':
            self.echograms = [e for e in self.echograms if len(e.objects)>0]
            for e in self.echograms:
                for o in e.objects:
                    self.shools.append((e,o))

        elif type(self.fish_type) == int:
            self.echograms = [e for e in self.echograms if any([o['fish_type_index'] == self.fish_type for o in e.objects])]
            for e in self.echograms:
                for o in e.objects:
                    if o['fish_type_index'] == self.fish_type:
                        self.shools.append((e,o))

        elif type(self.fish_type) == list:
            self.echograms = [e for e in self.echograms if any([o['fish_type_index'] in self.fish_type for o in e.objects])]
            for e in self.echograms:
                for o in e.objects:
                    if o['fish_type_index'] in self.fish_type:
                        self.shools.append((e,o))

        else:
            class UnknownFishType(Exception):pass
            raise UnknownFishType('Should be int, list of ints or "all"')

        if len(self.echograms) == 0:
            class EmptyListOfEchograms(Exception):pass
            raise EmptyListOfEchograms('fish_type not found in any echograms')
    
    def get_name(self):
        if self.fish_type == 27:
            return f"{self.__class__.__name__}, Sandeel"
        elif self.fish_type == 1:
            return f"{self.__class__.__name__}, Other"
        else:
            return f"{self.__class__.__name__}, fish_type: {self.fish_type}"

    def get_label(self):
        if self.fish_type == 1:
            return 2
        elif self.fish_type == 27:
            return 3
    
    def get_sample(self):
        """

        :return: [(int) y-coordinate, (int) x-coordinate], (Echogram) selected echogram
        """
        #Random object

        oi = np.random.randint(len(self.shools))
        e,o  = self.shools[oi]

        #Random pixel in object
        if self.random_sample:
            pi = np.random.randint(o['n_pixels'])
            y,x = o['indexes'][pi,:]
        # middle pixel in the object
        else:
            ymax, xmax = np.amax(o["indexes"], 0)
            ymin, xmin = np.amin(o["indexes"], 0)

            
            xdim, ydim = xmax - xmin, ymax - ymin
            
            #if xmax == xmin or ymax == ymin:
            #    return self.get_sample()
            xmid = np.random.randint(xmin - 20, xmax + 20)
            ymid = np.random.randint(ymin - 20, ymax + 20)
            #xmid, ymid = (xmax + xmin) / 2, (ymax + ymin) / 2


            # find largest axis and make the other axis as big as that one
            if xdim > ydim:
                ymax = ymid + (xdim/2)
                ymin = ymid - (xdim/2)
            else:
                xmax = xmid + (ydim/2)
                xmin = xmid - (ydim/2)
        
            offset = 32
            ymax, xmax = ymax + offset, xmax + offset
            ymin, xmin = ymin - offset, xmin - offset
            
            ymin, ymax = max(ymin, 0), min(ymax, self.shools[oi][0].shape[0])
            xmin, xmax = max(xmin, 0), min(xmax, self.shools[oi][0].shape[1])
            
            y = [ymin, ymax]
            x = [xmin, xmax]


        return [y,x], e
