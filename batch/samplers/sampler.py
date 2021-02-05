class Sampler():
    def __init__(self, *args, **kwargs):
        if "sample_probs" in kwargs:
            self.sample_probability = kwargs["sample_probs"]
        else:
            self.sample_probability = 1

    def get_name(self):
        return self.__class__.__name__

    def __str__(self):
        return f"{self.get_name()}, sample_prob: {self.sample_probability}"
        
    def __repr__(self):
        return f"{self.get_name()}, sample_prob: {self.sample_probability}"
    def get_sample_probability(self):
        return self.sample_probability 
