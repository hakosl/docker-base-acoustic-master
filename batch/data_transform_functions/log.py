import numpy as np
EPS = 1e-10
def log(data, *args):
    return (np.log10(data + EPS), *args)