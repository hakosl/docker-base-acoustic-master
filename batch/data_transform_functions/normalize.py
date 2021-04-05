from torchvision import transforms
import numpy as np
means = np.array([-61.2922, -62.5713, -65.3220, -66.7303,  50.8511])
stds = np.array([19.6664, 18.1856, 17.4163, 16.5581, 23.7778])

norm = transforms.Normalize(means, stds)
norm4 = transforms.Normalize(means[:4], stds[:4])
def normalize(data, labels, echogram, frequencies):
    
    if data.shape[1] == 4:
        return (data - means[:4].reshape(-1, 1, 1)) / stds[:4].reshape(-1, 1, 1), labels, echogram, frequencies
    return (data - means.reshape(-1, 1, 1)) / stds.reshape(-1, 1, 1), labels, echogram, frequencies
