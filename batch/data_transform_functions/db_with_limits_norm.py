from data.normalization import db
import warnings

def db_with_limits_norm(data, labels, echogram, frequencies):
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            data[data < 0.0] = 0.0
            ltz = data[data < 0.0]
            
            if ltz.shape[0] != 0:
                print("elements more than zero ", ltz)
            data = db(data)
            data[data>0] = 0
            data[data< -75] = -75
            data += 75
            data /= 75
        except Warning as w:
            print(data.shape, data, w)
    return data, labels, echogram, frequencies

def db_with_limits_norm_MSE(data, labels, echogram, frequencies):
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            data[data < 0.0] = 0.0
            ltz = data[data < 0.0]
            
            if ltz.shape[0] != 0:
                print("elements more than zero ", ltz)
            data = db(data)
            data[data>0] = 0
            data[data< -75] = -75
            data += 75
            data /= 75
            data *= 2
            data -= 1
        except Warning as w:
            print(data.shape, data, w)
    return data, labels, echogram, frequencies
