# Test cost functions

import numpy as np
from models.keras.unet import jaccard_distance_loss
from keras import backend as K


y_t = np.array([[0, 1], [0, 0]])
w = np.array([[0, 5], [1000, 1000]])

# correct
y_p = np.array([[0, 1], [0, 0]])
loss = jaccard_distance_loss(y_t, y_p, w)
K.get_value(loss)

# missed the fish - bad
y_p = np.array([[0, 0], [0, 0]])
loss = jaccard_distance_loss(y_t, y_p, w)
K.get_value(loss)

# misclass empty water as  fish - should be fine
y_p = np.array([[1, 1], [0, 0]])
loss = jaccard_distance_loss(y_t, y_p, w)
K.get_value(loss)

# bottom - VERY bad
y_p = np.array([[0, 1], [1, 1]])
loss = jaccard_distance_loss(y_t, y_p, w)
K.get_value(loss)
