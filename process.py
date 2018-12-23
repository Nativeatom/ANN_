""" Generate emnist-mini.mat



Before you run the code:

1) install numpy, scipy, tensorflow

2) download emnist-letters.mat into data folder

"""



from os import path

from scipy.io import loadmat, savemat

import numpy as np

import tensorflow as tf



m = loadmat(path.join(path.dirname(__file__), "..", "data", "emnist-letters.mat"), struct_as_record=False)

x = m["dataset"][0, 0].train[0, 0].images

y = m["dataset"][0, 0].train[0, 0].labels.T[0]



x_training = []

y_training = []

x_validate = []

y_validate = []

for i in range(1, 27):

    letter_i = x[y == i]

    x_training.append(letter_i[4000:])

    x_validate.append(letter_i[:50])

    y_t_i = np.zeros(shape=(800, 26), dtype=np.float32)

    y_t_i[:, i-1] = 1

    y_training.append(y_t_i)

    y_v_i = np.zeros(shape=(50, 26), dtype=np.float32)

    y_v_i[:, i-1] = 1

    y_validate.append(y_v_i)



x_t = np.concatenate(x_training)

y_t = np.concatenate(y_training)

x_v = np.concatenate(x_validate)

y_v = np.concatenate(y_validate)



savemat(path.join(path.dirname(__file__), "..", "data", "emnist-mini.mat"),

    {"x_t": x_t, "y_t": y_t, "x_v": x_v, "y_v": y_v})