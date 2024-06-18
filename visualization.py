import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import random

import models_approx as mod_ap

def uniform_maximum_error(y_true, y_pred):
    return max(abs(y_true - y_pred))

def model_test(x, name, min_iter = 1, max_iter = 15, show = True):

    name_to_params = {
        'square': (mod_ap.square_approximation, x**2),
        'cube': (mod_ap.cube_approximation, x**3),
        'exponent': (mod_ap.exponent_approximation, torch.exp(x))
    }
    
    try:
        model, y_true = name_to_params[name][0], name_to_params[name][1] 
    except KeyError as e:
        raise ValueError('Wrong model name: {}'.format(e.args[0]))
    
    input_size = len(x)
    errors = []
    predictions = []
    if show:
        plt.figure(figsize = (6, 6))
        plt.plot(np.sort(x.detach().numpy()), np.sort(y_true.detach().numpy()), label = "ground truth")

    for k_iter in range(min_iter, max_iter):
        model = name_to_params[name][0](k_iter, input_size)
        y_pred = model(x)
        errors.append(uniform_maximum_error(y_true, y_pred))
        #predictions.append(y_pred)
        if show:
            plt.plot(np.sort(x.detach().numpy()), np.sort(y_pred.detach().numpy()), label = "after {} iterations".format(k_iter))
    if show:
        plt.legend()
        plt.show()

    return errors