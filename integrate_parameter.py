import numpy as np
from numba import jit, types
from numba.typed import Dict
import matplotlib.pyplot as plt

from .neuron_model import NeuronModel

class IntegrateParameter:
    def __init__(self):
        self.model = NeuronModel

    def euler_int(self, x, dxdt):
        return x + dxdt * self.model.global_par['dt']