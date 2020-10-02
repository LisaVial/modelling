from numba import jit, prange
import numpy as np
from scipy.integrate import odeint
from copy import deepcopy

from .neuron_model import NeuronModel


class Simulation:
    def __init__(self, model):
        self.dt = NeuronModel.global_par['dt']

@jit(nopython=True, cache=True)
def run_voltageclamp(v):
    alpha_m = 0.32 * (v + 54) / (1 - np.exp(-(v + 54) / 4))   # v may be equal to V_0
    beta_m = 0.28 * (v + 27) / (np.exp((v + 27) / 5) - 1)
    alpha_h = 0.128 * np.exp(-(v + 50) / 18)
    beta_h = 4 / (1 + np.exp(-(v + 27)/5))
    alpha_n = 0.032 * (v + 52) / (1 - np.exp(-(v + 52) / 5))
    m_inf1 = 1 / (1 + np.exp(-(v + 25)/2.5))
    beta_n = 0.5 * np.exp(-(v + 57)/40)

    m_inf = alpha_m/(alpha_m + beta_m)


# @jit(nopython=True, cache=True)
# def run_current_clamp():
