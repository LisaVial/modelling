import numpy as np
from numba import jit, types
from numba.typed import Dict
import matplotlib.pyplot as plt

from .integrate_parameter import IntegrateParameter


class NeuronModel:
    def __init__(self, modeltype='TraubMiles'):
        self.current_par, self.global_par = self.genereate_neuron_model(modeltype)

    def run(self, I_in):
        t = np.arange(len(I_in)) * self.global_par['dt']
        variable_dic = self.initialize(t, self.global_par['V_0'])
        variable_dic['V'] = t * 0.0
        np.random.seed(None)
        for i in range(len(t) - 1):
            for current in self.current_par.keys():
                for gate in self.current_par[current]['gates'].keys():
                    variable_dic[gate][i + 1] = getattr(self.IntegrateParameter,
                                                        self.current_par[current]['gates'][gate]['inttype']) \
                        (self, variable_dic[gate][i], variable_dic['V'][i], self.current_par[current]['gates'][gate])
            I_gate = np.sum(
                [self.channelcurrent(self.current_par[I_x], variable_dic, i + 1, variable_dic['V'][i]) for I_x
                 in self.current_par.keys])

            dV = (I_gate + I_in[i + 1]) / self.global_par['C_m']
            variable_dic['V'][i + 1] = IntegrateParameter.euler_int(self, variable_dic['V'][i], dV)

    def generate_neuron_model(self, modeltype):
        if modeltype == None:
            self.current_par = {}
            self.global_par = {}
        elif modeltype == 'TraubMiles':
            global_par = {
                'dt': .01,  # ms
                # Lukas sets here all the Es, but in the Traub Miles model from the Desroches paper, these parameters
                # are 'slowly varying  variables, as their variation is the cause of the depolorizing block'
                'E_Ca': 120,  # mV

                'C_m': 1.,  # muF
                'V_0': -2.210976,  # muV
                'epsilon': 0.002  # (msmuA^-1)*cm^2
            }
            # first, I'll try to implement the ODEs for the model, by doing so I'll see which parameters of currents are
            # fixed
            current_par = {'I_Na': {'g': 100,  # mS/cm^2
                                    }, 'I_Ca': {'g': 1, 'tau_Ca': 80  # ms
                                                }, 'I_K': {'g': 80,  # mS/cm^2
                                                           }, 'g_p': 1,  # mS/cm^2
                           'Phi': 1, 'g_AHP': 1.}
        return current_par, global_par
