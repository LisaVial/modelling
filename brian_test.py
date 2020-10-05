from brian2 import *
import brian2.units.constants as con
import numpy as np


C = 1*uF

g_Na = 100 * mS/cm2
g_p = 1 * mS/cm2
g_K = 80 * mS/cm2
g_AHP = 1.5 * mS/cm2

phi = 1
E_Ca = 120 * mV
g_Ca = 1 * mS/cm2
tau_Ca = 80 * ms

epsilon = 0.002 * ms * uA ** -1 * cm2
J_E = np.arange(3.4, 4, 0.1) * mA
rho_KCC = 0.3 * mM/second
rho_NKCC = 0.1 * mM/second

rho_pump = 0.25 * mM/second
Na_sat = 22 * mM
K_sat = 3.5 * mM

# conversion factors
beta = 4.0
gamma = 4 * np.pi * (3 * (1.4368 * 10 ** -9 * cm3)) / con.faraday_constant * (1.4368 * 10 ** -9 * cm3)
# longer term in brackets is volume of neuron
gamma_i = 0.65 * gamma
tau = 1000

epsilon_K = 0.4 * second**-1
K_bath = 3.5 * mM

g_ClL = 0.15 * mS/cm2
g_KL = 0.05 * mS/cm2
g_NaL = 0.0015 * mS/cm2


eqs = Equations('''
# membrane equation with 
dv/dt = J_E - I_L - I_K - I_Na - I_Nap - I_AHP: volt
dn/dt = phi * (alpha_n * (1 - n) - beta_n * n): 1
dh/dt = phi * (alpha_h * (1 - h) - beta_h * h): 1
dCa/dt = -epsilon * g_Ca * m_inf1 * (v - E_Ca) - (Ca/tau_Ca): 1

# current equations
I_L = g_L (v - E_L): amp
I_K = g_K * n ** 4 * (v - E_K): amp
I_Na = g_Na * m ** 3 * h * (v - E_Na): amp
I_Nap = g_p * m ** 3 * (v - E_Na): amp
I_AHP = g_AHP * (Ca / Ca + 1) * (v - E_K): amp
m_inf = alpha_m / (alpha_m + beta_m): Hz
alpha_m = 0.32 * (v + 54)/(1 - exp(-(v+54)/4): Hz
beta_m = 0.28 * (v + 27)/(exp(v+27/5) - 1): Hz
alpha_h = 0.128 * exp(-(v+50)/18): Hz
beta_h = 4/(1 + exp(-(v +27)/5)): Hz
alpha_n = 0.032 * (v + 52) / (1 - exp(-(v + 52)/5): Hz
m_inf1 = 1 / (1 + exp(-(v+25)/2.5)): Hz
beta_n = 0.5 * exp(-(v+57)/40): Hz

# evolution equations for ionic currents
d[K]o/dt = 1/tau * (gamma * beta * (I_K + I_AHP + I_KL - 2 * I_pump) + beta * (I_KCC + INKCC) - I_sink + gamma_i * beta 
            * I_Ki): mM
d[Na]i/dt = 1/tau * (-gamma * (I_Na + I_Nap + I_NaL + 3*I_pump) - I_NKCC): mM
d[Cl]i/dt = 1/tau * (gamma * (I_GABA + I_ClL) - I_KCC - 2*I_NKCC): mM 

# co transporters
I_KCC = rho_KCC * log(([K]i*[Cl]i)/([K]o*[Cl]o)): mM
I_NKCC = rho_NKCC * f_Ko * (log(([K]i*[Cl]i)/([K]o*[Cl]o)) + log(([Na]i*[Cl]i)/([Na]o*[Cl]o))): mM
f_Ko = 1 / (1 + exp(16-[K]o): mM

# sodium potassium pump
I_pump = rho_pump * (1/gamma) * (1/(1+exp((Na_sat-[Na]i)/3) * ( 1/ (1 + exp(K_sat - [K]o))): amp

# potassium sink
I_sink = epsilon_K * ([K]o - K_bath): amp

# dynamics of complementary ion concentrations
d[K]i/dt = - 1/tau * (gamma * (I_K + I_AHP + I_KL - 2 * I_pump) + I_KCC + I_NKCC): mM
d/dt * (beta * [Na]i + [Na]o) = d/dt * (beta * [Cl]i + [Cl]o) = 0: mM

# reversal potentials
E_K = (RT/F) * log([K]o/[K]i): mV
E_Na = (RT/F) * log([Na]o/[Na]i): mV
E_Cl = (RT/F) * log([Cl]o/[Cl]i): mV

# leak current
I_L = g_L * (v - E_L): amp
g_L = (g_NaL + g_KL + g_ClL): 1
E_L = 1/g_L * (g_NaL * E_Na + g_KL * E_K + g_ClL * E_Cl): mV

''')

P = NeuronGroup(4000, model=eqs, threshold='v>-20*mV', refractory=3*ms,
                method='exponential_euler')
Pe = P[:3200]
Pi = P[3200:]
Ce = Synapses(Pe, P, on_pre='ge+=we')
Ci = Synapses(Pi, P, on_pre='gi+=wi')
Ce.connect(p=0.02)
Ci.connect(p=0.02)

# Initialization
P.v = 'El + (randn() * 5 - 5)*mV'
P.ge = '(randn() * 1.5 + 4) * 10.*nS'
P.gi = '(randn() * 12 + 20) * 10.*nS'

# Record a few traces
trace = StateMonitor(P, 'v', record=[1, 10, 100])
run(1 * second, report='text')
plot(trace.t/ms, trace[1].v/mV)
plot(trace.t/ms, trace[10].v/mV)
plot(trace.t/ms, trace[100].v/mV)
xlabel('t (ms)')
ylabel('v (mV)')
show()