from brian2 import *
from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
from IPython import embed

# Simulations
# multiple runs
# start_scope()
#
# # parameters
# num_inputs = 100
# input_rate = 10 * Hz
# weight = 0.1
#
# # range of time constants
# tau_range = linspace(1, 10, 30) * ms
# # use this list to store output rates
# output_rates = []
# # construct the network each time
# P = PoissonGroup(num_inputs, rates=input_rate)
# eqs = '''
#    dv/dt = -v/tau: 1
#    '''
# G = NeuronGroup(1, eqs, threshold='v>1', reset='v=0', method='exact')
# S = Synapses(P, G, on_pre='v += weight')
# S.connect()
# M = SpikeMonitor(G)
# # store the current state of the network
# store()
# # iterate over range of time constants
# for tau in tau_range:
#     # restore the original state of the network
#     restore()
#     # run it and store the output firing rate in the list
#     run(1*second)
#     output_rates.append(M.num_spikes/second)
#
# plt.plot(tau_range/ms, output_rates)
# plt.xlabel(r'$\tau$ [ms]')
# plt.ylabel('firing rate [spikes/s]')
# plt.show()
#
# # reduce noise to investigate tau
# start_scope()
# num_inputs = 100
# input_rate = 10 * Hz
# weight = 0.1
# tau_range = linspace(1, 10, 30) * ms
# output_rates = []
#
# # construct poisson spikes just once
# P = PoissonGroup(num_inputs, rates=input_rate)
# MP = SpikeMonitor(P)
#
# # We use a network object because later on we don't want to include these objects
# net = Network(P, MP)
# net.run(1 * second)
#
# # and keep a copy of those spikes
# spikes_i = MP.i
# spikes_t = MP.t
# # now construct the network that we run each time
# # SpikeGeneratorGroup gets the spikes that we created before
# SGG = SpikeGeneratorGroup(num_inputs, spikes_i, spikes_t)
# eqs = '''
# dv/dt = -v/tau: 1
# '''
#
# G = NeuronGroup(1, eqs, threshold='v>1', reset='v=0', method='exact')
# S = Synapses(SGG, G, on_pre=' v+= weight')
# S.connect()
# M = SpikeMonitor(G)
#
# # store the current state of the network
# net = Network(SGG, G, S, M)
# net.store()
#
# for tau in tau_range:
#     # restore the original state of the network
#     net.restore()
#     # run it with the new value of tau
#     net.run(1*second)
#     output_rates.append(M.num_spikes/second)
# plt.plot(tau_range/ms, output_rates)
# plt.xlabel(r'$\tau$ [ms]')
# plt.ylabel('firing rate [spikes/s]')
# plt.show()

# conceptually more difficult, but also more efficient
# start_scope()
#
# num_inputs = 100
# input_rate = 10 * Hz
# weight = 0.1
# tau_range = linspace(1, 10, 30) * ms
# num_tau = len(tau_range)
# P = PoissonGroup(num_inputs, rates=input_rate)
# # We make a tau parameter of the group
# eqs = '''
# dv/dt = -v/tau: 1
# tau: second
# '''
# # and we have num_tau output neurons, each with a different tau
# G = NeuronGroup(num_tau, eqs, threshold='v>1', reset='v=0', method='exact')
# G.tau = tau_range
# S = Synapses(P, G, on_pre=' v += weight')
# S.connect()
# M = SpikeMonitor(G)
#
# # # now we can just run it without a loop
# run(1 * second)
# # output_rates = M.count/second   # firing rate is count/duration
# # plt.plot(tau_range/ms, output_rates)
# # plt.xlabel(r'$\tau$ [ms]')
# # plt.ylabel('firing rate [spikes/s]')
# # plt.show()
#
# # dependency of interspike intervals on tau
# trains = M.spike_trains()
# isi_mu = full(num_tau, nan) * second
# isi_std = full(num_tau, nan) * second
#
# for idx in range(num_tau):
#     train = diff(trains[idx])
#     if len(train) > 1:
#         isi_mu[idx] = mean(train)
#         isi_std[idx] = std(train)
# plt.errorbar(tau_range/ms, isi_mu/ms, yerr=isi_std/ms)
# plt.xlabel(r'$\tau$ [ms]')
# plt.ylabel('interspike interval [ms]')
# plt.show()
#
# changing things during a run
start_scope()
# parameters
area = 20000 * umetre ** 2
Cm = 1 * ufarad * cm ** -2 * area
gl = 5e-5 * siemens * cm ** -2 * area
El = -65 * mV
EK = -90 * mV
ENa = 50 * mV
g_na = 100 * msiemens * cm ** -2 * area
g_kd = 30 * msiemens * cm ** -2 * area
VT = -63 * mV
# the model
# eqs_HH = '''
# dv/dt = (gl * (El - v) - g_na * (m * m * m) * h * (v - ENa) - g_kd * (n * n * n * n) * (v - EK) + I)/Cm: volt
# dm/dt = 0.32 * (mV ** -1) * (13. * mV - v + VT) / (exp((13. * mV - v + VT)/
# (4. * mV)) - 1.)/ms * (1 - m) - 0.28 * (mV ** -1) * (v - VT - 40.*mV)/(exp((v - VT - 40.*mV)/(5.*mV))-1)/ms*m: 1
# dn/dt = 0.032 * (mV ** -1) * (15. * mV - v + VT)/
# (exp((15. * mV - v + VT)/(5. * mV)) - 1.) / ms * (1. - n) - .5 * exp((10. * mV - v + VT)/(40.*mV))/ms*n: 1
# dh/dt = 0.128 * exp((17. * mV - v + VT)/(18. * mV))/ms * (1. - h) - 4./ (1+exp((40. * mV - v + VT)/
# (5. * mV)))/ms*h: 1
# I: amp
# '''
# group = NeuronGroup(1, eqs_HH, threshold='v > -40*mV', refractory='v > -40*mV', method='exponential_euler')
# group.v = El
# statemon = StateMonitor(group, 'v', record=True)
# spikemon = SpikeMonitor(group, variables='v')
#
# plt.figure(figsize=(9, 4))
# for l in range(5):
#     group.I = rand() * 50 * nA
#     run(10 * ms)
#     plt.axvline(l*10, ls='--', c='k')
# plt.axhline(El/mV, ls='-', c='lightgray', lw=3)
# plt.plot(statemon.t/ms, statemon.v[0]/mV, '-b')
# plt.plot(spikemon.t/ms, spikemon.v/mV, 'ob')
# plt.xlabel('time [ms]')
# plt.ylabel('v [mV]')
# plt.show()
#
# # my network above behaves really strange in comparision to what is shown in the tutorial
#
# start_scope()
#
# group = NeuronGroup(1, eqs_HH, threshold='v > -40 * mV', refractory='v > -40 * mV', method='exponential_euler')
# group.v = El
#
# statemon = StateMonitor(group, 'v', record=True)
# spikemon = SpikeMonitor(group, variables='v')
#
# # we replace the loop from above with a run_regularly
# group.run_regularly('I = rand() * 50 * nA', dt = 10*ms)
# run(50*ms)
# plt.figure(figsize=(9,4))
# # we keep the loop just to draw vertical lines
# for l in range(5):
#     plt.axvline(l*10, ls='--', c='k')
# plt.axhline(El/mV, ls='-', c='lightgray', lw=3)
# plt.plot(statemon.t/ms, statemon.v[0]/mV, '-b')
# plt.plot(spikemon.t/ms, spikemon.v/mV, 'ob')
# plt.xlabel('time [ms]')
# plt.ylabel('V [mV]')
# plt.show()
#
# start_scope()
#
# group = NeuronGroup(1, eqs_HH, threshold='v > -40*mV', refractory='v > -40*mV', method='exponential_euler')
# group.v = El
#
# statemon = StateMonitor(group, 'v', record=True)
# spikemon = SpikeMonitor(group, variables='v')
#
# @network_operation(dt=10*ms)
# def change_I():
#     group.I = rand()*50*nA
# run(50*ms)
# plt.figure(figsize=(9, 4))
# for l in range(5):
#     plt.axvline(l*10, ls='--', c='k')
# plt.axhline(El/mV, ls='-', c='lightgray', lw=3)
# plt.plot(statemon.t/ms, statemon.v[0]/mV, '-b')
# plt.plot(spikemon.t/ms, spikemon.v/mV, 'ob')
# plt.xlabel('time [ms]')
# plt.ylabel('v [mv]')
# plt.show()

# multiple neurons
# start_scope()
# N = 3
# eqs_HH_2 = '''
# dv/dt = (gl*(El-v) - g_na*(m*m*m)*h*(v-ENa) - g_kd*(n*n*n*n)*(v-EK) + I)/C : volt
# dm/dt = 0.32*(mV**-1)*(13.*mV-v+VT)/
#     (exp((13.*mV-v+VT)/(4.*mV))-1.)/ms*(1-m)-0.28*(mV**-1)*(v-VT-40.*mV)/
#     (exp((v-VT-40.*mV)/(5.*mV))-1.)/ms*m : 1
# dn/dt = 0.032*(mV**-1)*(15.*mV-v+VT)/
#     (exp((15.*mV-v+VT)/(5.*mV))-1.)/ms*(1.-n)-.5*exp((10.*mV-v+VT)/(40.*mV))/ms*n : 1
# dh/dt = 0.128*exp((17.*mV-v+VT)/(18.*mV))/ms*(1.-h)-4./(1+exp((40.*mV-v+VT)/(5.*mV)))/ms*h : 1
# I : amp
# C : farad
# '''
# group = NeuronGroup(N, eqs_HH_2, threshold='v > -40*mV', refractory='v > -40*mV', method='exponential_euler')
# group.v = El
# # initialize with some different capacitances
# group.C = array([0.8, 1, 1.3])*ufarad*cm**-2*area
#
# statemon = StateMonitor(group, variables=True, record=True)
# group.run_regularly('I = rand()*50*nA', dt=10*ms)
# run(50*ms)
#
# plt.figure(figsize=(9, 4))
# plt.subplot(121)
# for l in range(5):
#     plt.axvline(l*10, ls='--', c='k')
# plt.axhline(El/mV, ls='-', c='lightgray', lw=3)
# plt.plot(statemon.t/ms, statemon.v.T/mV, '-')
# plt.xlabel('time [ms]')
# plt.ylabel('v [mV]')
# plt.subplot(122)
# plt.plot(statemon.t/ms, statemon.I.T/nA, '-')
# plt.xlabel('time [ms]')
# plt.ylabel('I [nA]')
# plt.show()
#
# start_scope()
# N = 3
# eqs_HH_3 = '''
# dv/dt = (gl*(El-v) - g_na*(m*m*m)*h*(v-ENa) - g_kd*(n*n*n*n)*(v-EK) + I)/C : volt
# dm/dt = 0.32*(mV**-1)*(13.*mV-v+VT)/
#     (exp((13.*mV-v+VT)/(4.*mV))-1.)/ms*(1-m)-0.28*(mV**-1)*(v-VT-40.*mV)/
#     (exp((v-VT-40.*mV)/(5.*mV))-1.)/ms*m : 1
# dn/dt = 0.032*(mV**-1)*(15.*mV-v+VT)/
#     (exp((15.*mV-v+VT)/(5.*mV))-1.)/ms*(1.-n)-.5*exp((10.*mV-v+VT)/(40.*mV))/ms*n : 1
# dh/dt = 0.128*exp((17.*mV-v+VT)/(18.*mV))/ms*(1.-h)-4./(1+exp((40.*mV-v+VT)/(5.*mV)))/ms*h : 1
# I : amp (shared) # everything is the same except we've added this shared
# C : farad
# '''
# group = NeuronGroup(N, eqs_HH_3,
#                     threshold='v > -40*mV',
#                     refractory='v > -40*mV',
#                     method='exponential_euler')
# group.v = El
# group.C = array([0.8, 1, 1.2]) * ufarad * cm ** -2 * area
# statemon = StateMonitor(group, 'v', record=True)
# group.run_regularly('I = rand()*50*nA', dt=10*ms)
# run(50*ms)
# plt.figure(figsize=(9, 4))
# for l in range(5):
#     plt.axvline(l*10, ls='--', c='k')
# plt.axhline(El/mV, ls='-', c='lightgray', lw=3)
# plt.plot(statemon.t/ms, statemon.v.T/mV, '-')
# plt.xlabel('Time (ms)')
# plt.ylabel('v (mV)')
# plt.show()

# adding input
# A = 2.5
# f = 10*Hz
# tau = 5*ms
# # create a TimedArray and set the equations to use it
# t_recorded = arange(int(200*ms/defaultclock.dt))*defaultclock.dt
# I_recorded = TimedArray(A*sin(2*pi*f*t_recorded), dt=defaultclock.dt)
# eqs = '''
# dv/dt = (I - v)/tau: 1
# I = I_recorded(t): 1
# '''
# G = NeuronGroup(1, eqs, threshold='v > 1', reset='v = 0', method='exact')
# M = StateMonitor(G, variables=True, record=True)
# run(200*ms)
# plt.plot(M.t/ms, M.v[0], label='v')
# plt.plot(M.t/ms, M.I[0], label='I')
# plt.xlabel('time [ms]')
# plt.ylabel('v')
# plt.legend(loc='best')
# plt.show()

# create TimedArray that could not be reproduced with a formula
# A = 2.5
# f = 10*Hz
# tau = 5*ms
# # create a TimedArray and set the equations to use it
# num_samples = int(200*ms/defaultclock.dt)
# I_arr = zeros(num_samples)
# for _ in range(100):
#     a = randint(num_samples)
#     I_arr[a:a+100] = rand()
# I_recorded = TimedArray(A*I_arr, dt=defaultclock.dt)
# eqs = '''
# dv/dt = (I - v)/tau: 1
# I = I_recorded(t): 1
# '''
# G = NeuronGroup(1, eqs, threshold='v > 1', reset='v = 0', method='exact')
# M = StateMonitor(G, variables=True, record=True)
# run(200*ms)
# plt.plot(M.t/ms, M.v[0], label='v')
# plt.plot(M.t/ms, M.I[0], label='I')
# plt.xlabel('time [ms]')
# plt.ylabel('v')
# plt.legend(loc='best')
# plt.show()

start_scope()
img = (1-imread('brian.png'))[::-1, :, 0].T
num_samples, N = img.shape
ta = TimedArray(img, dt=1*ms) # 228
A = 1.5
tau = 2*ms
eqs = '''
dv/dt = (A*ta(t, i)-v)/tau+0.8*xi*tau**-0.5 : 1
'''
G = NeuronGroup(N, eqs, threshold='v>1', reset='v=0', method='euler')
M = SpikeMonitor(G)
run(num_samples*ms)
plot(M.t/ms, M.i, '.k', ms=3)
xlim(0, num_samples)
ylim(0, N)
xlabel('Time (ms)')
ylabel('Neuron index');