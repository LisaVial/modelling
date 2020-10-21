from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

# brian2.test()

# # simple neuron model:
#
# start_scope()
#
# tau = 10 * ms
# eqs = '''
# dv/dt = (1-v)/tau: 1
# '''
#
# # definition to create a neuron:
# G = NeuronGroup(1, eqs, method='exact')  # first number of neurons and then differential equations
# M = StateMonitor(G, 'v', record=True)   # StateMonitor records values of a neuron variable while the simulation runs
# print('Before v = s' % G.v[0])
# run(100 * ms)
# print('after v = %s' % G.v[0])
# print('Expected value of v = %s' % (1-exp(-100*ms/tau)))
#
# # graph of evolution of v; analytic would be expection of v:
# plt.plot(M.t/ms, M.v[0], 'C0', label='Brian')
# plt.plot(M.t/ms, 1-exp(-M.t/tau), 'C1--', label='Analytic')
# plt.xlabel('Time (ms)')
# plt.ylabel('v')
# plt.legend()
# plt.show()
#
# start_scope()
# tau = 10*ms
# eqs = '''
# dv/dt = (sin(2*pi*100*Hz*t)-v)/tau : 1
# '''
# # use the Euler method:
# G = NeuronGroup(1, eqs, method='euler')
# M = StateMonitor(G, 'v', record=0)
#
# G.v = 5 # initial value
#
# run(60*ms)
#
# plt.plot(M.t/ms, M.v[0])
# plt.xlabel('Time (ms)')
# plt.ylabel('v')
# plt.show()
#
# # add spiking
# start_scope()
#
# tau = 10*ms
# eqs = '''
# dv/dt = (1-v)/tau : 1 (unless refractory)
# '''
#
# G = NeuronGroup(1, eqs, threshold='v>0.8', reset='v = 0', method='exact')
#
# statemon = StateMonitor(G, 'v', record=0)
#
# spikemon = SpikeMonitor(G)
# run(50*ms)
#
# print('Spike times: %s' % spikemon.t[:])
# plt.plot(statemon.t/ms, statemon.v[0])
# for t in spikemon.t:
#     axvline(t/ms, ls='--', c='C1', lw=3)
# plt.xlabel('Time (ms)')
# plt.ylabel('v')
# plt.show()

# multiple neurons and play around with parameters

# start_scope()
#
# N = 100     # number of neurons
# tau = 10 * ms
# v0_max = 3.
# duration = 1000 * ms
# sigma = 0.2
# eqs = '''
# dv/dt = (v0-v)/tau+sigma*xi*tau**-0.5: 1 (unless refractory)
# v0 : 1
# '''
#
# G = NeuronGroup(N, eqs, threshold='v>1', reset='v=0', refractory=5*ms, method='euler')
# spikemon = SpikeMonitor(G)
#
# G.v0 = 'i*v0_max/(N-1)'
#
# run(duration)

# plt.figure(figsize=(12, 4))
# plt.subplot(121)
# plt.plot(spikemon.t/ms, spikemon.i, '.k')
# plt.xlabel('time [ms]')
# plt.ylabel('neuron index')
# plt.subplot(122)
# plt.plot(G.v0, spikemon.count/duration)
# plt.xlabel('v0')
# plt.ylabel('Firing rate [spikes/s]')
# plt.show()

# excercise at the end of tutorial

start_scope()

N = 1000
tau = 10 * ms
vr = -70 * mV
vt0 = -50 * mV
delta_vt0 = 5 * mV
tau_t = 100 * ms
sigma = 0.5 * (vt0 - vr)
v_drive = 2 * (vt0 - vr)
duration = 100 * ms

eqs = '''
dv/dt = (v_drive + vr - v) / tau + sigma * xi * tau ** -0.5: volt
dvt/dt = (vt0 - vt) / tau_t: volt
'''

reset = '''
v = vr
vt += delta_vt0
'''

G = NeuronGroup(N, eqs, threshold='v>vt', reset=reset, refractory=5 * ms, method='euler')

statemon = StateMonitor(G, 'v', record=50)
spikemon = SpikeMonitor(G)

G.v = 'rand()*(vt0-vr)+vr'
G.vt = vt0
run(duration)

plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(spikemon.t/ms, spikemon.i, '.k')
plt.xlabel('time [ms]')
plt.ylabel('neuron index')
plt.subplot(122)
plt.hist(spikemon.t/ms, 100, histtype='stepfilled', facecolor='k')
# according to tutorial, there also should be this parameter:
# weights=list(np.ones(len(spikemon.t/ms))/(N*defaultclock.dt))
# for me, that does not work. Maybe check out at another computer?
# https://brian2.readthedocs.io/en/stable/resources/tutorials/1-intro-to-brian-neurons.html
plt.xlabel('time [ms]')
plt.ylabel('instantaneous firing rate [spikes/s]')
plt.show()
