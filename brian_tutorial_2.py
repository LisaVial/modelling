from brian2 import *
import matplotlib.pyplot as plt


# the simplest synapse
# start_scope()
# eqs = '''
# dv/dt = (I-v)/tau: 1
# I: 1
# tau: second
# '''
#
# G = NeuronGroup(2, eqs, threshold='v>1', reset='v = 0', method='exact')
# G.I = [2, 0]
# G.tau = [10, 100] * ms
#
# S = Synapses(G, G, on_pre='v_post += 0.2')
# S.connect(i=0, j=1)
#
# M = StateMonitor(G, 'v', record=True)
#
# run(100*ms)
#
#
# plt.plot(M.t/ms, M.v[0], label='Neuron 0')
# plt.plot(M.t/ms, M.v[1], label='Neuron 1')
# plt.xlabel('time [ms]')
# plt.ylabel('v')
# plt.legend()
# plt.show()

# adding a weight to the synapse

# start_scope()
#
# eqs = '''
# dv/dt = (I-v)/tau: 1
# I: 1
# tau: second
# '''
# G = NeuronGroup(3, eqs, threshold='v>1', reset='v = 0', method='exact')
# G.I = [2, 0, 0]
# G.tau = [10, 100, 1000] * ms
#
# S = Synapses(G, G, 'w: 1', on_pre='v_post += w')
# S.connect(i=0, j=[1, 2])
# S.w = 'j * 0.2'
#
# M = StateMonitor(G, 'v', record=True)
#
# run(50 * ms)
#
# plt.plot(M.t/ms, M.v[0], label='Neuron 1')
# plt.plot(M.t/ms, M.v[1], label='Neuron 2')
# plt.plot(M.t/ms, M.v[2], label='Neuron 3')
# plt.xlabel('time [ms]')
# plt.ylabel('v')
# plt.legend()
# plt.show()

# introducing a delay
# start_scope()
#
# eqs = '''
# dv/dt = (I-v)/tau: 1
# I: 1
# tau: second
# '''
#
# G = NeuronGroup(3, eqs, threshold='v>1', reset='v = 0', method='exact')
# G.I = [2, 0, 0]
# G.tau = [10, 100, 1000] * ms
#
# S = Synapses(G, G, 'w: 1', on_pre='v_post += w')
# S.connect(i=0, j=[1, 2])
# S.w = 'j * 0.2'
# S.delay = 'j * 2* ms'
#
# M = StateMonitor(G, 'v', record=True)
#
# run(50* ms)
#
# plt.plot(M.t/ms, M.v[0], label='Neuron 1')
# plt.plot(M.t/ms, M.v[1], label='Neuron 2')
# plt.plot(M.t/ms, M.v[2], label='Neuron 3')
#
# plt.xlabel('time [ms]')
# plt.ylabel('v')
# plt.legend()
# plt.show()

# more complex connectivity
def visualise_connectivity(S):
    Ns = len(S.source)
    Nt = len(S.target)
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.plot(zeros(Ns), arange(Ns), 'ok', ms=10)
    plt.plot(ones(Nt), arange(Nt), 'ok', ms=10)
    for i, j in zip(S.i, S.j):
        plt.plot([0, 1], [i, j], '-k')
    plt.xticks([0, 1], ['Source', 'Target'])
    plt.ylabel('Neuron index')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-1, max(Ns, Nt))
    plt.subplot(122)
    plt.plot(S.i, S.j, 'ok')
    plt.xlim(-1, Ns)
    plt.ylim(-1, Nt)
    plt.xlabel('Source neuron index')
    plt.ylabel('Target neuron index')
    plt.show()

start_scope()
N = 10
G = NeuronGroup(N, 'v:1')

S = Synapses(G, G)
S.connect(condition='abs(i-j)<4 and i!=j')
visualise_connectivity(S)

