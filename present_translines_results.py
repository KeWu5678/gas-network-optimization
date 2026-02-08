'''
Present results of transmission lines example computed with pocadm.py and
saved in translines_results
'''

from translines import extract_solution
import numpy as np
import matplotlib.pyplot as plt
import shelve
from matplotlib2tikz import save as tikz_save
# Install matplotlib2tikz via "pip3 install -U matplotlib2tikz"

with shelve.open('translines_subgrid_results') as db:
    net_name = db['net_name']
    net_data = db['net']
    demand = db['demand']
    T = db['T']
    tau_min = db['tau_min']
    nt = db['nt']
    nx = db['nx']
    poc = db['poc']
    sur = db['sur']
    comb = db['comb']
    adm_wo_ciap = db['adm_wo_ciap']
    adm_with_ciap = db['adm_with_ciap']

t = np.linspace(0, T, nt)
_, A, producers, _, consumers, configs = net_data
n_ctrls = len(producers)
n_confg = len(configs)
n_consumer = len(consumers)
def net(): return net_data
r = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

def step_fmt(v): return np.concatenate((v.flatten(), [np.nan]))

results = [poc, sur, comb, adm_wo_ciap, adm_with_ciap]
labels = ['POC', 'SUR', 'COMB-CIAP', 'ADM', 'ADM + CIAP']

plt.figure(1).clear()
ax = plt.gca()
for i, (result, label) in enumerate(zip(results, labels)):
    u, alpha, xi_p, xi_m = extract_solution({'x': result[-2]}, net, nt, nx)
    v = alpha.dot(r)
    bl = 1.1*(4-i)
    p = ax.step(t, step_fmt(v[:,0]+bl), where='post', label=label)
    ax.fill_between(t, step_fmt(v[:,0]+bl), y2=bl, color=p[0].get_color(),
            step='post', alpha=0.1)
#ax.legend(loc='lower right')
ax.set_ylim(-0.02, 5.42)
ax.set_xlim(0, T)
ax.set_xlabel(r'time $t$')
ax.yaxis.set_visible(False)
tikz_save('translines_discrete_control_1.tex')

plt.figure(2).clear()
ax = plt.gca()
for i, (result, label) in enumerate(zip(results, labels)):
    u, alpha, xi_p, xi_m = extract_solution({'x': result[-2]}, net, nt, nx)
    v = alpha.dot(r)
    bl = 1.1*(4-i)
    p = ax.step(t, step_fmt(v[:,1])+bl, where='post', label=label)
    ax.fill_between(t, step_fmt(v[:,1]+bl), y2=bl, color=p[0].get_color(),
            step='post', alpha=0.1)
ax.legend(loc='lower right')
ax.set_ylim(-0.02, 5.42)
ax.set_xlim(0, T)
ax.set_xlabel(r'time $t$')
ax.yaxis.set_visible(False)
tikz_save('translines_discrete_control_2.tex')

plt.figure(3).clear()
ax = plt.gca()
for result, label in zip(results, labels):
    u, alpha, xi_p, xi_m = extract_solution({'x': result[-2]}, net, nt, nx)
    ax.step(t, step_fmt(u[:,0]), label=label)
#ax.legend(loc='upper left')
ax.set_ylim(0, 120)
ax.set_xlim(0, T)
ax.set_xlabel(r'time $t$')
ax.set_ylabel(r'Continuous control $u_1$')
tikz_save('translines_continuous_control_1.tex')

plt.figure(4).clear()
ax = plt.gca()
for result, label in zip(results, labels):
    u, alpha, xi_p, xi_m = extract_solution({'x': result[-2]}, net, nt, nx)
    ax.step(t, step_fmt(u[:,1]), label=label)
ax.legend(loc='lower center')
ax.set_ylim(0, 80)
ax.set_xlim(0, T)
ax.set_xlabel(r'time $t$')
ax.set_ylabel(r'Continuous control $u_2$')
tikz_save('translines_continuous_control_2.tex')

plt.figure(5).clear()
fig, axes = plt.subplots(5, 1, num=5)
end_vertices = [j for (i,j) in A]
fmt = r'Consumer {}'
for i, consumer in enumerate(consumers[0:]):
    line = end_vertices.index(consumer)
    axes[i].plot(t[:-1], demand[i,:], 'k--', label='demand')
    for result, label in zip(results[0:], labels[0:]):
        u, alpha, xi_p, xi_m = extract_solution({'x': result[-2]}, net, nt, nx)
        axes[i].plot(t, xi_p[line][-1,:], label=label)
    if i == 0:
        axes[i].legend(loc='lower center')
    ax.set_xlim(0, T)
    axes[i].set_xlim(0, T)
    axes[i].set_ylim(0, 60)
    axes[i].set_xlabel(r'time $t$')
    axes[i].set_ylabel(fmt.format(i+1))
tikz_save('translines_demand_and_delivery.tex')

