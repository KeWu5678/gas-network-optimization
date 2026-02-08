#!/usr/env python
'''
Partial Outer Convexification with Alternating Directions Method
for Mixed-Integer Optimal Control problems with combinatorial constraints that
prohibit chattering.
'''

import numpy as np
import casadi as cas
import gurobipy as grb
from matplotlib import pyplot as plt
import shelve
import time
from collocation import pairwise
import collocation
import translines

def resolve_with_fixed_controls(ocp_model, solver, w0, p,
        u_fix=None, v_fix=None, alpha_fix=None):
    '''Solve problem with some controls fixed by bounds. Can be used to emulate
    a forward simulation.'''
    # adjust bounds of last POC mode to avoid linear dependence with SOS1
    # constraint
    if alpha_fix is not None:
        alpha_bound = alpha_fix.copy()
        alpha_bound[:,-1] = -1. # < 0
    else:
        alpha_bound = None
    lbx = ocp_model.overwrite(ocp_model.lbw.copy(),
            u=u_fix, v=v_fix, alpha=alpha_bound)
    if alpha_fix is not None:
        alpha_bound[:,-1] = +2. # > 1
    ubx = ocp_model.overwrite(ocp_model.ubw.copy(),
            u=u_fix, v=v_fix, alpha=alpha_bound)
    sol = solver(x0=w0, p=p, lbx=lbx, ubx=ubx,
            lbg=ocp_model.lbg, ubg=ocp_model.ubg)
    ret = solver.stats()['return_status']
    solved = ['Solve_Succeeded', 'Solved_To_Acceptable_Level']
#    assert ret in solved, 'Solution of NLP failed'
    if ret not in solved:
        print('Solution of NLP failed')
    w = sol['x'].full().flatten()
    y, u, alpha, v, nodes = ocp_model.extract(w)
    stats = solver.stats()
    wall_time = 0.0
    for key in stats.keys():
        if 't_wall' in key:
            wall_time += stats[key]
    return y, u, alpha, v, nodes, w, wall_time

def solve_ciap(t, alpha, strategy='SUR', tau_min=None):
    '''Solve Combinatorial Integral Approximation MILP to reconstruct binary
    feasible convex multipliers beta from relaxed convex multipliers alpha.
    Enforce dwell-time constraints if strategy is COMB.'''
    assert strategy in ['SUR', 'MIP', 'SUR+MIP', 'COMB'], \
        'Unknown CIAP strategy "{}"'.format(strategy)
    N = len(t)
    dt = np.diff(t)
    modes = alpha.shape[1]
    
    wall_time = 0.0

    if strategy in ['SUR', 'SUR+MIP']:
        # Sum-Up-Rounding with SOS1 constraint
        p = np.zeros((N-1, modes))
        phat = np.zeros(modes)
        wall_time -= time.time()
        for i in range(N-1):
            phat += dt[i] * alpha[i,:]
            j = np.argmax(phat)
            p[i,j] = 1
            phat[j] -= dt[i]
        wall_time += time.time()
        if strategy == 'SUR':
            return p, wall_time

    if strategy in ['MIP', 'SUR+MIP', 'COMB']:
        # solve CIAP
        m = grb.Model('CIAP')
        m.setParam('LogToConsole', 0) # suppress screen output
        m.setParam('Threads', 1) # Disable multi-threading

        beta = m.addVars(N-1, modes, name='beta', vtype=grb.GRB.BINARY)
        weighted_dev = m.addVars(N-1, modes, lb=-grb.GRB.INFINITY,
                name='weighted_dev')
        delta = m.addVar(name='delta', lb=-grb.GRB.INFINITY)
        eta = m.addVar(name='eta', obj=1.)

        m.addConstrs((weighted_dev[k,i] == dt[k] * (alpha[k,i] - beta[k,i])
            for i in range(modes) for k in range(N-1)), name='weighted_dev')
        m.addConstrs((delta + weighted_dev.sum(range(k),i) <= eta
            for i in range(modes) for k in range(N)), name='upper_max')
        m.addConstrs((delta + weighted_dev.sum(range(k),i) >= -eta
            for i in range(modes) for k in range(N)), name='lower_max')
        m.addConstrs((beta.sum(k,'*') == 1 for k in range(N-1)), 'SOS1')

        if strategy == 'SUR+MIP':
            # initial guess from SUR
            for k in range(N-1):
                for i in range(modes):
                    beta[k,i].start = p[k,i]

        if strategy == 'COMB':
            # generate standard min-up constraints
            for mode in range(modes):
                # first time step
                l = 1
                while (t[l] <= tau_min) and (l < N-1):
                    m.addConstr(beta[0, mode] <= beta[l, mode])
                    l += 1
                # all other time steps
                for k in range(1, N-1):
                    tk = t[k]
                    l = k + 1
                    while (t[l] <= tk + tau_min) and (l < N-1):
                        ineq = beta[k, mode]-beta[k-1, mode] <= beta[l, mode]
                        m.addConstr(ineq)
                        ineq = beta[k-1, mode]-beta[k, mode] <= 1-beta[l, mode]
                        m.addConstr(ineq)
                        l += 1

        m.optimize()
        assert m.status == grb.GRB.Status.OPTIMAL, 'Solution of CIAP failed'

        sol = m.getAttr('x', beta)
        beta = np.zeros((N-1, modes))
        for i, j in sol:
            beta[i,j] = sol[(i,j)]

        wall_time += m.getAttr('Runtime')

    return beta, wall_time

class tComb(object):
    """MILP model for the combinatorial constraints."""

    def __init__(self, t, modes, tau_min):
        N = len(t)

        # TODO(LS) use on/off variables as extended formulation
        model = grb.Model("tCOMB")
        x = model.addVars(N-1, modes, name='x', vtype=grb.GRB.BINARY)

        # generate standard min-up constraints
        for mode in range(modes):

            # first time step
            l = 1
            while (t[l] <= tau_min) and (l < N-1):
                model.addConstr(x[0, mode] <= x[l, mode])
                l += 1

            # all other time steps
            for k in range(1, N-1):
                time = t[k]
                l = k + 1
                while (t[l] <= time + tau_min) and (l < N-1):
                    model.addConstr(x[k, mode] - x[k - 1, mode] <= x[l, mode])
                    model.addConstr(x[k-1, mode] - x[k, mode] <= 1-x[l, mode])
                    l += 1

        # only one mode is on for each time step
        for k in range(N-1):
            model.addConstr(grb.quicksum(x[k,mode] for mode in range(modes))==1)
        model.write("C:/Users/mosa/fuller.lp")
            
        self.N = N
        self.modes = modes
        self.model = model
        self.t = t
        self.x = x

    def solve(self, v, r):
        """ solve the model for given v, encoding realizations r"""
        N = self.N
        model = self.model
        x = self.x

        eps = model.params.IntFeasTol
        for i in range(self.modes):
            for k in range(N-1):
                hk = self.t[k+1] - self.t[k]
                x[k, i].Obj = hk * np.linalg.norm(v[k,:] - r[i,:], 1)

        model.optimize()
        assert model.status == grb.GRB.Status.OPTIMAL, \
                'Solution of tCOMB failed'

        obj_val = model.getAttr('objVal')
        sol = model.getAttr('x', x)
        result = np.zeros((N-1, self.modes))
        for i, j in sol:
            result[i, j] = sol[i, j]
        wall_time = model.getAttr('Runtime')
        return result.dot(r), obj_val, wall_time

def online_plot(t, rho, obj_val, y, alpha, beta, v_ref=None):
    fig = plt.figure(1)
    fig.clear()

    plt.subplot(1, 2, 1)
    plt.plot(t, y[:,:2])
    plt.legend(('$y_0$', '$y_1$'))
    plt.xlabel('time $t$')
    plt.title(r'States (objective value {:.8g})'.format(obj_val))

    plt.subplot(1, 2, 2)
    plt.step(t, np.concatenate(([np.nan],
        alpha.dot(np.arange(alpha.shape[1])))), linewidth=5)
    plt.step(t, np.concatenate(([np.nan],
        beta.dot(np.arange(beta.shape[1])))), linewidth=3)
    legend_strings = (r'$\alpha$ (NLP)', r'$\beta$ (CIAP)')
    if v_ref is not None:
        plt.step(t, np.concatenate(([np.nan], v_ref.flatten())), linewidth=2)
        legend_strings += (r'$v_{ref}$ (tCOMB)',)
    plt.legend(legend_strings)
    plt.xlabel('time $t$')
    plt.title(r'Controls, $\rho$ = {:.0g}'.format(rho))
    plt.axis([0, 1, -0.01, 1.01])

    plt.draw()
    fig.canvas.draw()
    fig.canvas.flush_events()
    fig.canvas.set_window_title('Fuller ADM')
    #plt.pause(1e-4)

def miocp_adm(ocp_model, tau_min=0.01, with_CIAP=True, online_plot=None):
    '''ADM method for mixed-integer optimal control problems with additional
    combinatorial constraints that couple over time.'''

    nlp_solver_name = 'ipopt'
    solver = ocp_model.create_NLP_solver(nlp_solver_name, tol=1e-8)
    v_ref = np.array([[1] + [0]*(ocp_model.nv_ref-1)]*(len(ocp_model.t)-1))

    timings = {'poc_nlp': 0.0, 'reopt_nlp': 0.0, 'comb': 0.0, 'sur': 0.0}

    w0 = ocp_model.w0.copy()
    y, u, alpha, v, nodes = ocp_model.extract(w0)
    if with_CIAP:
        beta, wall_t = solve_ciap(ocp_model.t, alpha, strategy='SUR')
        timings['sur'] += wall_t
    else:
        beta = alpha

    # setup tComb
    tcomb = tComb(ocp_model.t, ocp_model.nalpha, tau_min)
    tcomb.model.setParam('LogToConsole', 0) # suppress screen output
    tcomb.model.setParam('Threads', 1) # Disable multi-threading

    # set up display
    out_hdr = '{:>9} {:>9} {:>10} {:>11} {:>9} {:>9}'
    output = '{:9.2e} {:9.2e} {:>10s} {:>11s} {:>9s} {:>9s}'
    print(out_hdr.format('rho', 'Psi_l_l', 'Psi_l+1_l', 'Psi_l+1_l+1', 'L1 pen',
        'termcond'))

    # penalty loop
    epsilon = 1e-3 # TODO: recover from CIAP objective value
    for rho in np.logspace(-3, 6, num=10):
        # forward simulation to obtain correct Psi_l_l
        p = np.concatenate(([rho], v_ref.flatten()))
        y, u, beta, v, nodes, w0, wall_t = resolve_with_fixed_controls(
                ocp_model, solver, w0.copy(), p, u_fix=u, v_fix=v,
                alpha_fix=beta)
        Psi_l_l = ocp_model.evaluate_objective(rho, v_ref, w0)
        timings['reopt_nlp'] += wall_t

        # ADM loop
        for iadm in range(100):
            # solve POC relaxed for y, u, alpha for given v_ref
            p = np.concatenate(([rho], v_ref.flatten()))
            # optionally use v_ref as initial value for alpha
            #w0 = ocp_model.overwrite(w0, alpha=v_ref)
            sol = solver(x0=w0, p=p, lbx=ocp_model.lbw, ubx=ocp_model.ubw,
                    lbg=ocp_model.lbg, ubg=ocp_model.ubg)
            if nlp_solver_name == 'ipopt':
                ret = solver.stats()['return_status']
                solved = ['Solve_Succeeded', 'Solved_To_Acceptable_Level']
                assert ret in solved, 'Solution of NLP failed'
            w0 = sol['x'].full().flatten()
            y, u, alpha, v, nodes = ocp_model.extract(w0)
            #timings['poc_nlp'] += solver.stats()['t_wall_solver']
            stats = solver.stats()
            t_wall = 0.0
            for key in stats.keys():
                if 't_wall' in key:
                    t_wall += stats[key]
            timings['poc_nlp'] = t_wall

            if with_CIAP:
                # solve integral approximation problem to obtain beta from alpha
                beta, wall_t = solve_ciap(ocp_model.t, alpha, strategy='SUR')
                timings['sur'] += wall_t
                # Reoptimize continuous controls
                y, u, beta, v, nodes, w0, wall_t = resolve_with_fixed_controls(
                        ocp_model, solver, w0.copy(), p, v_fix=v,
                        alpha_fix=beta)
                timings['reopt_nlp'] += wall_t
            else:
                beta = alpha

            # compute objective Psi(u^{k,l+1}, v^{k,l+1}, \tilde{v}^{k,l})
            Psi_lp_l = ocp_model.evaluate_objective(rho, v_ref, w0)

            # first termination criterion
            if Psi_lp_l >= Psi_l_l - epsilon:
                if online_plot is not None:
                    online_plot(ocp_model.t, rho, Psi_lp_l, y, alpha, beta,
                            v_ref)
                print(output.format(rho, Psi_l_l, '{:9.2e}'.format(Psi_lp_l),
                    '', '', '(i)'))
                break

            # solve combinatorial constraints
            prev_v_ref = v_ref.copy()
            v_ref, beta_deviation, wall_t = tcomb.solve(beta.dot(ocp_model.r),
                    ocp_model.r)
            timings['comb'] += wall_t

            # compute objective Psi(u^{k,l+1}, v^{k,l+1}, \tilde{v}^{k,l+1})
            Psi_lp_lp = ocp_model.evaluate_objective(rho, v_ref, w0)

            if online_plot is not None:
                online_plot(ocp_model.t, rho, Psi_lp_lp, y, alpha, beta, v_ref)

            # second termination criterion
            if Psi_lp_lp >= Psi_lp_l - epsilon:
                if Psi_lp_lp > Psi_lp_l:
                    v_ref = prev_v_ref
                print(output.format(rho, Psi_l_l, '{:9.2e}'.format(Psi_lp_l),
                    '{:9.2e}'.format(Psi_lp_lp),
                    '{:9.2e}'.format(beta_deviation), '(ii)'))
                break

            # update current objective
            Psi_l_l = Psi_lp_lp
            print(output.format(rho, Psi_l_l, '{:9.2e}'.format(Psi_lp_l),
                '{:9.2e}'.format(Psi_lp_lp), '{:9.2e}'.format(beta_deviation),
                ''))

        # stop increasing rho?
        if np.linalg.norm(alpha.dot(ocp_model.r) - v_ref, np.inf) < 1e-4:
            break

    # compute feasible point and reoptimize continuous controls
    y, u, alpha, v, nodes, w0, wall_t = resolve_with_fixed_controls(ocp_model,
            solver, w0.copy(), p, v_fix=v, alpha_fix=beta)
    timings['reopt_nlp'] += wall_t
    obj_val = ocp_model.evaluate_objective(0, v_ref, w0)
    print('\nFinal objective value: {:.10e}'.format(obj_val))

    if online_plot is not None:
        online_plot(ocp_model.t, rho, obj_val, y, alpha, beta, v_ref)

    return y, u, beta, v_ref, nodes, obj_val, w0, timings

class TranslinesOCModel(collocation.OCModel):
    '''Optimal control model interface for transmission lines example.'''

    def __init__(self, network_name, coarse=True):
        if network_name == 'extended tree':
            self.net = translines.network_extended_tree
            T = 26.
            if not coarse: # dt = dx = 0.25
                self.demand = np.loadtxt('demand_extended_tree.dat')
                nt = 4*26+1 
                nx = 4
            else: # dt = dx = 0.5
                self.demand = np.loadtxt('demand_extended_tree_coarse.dat')
                nt = 2*26+1 
                nx = 2
        elif network_name == 'subgrid':
            self.net = translines.network_subgrid
            T = 26.
            if not coarse: # dt = dx = 0.25
                self.demand = np.loadtxt('demand_subgrid.dat')
                nt = 4*26+1 
                nx = 4
            else: # dt = dx = 0.5
                self.demand = np.loadtxt('demand_subgrid_coarse.dat')
                nt = 2*26+1 
                nx = 2
        else:
            raise Exception('Unkown network name "{}"'.format(network_name))
        self.T = T
        self.t = np.linspace(0, T, nt)
        self.nt = nt
        self.nx = nx
        d = translines.translines_nlp(self.net, self.demand, T, nt=nt, nx=nx)
        self.nlp, self.lbw, self.ubw, self.lbg, self.ubg, w0 = d
        self.w0 = np.array(w0)
        _, self.A, self.producers, _, _, self.configs = self.net()
        self.n_ctrls = len(self.producers)
        self.n_confg = len(self.configs)
        self.nv_ref = 2
        self.nalpha = 4
        self.r = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        assert self.n_confg == self.nalpha, 'Implement case nalpha != 4'

        # augment problem with outer-convexified L1 penalization term
        rho = cas.MX.sym('rho')
        V_ref = cas.MX.sym('V_ref', self.nv_ref, nt-1)
        self.nlp['p'] = cas.vertcat(rho, cas.vec(V_ref))
        alpha = cas.reshape(self.nlp['x'][:(nt-1)*self.n_confg],
                nt-1, self.n_confg)
        pen_L1 = 0
        for k, (tk, tkp1) in enumerate(pairwise(self.t)):
            integrand = 0
            for i in range(self.nalpha):
                for j in range(self.nv_ref):
                    # explicit cases for signs in absolute value function
                    if self.r[i,j] == 0: # -> |x| = +x >= 0
                        integrand += V_ref[j,k] * alpha[k,i]
                    elif self.r[i,j] == 1: # -> |x| = -x >= 0
                        integrand += (1 - V_ref[j,k]) * alpha[k,i]
                    else:
                        raise Exception('Array r not binary')
            pen_L1 = pen_L1 + (tkp1 - tk) * integrand
        obj_sca = 1e-3
        self.nlp['f'] = obj_sca * self.nlp['f'] + rho * pen_L1

    def extract(self, w):
        'Extract and return (y, u, alpha, v, nodes) from NLP variable w.'
        offset = 0
        alpha = np.reshape(w[offset:offset+(self.nt-1)*self.n_confg],
            (self.nt-1, self.n_confg), order='F')
        offset += (self.nt - 1) * self.n_confg
        u = np.reshape(w[offset:offset+(self.nt-1)*self.n_ctrls],
            (self.nt-1, self.n_ctrls), order='F')
        offset += (self.nt - 1) * self.n_ctrls
        y = w[offset:offset+2*len(self.A)*self.nt*self.nx]
        return y, u, alpha, None, None

    def overwrite(self, w, y=None, u=None, alpha=None, v=None, nodes=None):
        '''Overwrite given components (y, u, alpha, v, nodes) in NLP variable w.
        Returns overwritten w.'''
        offset = 0
        if v is not None or nodes is not None:
            raise Exception('Cannot overwrite unknown variables')
        if alpha is not None:
            w[offset:offset+(self.nt-1)*self.n_confg] = alpha.flatten(order='F')
        offset += (self.nt - 1) * self.n_confg
        if u is not None:
            w[offset:offset+(self.nt-1)*self.n_ctrls] = u.flatten(order='F')
        offset += (self.nt - 1) * self.n_ctrls
        if y is not None:
            w[offset:offset+2*len(self.A)*self.nt*self.nx] = y
        return w

def run_translines_relaxed():
    """
    Solve translines problem only with POC and plot the results
    """
    ocp_model = TranslinesOCModel('extended tree', coarse=True)
    rho = 0.0
    v_ref = np.array([[1] + [0]*(ocp_model.nv_ref-1)]*(len(ocp_model.t)-1))
    p = np.concatenate(([rho], v_ref.flatten()))
    solver = ocp_model.create_NLP_solver()

    sol = solver(x0=ocp_model.w0, p=p, lbx=ocp_model.lbw, ubx=ocp_model.ubw,
            lbg=ocp_model.lbg, ubg=ocp_model.ubg)
    ret = solver.stats()['return_status']
    solved = ['Solve_Succeeded', 'Solved_To_Acceptable_Level']
    assert ret in solved, 'Solution of NLP failed'

    u, alpha, xi_p, xi_m = translines.extract_solution(sol, ocp_model.net,
            ocp_model.nt, ocp_model.nx)
    translines.plot_solution(ocp_model.net, u, alpha, xi_p, xi_m,
            ocp_model.demand, ocp_model.T, ocp_model.nt)

def translines_plot(t, rho, obj_val, y, alpha, beta, v_ref=None):
    fig = plt.figure(6)
    fig.clear()

    r = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    v_nlp = alpha.dot(r)
    v_ciap = beta.dot(r)

    def fmt(v): return np.concatenate(([np.nan], v.flatten()))

    for i in [0, 1]:
        plt.subplot(2, 1, i+1)
        plt.step(t, fmt(v_ref[:,i]), linewidth=5, label='tCOMB')
        plt.step(t, fmt(v_ciap[:,i]), linewidth=4, label='CIAP')
        plt.step(t, fmt(v_nlp[:,i]), linewidth=3, label='NLP')
        plt.legend()
        plt.xlabel('time $t$')
        plt.title(r'Integer control {}, $\rho$ = {:.0g}'.format(i+1, rho))
        plt.axis([0, t[-1], -0.01, 1.01])

    plt.draw()
    fig.canvas.draw()
    fig.canvas.flush_events()
    fig.canvas.set_window_title('Transmission lines ADM')
    #plt.pause(1e-4)

def prepare_translines_results(net_name='subgrid', methode='all'):
    """
    net_name: subgrid, extended tree
    methode: all, SUR, COMB CIAP, ADM, ADM without CIAP, ADM with CIAP
    """
    ocp_model = TranslinesOCModel(net_name, coarse=False)
    tau_min = 1.0
    print('Transmission lines example on {} net with tau_min={:9.3e}\n'.format(
        net_name, tau_min))

    out_file = 'translines_{}_results'.format('_'.join(net_name.split()))

    # save data
    with shelve.open(out_file) as db:
        db['net_name'] = net_name
        db['net'] = ocp_model.net()
        db['demand'] = ocp_model.demand
        db['T'] = ocp_model.T
        db['tau_min'] = tau_min
        db['nt'] = ocp_model.nt
        db['nx'] = ocp_model.nx
    

    timings = {'poc_nlp': 0.0, 'reopt_nlp': 0.0, 'comb': 0.0, 'sur': 0.0}
    solver = ocp_model.create_NLP_solver(tol=1e-8)
    v_ref = np.array([[1] + [0]*(ocp_model.nv_ref-1)]*(len(ocp_model.t)-1))
    p = np.concatenate(([0], v_ref.flatten()))
    
    print('Partial Outer Convexification Relaxation:')
    # solve relaxed NLP
    sol = solver(x0=ocp_model.w0, p=p, lbx=ocp_model.lbw, ubx=ocp_model.ubw,
            lbg=ocp_model.lbg, ubg=ocp_model.ubg)
    ret = solver.stats()['return_status']
    solved = ['Solve_Succeeded', 'Solved_To_Acceptable_Level']
    assert ret in solved, 'Solution of NLP failed'
    w_poc = sol['x'].full().flatten()
    y, u, alpha, v, nodes = ocp_model.extract(w_poc)
    obj_val = float(sol['f'])
    #timings['poc_nlp'] = solver.stats()['t_wall_solver']
    stats = solver.stats()
    t_wall = 0.0
    for key in stats.keys():
        if 't_wall' in key:
            t_wall += stats[key]
    timings['poc_nlp'] = t_wall
    with shelve.open(out_file) as db:
        db['poc'] = (y, u, alpha, v, nodes, obj_val, w_poc, timings)
    print('Final objective value: {:.10e}\n'.format(obj_val))
    print('Timings: {}\n'.format(timings))

    if methode in ['all', 'SUR']:
        print('Sum-Up Rounding:')
        # solve CIAP without dwell-time constraints
        beta, timings['sur'] = solve_ciap(ocp_model.t, alpha, strategy='SUR')
        y, u, beta, v, nodes, w_sur, time_reopt = resolve_with_fixed_controls(
                ocp_model, solver, w_poc.copy(), p, alpha_fix=beta)
        obj_val = ocp_model.evaluate_objective(0., v_ref, w_sur)
        with shelve.open(out_file) as db:
            db['sur'] = (y, u, beta, v, nodes, obj_val, w_sur, timings)
        print('Final objective value: {:.10e}'.format(obj_val))
        print('Timings: {}\n'.format(timings))

    if methode in ['all', 'COMB CIAP']:
        print('COMB CIAP:')
        # solve CIAP with dwell-time constraints
        beta, timings['sur'] = solve_ciap(ocp_model.t, alpha, strategy='COMB',
                tau_min=tau_min)
        y, u, beta, v, nodes, w_comb, time_reopt = resolve_with_fixed_controls(
                ocp_model, solver, w_poc.copy(), p, alpha_fix=beta)
        obj_val = ocp_model.evaluate_objective(0., v_ref, w_comb)
        with shelve.open(out_file) as db:
            db['comb'] = (y, u, beta, v, nodes, obj_val, w_comb, timings)
        print('Final objective value: {:.10e}'.format(obj_val))
        print('Timings: {}\n'.format(timings))

    if methode in ['all', 'ADM', 'ADM without CIAP']:
        print('ADM without CIAP:', '\n')
        adm_wo_ciap = miocp_adm(ocp_model, tau_min=tau_min, with_CIAP=False,
                online_plot=translines_plot)
        with shelve.open(out_file) as db:
            db['adm_wo_ciap'] = adm_wo_ciap
        print('Timings: {}\n'.format(adm_wo_ciap[-1]))

    if methode in ['all', 'ADM', 'ADM with CIAP']:
        print('ADM with CIAP:', '\n')
        adm_with_ciap = miocp_adm(ocp_model, tau_min=tau_min, with_CIAP=True,
                online_plot=translines_plot)
        with shelve.open(out_file) as db:
            db['adm_with_ciap'] = adm_with_ciap
        print('Timings: {}\n'.format(adm_with_ciap[-1]))

        # call plotting routines from translines module
        # plot for last methode (ADM with CIAP)
        w = adm_with_ciap[-2]
        u, alpha, xi_p, xi_m = translines.extract_solution({'x': w}, ocp_model.net,
                ocp_model.nt, ocp_model.nx)
        translines.plot_solution(ocp_model.net, u, alpha, xi_p, xi_m,
                ocp_model.demand, ocp_model.T, ocp_model.nt)

if __name__ == '__main__':
    methodes = ''
    while methodes not in ['all', 'SUR', 'COMB CIAP', 'ADM', 'ADM without CIAP', 'ADM with CIAP']:
        methodes = input('Please choose the methode (all, MIQP, COMB CIAP, ADM, ADM without CIAP, ADM with CIAP):')
        prepare_translines_results(net_name='extended tree', methode=methodes)
    

