'''
Discretized mixed-integer optimal control problem for transient gas networks
with partial outer convexification (POC) of the switching decisions.

Model: semilinear isothermal Euler equations (model ISO2 in Domschke, Hiller,
Lang, Tischendorf: "Modellierung von Gasnetzwerken: Eine Uebersicht",
TRR154 preprint 2717, 2017) per pipe,

    rho_t + m_x = 0,
    m_t + p_x = -lambda/(2D) m|m|/rho,      p = c^2 rho,

with mass flux density m = rho v [kg/(m^2 s)] and constant speed of sound c
(the TRR154 .bcd files prescribe c = 340 m/s). In characteristic variables
xi_pm = m +- p/c the system is a 2x2 hyperbolic system with transport speeds
+-c and a nonlinear friction source -- the exact analogue of the telegraph
system of the transmission-lines example, which is why the same first-order
upwind discretization is used. The stiff friction term is treated implicitly
(IMEX): advection explicit under the CFL condition dt <= dx/c, friction
evaluated at the new time level (the friction relaxation time ~ D/(lambda |v|)
is of the order of seconds, far below practical dt).

Nondimensionalization: p_ref (default 50 bar), rho_ref = p_ref/c^2,
m_ref = rho_ref c, L_ref = max pipe length, T_ref = L_ref/c,
flow scale Q_ref = m_ref * A_ref.

Network coupling (replaces the equal-distribution matrices of the telegraph
model): per node a pressure variable, per pipe end a boundary mass flux;
the outgoing characteristic of each pipe is matched to the node state and
mass is conserved at every node. Sources have controlled pressure (continuous
control u, bounds from the .net file) and bounded free supply; sink delivery
is a free expression tracked against the .bcd demand in the objective
(least squares, as in the transmission-lines example).

Switching (POC): valves (open/closed) and compressor stations (on/bypass)
are the binary controls. All 2^s on/off combinations form the configuration
set; convex multipliers alpha(t) with SOS1 constraint relax the choice.
Valve: |q_V| <= w q_max, |p_a - p_b| <= (1-w) M.
Compressor: p_b = p_a + dp, 0 <= dp <= w dp_max, 0 <= q_C <= q_max,
where w(t) = sum_c alpha_c(t) r_{c,j} is the relaxed switch state.
'''

from dataclasses import dataclass, field
from math import sqrt
from typing import Dict, List, Optional

import casadi as cas
import numpy as np


@dataclass
class GasScaling:
    'Reference quantities of the nondimensionalization.'
    c: float            # speed of sound [m/s]
    p_ref: float        # pressure scale [Pa]
    L_ref: float        # length scale [m]
    A_ref: float        # area scale [m^2]

    @property
    def rho_ref(self):
        return self.p_ref / self.c ** 2

    @property
    def m_ref(self):
        return self.rho_ref * self.c     # mass flux density scale

    @property
    def T_ref(self):
        return self.L_ref / self.c       # time scale [s]

    @property
    def Q_ref(self):
        return self.m_ref * self.A_ref   # mass flow scale [kg/s]


def switch_encoding(n_switches):
    '''Binary encoding of all 2^s switch configurations: returns r of shape
    (2^s, s) with r[c, j] = state of switch j in configuration c.'''
    n_confg = 2 ** n_switches
    r = np.zeros((n_confg, n_switches), dtype=int)
    for conf in range(n_confg):
        for j in range(n_switches):
            r[conf, j] = (conf >> j) & 1
    return r


def build_gas_nlp(net, bc, T=7200., nt=121, nx=2, p_ref=50e5,
                  initial_state=None, p_init=None, gamma_comp=0.02,
                  eps_fric=1e-3):
    '''Build the POC-relaxed NLP for gas network `net` (gaslib_io.GasNetwork)
    with boundary conditions `bc` (gaslib_io.BoundaryConditions).

    T: horizon [s]; nt: time grid points; nx: cells per pipe.
    initial_state: optional gaslib_io.NetworkState for the initial condition;
    if None, a flat state at pressure p_init (default: first entry pressure
    of the .bcd) and zero flow is used.
    gamma_comp: weight of the compression cost term.

    Returns a dict with the NLP, bounds, initial guess, scaling, time grid
    and index bookkeeping.'''
    c = bc.speed_of_sound
    sca = GasScaling(c=c, p_ref=p_ref,
                     L_ref=max(p.length for p in net.pipes),
                     A_ref=float(np.mean([p.area for p in net.pipes])))

    t_grid = np.linspace(0., T, nt) / sca.T_ref      # scaled time grid
    dt = t_grid[1] - t_grid[0]
    t_sec = np.linspace(0., T, nt)                   # physical time grid

    # CFL on the scaled equations (transport speed 1)
    dx_min = min(p.length / sca.L_ref / nx for p in net.pipes)
    assert dt <= dx_min + 1e-12, \
        'CFL violated: dt={:.3g} > dx={:.3g} (scaled); increase nt or ' \
        'decrease nx'.format(dt, dx_min)

    sources = net.sources
    sinks = net.sinks
    src_ids = [n.id for n in sources]
    switches = net.switches
    n_src = len(sources)
    n_sw = len(switches)
    r = switch_encoding(n_sw)
    n_confg = r.shape[0]

    # initial condition per pipe: (m0, rho0) arrays of length nx
    if p_init is None:
        if bc.pressure:
            first = sorted(bc.pressure)[0]
            p_init = bc.pressure[first][1][0]
        else:
            p_init = p_ref
    init = {}
    for pipe in net.pipes:
        if initial_state is not None and pipe.id in \
                initial_state.pipe_profiles:
            xs, qs, ps = initial_state.pipe_profiles[pipe.id]
            x_cells = (np.arange(nx) + .5) * pipe.length / nx
            m0 = np.interp(x_cells, xs, qs) / pipe.area / sca.m_ref
            rho0 = np.interp(x_cells, xs, ps) / p_ref
        else:
            m0 = np.zeros(nx)
            rho0 = np.full(nx, p_init / p_ref)
        init[pipe.id] = (m0, rho0)

    # ------------------------------------------------------------------ #
    # NLP variables (column-major layout, alpha and u first for the ADM)  #
    # ------------------------------------------------------------------ #
    w, w0, lbw, ubw = [], [], [], []
    g, lbg, ubg = [], [], []
    layout = []          # (name, (rows, cols)) in storage order
    J = 0

    def add_var(sym, init_vals, lb, ub):
        layout.append((sym.name(), (sym.size1(), sym.size2())))
        w.append(cas.reshape(sym, -1, 1))
        w0.extend(np.asarray(init_vals).flatten(order='F').tolist())
        lbw.extend(np.asarray(lb).flatten(order='F').tolist())
        ubw.extend(np.asarray(ub).flatten(order='F').tolist())

    def add_eq(expr):
        g.append(expr)
        n = expr.numel()
        lbg.extend([0.] * n)
        ubg.extend([0.] * n)

    def add_le(expr):
        'expr <= 0'
        g.append(expr)
        n = expr.numel()
        lbg.extend([-np.inf] * n)
        ubg.extend([0.] * n)

    # convex configuration multipliers
    alpha = cas.MX.sym('alpha', nt - 1, n_confg)
    add_var(alpha, np.full((nt - 1, n_confg), 1. / n_confg),
            np.zeros((nt - 1, n_confg)), np.full((nt - 1, n_confg), 2.))
    add_eq(cas.mtimes(alpha, cas.DM.ones(n_confg)) - 1.)    # SOS1

    # continuous controls: source pressures (scaled)
    u = cas.MX.sym('u', nt - 1, n_src)
    u0 = np.column_stack([
        bc.pressure_at(n.id, t_sec[:-1]) / p_ref if n.id in bc.pressure
        else np.full(nt - 1, p_init / p_ref) for n in sources])
    add_var(u, u0,
            np.column_stack([np.full(nt - 1, n.pressure_min / p_ref)
                             for n in sources]),
            np.column_stack([np.full(nt - 1, n.pressure_max / p_ref)
                             for n in sources]))

    n_ctrl_vars = (nt - 1) * (n_confg + n_src)

    # pipe characteristic states xi_pm = m +- rho (scaled), initial column
    # fixed by bounds
    xi_p, xi_m = {}, {}
    for pipe in net.pipes:
        m0, rho0 = init[pipe.id]
        for sign, store in (('p', xi_p), ('m', xi_m)):
            s0 = m0 + rho0 if sign == 'p' else m0 - rho0
            sym = cas.MX.sym('xi_{}_{}'.format(sign, pipe.id), nx, nt)
            store[pipe.id] = sym
            iv = np.tile(s0[:, None], (1, nt))
            lb = np.full((nx, nt), -np.inf)
            ub = np.full((nx, nt), np.inf)
            lb[:, 0] = s0
            ub[:, 0] = s0
            add_var(sym, iv, lb, ub)

    # node pressures (scaled) for non-source nodes, defined on t = 0..nt-2
    rho_node = {}
    for node in net.nodes.values():
        if node.type == 'source':
            continue
        sym = cas.MX.sym('rho_{}'.format(node.id), nt - 1)
        rho_node[node.id] = sym
        add_var(sym, np.full(nt - 1, p_init / p_ref),
                np.full(nt - 1, node.pressure_min / p_ref),
                np.full(nt - 1, node.pressure_max / p_ref))

    def node_pressure(node_id):
        'Pressure expression of a node (control column for sources).'
        if node_id in src_ids:
            return u[:, src_ids.index(node_id)]
        return rho_node[node_id]

    # pipe boundary mass flux densities (scaled), t = 0..nt-2
    m_L, m_R = {}, {}
    for pipe in net.pipes:
        mb = pipe.flow_max / (sca.m_ref * pipe.area)
        for name, store in (('L', m_L), ('R', m_R)):
            sym = cas.MX.sym('m{}_{}'.format(name, pipe.id), nt - 1)
            store[pipe.id] = sym
            add_var(sym, np.zeros(nt - 1), np.full(nt - 1, -mb),
                    np.full(nt - 1, mb))

    # source supplies (scaled mass flow), t = 0..nt-2
    supply = {}
    for node in sources:
        sym = cas.MX.sym('s_{}'.format(node.id), nt - 1)
        supply[node.id] = sym
        add_var(sym, np.full(nt - 1, node.flow_min / sca.Q_ref),
                np.full(nt - 1, node.flow_min / sca.Q_ref),
                np.full(nt - 1, node.flow_max / sca.Q_ref))

    # valve flows (scaled mass flow), t = 0..nt-2
    q_valve = {}
    for valve in net.valves:
        sym = cas.MX.sym('qV_{}'.format(valve.id), nt - 1)
        q_valve[valve.id] = sym
        qb = valve.flow_max / sca.Q_ref
        add_var(sym, np.zeros(nt - 1), np.full(nt - 1, -qb),
                np.full(nt - 1, qb))

    # compressor flows and pressure increases (scaled), t = 0..nt-2
    q_comp, dp_comp = {}, {}
    for comp in net.compressors:
        sym = cas.MX.sym('qC_{}'.format(comp.id), nt - 1)
        q_comp[comp.id] = sym
        add_var(sym, np.zeros(nt - 1),
                np.full(nt - 1, comp.flow_min / sca.Q_ref),
                np.full(nt - 1, comp.flow_max / sca.Q_ref))
        dp_max = (comp.pressure_out_max - comp.pressure_in_min) / p_ref
        sym = cas.MX.sym('dp_{}'.format(comp.id), nt - 1)
        dp_comp[comp.id] = sym
        add_var(sym, np.zeros(nt - 1), np.zeros(nt - 1),
                np.full(nt - 1, dp_max))

    # ------------------------------------------------------------------ #
    # pipe dynamics: explicit upwind advection + implicit friction (IMEX) #
    # ------------------------------------------------------------------ #
    for pipe in net.pipes:
        dx = pipe.length / sca.L_ref / nx
        kappa = pipe.friction_factor() * sca.L_ref / (2. * pipe.diameter)
        xp, xm = xi_p[pipe.id], xi_m[pipe.id]
        rho_l = node_pressure(pipe.from_node)
        rho_r = node_pressure(pipe.to_node)
        mL, mR = m_L[pipe.id], m_R[pipe.id]

        for k in range(nt - 1):
            # friction at the new time level (implicit)
            m_new = 0.5 * (xp[:, k + 1] + xm[:, k + 1])
            rho_new = 0.5 * (xp[:, k + 1] - xm[:, k + 1])
            fric = kappa * m_new * cas.sqrt(m_new ** 2 + eps_fric ** 2) \
                / rho_new

            # xi_p: transport to the right, ghost value from left node
            ghost_p = mL[k] + rho_l[k]
            theta = cas.vertcat(ghost_p, xp[:-1, k])
            adv = -(xp[:, k] - theta) / dx
            add_eq(xp[:, k + 1] - xp[:, k] - dt * (adv - fric))

            # xi_m: transport to the left, ghost value from right node
            ghost_m = mR[k] - rho_r[k]
            theta = cas.vertcat(xm[1:, k], ghost_m)
            adv = (theta - xm[:, k]) / dx
            add_eq(xm[:, k + 1] - xm[:, k] - dt * (adv - fric))

            # outgoing characteristics matched to the node state
            add_eq(xm[0, k] - (mL[k] - rho_l[k]))
            add_eq(xp[-1, k] - (mR[k] + rho_r[k]))

    # ------------------------------------------------------------------ #
    # switching elements (POC via big-M with relaxed switch state w)      #
    # ------------------------------------------------------------------ #
    def switch_state(j):
        'Relaxed state w_j(t) of switch j as MX column (nt-1).'
        cols = [alpha[:, conf] for conf in range(n_confg) if r[conf, j]]
        return sum(cols) if cols else cas.DM.zeros(nt - 1)

    rho_span = (max(n.pressure_max for n in net.nodes.values())
                - min(n.pressure_min for n in net.nodes.values())) / p_ref

    for j, valve in enumerate(net.valves):
        wv = switch_state(j)
        qv = q_valve[valve.id]
        qb = valve.flow_max / sca.Q_ref
        dpb = min(valve.pressure_diff_max / p_ref, rho_span)
        dp = node_pressure(valve.from_node) - node_pressure(valve.to_node)
        add_le(qv - wv * qb)                 # |q_V| <= w q_max
        add_le(-qv - wv * qb)
        add_le(dp - (1. - wv) * dpb)         # |p_a-p_b| <= (1-w) M
        add_le(-dp - (1. - wv) * dpb)

    for j, comp in enumerate(net.compressors, start=len(net.valves)):
        wc = switch_state(j)
        dp_max = (comp.pressure_out_max - comp.pressure_in_min) / p_ref
        add_eq(node_pressure(comp.to_node) - node_pressure(comp.from_node)
               - dp_comp[comp.id])           # p_out = p_in + dp
        add_le(dp_comp[comp.id] - wc * dp_max)   # dp <= w dp_max

    # ------------------------------------------------------------------ #
    # node mass balances and objective                                    #
    # ------------------------------------------------------------------ #
    def net_inflow(node_id):
        'Net mass inflow (scaled by Q_ref) into a node as MX column.'
        expr = cas.DM.zeros(nt - 1)
        for pipe in net.pipes:
            a = pipe.area / sca.A_ref
            if pipe.to_node == node_id:
                expr = expr + a * m_R[pipe.id]
            if pipe.from_node == node_id:
                expr = expr - a * m_L[pipe.id]
        for valve in net.valves:
            if valve.to_node == node_id:
                expr = expr + q_valve[valve.id]
            if valve.from_node == node_id:
                expr = expr - q_valve[valve.id]
        for comp in net.compressors:
            if comp.to_node == node_id:
                expr = expr + q_comp[comp.id]
            if comp.from_node == node_id:
                expr = expr - q_comp[comp.id]
        return expr

    delivered = {}
    d_refs = {}
    for node in net.nodes.values():
        inflow = net_inflow(node.id)
        if node.type == 'innode':
            add_eq(inflow)
        elif node.type == 'source':
            add_eq(inflow + supply[node.id])
        else:  # sink: delivery tracked in the objective
            delivered[node.id] = inflow
            d_refs[node.id] = bc.demand(node.id, t_sec[:-1]) / sca.Q_ref

    d_scale = max(d.max() for d in d_refs.values())
    for sink_id, d_ref in d_refs.items():
        mis = (delivered[sink_id] - cas.DM(d_ref)) / d_scale
        J = J + 0.5 * dt * cas.dot(mis, mis)

    # compression cost
    for comp in net.compressors:
        J = J + gamma_comp * dt * cas.dot(dp_comp[comp.id],
                                          q_comp[comp.id]) / \
            (rho_span * d_scale)

    nlp = {'f': J, 'x': cas.vertcat(*w), 'g': cas.vertcat(*g)}
    return {
        'nlp': nlp, 'lbw': lbw, 'ubw': ubw, 'lbg': lbg, 'ubg': ubg,
        'w0': w0, 'scaling': sca, 't_scaled': t_grid, 't_sec': t_sec,
        'r': r, 'n_confg': n_confg, 'n_src': n_src, 'n_switches': n_sw,
        'n_ctrl_vars': n_ctrl_vars, 'nt': nt, 'nx': nx,
        'd_refs': d_refs, 'd_scale': d_scale,
        'src_ids': src_ids, 'sink_ids': [n.id for n in sinks],
        'switch_ids': [s.id for s in switches],
        'delivered': delivered, 'layout': layout,
        'pipe_ids': [p.id for p in net.pipes],
    }
