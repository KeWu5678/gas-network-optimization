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
Valve: |q_V| <= s q_max, |p_a - p_b| <= (1-s) M.
Compressor: p_b = p_a + dp, 0 <= dp <= s dp_max, 0 <= q_C <= q_max,
with the station pressure envelope enforced both while running and in bypass;
where s_j(t) = sum_c alpha_c(t) r_{c,j} is the relaxed switch state.
'''

from __future__ import annotations

from dataclasses import dataclass, field

import casadi as cas
import numpy as np

from ..core.nlp import NlpBuilder, NlpProblem
from ..core.ocmodel import OCModel
from . import data


@dataclass(frozen=True)
class GasModelSettings:
    horizon: float = 7200.          # s
    nt: int = 121                   # time grid points
    nx: int = 2                     # cells per pipe
    p_ref: float = 50e5             # pressure scale [Pa]
    p_init: float | None = None     # flat initial pressure [Pa]; None: first
                                    # entry pressure of the .bcd
    gamma_comp: float = 0.02        # weight of the compression cost
    eps_fric: float = 1e-3          # smoothing of |m| in the friction term


class GasOCModel(OCModel):
    '''Gas network optimal control model with POC of valve/compressor
    switching (see build_gas_nlp for the discretization).'''

    def __init__(self, net: data.GasNetwork,
                 bc: data.BoundaryConditions,
                 settings: GasModelSettings = GasModelSettings(),
                 initial_state=None) -> None:
        gas = build_gas_nlp(net, bc, settings, initial_state)
        super().__init__(gas.problem, t=gas.t, r=gas.r,
                         n_controls=len(gas.source_ids))
        self._gas = gas
        self.scaling = gas.scaling
        self.t_sec = gas.t_sec
        self._delivered = cas.Function(
            'delivered', [gas.problem.x],
            [gas.delivered[sid] for sid in gas.sink_ids])

    def solution_dict(self, w: np.ndarray) -> dict:
        '''Physical-unit solution arrays: pressures [bar], mass flows [kg/s],
        switch states, deliveries and demands per sink.'''
        gas = self._gas
        sca = self.scaling
        w = np.asarray(w).flatten()
        alpha, u, _ = self.extract(w)
        to_bar = sca.p_ref / 1e5

        def family(name: str, unit: float) -> dict[str, np.ndarray]:
            return {eid: self._variable(w, var).flatten() * unit
                    for eid, var in gas.variable_names[name].items()}

        source_pressure = u * to_bar
        node_pressure = family('node_pressure', to_bar)
        for i, sid in enumerate(gas.source_ids):
            node_pressure[sid] = source_pressure[:, i]
        pipe_mass_flux, pipe_pressure = {}, {}
        for pid in gas.pipe_ids:
            xp = self._variable(w, gas.variable_names['xi_p'][pid])
            xm = self._variable(w, gas.variable_names['xi_m'][pid])
            pipe_mass_flux[pid] = 0.5 * (xp + xm) * sca.m_ref
            pipe_pressure[pid] = 0.5 * (xp - xm) * to_bar
        delivered = self._delivered(w)
        if len(gas.sink_ids) == 1:
            delivered = [delivered]
        return {
            't': self.t_sec,
            'alpha': alpha,
            'switch_state': alpha.dot(self.r),
            'switch_ids': gas.switch_ids,
            'source_ids': gas.source_ids,
            'sink_ids': gas.sink_ids,
            'source_pressure_bar': source_pressure,
            'node_pressure_bar': node_pressure,
            'supply_kg_s': family('supply', sca.Q_ref),
            'valve_flow_kg_s': family('valve_flow', sca.Q_ref),
            'compressor_flow_kg_s': family('compressor_flow', sca.Q_ref),
            'compressor_dp_bar': family('compressor_dp', to_bar),
            'pipe_mass_flux': pipe_mass_flux,
            'pipe_pressure_bar': pipe_pressure,
            'delivered_kg_s': {
                sid: np.array(val).flatten() * sca.Q_ref
                for sid, val in zip(gas.sink_ids, delivered)},
            'demand_kg_s': {sid: gas.demand[sid] * sca.Q_ref
                            for sid in gas.sink_ids},
        }

    def _variable(self, w: np.ndarray, name: str) -> np.ndarray:
        'Extract variable `name` from NLP vector w as a 2D array.'
        pos, rows, cols = self._gas.offsets[name]
        return np.reshape(w[pos:pos + rows * cols], (rows, cols), order='F')


def build_gas_nlp(net: data.GasNetwork,
                  bc: data.BoundaryConditions,
                  settings: GasModelSettings = GasModelSettings(),
                  initial_state=None) -> GasNlp:
    '''Build the POC-relaxed NLP for gas network `net` (data.GasNetwork)
    with boundary conditions `bc` (data.BoundaryConditions).

    initial_state: optional data.NetworkState for the initial condition;
    if None, a flat state at pressure settings.p_init and zero flow is used.

    Returns the NLP with the data needed to interpret its solution.'''
    T, nt, nx = settings.horizon, settings.nt, settings.nx
    p_ref = settings.p_ref
    missing = data.missing_big_m_bounds(net)
    if missing:
        raise ValueError('big-M bounds missing in the .net file: '
                         + '; '.join(missing))
    sca = GasScaling(c=bc.speed_of_sound, p_ref=p_ref,
                     L_ref=max(pipe.length for pipe in net.pipes),
                     A_ref=float(np.mean([pipe.area for pipe in net.pipes])))
    t_sec = np.linspace(0., T, nt)                   # physical time grid
    t = t_sec / sca.T_ref                            # scaled time grid
    dt = t[1] - t[0]
    _check_cfl(net, sca, dt, nx)
    p_init = settings.p_init
    if p_init is None:
        p_init = _default_initial_pressure(bc, p_ref)
    r = switch_encoding(len(net.switches))
    # network pressure span: big-M of the valves, scale of the compression
    # cost
    p_span = (max(n.pressure_max for n in net.nodes.values())
              - min(n.pressure_min for n in net.nodes.values())) / p_ref

    b = NlpBuilder()
    v = _add_variables(b, net, bc, sca, t_sec, nx, r.shape[0], p_init,
                       _initial_pipe_states(net, sca, nx, p_init,
                                            initial_state))
    _add_pipe_dynamics(b, net, v, sca, dt, nx, settings.eps_fric)
    _add_switching_elements(b, net, v, r, sca, p_span)
    delivered = _add_node_balances(b, net, v, sca)
    if not delivered:
        raise ValueError('Network has no sink nodes; the objective tracks '
                         'sink deliveries against the .bcd demand.')
    demand = {sid: bc.demand(sid, t_sec[:-1]) / sca.Q_ref
              for sid in delivered}

    return GasNlp(
        problem=b.build(_objective(net, v, delivered, demand, dt,
                                   settings.gamma_comp, p_span)),
        scaling=sca, t=t, t_sec=t_sec, r=r,
        source_ids=v.source_ids, sink_ids=[n.id for n in net.sinks],
        switch_ids=[s.id for s in net.switches],
        pipe_ids=[pipe.id for pipe in net.pipes],
        variable_names=v.names(), offsets=b.offsets,
        delivered=delivered, demand=demand)


@dataclass(frozen=True)
class GasNlp:
    problem: NlpProblem
    scaling: GasScaling
    t: np.ndarray                   # scaled time grid
    t_sec: np.ndarray               # physical time grid [s]
    r: np.ndarray                   # configuration encoding
    source_ids: list[str]
    sink_ids: list[str]
    switch_ids: list[str]
    pipe_ids: list[str]
    variable_names: dict[str, dict[str, str]]   # family -> id -> name
    offsets: dict[str, tuple[int, int, int]]    # name -> (pos, rows, cols)
    delivered: dict[str, cas.MX]                # per sink, scaled by Q_ref
    demand: dict[str, np.ndarray]               # per sink, scaled by Q_ref


@dataclass
class GasScaling:
    'Reference quantities of the nondimensionalization.'
    c: float            # speed of sound [m/s]
    p_ref: float        # pressure scale [Pa]
    L_ref: float        # length scale [m]
    A_ref: float        # area scale [m^2]

    @property
    def rho_ref(self) -> float:
        return self.p_ref / self.c ** 2

    @property
    def m_ref(self) -> float:
        return self.rho_ref * self.c     # mass flux density scale

    @property
    def T_ref(self) -> float:
        return self.L_ref / self.c       # time scale [s]

    @property
    def Q_ref(self) -> float:
        return self.m_ref * self.A_ref   # mass flow scale [kg/s]


def _check_cfl(net: data.GasNetwork, sca: GasScaling, dt: float,
               nx: int) -> None:
    # transport speed 1 in the scaled equations
    dx_min = min(pipe.length / sca.L_ref / nx for pipe in net.pipes)
    if dt > dx_min + 1e-12:
        raise ValueError('CFL violated: dt={:.3g} > dx={:.3g} (scaled); '
                         'increase nt or decrease nx'.format(dt, dx_min))


def _default_initial_pressure(bc: data.BoundaryConditions,
                              p_ref: float) -> float:
    if not bc.pressure:
        return p_ref
    first = sorted(bc.pressure)[0]
    return bc.pressure[first][1][0]


def switch_encoding(n_switches: int) -> np.ndarray:
    '''Binary encoding of all 2^s switch configurations: returns r of shape
    (2^s, s) with r[c, j] = state of switch j in configuration c.'''
    n_configs = 2 ** n_switches
    r = np.zeros((n_configs, n_switches), dtype=int)
    for conf in range(n_configs):
        for j in range(n_switches):
            r[conf, j] = (conf >> j) & 1
    return r


def _initial_pipe_states(net: data.GasNetwork, sca: GasScaling, nx: int,
                         p_init: float, initial_state) -> dict:
    init = {}
    for pipe in net.pipes:
        if initial_state is not None and pipe.id in \
                initial_state.pipe_profiles:
            xs, qs, ps = initial_state.pipe_profiles[pipe.id]
            x_cells = (np.arange(nx) + .5) * pipe.length / nx
            m0 = np.interp(x_cells, xs, qs) / pipe.area / sca.m_ref
            p0 = np.interp(x_cells, xs, ps) / sca.p_ref
        else:
            m0 = np.zeros(nx)
            p0 = np.full(nx, p_init / sca.p_ref)
        init[pipe.id] = (m0, p0)
    return init


@dataclass
class _GasVariables:
    alpha: cas.MX                   # (nt-1, n_configs)
    u: cas.MX                       # source pressures (nt-1, n_sources)
    source_ids: list[str]
    xi_p: dict = field(default_factory=dict)     # pipe -> (nx, nt)
    xi_m: dict = field(default_factory=dict)
    p_node: dict = field(default_factory=dict)   # non-source node pressure
    m_L: dict = field(default_factory=dict)      # pipe boundary fluxes
    m_R: dict = field(default_factory=dict)
    supply: dict = field(default_factory=dict)   # source mass flows
    q_valve: dict = field(default_factory=dict)
    q_comp: dict = field(default_factory=dict)
    dp_comp: dict = field(default_factory=dict)

    def pressure(self, node_id: str) -> cas.MX:
        'Pressure expression of a node (control column for sources).'
        if node_id in self.source_ids:
            return self.u[:, self.source_ids.index(node_id)]
        return self.p_node[node_id]

    def names(self) -> dict[str, dict[str, str]]:
        families = {'node_pressure': self.p_node, 'supply': self.supply,
                    'valve_flow': self.q_valve, 'compressor_flow': self.q_comp,
                    'compressor_dp': self.dp_comp, 'xi_p': self.xi_p,
                    'xi_m': self.xi_m}
        return {family: {eid: sym.name() for eid, sym in syms.items()}
                for family, syms in families.items()}


def _add_variables(b: NlpBuilder, net: data.GasNetwork,
                   bc: data.BoundaryConditions, sca: GasScaling,
                   t_sec: np.ndarray, nx: int, n_configs: int, p_init: float,
                   init: dict) -> _GasVariables:
    # column-major layout, alpha and u first (the OCModel layout)
    nt = len(t_sec)
    p_ref = sca.p_ref
    sources = net.sources

    # convex configuration multipliers
    alpha = b.add_variable(cas.MX.sym('alpha', nt - 1, n_configs),
                           np.full((nt - 1, n_configs), 1. / n_configs),
                           np.zeros((nt - 1, n_configs)),
                           np.full((nt - 1, n_configs), 2.))
    b.add_equality(cas.mtimes(alpha, cas.DM.ones(n_configs)) - 1.)  # SOS1

    # continuous controls: source pressures (scaled)
    u0 = np.column_stack([
        bc.pressure_at(n.id, t_sec[:-1]) / p_ref if n.id in bc.pressure
        else np.full(nt - 1, p_init / p_ref) for n in sources])
    u = b.add_variable(cas.MX.sym('u', nt - 1, len(sources)), u0,
                       np.column_stack([np.full(nt - 1, n.pressure_min / p_ref)
                                        for n in sources]),
                       np.column_stack([np.full(nt - 1, n.pressure_max / p_ref)
                                        for n in sources]))
    v = _GasVariables(alpha=alpha, u=u, source_ids=[n.id for n in sources])

    # pipe characteristic states xi_pm = m +- p (scaled), initial column
    # fixed by bounds
    for pipe in net.pipes:
        m0, p0 = init[pipe.id]
        for sign, store in (('p', v.xi_p), ('m', v.xi_m)):
            s0 = m0 + p0 if sign == 'p' else m0 - p0
            lb = np.full((nx, nt), -np.inf)
            ub = np.full((nx, nt), np.inf)
            lb[:, 0] = s0
            ub[:, 0] = s0
            store[pipe.id] = b.add_variable(
                cas.MX.sym('xi_{}_{}'.format(sign, pipe.id), nx, nt),
                np.tile(s0[:, None], (1, nt)), lb, ub)

    # node pressures (scaled) for non-source nodes, defined on t = 0..nt-2
    for node in net.nodes.values():
        if node.type == 'source':
            continue
        v.p_node[node.id] = b.add_variable(
            cas.MX.sym('p_{}'.format(node.id), nt - 1),
            np.full(nt - 1, p_init / p_ref),
            np.full(nt - 1, node.pressure_min / p_ref),
            np.full(nt - 1, node.pressure_max / p_ref))

    # pipe boundary mass flux densities (scaled), t = 0..nt-2
    for pipe in net.pipes:
        mb = pipe.flow_max / (sca.m_ref * pipe.area)
        for name, store in (('L', v.m_L), ('R', v.m_R)):
            store[pipe.id] = b.add_variable(
                cas.MX.sym('m{}_{}'.format(name, pipe.id), nt - 1),
                np.zeros(nt - 1), np.full(nt - 1, -mb), np.full(nt - 1, mb))

    # source supplies (scaled mass flow), t = 0..nt-2
    for node in sources:
        v.supply[node.id] = b.add_variable(
            cas.MX.sym('s_{}'.format(node.id), nt - 1),
            np.full(nt - 1, node.flow_min / sca.Q_ref),
            np.full(nt - 1, node.flow_min / sca.Q_ref),
            np.full(nt - 1, node.flow_max / sca.Q_ref))

    # valve flows (scaled mass flow), t = 0..nt-2
    for valve in net.valves:
        qb = valve.flow_max / sca.Q_ref
        v.q_valve[valve.id] = b.add_variable(
            cas.MX.sym('qV_{}'.format(valve.id), nt - 1),
            np.zeros(nt - 1), np.full(nt - 1, -qb), np.full(nt - 1, qb))

    # compressor flows and pressure increases (scaled), t = 0..nt-2
    for comp in net.compressors:
        v.q_comp[comp.id] = b.add_variable(
            cas.MX.sym('qC_{}'.format(comp.id), nt - 1), np.zeros(nt - 1),
            np.full(nt - 1, comp.flow_min / sca.Q_ref),
            np.full(nt - 1, comp.flow_max / sca.Q_ref))
        dp_max = (comp.pressure_out_max - comp.pressure_in_min) / p_ref
        v.dp_comp[comp.id] = b.add_variable(
            cas.MX.sym('dp_{}'.format(comp.id), nt - 1), np.zeros(nt - 1),
            np.zeros(nt - 1), np.full(nt - 1, dp_max))
    return v


def _add_pipe_dynamics(b: NlpBuilder, net: data.GasNetwork,
                       v: _GasVariables, sca: GasScaling, dt: float, nx: int,
                       eps_fric: float) -> None:
    # strictly positive density inside every pipe cell: rho appears in the
    # friction denominator, and the characteristic states carry no implicit
    # positivity (their box bounds are infinite past the initial column)
    rho_lo = max(0.5 * min(n.pressure_min for n in net.nodes.values())
                 / sca.p_ref, 1e-2)

    for pipe in net.pipes:
        dx = pipe.length / sca.L_ref / nx
        kappa = pipe.friction_factor() * sca.L_ref / (2. * pipe.diameter)
        xp, xm = v.xi_p[pipe.id], v.xi_m[pipe.id]
        p_l = v.pressure(pipe.from_node)
        p_r = v.pressure(pipe.to_node)
        mL, mR = v.m_L[pipe.id], v.m_R[pipe.id]

        for k in range(xp.size2() - 1):
            # friction at the new time level (implicit)
            m_new = 0.5 * (xp[:, k + 1] + xm[:, k + 1])
            rho_new = 0.5 * (xp[:, k + 1] - xm[:, k + 1])  # scaled rho = p
            fric = kappa * m_new * cas.sqrt(m_new ** 2 + eps_fric ** 2) \
                / rho_new
            b.add_upper_bound(rho_lo - rho_new)      # rho >= rho_lo > 0

            # xi_p: transport to the right, ghost value from left node
            ghost_p = mL[k] + p_l[k]
            theta = cas.vertcat(ghost_p, xp[:-1, k])
            adv = -(xp[:, k] - theta) / dx
            b.add_equality(xp[:, k + 1] - xp[:, k] - dt * (adv - fric))

            # xi_m: transport to the left, ghost value from right node
            ghost_m = mR[k] - p_r[k]
            theta = cas.vertcat(xm[1:, k], ghost_m)
            adv = (theta - xm[:, k]) / dx
            b.add_equality(xm[:, k + 1] - xm[:, k] - dt * (adv - fric))

            # outgoing characteristics matched to the node state
            b.add_equality(xm[0, k] - (mL[k] - p_l[k]))
            b.add_equality(xp[-1, k] - (mR[k] + p_r[k]))


def _add_switching_elements(b: NlpBuilder, net: data.GasNetwork,
                            v: _GasVariables, r: np.ndarray,
                            sca: GasScaling, p_span: float) -> None:
    n_steps = v.alpha.size1()
    switch_index = {s.id: j for j, s in enumerate(net.switches)}

    def switch_state(j):
        'Relaxed state s_j(t) of switch j as MX column (nt-1).'
        cols = [v.alpha[:, conf] for conf in range(r.shape[0]) if r[conf, j]]
        return sum(cols) if cols else cas.DM.zeros(n_steps)

    for valve in net.valves:
        s = switch_state(switch_index[valve.id])
        qv = v.q_valve[valve.id]
        qb = valve.flow_max / sca.Q_ref
        dpb = min(valve.pressure_diff_max / sca.p_ref, p_span)
        dp = v.pressure(valve.from_node) - v.pressure(valve.to_node)
        b.add_upper_bound(qv - s * qb)               # |q_V| <= s q_max
        b.add_upper_bound(-qv - s * qb)
        b.add_upper_bound(dp - (1. - s) * dpb)       # |p_a-p_b| <= (1-s) M
        b.add_upper_bound(-dp - (1. - s) * dpb)

    for comp in net.compressors:
        s = switch_state(switch_index[comp.id])
        dp_max = (comp.pressure_out_max - comp.pressure_in_min) / sca.p_ref
        p_in = v.pressure(comp.from_node)
        p_out = v.pressure(comp.to_node)
        b.add_equality(p_out - p_in - v.dp_comp[comp.id])   # p_out = p_in + dp
        b.add_upper_bound(v.dp_comp[comp.id] - s * dp_max)  # dp <= s dp_max
        # The same station carries flow in compression and bypass, so its
        # equipment pressure envelope applies in both states.
        b.add_upper_bound(comp.pressure_in_min / sca.p_ref - p_in)
        b.add_upper_bound(p_out - comp.pressure_out_max / sca.p_ref)


def _add_node_balances(b: NlpBuilder, net: data.GasNetwork,
                       v: _GasVariables, sca: GasScaling
                       ) -> dict[str, cas.MX]:
    delivered = {}
    for node in net.nodes.values():
        inflow = _net_inflow(net, v, node.id, sca)
        if node.type == 'innode':
            b.add_equality(inflow)
        elif node.type == 'source':
            b.add_equality(inflow + v.supply[node.id])
        else:
            delivered[node.id] = inflow
    return delivered


def _net_inflow(net: data.GasNetwork, v: _GasVariables, node_id: str,
                sca: GasScaling) -> cas.MX:
    'Net mass inflow (scaled by Q_ref) into a node as MX column.'
    expr = cas.DM.zeros(v.alpha.size1())
    for pipe in net.pipes:
        a = pipe.area / sca.A_ref
        if pipe.to_node == node_id:
            expr = expr + a * v.m_R[pipe.id]
        if pipe.from_node == node_id:
            expr = expr - a * v.m_L[pipe.id]
    for valve in net.valves:
        if valve.to_node == node_id:
            expr = expr + v.q_valve[valve.id]
        if valve.from_node == node_id:
            expr = expr - v.q_valve[valve.id]
    for comp in net.compressors:
        if comp.to_node == node_id:
            expr = expr + v.q_comp[comp.id]
        if comp.from_node == node_id:
            expr = expr - v.q_comp[comp.id]
    return expr


def _objective(net: data.GasNetwork, v: _GasVariables,
               delivered: dict, demand: dict, dt: float, gamma_comp: float,
               p_span: float) -> cas.MX:
    # guard against identically zero demand (objective is scale-invariant
    # only up to this normalization)
    d_scale = max(d.max() for d in demand.values()) or 1.
    J = 0
    for sink_id, d_ref in demand.items():
        mis = (delivered[sink_id] - cas.DM(d_ref)) / d_scale
        J = J + 0.5 * dt * cas.dot(mis, mis)
    for comp in net.compressors:
        J = J + gamma_comp * dt * cas.dot(v.dp_comp[comp.id],
                                          v.q_comp[comp.id]) / \
            (p_span * d_scale)
    return J
