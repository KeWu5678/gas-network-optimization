'''
OCModel interface for the gas network MIOCP, so that the CIAP and penalty-ADM
algorithms (gasnetopt.ciap / gasnetopt.adm) run unchanged on gas networks.

As in the transmission-lines model, the NLP objective is augmented with the
outer-convexified L1 coupling penalty rho * sum_k h_k sum_i alpha_{k,i}
sum_j |r_{i,j} - V_ref_{j,k}|, parametric in (rho, V_ref).
'''

from __future__ import annotations

import casadi as cas
import numpy as np

from .. import collocation
from . import gaslib_io
from .model import build_gas_nlp


class GasOCModel(collocation.OCModel):
    '''Gas network optimal control model with POC of valve/compressor
    switching (see gas.model.build_gas_nlp for the discretization).'''

    def __init__(self, net: gaslib_io.GasNetwork,
                 bc: gaslib_io.BoundaryConditions, **kwargs) -> None:
        self.net = net
        self.bc = bc
        d = build_gas_nlp(net, bc, **kwargs)
        self.data = d
        self.nlp = d['nlp']
        self.lbw, self.ubw = d['lbw'], d['ubw']
        self.lbg, self.ubg = d['lbg'], d['ubg']
        self.w0 = np.array(d['w0'])
        self.scaling = d['scaling']
        self.t = d['t_scaled']
        self.t_sec = d['t_sec']
        self.nt, self.nx = d['nt'], d['nx']
        self.n_confg = d['n_confg']
        self.n_src = d['n_src']
        self.nalpha = d['n_confg']
        self.nv_ref = d['n_switches']
        self.r = d['r']

        # variable offsets from the storage layout
        self.offsets = {}
        pos = 0
        for name, (rows, cols) in d['layout']:
            self.offsets[name] = (pos, rows, cols)
            pos += rows * cols
        self.nw = pos

        # augment with the outer-convexified L1 penalization term
        self.add_l1_penalty()

    def var(self, w: np.ndarray, name: str) -> np.ndarray:
        'Extract variable `name` from NLP vector w as a 2D array.'
        pos, rows, cols = self.offsets[name]
        return np.reshape(w[pos:pos + rows * cols], (rows, cols), order='F')

    def extract(self, w: np.ndarray) -> tuple:
        'Extract and return (y, u, alpha, v, nodes) from NLP variable w.'
        alpha = self.var(w, 'alpha')
        u = self.var(w, 'u')
        y = w[(self.nt - 1) * (self.n_confg + self.n_src):]
        return y, u, alpha, None, None

    def overwrite(self, w: np.ndarray, y=None, u=None, alpha=None,
                  v=None, nodes=None) -> np.ndarray:
        'Overwrite given components in NLP variable w. Returns w.'
        if v is not None or nodes is not None:
            raise Exception('Cannot overwrite unknown variables')
        nt = self.nt
        if alpha is not None:
            w[:(nt - 1) * self.n_confg] = alpha.flatten(order='F')
        if u is not None:
            pos = (nt - 1) * self.n_confg
            w[pos:pos + (nt - 1) * self.n_src] = u.flatten(order='F')
        if y is not None:
            pos = (nt - 1) * (self.n_confg + self.n_src)
            w[pos:] = y
        return w

    def solution_dict(self, w: np.ndarray) -> dict:
        '''Physical-unit solution arrays: pressures [bar], mass flows [kg/s],
        switch states, deliveries and demands per sink.'''
        d = self.data
        sca = self.scaling
        w = np.asarray(w).flatten()
        out = {
            't': self.t_sec,
            'alpha': self.var(w, 'alpha'),
            'switch_ids': d['switch_ids'],
            'src_ids': d['src_ids'],
            'sink_ids': d['sink_ids'],
        }
        out['w_switch'] = out['alpha'].dot(self.r)   # relaxed switch states
        out['u_bar'] = self.var(w, 'u') * sca.p_ref / 1e5
        # node pressures
        pressures = {}
        for name in self.offsets:
            if name.startswith('rho_'):
                pressures[name[4:]] = \
                    self.var(w, name).flatten() * sca.p_ref / 1e5
        for i, sid in enumerate(d['src_ids']):
            pressures[sid] = out['u_bar'][:, i]
        out['node_pressure_bar'] = pressures
        # element flows
        flows = {}
        for name in self.offsets:
            if name.startswith('qV_') or name.startswith('qC_') \
                    or name.startswith('s_'):
                key = name.split('_', 1)[1]
                flows[key] = self.var(w, name).flatten() * sca.Q_ref
        out['flows_kg_s'] = flows
        out['dp_bar'] = {
            name[3:]: self.var(w, name).flatten() * sca.p_ref / 1e5
            for name in self.offsets if name.startswith('dp_')}
        # pipe states
        out['pipe_m'] = {}
        out['pipe_p_bar'] = {}
        for pid in d['pipe_ids']:
            xp = self.var(w, 'xi_p_' + pid)
            xm = self.var(w, 'xi_m_' + pid)
            out['pipe_m'][pid] = 0.5 * (xp + xm) * sca.m_ref
            out['pipe_p_bar'][pid] = 0.5 * (xp - xm) * sca.p_ref / 1e5
        # deliveries: evaluate the delivered expressions
        delivered = cas.Function(
            'delivered', [self.nlp['x']],
            [d['delivered'][sid] for sid in d['sink_ids']])
        vals = delivered(w)
        if len(d['sink_ids']) == 1:
            vals = [vals]
        out['delivered_kg_s'] = {
            sid: np.array(val).flatten() * sca.Q_ref
            for sid, val in zip(d['sink_ids'], vals)}
        out['demand_kg_s'] = {
            sid: d['d_refs'][sid] * sca.Q_ref for sid in d['sink_ids']}
        return out
