'''
Utility functions for direct collocation discretizations.
'''

from __future__ import annotations

from itertools import tee

import casadi as cas
import numpy as np


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def gauss_collocation(d: int, points: str = 'legendre'
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    'Compute matrices for Gauss collocation of degree d'

    # Get collocation points
    tau_root = [0]+cas.collocation_points(d, points)

    # Coefficients of the collocation equation
    C = np.zeros((d+1,d+1))

    # Coefficients of the continuity equation
    D = np.zeros(d+1)

    # Coefficients of the quadrature function
    B = np.zeros(d+1)

    # Construct polynomial basis
    for j in range(d+1):
        # Construct Lagrange polynomials to get the polynomial basis at the
        # collocation point
        p = np.poly1d([1])
        for r in range(d+1):
            if r != j:
                p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j]-tau_root[r])

        # Evaluate the polynomial at the final time to get the coefficients of
        # the continuity equation
        D[j] = p(1.0)

        # Evaluate the time derivative of the polynomial at all collocation
        # points to get the coefficients of the continuity equation
        pder = np.polyder(p)
        for r in range(d+1):
            C[j,r] = pder(tau_root[r])

        # Evaluate the integral of the polynomial to get the coefficients of the
        # quadrature function
        pint = np.polyint(p)
        B[j] = pint(1.0)

    return B, C, D

def direct_transcription(prob, d, t):
    '''Direct transcription of problem prob via Gauss-Legendre collocation of
    degree d on the grid determined by t. Adds simplex SOS1 constraint for
    variables in prob['SOS1']. Returns an NLP.
    '''
    # obtain collocation matrices
    B, C, D = gauss_collocation(d)

    # construct functions
    arg_list = ['y', 'u', 'alpha', 'v', 'rho', 'v_ref']
    args = [prob[var_name] for var_name in arg_list if var_name in prob]
    f = cas.Function('f', args, [prob['ydot'], prob['L']])

    # Start with an empty NLP
    w = []
    w0 = []
    lbw = []
    ubw = []
    J = 0
    g = []
    lbg = []
    ubg = []
    v_indices = []

    # Make it parametric if v_ref is part of the problem
    if 'v_ref' in prob:
        rho = cas.MX.sym('rho')
        V_ref = cas.MX.sym('V_ref', prob['nv_ref'], len(t)-1)

    # Initial conditions
    Yk = cas.MX.sym('Y_0000', prob['ny'])
    w += [Yk]
    lbw += prob['lb_ys']
    ubw += prob['ub_ys']
    w0 += [0] * prob['ny']

    # Formulate the NLP
    for k, (tk, tkp1) in enumerate(pairwise(t)):
        # New NLP variables for the controls
        Cks = []
        for cvar in [cvar for cvar in ['u', 'alpha', 'v'] if cvar in prob]:
            nc = prob['n'+cvar]
            Ck = cas.MX.sym('{}_{:04d}'.format(cvar.upper(), k), nc)
            Cks += [Ck]
            w += [Ck]
            lbw += [prob['lb_'+cvar]] * nc
            ubw += [prob['ub_'+cvar]] * nc
            w0  += [.5] * nc
            # SOS1 simplex constraints for marked variables
            if cvar in prob['SOS1']: #alpha or v
                g += [cas.dot(cas.MX.ones(nc,1), Ck)] #inner product
                lbg += [1]
                ubg += [1]
            # remember v variable indices
            if cvar == 'v':
                v_indices += list(range(len(w0)-nc, len(w0)))

        # possibly add parametric dependency on penalty and reference for rhs
        if 'v_ref' in prob:
            Cks += [rho, V_ref[:,k]]

        # State at collocation points
        Ykj = []
        for j in range(d):
            Ykj += [cas.MX.sym('Y_{:04d}_{:02d}'.format(k, j), prob['ny'])]
            w += [Ykj[-1]]
            lbw += [-cas.inf] * prob['ny']
            ubw += [cas.inf] * prob['ny']
            w0 += [0] * prob['ny']

        # Loop over collocation points
        Yk_end = D[0]*Yk
        for j in range(1,d+1):
            # Expression for the state derivative at the collocation point
            yp = C[0,j]*Yk
            for r in range(d):
                yp = yp + C[r+1,j]*Ykj[r]

            # Append collocation equations
            h = tkp1 - tk
            fj, qj = f(*([Ykj[j-1]]+Cks))
            g += [h*fj - yp]
            lbg += [0] * prob['ny']
            ubg += [0] * prob['ny']

            # Add contribution to the end state
            Yk_end = Yk_end + D[j]*Ykj[j-1]

            # Add contribution to quadrature function
            J = J + B[j]*qj*h

        # New NLP variable for state at end of interval
        Yk = cas.MX.sym('Y_{:04d}'.format(k+1), prob['ny'])
        w += [Yk]
        lbw += [-cas.inf] * prob['ny']
        ubw += [cas.inf] * prob['ny']
        w0 += [0] * prob['ny']

        # Add equality constraint
        g += [Yk_end-Yk]
        lbg += [0] * prob['ny']
        ubg += [0] * prob['ny']

    # add Mayer objective term
    if 'M' in prob:
        if 'rho' in prob:
            M = cas.Function('M', [prob['y'], prob['rho']], [prob['M']])
            J = J + M(Yk, rho)
        else:
            M = cas.Function('M', [prob['y']], [prob['M']])
            J = J + M(Yk)

    # modify terminal bounds
    if 'lb_ye' in prob:
        lbw[-prob['ny']:] = prob['lb_ye']
    if 'ub_ye' in prob:
        ubw[-prob['ny']:] = prob['ub_ye']
    w0[-prob['ny']:] = [0] * prob['ny']

    # create NLP dictionary
    nlp = {}
    nlp['f'] = J
    nlp['x'] = cas.vertcat(*w)
    if 'v_ref' in prob:
        nlp['p'] = cas.vertcat(rho, cas.vec(V_ref))
    nlp['g'] = cas.vertcat(*g)

    return nlp, lbw, ubw, lbg, ubg, w0, v_indices

def get_NLP_vars(w, p, d):
    'Recover y, u, alpha, v of problem p from NLP solution vector w.'
    # pad with non-existing nodes and controls on final grid point
    ny = p.get('ny')
    nu = p.get('nu', 0)
    na = p.get('nalpha', 0)
    nv = p.get('nv', 0)
    w = np.concatenate((w, [0] * (nu + na + nv + ny*d)))
    # reshape and extract
    w = np.reshape(w, (-1, ny + nu + na + nv + ny*d))
    y, u, alpha, v, nodes = np.hsplit(w, np.cumsum([ny,nu,na,nv]))
    u = u[:-1,:]
    alpha = alpha[:-1,:]
    v = v[:-1,:]
    nodes = nodes[:-1,:]
    return y, u, alpha, v, nodes

def set_NLP_vars(w, p, d, y=None, u=None, alpha=None, v=None, nodes=None):
    'Set NLP variables in w (for problem p, degree d).'
    # pad with non-existing nodes and controls on final grid point
    ny = p.get('ny')
    nu = p.get('nu', 0)
    na = p.get('nalpha', 0)
    nv = p.get('nv', 0)
    w = np.concatenate((w, [0] * (nu + na + nv + ny*d))) #[0]*x: x-dim zero vector
    # reshape and extract
    w = np.reshape(w, (-1, ny + nu + na + nv + ny*d))
    if y is not None:
        w[:,:ny] = y
    if u is not None:
        w[:-1,ny:ny+nu] = u #-1:last row all 0
    if alpha is not None:
        w[:-1,ny+nu:ny+nu+na] = alpha
    if v is not None:
        w[:-1,ny+nu+na:ny+nu+na+nv] = v
    if nodes is not None:
        w[:-1,ny+nu+na+nv:ny+nu+na+nv+ny*d] = nodes
    w = w.flatten()
    w = w[:-(nu+na+nv+ny*d)]
    return w

class OCModel:
    '''Class interface for optimal control models that we generate by direct
    collocation from the problems in problems.py.'''

    def __init__(self, problem: dict, degree: int,
                 t: np.ndarray) -> None:
        '''Constructor for problem with collocation discretization of given
        degree on time grid t.
        t: numpy array using linspace'''
        self.problem = problem
        self.degree = degree
        self.t = t
        c = direct_transcription(problem, degree, t)
        self.nlp, self.lbw, self.ubw, self.lbg, self.ubg, self.w0, self.vidx = c
        self.nv_ref = problem.get('nv_ref', 0)
        self.nalpha = problem.get('nalpha', 0)
        self.r = problem.get('r', np.array([]))

    def extract(self, w: np.ndarray) -> tuple:
        'Extract and return (y, u, alpha, v, nodes) from NLP variable w.'
        return get_NLP_vars(w, self.problem, self.degree)

    def overwrite(self, w: np.ndarray, y=None, u=None, alpha=None,
                  v=None, nodes=None) -> np.ndarray:
        '''Overwrite given components (y, u, alpha, v, nodes) in NLP variable w.
        Returns overwritten w.'''
        return set_NLP_vars(w, self.problem, self.degree, y, u, alpha, v, nodes)

    def create_NLP_solver(self, nlp_solver_name: str = 'ipopt',
                          tol: float = 1e-13) -> cas.Function:
        options: dict = {'print_time': False}
        if nlp_solver_name == 'ipopt':
            options['ipopt'] = {'tol': tol, 'print_level': 0}
        elif nlp_solver_name == 'blocksqp':
            options['opttol'] = tol
        return cas.nlpsol('solver', nlp_solver_name, self.nlp, options)

    def add_l1_penalty(self, obj_sca: float = 1.) -> None:
        '''Augment the NLP objective with the outer-convexified L1 coupling
        penalty rho * sum_k h_k sum_i alpha_{k,i} sum_j |r_{i,j} - V_ref_{j,k}|
        of the penalty ADM, parametric in (rho, V_ref). Assumes alpha (shape
        (nt-1, nalpha)) is stored first in nlp['x'] and r is binary. The
        original objective is scaled by obj_sca.'''
        nt = len(self.t)
        rho = cas.MX.sym('rho')
        V_ref = cas.MX.sym('V_ref', self.nv_ref, nt - 1)
        self.nlp['p'] = cas.vertcat(rho, cas.vec(V_ref))
        alpha = cas.reshape(self.nlp['x'][:(nt - 1) * self.nalpha],
                            nt - 1, self.nalpha)
        pen_L1 = 0
        for k, (tk, tkp1) in enumerate(pairwise(self.t)):
            integrand = 0
            for i in range(self.nalpha):
                for j in range(self.nv_ref):
                    # explicit cases for signs in absolute value function
                    if self.r[i, j] == 0:  # -> |x| = +x >= 0
                        integrand += V_ref[j, k] * alpha[k, i]
                    elif self.r[i, j] == 1:  # -> |x| = -x >= 0
                        integrand += (1 - V_ref[j, k]) * alpha[k, i]
                    else:
                        raise Exception('Array r not binary')
            pen_L1 = pen_L1 + (tkp1 - tk) * integrand
        self.nlp['f'] = obj_sca * self.nlp['f'] + rho * pen_L1
        self._f_obj = None

    def evaluate_objective(self, rho: float, v_ref: np.ndarray,
                           w: np.ndarray) -> float:
        '''Evaluate objective with L1 penalization term.'''
        f_obj = getattr(self, '_f_obj', None)
        if f_obj is None:
            f_obj = self._f_obj = cas.Function(
                'obj', [self.nlp['x'], self.nlp['p']], [self.nlp['f']])
        p = np.concatenate(([rho], v_ref.flatten()))
        return float(f_obj(w, p))

