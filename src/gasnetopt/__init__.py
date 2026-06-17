'''
gasnetopt: mixed-integer optimal control on networks via partial outer
convexification (POC), combinatorial integral approximation (CIAP) and
penalty alternating direction methods (ADM).

Implements the algorithms of

[1] Göttlich, Potschka, Teuber: A partial outer convexification approach to
    control transmission lines. Comput. Optim. Appl. (2019).
[2] Göttlich, Hante, Potschka, Schewe: Penalty alternating direction methods
    for mixed-integer optimal control with combinatorial constraints.
    Math. Program. 188 (2021) 599-619.

for electric transmission lines (telegraph equations) and transient gas
networks (semilinear isothermal Euler equations, TRR154/GasLib data).
'''

from pathlib import Path

#: Repository data directory (works for editable installs from the repo).
DATA_DIR = Path(__file__).resolve().parents[2] / 'data'
