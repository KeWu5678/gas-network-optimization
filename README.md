# gasnetopt — mixed-integer optimal control of transmission lines and gas networks

Mixed-integer optimal control of networked hyperbolic PDEs via **partial
outer convexification (POC)**, **combinatorial integral approximation
(CIAP)** and the **penalty alternating direction method (ADM)**, applied to

- electric transmission lines (telegraph equations), and
- transient gas networks (semilinear isothermal Euler equations) on
  [GasLib](https://gaslib.zib.de) instances with
  [TRR154 transient data](https://www.trr154.fau.de/transient-data/).

## References

1. S. Göttlich, A. Potschka, C. Teuber: *A partial outer convexification
   approach to control transmission lines*, Comput. Optim. Appl. (2019).
2. S. Göttlich, F. M. Hante, A. Potschka, L. Schewe: *Penalty alternating
   direction methods for mixed-integer optimal control with combinatorial
   constraints*, Math. Program. 188 (2021) 599–619.
3. P. Domschke, B. Hiller, J. Lang, C. Tischendorf: *Modellierung von
   Gasnetzwerken: Eine Übersicht*, TRR154 preprint 2717 (2017).

PDFs in `docs/references/`. The gas extension is documented in
[`docs/gas-network-design.md`](docs/gas-network-design.md).

## Layout

```
data/
  translines/        demand profiles for the transmission-lines example
  gaslib/            GasLib-11/-40 networks + TRR154 .bcd/.state + XSDs
docs/
  gas-network-design.md   data-structure analysis & gas model design
  references/             papers and the TRR154 format report
examples/            runnable scripts (see below)
src/gasnetopt/
  collocation.py     Gauss collocation, direct transcription, OCModel base
  milp.py            solver-agnostic MILP layer (HiGHS default, Gurobi optional)
  ciap.py            sum-up rounding, CIAP-MILP, dwell-time COMB
  adm.py             penalty ADM [2], tCOMB projection MILP
  results_io.py      portable result persistence
  translines/        telegraph-equation example [1]
  gas/               GasLib/TRR154 parsers, ISO2 gas model, GasOCModel
tests/               pytest suite
```

## Quickstart

Requires [uv](https://docs.astral.sh/uv/):

```sh
uv sync            # creates .venv with casadi/ipopt, numpy, scipy, matplotlib
uv run pytest      # run the test suite

# transmission lines: POC relaxation + sum-up rounding
uv run python examples/run_translines.py

# transmission lines: full benchmark (POC, SUR, COMB-CIAP, penalty ADM)
uv run python examples/run_translines_adm.py --net "extended tree" --coarse
uv run python examples/present_translines_results.py \
    results/translines/translines_extended_tree_results

# gas network: GasLib-11 with TRR154 sinus boundary data
uv run python examples/run_gaslib11.py --horizon 7200 --tau-min 900
```

Figures and result files are written to `results/` (gitignored).

The MILP subproblems use HiGHS (via `scipy.optimize.milp`) by default.
With a Gurobi license: `uv sync --extra gurobi` and pass
`--milp-backend gurobi` (or set `GASNETOPT_MILP_BACKEND=gurobi`).

## Data licenses

GasLib data (`data/gaslib/*/**.net`, `.scn`, `.cs`) is licensed CC-BY 3.0;
please cite the GasLib paper (M. Schmidt et al., *GasLib — A Library of Gas
Network Instances*, Data 2(4):40, 2017). The `.bcd`/`.state` files originate
from the TRR154 transient-data collection. The original transmission-lines
code is Copyright 2018 A. Potschka, C. Teuber (GPL-3.0-or-later).
