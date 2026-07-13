# Applying the POC / CIAP / penalty-ADM pipeline to transient gas networks

This document records the analysis behind the gas-network extension in
`src/gasnetopt/gas/`: the structure of the TRR154 transient data, the model
mapping from the electric transmission-lines example to gas pipelines, the
discretization and formulation choices, and the dependency assessment for the
whole code base.

## 1. Background

The repository implements two papers on mixed-integer optimal control (MIOC)
of networked hyperbolic PDEs:

1. **Göttlich, Potschka, Teuber** (Comput. Optim. Appl., 2019): partial outer
   convexification (POC) for switching the topology of electric transmission
   lines governed by the telegraph equations; first-order upwind
   discretization in characteristic variables; sum-up rounding (SUR) with
   reoptimization. → `gasnetopt/translines/`.
2. **Göttlich, Hante, Potschka, Schewe** (Math. Program. 188, 2021): penalty
   alternating direction method (ADM) for MIOC with combinatorial (dwell-time)
   constraints; CIAP variants (SUR / MILP / dwell-time MILP "COMB") and the
   tCOMB projection MILP. → `gasnetopt/ciap.py`, `gasnetopt/adm.py`.

The algorithms only interact with a model through the `OCModel` interface
(`create_NLP_solver`, `extract`, `overwrite`, `evaluate_objective`, time grid
`t`, mode encoding `r`). The gas extension therefore consists of a new model
class (`gas/ocmodel.py:GasOCModel`) — the algorithm layer is unchanged.

## 2. TRR154 transient data (https://www.trr154.fau.de/transient-data/)

The data extends static [GasLib](https://gaslib.zib.de) instances with
transient information. Three XML formats are involved:

### `.net` — GasLib network topology (namespace `http://gaslib.zib.de/Gas`)

- `framework:nodes`: `source`, `sink`, `innode` elements with `height`,
  `pressureMin/Max` [bar], and for sources/sinks `flowMin/Max`
  [1000 m³/h at norm conditions] plus gas properties (`normDensity`,
  `calorificValue`, pseudocritical values, …).
- `framework:connections`: `pipe` (`length` [km], `diameter` [mm],
  `roughness` [mm], flow bounds), `valve` (`pressureDifferentialMax`),
  `compressorStation` (`pressureInMin`, `pressureOutMax`, flow bounds),
  and further elements (resistor, controlValve, …) not present in GasLib-11.
- Volumetric flow bounds are converted to mass flow with the norm density:
  q [kg/s] = Q [1000 m³/h] · 1000 · ρ₀ / 3600.

### `.bcd` — transient boundary conditions (TRR154, schema `bcd.xsd`)

- `metadata`: referenced network, `timeInterval` (0 … 86400 s for the sinus
  scenarios), and notably **`speedOfSound value="340"`** — the data is
  generated for the *semilinear isothermal* model with constant speed of
  sound, which fixes the model choice (ISO2 below).
- `nodes/node/nodedata`: time series of `pressure` (entries; constant 53 bar
  for GasLib-11) and `massflow` (exits; negative = withdrawal, 60 s
  resolution, smooth sinusoidal profiles).

### `.state` — network state at one time point (schema `state-1.xsd`)

- nodal `(massflow, pressure)` pairs,
- per-pipe spatial profiles `(space, massflow, pressure)` — a stationary
  initial condition for the PDE,
- compressor stations: `(massflow, pressureDifference)` — confirming that the
  intended compressor model is an additive pressure increase Δp.

**Data & CI policy (settled 2026-07-12):** GasLib (CC-BY 3.0, redistribution
would be permitted for the `.net`) and TRR154 `.bcd` (unclear license) stay
untracked under `data/gaslib/`; the project-own translines demand profiles
are tracked. CI covers the gas code path via a synthetic 3-node fixture
(`tests/fixtures/tiny.net`/`tiny.bcd`, hand-written in the GasLib/TRR154
formats); the GasLib-11/-40 tests skip when the data is absent.

**Data quirk:** on the TRR154 page the GasLib-11 `.state` link is broken (the
anchor labelled `GasLib-11-sinus_5000_60-initial.state` points to the `.bcd`
file, and the expected shared-file ID returns 404). The prototype therefore
starts from a flat initial state (uniform pressure, zero flow — an exact
steady state of the PDE); `gaslib_io.parse_state` is implemented and verified
against the GasLib-40 `.state`, and `build_gas_nlp(initial_state=...)`
accepts such profiles.

Files used here (downloaded 2026-06-13):
- `data/gaslib/GasLib-11/`: `.net` from gaslib.zib.de (`GasLib-11-v1-20211130`),
  `.bcd` from TRR154 (shared-files/6818).
- `data/gaslib/GasLib-40/`: `.net` (`GasLib-40-v1-20211130`), `.bcd`
  (shared-files/6820), `.state` (shared-files/6823).
- `data/gaslib/schemas/`: `bcd.xsd`, `state-1.xsd`.
- Format report: `docs/references/TRR154-transient-data-format-Report.pdf`.

GasLib data is CC-BY 3.0; cite Pfetsch et al., "Validation of Nominations in
Gas Network Optimization", and the GasLib paper (Schmidt et al., Data 2017).

## 3. Model mapping: telegraph equations → isothermal Euler (ISO2)

Following the model hierarchy of Domschke, Hiller, Lang, Tischendorf
("Modellierung von Gasnetzwerken: Eine Übersicht", TRR154 preprint 2717,
2017), the semilinear isothermal model **ISO2** is

ρ_t + m_x = 0,  m_t + p_x = −λ/(2D)·m|m|/ρ,  p = c²ρ,

with mass flux density m = ρv and constant speed of sound c (= 340 m/s per
the `.bcd` metadata). Its characteristic variables ξ± = m ± p/c satisfy

∂_t ξ± ± c ∂_x ξ± = −λ/(2D)·m|m|/ρ,

i.e. a 2×2 hyperbolic system with speeds ±c and a nonlinear source — the
exact structural analogue of the telegraph system ξ± of the transmission
lines example (there: linear damping −Bξ). The same first-order upwind
scheme on a per-pipe space-time grid is reused.

| | transmission lines | gas (ISO2) |
|---|---|---|
| states per edge | ξ± (characteristics) | ξ± = m ± p/c |
| transport speeds | ±1/√(LC) | ±c |
| source term | linear damping | nonlinear friction −λ/(2D)·m|m|/ρ |
| node coupling | equal-distribution matrices D_p, D_m per configuration | pressure continuity + mass conservation |
| discrete decisions | line on/off configurations | valve open/closed, compressor on/bypass |
| continuous controls | producer inflows u | entry pressures u, compressor Δp |
| objective | consumer demand tracking | sink demand tracking (+ compression cost) |

### Nondimensionalization

p_ref = 50 bar, ρ_ref = p_ref/c², m_ref = ρ_ref·c, L_ref = max pipe length,
T_ref = L_ref/c (≈ 162 s for GasLib-11), Q_ref = m_ref·A_ref. In scaled
variables the transport speed is 1 and the friction coefficient becomes
κ = λL_ref/(2D) (≈ 750 for GasLib-11 — large, see below). Typical scaled
magnitudes: ρ' ≈ 1, m' ≈ 0.01…0.08, so the NLP is well-conditioned for IPOPT.

### Stiffness: IMEX treatment of friction

Unlike the telegraph damping, gas friction is stiff: the relaxation rate
∂(friction)/∂m = λ|v|/D is ≈ 0.5 s⁻¹ at v ≈ 18 m/s, so a fully explicit
source (as in the translines code) is unstable for any practical Δt (60 s
data resolution). The gas discretization therefore evaluates the friction
term at the **new** time level (advection explicit under CFL Δt ≤ Δx/c,
friction implicit). In a direct-transcription NLP this costs nothing — the
constraint simply couples ξ(t+1) nonlinearly — and any forward simulation
through `resolve_with_fixed_controls` inherits the stability.

The nonsmooth |m| is smoothed as √(m² + ε²), ε = 10⁻³ (scaled).

### Network coupling

Instead of configuration-dependent distribution matrices, the gas model uses
the standard junction formulation: per node a pressure variable ρ_v(t) (for
sources, the control u itself), per pipe end a boundary mass flux variable.
Per pipe and time step:

- ghost values for the incoming characteristics from the node state:
  ξ₊(0) = m_L + ρ_v(left), ξ₋(L) = m_R − ρ_v(right);
- the outgoing characteristics are matched to the node state:
  ξ₋(first cell) = m_L − ρ_v(left), ξ₊(last cell) = m_R + ρ_v(right);
- per node: mass conservation Σ A_e·m (signed) + valve/compressor flows
  = −supply (sources) / = delivery (sinks) / = 0 (innodes).

### Switching via POC + big-M

All 2^s on/off combinations of the s switchable elements (valves +
compressor stations; s = 3 ⇒ 8 configurations for GasLib-11) form the
configuration set with convex multipliers α(t) (SOS1). The relaxed state of
switch j is w_j(t) = Σ_c α_c(t)·r_{c,j} with the binary encoding r — the same
r used by the L1 penalty and the tCOMB projection of the penalty ADM.
Element models:

- valve (a,b):   |q_V| ≤ w·q_max,  |p_a − p_b| ≤ (1−w)·M, with
  M = min(pressureDifferentialMax, global pressure span);
- compressor (a,b):  p_b = p_a + Δp,  0 ≤ Δp ≤ w·Δp_max,  0 ≤ q_C ≤ q_max
  ("off" = bypass with Δp = 0).

For binary w these constraints are exact; for relaxed w they are a valid
relaxation, so POC → CIAP → reoptimization and the penalty ADM apply
verbatim. Unlike the translines case the PDE itself is configuration-
independent; switching acts through these algebraic coupling constraints.

### Dwell-time semantics of tCOMB: per-configuration, by design

*(investigated 2026-07-11, during the PR #1 review)*

`adm.TComb` (and the `'COMB'` CIAP strategy) applies the min-up constraints
to the **whole network configuration** — the 2^s modes with the SOS1 row —
not to the s individual devices. `--tau-min 900` in `run_gaslib11.py`
therefore means: after *any* configuration change, no further change for
900 s. This is stricter than per-device dwell (per-config feasible ⇒
per-device feasible, not conversely) and was checked against the literature:

1. **Göttlich, Hante, Potschka, Schewe** (Math. Program. 188, 2021): the
   theory covers any combinatorial set Γ with the uniform finiteness
   property. Example 1 (eq. 2) happens to state dwell per component v_i,
   but the authors' own experiment code (`pocadm.py`, Potschka) built
   tCOMB over the nalpha *configurations* — the published transmission-lines
   results use per-configuration dwell. Our port keeps that.
2. **Zeile, Sager** (Math. Program. 2021, "MIOC under minimum dwell time
   constraints"): in the CIA methodology, MDT constraints are imposed on
   "the active integer control", which after outer convexification is the
   mode/configuration.
3. **Amin, Hante, Bayen** (IEEE TAC 2012, "Exponential stability of switched
   linear hyperbolic IBVPs"): the dwell-time bound that prevents
   destabilizing chattering in hyperbolic networks is a property of the
   switching signal = the active configuration of the whole system. Every
   configuration change re-excites pressure waves, whichever device caused
   it.
4. **Hennings, Anderson, Hoppmann-Baum, Turner, Koch** (Optim. Eng. 2021,
   "Controlling transient gas flow in real-world pipeline intersection
   areas", ZIB/OGE) and **Hoppmann-Baum et al.** (Optim. Eng. 2021): real
   dispatch switches the joint "operation mode" of a network station (all
   valve/compressor states combined, up to 2000 feasible modes) and imposes
   mode-pair transition times as per-mode minimum active durations —
   per-configuration dwell. Per-device concerns enter only as compressor
   *start-up penalties* in the objective, never as per-device dwell
   constraints ("the mode changes of valves and compressor stations are not
   tracked separately").

**Decision:** keep per-configuration dwell. If a refinement is ever needed,
the realistic one is mode-pair-dependent transition times θ(A,B) (a
compressor start needs longer than a valve-only change) plus per-unit
start-up penalties in the tCOMB objective — not per-device dwell.

### Objective, boundary data, initial condition

- Sink delivery (a free expression per sink node) tracks the `.bcd` demand in
  scaled least squares, exactly as the translines demand tracking.
- Entry pressures are continuous controls within the `.net` bounds
  ([40, 70] bar); the `.bcd` entry pressures provide the initial guess.
  Source supply is bounded by the `.net` flow bounds.
- A small compression cost γ·Σ∫ Δp·q dt regularizes compressor usage.
- Initial condition: flat steady state (no `.state` for GasLib-11, see §2),
  or interpolated `.state` profiles via `initial_state=`.

## 4. Validation (examples/run_gaslib11.py, tests/test_gas.py)

On GasLib-11 with the sinus `.bcd`, 2 h horizon, Δt = 60 s, 2 cells/pipe
(9032 variables, 7800 constraints):

- POC relaxation: IPOPT `Solve_Succeeded`, objective ≈ 2.2·10⁻⁵;
- SUR + reoptimization: binary multipliers, objective same order;
- penalty ADM (τ_min = 900 s, HiGHS MILP backend): converges in 2 penalty
  steps to a dwell-time-feasible binary schedule (valve open, compressors
  bypassed) with zero delivery error;
- pressures stay within the `.net` bounds; equality constraints (mass
  balances, dynamics) satisfied to < 10⁻⁶.

For this easy scenario the network can serve the demand in a constant
configuration — the discrete structure becomes binding for longer horizons
(24 h sinusoidal swing), tighter pressure/flow bounds, or supply outage
scenarios, which is the natural next experiment.

## 5. Dependency assessment

| dependency | verdict |
|---|---|
| **CasADi + IPOPT** | Keep. Sparse AD + interior point is the right tool for discretize-then-optimize PDE OCPs; pip wheels bundle IPOPT/MUMPS. Pyomo/GEKKO/AMPL offer no advantage here; a structure-exploiting SQP (e.g. blockSQP) could be revisited for speed later. |
| **gurobipy** | Was a hard dependency — now optional. The three MILPs (CIAP-MIP, COMB, tCOMB) are small; `gasnetopt/milp.py` solves them by default with **HiGHS** via `scipy.optimize.milp` (no extra dependency). `pip install gasnetopt[gurobi]` or `GASNETOPT_MILP_BACKEND=gurobi` re-enables Gurobi (warm starts supported there). |
| **matplotlib2tikz** | Removed (deprecated, successor archived). Figures are saved as PNG/PDF. |
| **shelve** | Replaced by portable pickle files (`gasnetopt/results_io.py`); the old `.bak/.dat/.dir` dbm artifacts were removed. |
| **numpy / scipy / matplotlib** | Standard; scipy ≥ 1.9 required for `milp`. |

## 6. Known limitations / next steps

- **Compressor model** is the idealized Δp booster suggested by the `.state`
  format (`pressureDifference`); no characteristic diagrams (Section 7.4.7 of
  the Domschke et al. overview) or fuel gas consumption.
- **Sink flow bounds** from the `.net` file are not imposed (delivery is the
  tracked quantity, mirroring the translines objective).
- **Gravity term** −gρ sin(α) is omitted (all GasLib-11 heights are 0);
  heights are parsed, so adding it is a one-line source-term change.
- First-order upwind: dissipative; fine for the smooth sinus scenarios, but a
  second-order or implicit box scheme would allow coarser grids / longer
  horizons (the 24 h horizon with Δt = 60 s gives nt = 1441, i.e. ≈ 10⁵ NLP
  variables — still tractable but slow inside the ADM loop).
- GasLib-40 (.net/.bcd/.state all available) is parsed and ready as the next
  test case: 6 compressor stations ⇒ 64 configurations; the tCOMB MILP and
  the configuration enumeration still work, but a sparse switch encoding
  (per-switch SOS1 blocks) would scale better than full enumeration for
  networks with many switches.
- The GasLib-11 initial `.state` should be requested from the TRR154
  maintainers (broken link), or generated by solving the stationary model
  (ISO4) — `parse_state` and `initial_state=` are already in place.
