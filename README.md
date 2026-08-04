# Carbon-Aware, Privacy-Preserving, Thermal-Active DNN Split Scheduling — Simulation Suite

Simulation code accompanying *"An Adaptive Spatio-Temporal Framework for
Carbon-Aware and Privacy-Preserving DNN Partitioning"* (Goel & Veeragandham,
Vellore Institute of Technology). This repository contains the full,
runnable simulation model used to generate every numerical result reported
in the paper: latency, carbon emissions, privacy compliance, thermal
throttle risk, the adaptive weight controller evaluation, cross-architecture
generalisation, and the SOTA-adjacent baseline comparison.

**This is a simulation study.** All per-layer power, timing, and
tensor-size figures, along with the thermal and battery model constants,
are documented modelling assumptions chosen to be physically plausible and
internally consistent — not measurements from physical hardware. Every
assumption is stated explicitly in-line as a comment/description at its
point of use.

## Contents

| File | Description |
|---|---|
| `simulation_suite.ipynb` | All four sections merged into a single runnable notebook, in dependency order. **Start here.** |
| `thermal_sim.py` | Core cost model (latency, carbon, privacy, active thermal-battery conduction) and the sustained-load thermal throttle experiment. |
| `awc_sim.py` | Adaptive Weight Controller (AWC) evaluation against a static weight preset. |
| `multi_arch_sim.py` | Cross-architecture generalisation test (ResNet-50, ViT-Base, a multimodal profile). |
| `sota_baselines_sim.py` | Comparison against two SOTA-adjacent baselines (carbon-aware-greedy, privacy-aware-static). |

The four `.py` files are standalone and can be run independently
(`multi_arch_sim.py`, `awc_sim.py`, and `sota_baselines_sim.py` each
`import thermal_sim as ts`). The notebook merges the same logic into one
shared namespace — see [Notes on the notebook merge](#notes-on-the-notebook-merge)
below if you plan to extend it.

## Requirements

- Python 3.9+
- `numpy`

No other dependencies. To run the notebook itself you'll also need
`jupyter` (or just open it in VS Code / Google Colab / GitHub's built-in
notebook viewer).

```bash
pip install numpy jupyter
```

## Usage

**Notebook (recommended):**
```bash
jupyter notebook simulation_suite.ipynb
```
Run all cells top to bottom — later sections depend on definitions from
earlier ones.

**Standalone scripts:**
```bash
python thermal_sim.py
python awc_sim.py
python multi_arch_sim.py
python sota_baselines_sim.py
```
Each prints its own results table when run directly.

## What each section does

### 1. Core simulation model (`thermal_sim.py`)
Defines the split-point cost function `J(k, r)` over four normalised,
min-max-scaled objectives — latency, carbon, privacy, and thermal risk —
combined with an operator-set weight vector. The thermal term is a
**forecast**, not a reactive threshold: a lumped thermal-capacitance model
(coupled to a state-of-charge-dependent battery self-heating term) predicts
device temperature one control interval ahead for each candidate split, so
thermal cost genuinely participates in split selection rather than only
triggering a DVFS override after the fact. Runs a 50-trial, 60-second
sustained-inference-stream experiment comparing:

| Strategy | Throttle risk | Mean latency | Carbon (norm.) |
|---|---|---|---|
| Local-only | 100% | 450.0 ms | 100% |
| Latency-first | 100% | 300.9 ms | 46.3% |
| **Proposed (active)** | **2%** | 426.6 ms | 11.3% |

### 2. Adaptive Weight Controller evaluation (`awc_sim.py`)
Rather than a fixed weight vector, the AWC adjusts weights online via a
multiplicative feedback update against operator-specified per-objective
targets. Evaluated over a 200-cycle scenario combining a carbon-intensity
spike with a concurrent thermal excursion, against the best static preset.
The AWC reduces cumulative cross-objective target violation by **72%**
relative to the static preset — not because it improves every axis, but
because it corrects the static preset's chronic, uncorrected neglect of
whichever objective isn't explicitly weighted.

### 3. Cross-architecture generalisation (`multi_arch_sim.py`)
Extends the same cost model to ResNet-50 (large CNN), ViT-Base
(Transformer), and a CLIP-style two-tower multimodal profile. Carbon
reduction generalises well (87–92% across all four architectures tested).
The MobileNetV2-specific finding of a genuine *interior-optimum* split
point does **not** generalise: it depends on transmitted-tensor size
shrinking with network depth, a property of CNN spatial downsampling that
Transformer-style architectures largely lack — for those, the optimiser
instead collapses to the shallowest privacy-feasible split.

### 4. SOTA-adjacent baseline comparison (`sota_baselines_sim.py`)
Adds two further baselines, each a re-implementation of a *design
principle* from prior work (not a reproduction of that work's own
published numbers, since neither targets this exact task):

- **Carbon-Aware Greedy (CAG)** — informed by geographic carbon-shifting
  (Wiesner et al.): route to the lowest-CI region, then minimise latency.
- **Privacy-Aware Static Split (PASS)** — informed by shallow-split /
  fixed-threshold privacy schemes (Osia et al.; Mireshghallah et al.'s
  Shredder): fix the split once, at the shallowest layer meeting a
  disentanglement threshold.

| Strategy | Latency | Carbon (norm.) | Privacy violations | Throttle risk |
|---|---|---|---|---|
| Local-only | 450.0 ms | 100% | 0% | 100% |
| Latency-first | 297.4 ms | 45.5% | 8% | 98% |
| CAG | 377.4 ms | 11.4% | 8% | 96% |
| PASS | 309.0 ms | 51.3% | 0% | 100% |
| **Proposed** | 385.6 ms | 11.7% | **0%** | **4%** |

The honest reading: the proposed method matches CAG's carbon performance
while eliminating its privacy violations, and matches PASS's privacy
guarantee while cutting carbon cost by 77% and throttle risk by 96 points.
It does not dominate every baseline on every axis (it has the highest
latency of the five) — it sits on the Pareto front, trading latency for
carbon and privacy guarantees.

## Notes on the notebook merge

`multi_arch_sim.py` defines functions with the same names as
`thermal_sim.py` (`latency_ms`, `carbon_gco2`, `edge_time_ms`,
`p_compute`, `normalise`) but a different signature — they take an extra
`profile` argument, since that script generalises the cost model across
architectures. In separate files this is fine; merged into one notebook's
shared namespace it would silently overwrite Section 1's functions and
break Section 4. The notebook resolves this by prefixing Section 3's
colliding functions with `ma_` (`ma_latency_ms`, `ma_carbon_gco2`, etc.).
If you edit Section 3, keep this prefix convention or reintroduce the
collision.

## Reproducibility

Every experiment uses a fixed `numpy` random seed (visible at the top of
each section), so re-running should reproduce the tables above exactly.
Random seeds differ slightly between sections (`thermal_sim.py` and
`sota_baselines_sim.py` run overlapping but not identical trial
protocols), which accounts for minor throttle-rate differences (e.g. 2%
vs. 4%) between Section 1 and Section 4 for the same "proposed" strategy —
both are real, independently seeded 50-trial runs, not a discrepancy.

## Citation

If you use this code, please cite:

```bibtex
@article{goel2026adaptive,
  title   = {An Adaptive Spatio-Temporal Framework for Carbon-Aware and
             Privacy-Preserving DNN Partitioning},
  author  = {Goel, Saanvi and Veeragandham, Syamasudha},
  journal = {IEEE Access},
  year    = {2026}
}
```

## License

Add a license (e.g. MIT) before making this repository public if you
intend for others to reuse the code.
