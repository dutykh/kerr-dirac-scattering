# kerr-dirac-scattering

**Authors:** Dr. Davide Batić and Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)

Companion code repository for the paper

> D. Batić and D. Dutykh, *Two-channel Dollard scattering and partial-wave
> resonances for massive Dirac fields on Kerr*, submitted (2026).

Every numerical statement and every figure of the paper is reproducible from
this repository: the benchmark tables of Section XI, the machine-precision
structure certificates, the gate-certified quasi-bound computation, and all
seven manuscript figures (including the TikZ diagram sources).

## Quickstart

```bash
python3 -m pip install -r requirements.txt
make            # benchmarks + all figures (~15 min on a laptop)
make benchmarks # regenerate results/ only          (~12 min)
make figures    # recompile/regenerate the figures  (~1 min)
make test       # fast smoke test of the pipeline   (<1 min)
```

The canonical outputs are checked into `results/`; after `make benchmarks`,
`git diff results/` is the reproduction check (differences should be at the
last quoted digit at most).

## Script → paper map

| Script | Reproduces |
|---|---|
| `dirac_radial_common.py` | shared machinery: tortoise maps, radial matrix (VIII.3), infinity transport/Dollard phase (X.1), angular shooting, S-matrix matching (XI.4), diagnostics |
| `schwarzschild_dirac_matching.py` | Section XI.A benchmark table (unitarity defect, Wronskian, matching-point diagnostics) |
| `kerr_dirac_matching.py` | Section XI.B rotating benchmark table, angular eigenvalue λ₀₊(0.5) = 1.1609940906 |
| `convergence_study.py` | the two-tier convergence data of Fig. 6(b): u₊⁻² invariants vs u₊⁻¹ log u₊ channel-gauge phases |
| `massgap_reflection.py` | Section XI.C: mass-gap unimodularity \|\|S_HH\|−1\| ≤ 2.3×10⁻¹⁶ at gap energies, both geometries |
| `flat_limit_check.py` | Section XI.C: exact flat (M=0) certification and oblate a=0.4 flat case |
| `quasibound_search.py` | Section XI.D: gate-certified quasi-bound levels ω₁, ω₂ (XI.25) and the Breit–Wigner fingerprint data |
| `make_fig06.py` | Fig. 6(b,c) (convergence and tolerance-floor panels) |
| `make_fig07.py` | Fig. 7 (refuses to run unless the quasi-bound gate is certified) |
| `figsrc/fig01…fig06a…tex` | TikZ sources of Figs. 1–5 and 6(a) (compiled by `make figures`) |
| `ku_style.py` | shared Matplotlib style (vector PDF, TrueType fonts, house palette) |

## Requirements

Python ≥ 3.12 with `numpy`, `scipy`, `matplotlib` (see `requirements.txt`);
a TeX Live installation with `pdflatex`, TikZ, and the `standalone` class for
the diagram figures. No revtex installation is needed here (the manuscript
folder ships its own revtex4-2 runtime).

## Output formats

* `results/*_output.json` — full benchmark records: parameters, per-`u₊` runs
  with the complex S-matrix (`[re, im]` pairs), unitarity defect, Wronskian
  variation, matching-point variation, conjugation/reciprocity defects.
* `results/*_summary.csv` — the corresponding flat tables printed in the paper.
* `results/quasibound_search.json` — scan, certified roots, stability grid,
  Breit–Wigner overlay data, and the gate verdicts (`gate.certified`).

## Structure of the numerics

All radial integrations use `scipy.integrate.solve_ivp` (DOP853) on the
first-order 2×2 system ∂ᵤR = A(u)R with Jost start data at both ends:
horizon plane waves at u₋ and Dollard-normalized transport-corrected columns
at u₊ (decaying columns with the Coulomb power (u/ℓ∞)^α̃ in the mass gap).
The scattering matrix is the finite matching S = (F_out)⁻¹F_in of eq. (XI.4);
every run monitors unitarity, Wronskian conservation, matching-point
independence, and the conjugation identity det F_in = −conj(det F_out).

## Citation

If you use this code, please cite the paper above.

## License

To be chosen by the authors before publication.
