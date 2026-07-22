# kerr-dirac-scattering

**Authors**

* Dr. Davide Batić — <davide.batic@ku.ac.ae>
* Dr. Denys Dutykh — <denys.dutykh@ku.ac.ae>

Mathematics Department, Khalifa University of Science and Technology,
PO Box 127788, Abu Dhabi, United Arab Emirates.

Companion code repository for the paper

> D. Batić and D. Dutykh, *Two-Channel Dollard Scattering and Partial-Wave
> Resonances for Massive Dirac Fields on Kerr*, submitted (2026).

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
make clean      # remove build litter (never touches results/)
```

The canonical outputs are checked into `results/`; after `make benchmarks`,
`git diff results/` is the reproduction check (differences should be at the
last quoted digit at most).

`make figures` reads the JSON files in `results/`, so run `make benchmarks`
(or plain `make`) first on a clean checkout.

## Repository layout

```
dirac_radial_common.py            shared numerical machinery (imported by all scripts)
schwarzschild_dirac_matching.py   Section XI.A benchmark
kerr_dirac_matching.py            Section XI.B benchmark
convergence_study.py              two-tier convergence data
massgap_reflection.py             mass-gap unimodularity certificate
flat_limit_check.py               flat (M = 0) certification
quasibound_search.py              gated quasi-bound resonance search
make_fig06.py, make_fig07.py      data figures (Matplotlib)
ku_style.py                       shared Matplotlib style
figsrc/                           TikZ sources of the diagram figures
results/                          canonical JSON/CSV outputs (checked in)
Makefile, requirements.txt        build and dependencies
```

## Script → paper map

| Script | Reproduces |
|---|---|
| `dirac_radial_common.py` | shared machinery: tortoise maps (Lambert W for Schwarzschild, Brent inversion for Kerr), radial matrix A(u, ω) (VIII.5), infinity diagonalizer and transport coefficients (Lemma X.4), Dollard phase Φ and refined phase Φ⋆ (II.10), Chandrasekhar–Page angular shooting, finite matching S = (F_out)⁻¹F_in with the conjugation identities of Props. VIII.1 and IX.1, JSON/CSV writers |
| `schwarzschild_dirac_matching.py` | Section XI.A benchmark table (unitarity defect, Wronskian, matching-point diagnostics) |
| `kerr_dirac_matching.py` | Section XI.B rotating benchmark table, angular eigenvalue λ₀₊ = 1.1609940906 |
| `convergence_study.py` | the two-tier convergence data of Fig. 6(b): u₊⁻² invariants vs u₊⁻¹ log u₊ channel-gauge phases |
| `massgap_reflection.py` | Section XI.C: mass-gap unimodularity \|\|S_HH\|−1\| ≤ 2.22×10⁻¹⁶ at gap energies, both geometries |
| `flat_limit_check.py` | Section XI.C: exact flat (M=0) certification and oblate a=0.4 flat case |
| `quasibound_search.py` | Section XI.D: gate-certified quasi-bound levels ω₁ ≈ 0.1948345 + 7.51×10⁻⁴ i and ω₂ ≈ 0.1988022 + 9.97×10⁻⁵ i, plus the Breit–Wigner fingerprint data |
| `make_fig06.py` | Fig. 6(b,c) (convergence and tolerance-floor panels) → `fig06b-convergence.pdf` |
| `make_fig07.py` | Fig. 7 (refuses to run unless the quasi-bound gate is certified) → `fig07-quasibound.pdf` |
| `figsrc/fig01…fig06a…tex` | TikZ sources of Figs. 1–5 and 6(a), plus the shared `figstyle.tex` (compiled by `make figures`) |
| `ku_style.py` | shared Matplotlib style (vector PDF, Computer Modern fonts, house palette) |

Both matching scripts accept command-line options:

```bash
python3 schwarzschild_dirac_matching.py --refined            # Φ⋆ start data
python3 kerr_dirac_matching.py --u-rights 80,120,180         # endpoint list
python3 kerr_dirac_matching.py --outdir results              # output directory
```

The `--refined` runs write the `*_refined_*` files in `results/`.

## Benchmark parameters

| | Schwarzschild | Kerr |
|---|---|---|
| M, a | 1.0, 0 | 1.0, 0.4 |
| electron mass mₑ, ω | 0.2, 0.5 | 0.2, 0.5 |
| angular data | λ = 1 | k = 0 ⇒ λ₀₊ = 1.1609940906 |
| u₋, u_match | −80, 20 | −40, 20 |
| radial rtol/atol | 1e−10 / 1e−11 | 1e−10 / 1e−11 |

## Requirements

Python ≥ 3.12 with `numpy ≥ 1.26`, `scipy ≥ 1.13`, `matplotlib ≥ 3.9`
(see `requirements.txt`); a TeX Live installation with `pdflatex`, TikZ, and
the `standalone` class for the diagram figures. No revtex installation is
needed here (the manuscript folder ships its own revtex4-2 runtime).

**Figure output location.** All figures (compiled TikZ and Matplotlib alike)
are written to `../latex/figures/`, i.e. *outside* this repository, into the
manuscript tree that is expected to sit beside it (`Makefile: FIGDIR`,
`ku_style.FIG_DIR`). The directory is created on demand; only `results/` is
versioned here.

## Output formats

* `results/{schwarzschild,kerr}_dirac_matching[_refined]_output.json` — full
  benchmark records: parameters, geometry/angular data, per-`u₊` runs with
  the complex S-matrix (`[re, im]` pairs), unitarity defect, Wronskian
  variation, matching-point variation, oriented fluxes, and the
  conjugation/reciprocity/transmission-identity defects.
* `results/*_summary.csv` — the corresponding flat tables printed in the paper.
* `results/convergence_study.json` / `.csv` — unitarity defect and
  transmission-phase drift vs u₊ ∈ {60 … 360} (reference u_ref = 480) for both
  phase conventions (`first_order`, `refined`).
* `results/massgap_reflection.json` / `.csv` — S_HH, ||S_HH|−1| and arg S_HH
  on the gap grid ω ∈ {0.05, 0.10, 0.15, 0.18}, both geometries.
* `results/flat_limit_check.json` — the two flat cases with antidiagonality,
  transmission-modulus and conjugation defects.
* `results/quasibound_search.json` — scan, certified roots, stability grid,
  Breit–Wigner overlay data, and the gate verdicts (`gate.certified`);
  `results/quasibound_scan.csv` holds the real-axis residual scan.

## Structure of the numerics

All radial integrations use `scipy.integrate.solve_ivp` (DOP853) on the
first-order 2×2 system ∂ᵤR = A(u)R with Jost start data at both ends:
horizon plane waves at u₋ and Dollard-normalized transport-corrected columns
at u₊ (decaying columns with the Coulomb power (u/ℓ∞)^α̃ in the mass gap).
The scattering matrix is the finite matching S = (F_out)⁻¹F_in of Section XI;
every run monitors unitarity, Wronskian conservation, matching-point
independence, and the conjugation identity det F_in = −conj(det F_out).
The Kerr angular eigenvalue comes from two-sided Chandrasekhar–Page shooting
with an adaptive bracket, also integrated with DOP853.

The quasi-bound search is *gated*: results are marked certified only if the
relative Evans residual is below 1e−10 (G1), the root is stable under
(u₊, rtol) refinement to within 5 % of |γₙ| (G2), and the parameter-free
Breit–Wigner overlay deviates by less than 10 % in relative L² on the
resonant core (G3). `make_fig07.py` exits without producing a figure unless
`gate.certified` is true.

## Citation

If you use this code, please cite the paper:

> D. Batić and D. Dutykh, *Two-Channel Dollard Scattering and Partial-Wave
> Resonances for Massive Dirac Fields on Kerr*, submitted (2026).

```bibtex
@article{BaticDutykh2026KerrDirac,
  author  = {Bati{\'c}, Davide and Dutykh, Denys},
  title   = {Two-Channel {D}ollard Scattering and Partial-Wave Resonances
             for Massive {D}irac Fields on {K}err},
  note    = {Submitted},
  year    = {2026},
}
```

Until the article appears, please also reference this repository as the
source of the numerical results:

```bibtex
@misc{BaticDutykh2026KerrDiracCode,
  author = {Bati{\'c}, Davide and Dutykh, Denys},
  title  = {\texttt{kerr-dirac-scattering}: companion code for
            ``Two-Channel {D}ollard Scattering and Partial-Wave Resonances
            for Massive {D}irac Fields on {K}err''},
  year   = {2026},
  note   = {Version as of the submitted manuscript},
}
```

Update `year`/`note` with the journal, volume and pages (and the archival
DOI of the code release) once the paper is published.

## License

Distributed under the **GNU Lesser General Public License, version 2.1**
(February 1999); the full text is in [`LICENSE`](LICENSE).

Copyright © 2026 Davide Batić and Denys Dutykh.
