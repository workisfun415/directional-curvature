# Directional curvature recovery

Reproduction code for the manuscript

> **Geometry, conditioning, and limits of one-sided directional curvature recovery**
> R. Pasupuleti, Independent Researcher, Suryapet, India.
> Preprint: https://doi.org/10.21203/rs.3.rs-10592133
> Archive: https://doi.org/10.5281/zenodo.21793101

Everything reported in the manuscript — every theorem verification, every numerical
observation, both comparison tables and all figures — is produced by the two scripts
in this repository. No other code is required.

## Contents

| File | Purpose |
|---|---|
| `directional_curvature.py` | All numerical work: geometry, estimators, harmonic machinery, verification, comparison tables, figures |
| `verify_proofs.py` | Symbolic verification of the algebraic steps in the proofs |
| `requirements.txt` | Dependencies |
| `CITATION.cff` | Citation metadata |

## Requirements

Python 3.9 or later.

```
pip install -r requirements.txt
```

Dependencies are `numpy`, `matplotlib` and `sympy`. Nothing else is needed and there
is no installation step — both files are standalone scripts.

## Running

```bash
python directional_curvature.py --quick      # fast smoke test, reduced replicates
python directional_curvature.py --theorems   # theorem and observation verification
python directional_curvature.py --benchmark  # the comparison tables
python directional_curvature.py --figures    # all figures, as .pdf and .png
python directional_curvature.py --all        # everything (slowest)
python verify_proofs.py                      # symbolic proof checks; all must print True
```

`--quick` finishes in a few minutes and is the right first command. `--all` takes
considerably longer, most of it in the comparison tables, which sweep an
oracle-optimal sampling radius for every method in every cell.

## What each function verifies

| Function | Manuscript statement |
|---|---|
| `verify_hessians()` | Test-function Hessians against fourth-order central differences |
| `obs_identifiability()` | Theorem 2.1, the ten-evaluation floor |
| `obs_direction_sets()` | Rank deficiency of Fibonacci sets at m = 6 |
| `thm_cone_singular()` | Theorem 3.1, singular-value orders and constants |
| `obs_constants()` | Grid refinement of the constants, against 1/sqrt(24) |
| `obs_parity_scaling()` | Proposition 4.2 and Corollary 4.3, the parity split |
| `obs_order()` | Proposition 4.4, antipodal availability and attainable order |
| `obs_parity_separation()` | Degeneracy of the parity split on a narrow cap |
| `obs_plateau()` | Insensitivity of the estimate to the ridge parameter |
| `obs_tracking()` | That the regularised estimate tracks the third derivative |
| `obs_pilot_size()` | Minimum pilot size and the rank condition |
| `obs_R_sweep()` | Node placement: R = 1/2 at high noise, R ~ 0.3 at low |
| `obs_dopt()` | D-optimal direction selection inside a cone |
| `benchmark_main()` | Table 3, the controlled comparison |
| `benchmark_msweep()` | Table 5, redundant directions at matched budget |
| `benchmark_sensitivity()` | Base-point and cone-axis sensitivity |

## Conventions

Sym_3(R) uses the Frobenius-consistent coordinates
`(H11, H22, H33, sqrt2*H12, sqrt2*H13, sqrt2*H23)`, so the coordinate 2-norm equals
the Frobenius norm. Caps are sampled on a tensor grid in (phi, psi); *area-uniform*
means cos(phi) equispaced, *polar-uniform* means phi equispaced. The default probe
uses R = 1/2 and the evaluation point is x = (0.10, -0.05, 0.20) throughout. Noise is
independent N(0, sigma^2) per evaluation, with sigma quoted relative to |f(x)| + 1.

All random seeds are literals in the source. Within a benchmark cell every method
sees the same noise realisations. Grid-refinement studies report only digits stable
under a doubling of the grid in both coordinates.

## Reproducing the figures

`python directional_curvature.py --figures` writes, as both `.pdf` and `.png`:

- `fig0_roadmap` — organisation of the results
- `fig_cone_geometry` — admissible directions on a wide and a narrow cap
- `fig_thm_illustration` — response of the three curvature components against aperture
- `fig1_cone` — singular values and the constant
- `fig2_refine` — grid refinement of the constant
- `fig3_parity` — parity scaling
- `fig4_order` — antipodal availability and order
- `fig5_plateau` — the regularisation plateau
- `fig6_pilot` — pilot usability against aperture
- `fig_comparison` — error against admissible aperture

## A note on what this code shows

The comparison in `benchmark_main()` finds that the directional schemes studied do
**not** outperform quadratic regression on the same feasible sample points at matched
evaluation budget, in any configuration tested. That is the reported result, not a
bug. The manuscript's contribution is the geometry and the limits, not an estimator.

The manuscript also documents nine experimental artifacts that produced apparently
positive results during development and were removed. Several of the guards against
them are visible in this code: the oracle radius sweep applied to every method rather
than only the one under test, the shared noise realisations, the explicit reporting of
infeasible stencils, and the rank condition on the pilot basis.

## Citation

```bibtex
@software{pasupuleti_directional_curvature,
  author  = {Pasupuleti, Ramakrishna},
  title   = {Directional curvature recovery: reproduction code},
  year    = {2026},
  doi     = {10.5281/zenodo.21793101},
  url     = {https://github.com/workisfun415/directional-curvature}
}
```

## License

MIT. See `LICENSE`.
