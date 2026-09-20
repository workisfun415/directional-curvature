# Paper A validation package

Reproducible numerical validation for the manuscript
*Hessian Recovery under One-Sided Conical Access: Conditioning, Parity, Noise, and Radial Aliasing*
(R. Pasupuleti).

Every number and figure in Section 10 of the manuscript is produced by `run_all.py` from
`config/validation.yaml`. Nothing is entered by hand.

## Install and run (from the repository root)

```bash
python -m pip install -e ".[dev]"   # installs dircurv and the validation dependencies
pytest                              # fast tests: exact constants, symbolic coefficients,
                                    # A/B equivalence, noise identity, exact alias witness
pytest -m slow                      # Monte Carlo, bootstrap, span cap, parity and alias slopes
python run_design_search.py --batch 10   # repeat until it prints "complete"
python run_all.py                   # regenerates results/*.json, results/metadata.json, figures/*.pdf
python run_all.py --quick           # smoke test with reduced sizes (not for the paper)
```

`results/design_search.json` is already included, so `run_design_search.py` only needs to be
repeated if you want to regenerate the design search from scratch. Run `run_all.py` from a git
checkout so that `results/metadata.json` records the commit hash.

## Structure

```
config/validation.yaml        provenance (frozen record of exploratory runs) + regenerated settings
paperA_validation/
  common.py                   definitions only: cap parametrisation, Frobenius coordinates,
                              stencils, designs, test functions, seeds
  impl_a/factorized.py        Implementation A (stencils): (c, b~, T') coordinates, D_theta
                              factorisation, shear map P, Schur complement, exact covariance
  impl_a/regression.py        Implementation A (quadratic regression): factorisation
                              X = X~(theta) D_theta and the alias-matrix moment formula
  impl_b/physical.py          Implementation B (stencils): physical Frobenius basis,
                              evaluation-level noise simulation
  impl_b/regression_physical.py  Implementation B (quadratic regression): physical coordinates only
  impl_b/symbolic_bias.py     exact stencil constants and sector-bias coefficients (SymPy)
  impl_b/symbolic_alias.py    exact non-product witness for Gamma_01, Gamma_02, Gamma_12
  experiments/constants.py    conditioning constants, finite-theta ladders, design search
  experiments/bias.py         sector-bias coefficients, central-symmetry (parity) slopes
  experiments/accuracy.py     sector variance, span law (Monte Carlo + bootstrap), span cap,
                              stencil-order crossover
  experiments/aliasing.py     alias paths (product vs non-product designs), stencil vs regression
  figures.py                  Figures 3-7 from results/ only
docs/Section7_radial_aliasing.md   text of the radial-aliasing theorem and its proof
run_all.py                    regenerates everything; records seeds and environment
run_design_search.py          checkpointed design search (numerical best found)
tests/test_validation.py      stencil, conditioning and noise tests
tests/test_aliasing.py        quadratic-regression and aliasing tests
```

## Validation layers

| Layer | What is checked | Criterion |
|---|---|---|
| Symbolic | stencil constants; sector-bias coefficients for n = 3, 4 and orders 3, 4 | exact equality |
| Deterministic | uniform-cap limit 4/((d+2)^2(d+4)) (exact rationals); 5/72; optimum 2/(d+2)^2 (1/4 for d = 1) | exact / 1e-9 |
| Finite theta | lambda_min^F / theta^4 on a theta ladder, fitted a + b theta^2 | intercept rel. err. < 1e-3 |
| A vs B | eigenvalues, fitted Hessians (stencils and regression), b~ = sqrt(2) b | < 1e-9 |
| Noise identity | fit of all-ones data is the identity | < 1e-9 |
| Sector variance | Monte Carlo / exact formula | within 5 % |
| Span law | slopes of fitted h* and E* with bootstrap 95 % CI | CI contains prediction, or error <= 0.01 |
| Span cap | capped tangential error slope below theta_c | -2 +/- 0.1 |
| Parity | tangential bias correction slope: pentagon ~1, centrally symmetric ~2 | +/- 0.15 |
| Aliasing | exact non-product witness nonzero, product exactly zero; divergent blocks vanish for 2, 3, 4 common radii; slopes -1, -2, -1 for non-product designs | exact / < 1e-10 / +/- 0.05 |

## Conventions

* Primary Monte Carlo runs use pure-order test functions, so the leading-order span model is exact
  in h. A robustness run adds a higher-order term; its deviation is the remainder the theory omits.
* Slopes are compared with the finite-angle leading-order prediction; asymptotic exponents are
  reported alongside.
* Runs with the earlier independent-noise convention are included for provenance only.
* Finite-design optima (m = 6, 10) are numerical best-found results.
* The alias coefficient depends on the design; the values reported in the manuscript are those of
  the seeded validation design in `results/aliasing.json`.

## Citation

Please cite the Zenodo archive of the release you use (see `CITATION.cff`).
