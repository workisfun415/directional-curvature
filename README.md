# Paper A validation package

Reproducible numerical validation for the manuscript
*One-Sided Hessian Recovery under Restricted Conical Access: Conditioning, Noise, Bias, and Span Selection*
(R. Pasupuleti).

Every number and figure in Section 8 of the manuscript is produced by `run_all.py` from
`config/validation.yaml`. Nothing is entered by hand.

## Where this goes

Intended location: `dircurv/validation/paperA/` in the existing `dircurv` repository, released as a
**new version of concept DOI 10.5281/zenodo.21793101** (the DOI cited in the manuscript and preprint).

If you create a *new* GitHub repository instead, Zenodo will mint a **new** concept DOI when you
enable the integration. To keep one DOI lineage, either (a) put this directory inside `dircurv`, or
(b) upload the release manually on Zenodo as a *new version* of record 21793101.

## Install and run

```bash
python -m pip install -r requirements.txt
pytest                 # fast: exact constants, symbolic bias, A/B equivalence, noise identity (~2 s)
pytest -m slow         # Monte Carlo, bootstrap, span cap, parity (~10 s)
python run_design_search.py --batch 10   # repeat until it prints 'complete' (checkpointed; ~11 min total)
python run_all.py      # full regeneration -> results/*.json, results/metadata.json, figures/*.pdf
python run_all.py --quick   # smoke test with reduced sizes (not for the paper)
```

Commit the repository **before** the final `run_all.py`, so `metadata.json` records the commit hash.

## Structure

```
config/validation.yaml        provenance (frozen record of exploratory runs) + regenerated settings
paperA_validation/
  common.py                   definitions only: cap parametrisation, Frobenius coordinates,
                              stencils, designs, test functions, seeds
  impl_a/factorized.py        Implementation A: (c, b~, T') coordinates, D_theta factorisation,
                              shear map P, Schur complement, exact covariance
  impl_b/physical.py          Implementation B: direct physical Frobenius basis, evaluation-level
                              noise simulation (shared f(x0) or historical independent)
  impl_b/symbolic_bias.py     Implementation B, symbolic: exact stencil constants and Lemma 6.2
                              coefficients derived by series expansion (SymPy)
  experiments/constants.py    Cor. 4.4, 4.5, Prop. 4.6: exact planar limits, finite-theta ladders,
                              design search (labelled "best found")
  experiments/bias.py         Lemma 6.2 numerical check, Remark 6.3 parity slopes
  experiments/accuracy.py     Theorem 5.2 variance check, Theorem 7.1 Monte Carlo + bootstrap,
                              Cor. 7.2 span cap, stencil-order crossover (exact MSE)
  figures.py                  Figures 3-6 from results/ only
run_all.py                    regenerates everything; records seeds and environment
run_design_search.py          checkpointed m=6, m=10 design search (best found), merged by run_all.py
tests/test_validation.py      fast and slow (@pytest.mark.slow) tests
```

## Validation layers

| Layer | What is checked | Criterion |
|---|---|---|
| Symbolic | stencil beta, s_w^2, s_w'^2; Lemma 6.2 coefficients for n=3,4, nu=3,4 | exact equality (SymPy) |
| Deterministic | 4/((d+2)^2(d+4)) uniform planar limit (exact rationals, d=1,2,3); 5/72; optimum 2/(d+2)^2 (1/4 for d=1) | exact / 1e-9 |
| Finite theta | lambda_min^F/theta^4 on a theta ladder, fitted a + b theta^2 | intercept rel. err. < 1e-3 |
| A vs B | physical-Frobenius eigenvalues; fitted H; b~ = sqrt(2) b map | < 1e-9 |
| Noise identity | fit of the all-ones data is I_n | < 1e-9 |
| Theorem 5.2 | Monte Carlo sector variance (B, evaluation-level noise) / exact formula (A) | within 5 % |
| Theorem 7.1 | slopes of fitted h* and E* vs theta, bootstrap 95 % CI | CI contains finite-theta prediction, or |est - pred| <= 0.01 |
| Cor. 7.2 | slope of capped tangential error below theta_c | -2 +/- 0.1 |
| Remark 6.3 | tangential bias correction slope: pentagon ~1, centrally symmetric ~2 | +/- 0.15 |

## Important conventions

* **Primary Monte Carlo runs use pure-order test functions** (cubic for the 3-point stencil,
  quartic for the 4-point stencil), so that the leading-order model of Theorem 7.1 is exact in h.
  A separate robustness run adds a quartic term to the 3-point case; its measurable deviation from
  the leading-order slopes is the O(h^{p+1}) remainder that Theorem 7.1 omits, and is reported as such.
* **Slopes are compared with the finite-theta leading-order prediction**, computed from the exact
  covariance and exact bias coefficients at each theta. The asymptotic exponents -k/(p+2) and
  -kp/(p+2) are reported alongside; the small differences are the O(theta) finite-angle corrections.
* **Historical independent-noise runs** are reproduced for provenance only; the manuscript's model
  is the shared-f(x0) model (Assumption 5.1).
* **Design-search optima (m = 6, 10) are "numerical best found"**, never "optimal".
* The historical uniform-cap value 0.04188 was quadrature-limited; the primary 1/24 test is the
  exact planar limit.

## Section 7: radial aliasing (quadratic regression)

* `paperA_validation/impl_a/regression.py` - Implementation A: transformed coordinates with the explicit
  factorisation X = X~(theta) D_theta and the alias-moment formula (Schur complement of design moments).
* `paperA_validation/impl_b/regression_physical.py` - Implementation B: physical coordinates only.
* `paperA_validation/impl_b/symbolic_alias.py` - exact non-product witness for Gamma_01, Gamma_02, Gamma_12
  (rational design, exact arithmetic) and the exact zero for the matching product design.
* `paperA_validation/experiments/aliasing.py` - alias paths (slopes and limit coefficients) for product
  designs with 2, 3 and 4 common radii and a non-product design; stencil vs regression on identical points.
* `docs/Section7_radial_aliasing.md` - manuscript text of Theorem 7.1 and its proof.
* Results: `results/aliasing.json`, `results/stencil_vs_regression.json`, `figures/fig7_aliasing.pdf`.

## Not generated here

Figure 1 (geometry) and Figure 2 (sector schematic) are conceptual diagrams with no data.
