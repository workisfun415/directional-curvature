# Section 7. Quadratic regression under conical access: radial aliasing

*Manuscript-ready text for the merged paper (step 1, frozen pending human review).
Numbers quoted in 7.5 are those produced by `run_all.py` (results/aliasing.json).*

The distinction analysed here is between **sampling designs**, not estimator classes. A *product design* uses the same set of radii on every admissible ray; a *non-product design* allows the radii to depend on the ray. The directional stencils of Sections 5–6 are product designs; quadratic regression can be applied to product or non-product sample sets.

## 7.1 Model and alias matrix

Let $d=n-1$ and let the sample points be
$$x_i=x_0+h\,r_i\,u(v_i,\theta),\qquad u(v,\theta)=(\theta v,\ u_n),\qquad u_n=\sqrt{1-\theta^2|v|^2},$$
with $r_i>0$, $v_i\in B^d$ and weights $w_i>0$; the base point $x_0$ ($r=0$) may be included. The quadratic model $f_0+g^{\mathsf T}y+\tfrac12y^{\mathsf T}Hy$, $y=x-x_0$, is written with $g=(g_t,g_n)$ and $H$ in the coordinates $(c,\tilde b,t')$ of Section 2. By Proposition 2.1,
$$f_0+g^{\mathsf T}y+\tfrac12y^{\mathsf T}Hy
= f_0 + hr\left(g_nu_n+\theta\,g_t^{\mathsf T}v\right)+\tfrac12h^2r^2\left(c+\sqrt2\,\theta\,u_n\tilde b^{\mathsf T}v+\theta^2\psi(v)^{\mathsf T}t'\right),$$
so the design matrix factors exactly as
$$X_\theta=\widetilde X(\theta)\,D_\theta,\qquad D_\theta=\operatorname{diag}\left(1,\,1,\,\theta I_d,\,1,\,\theta I_d,\,\theta^2I_{d(d+1)/2}\right),\qquad \widetilde X(\theta)=\widetilde X_0+O(\theta^2),$$
where the rows of $\widetilde X_0$ are the scaled features
$$1,\quad hr,\quad hr\,v,\quad \tfrac12h^2r^2,\quad \tfrac{\sqrt2}{2}h^2r^2\,v,\quad \tfrac12h^2r^2\,\psi(v),$$
evaluated at $\theta=0$. We assume $\widetilde X_0^{\mathsf T}W\widetilde X_0\succ0$ (identifiability of the scaled design).

**Cubic sectors.** For $f\in C^4$, the omitted cubic term at a sample point is $\frac{h^3r^3}{6}D^3f(x_0)[u,u,u]$. Expanding in $\theta$ and grouping by the number $j$ of tangential indices,
$$D^3f[u,u,u]=\sum_{j=0}^{3}\theta^j P_j(v)+\sum_{j=0}^{3}O(\theta^{j+2}),$$
$$P_0=f_{nnn},\quad P_1=3f_{\alpha nn}v_\alpha,\quad P_2=3f_{\alpha\beta n}v_\alpha v_\beta,\quad P_3=f_{\alpha\beta\gamma}v_\alpha v_\beta v_\gamma ,$$
where the remainders come only from $u_n^{3-j}=1-\frac{3-j}{2}\theta^2|v|^2+O(\theta^4)$, so they are two powers of $\theta$ below the leading term. $P_j$ is a polynomial of degree $j$ in $v$.

**Alias matrix.** The bias of the weighted least-squares coefficients caused by an omitted term with data vector $d$ is $(X^{\mathsf T}WX)^{-1}X^{\mathsf T}Wd$; the matrix mapping omitted terms to fitted coefficients is the classical alias matrix of response-surface design (Box and Draper). We define $\Gamma_{jk}(\xi)$ as the block of the alias matrix of the **scaled design at $\theta=0$** that maps cubic sector $j$ (data $r^3P_j/6$) to Hessian sector $k$ ($k=0$: $c$; $k=1$: $\tilde b$; $k=2$: $t'$). Eliminating the intercept, linear, and lower-order Hessian features $F$ by a Schur complement, the block mapping into the tangential sector $Q=\tfrac12r^2\psi(v)$ is
$$\Gamma_{j2}=S^{-1}C_j,\qquad S=M_{QQ}-M_{FQ}^{\mathsf T}M_{FF}^{-1}M_{FQ},\qquad C_j=M_{Qd_j}-M_{FQ}^{\mathsf T}M_{FF}^{-1}M_{Fd_j},$$
where $M_{AB}=\sum_i w_iA(r_i,v_i)B(r_i,v_i)^{\mathsf T}$ are weighted moments of the design (Section 7.4).

## 7.2 Theorem

**Theorem 7.1 (Conical alias scaling and product-design cancellation).** *Assume $f\in C^4$ near $x_0$ and $\widetilde X_0^{\mathsf T}W\widetilde X_0\succ0$. Then, as $\theta\to0$, the bias of the fitted Hessian sectors satisfies*
$$\operatorname{Bias}_k=h\sum_{j=0}^{3}\theta^{\,j-k}\,\Gamma_{jk}(\xi)\,P_j\ +\ O\!\left(h\sum_{j}\theta^{\,j-k+2}\right)\ +\ O(h^2).$$
*(i) The potentially divergent paths are exactly those with $j<k$: $\theta^{-1}$ for $(j,k)=(0,1),(1,2)$ and $\theta^{-2}$ for $(0,2)$.*

*(ii) If the design is a product design, $\xi=\xi_r\otimes\xi_v$ (the same radial set and radial weights on every ray; the base point is permitted), then $\Gamma_{jk}(\xi)=0$ for every $j<k$, for any number of radii and any angular design. Hence all bias terms are $O(h)$ uniformly in $\theta$.*

*(iii) For non-product designs, $\Gamma_{01}$, $\Gamma_{02}$ and $\Gamma_{12}$ are generically nonzero: each is a rational function of the design coordinates that is not identically zero, so it vanishes only on a proper algebraic subset of non-product designs.*

## 7.3 Proof

*Step 1 (scaling, with all cross-sector coupling).* The fitted coefficient vector for data $d$ is
$$\hat w=(X_\theta^{\mathsf T}WX_\theta)^{-1}X_\theta^{\mathsf T}Wd=D_\theta^{-1}\left(\widetilde X^{\mathsf T}W\widetilde X\right)^{-1}\widetilde X^{\mathsf T}Wd .$$
For the cubic sector $j$ the data are $d=\frac{h^3r^3}{6}\left(\theta^jP_j+O(\theta^{j+2})\right)$. Since $\widetilde X(\theta)=\widetilde X_0+O(\theta^2)$ and the Gram matrix of $\widetilde X_0$ is nonsingular, $(\widetilde X^{\mathsf T}W\widetilde X)^{-1}\widetilde X^{\mathsf T}W=(\widetilde X_0^{\mathsf T}W\widetilde X_0)^{-1}\widetilde X_0^{\mathsf T}W+O(\theta^2)$. The factor $D_\theta^{-1}$ multiplies the Hessian sector $k$ by $\theta^{-k}$, and the $h$-scaling of the features gives one net power of $h$ for the Hessian coefficients. Hence the sector-$k$ bias is $h\,\theta^{j-k}\Gamma_{jk}P_j$ with relative corrections $O(\theta^2)$, which proves the expansion; the $O(h^2)$ term is the quartic Taylor remainder. No block-diagonal assumption on the Gram matrix is used.

*Step 2 (product structure of the model space).* For a product design, inner products of separable functions factorise: $\langle a(r)A(v),\,b(r)B(v)\rangle_\xi=\langle a,b\rangle_{\xi_r}\langle A,B\rangle_{\xi_v}$. The base point lies at $r=0$, where every feature except the intercept vanishes; it therefore contributes nothing to any inner product involving a feature with a positive power of $r$, and the factorisation of the relevant cross moments is unaffected. At $\theta=0$ the scaled feature space is
$$\operatorname{span}\{1,r,r^2\}\otimes\{1\}\ \oplus\ \operatorname{span}\{r,r^2\}\otimes\operatorname{span}\{v\}\ \oplus\ \{r^2\}\otimes\operatorname{span}\{\psi\}.$$

*Step 3 (tangential sector, $k=2$).* Let $A\psi=\psi-\Pi_{\operatorname{span}\{1,v\}}\psi$, the projection taken in $L^2(\xi_v)$. Then $r^2\otimes\psi=r^2\otimes\Pi\psi+r^2\otimes A\psi$, where $r^2\otimes\Pi\psi$ lies in the span of the other features ($r^2\otimes1$ and $r^2\otimes v$), and $r^2\otimes A\psi$ is orthogonal to every other feature, because $\langle r^a\otimes B,\,r^2\otimes A\psi\rangle=\langle r^a,r^2\rangle\langle B,A\psi\rangle=0$ for $B\in\operatorname{span}\{1,v\}$. Hence the residualised tangential feature is exactly $r^2\otimes A\psi$, and
$$C_j=\left\langle r^2\otimes A\psi,\ r^3\otimes P_j/6\right\rangle=\tfrac16\langle r^2,r^3\rangle_{\xi_r}\,\langle A\psi,P_j\rangle_{\xi_v}.$$
For $j=0,1$, $P_j\in\operatorname{span}\{1,v\}$, so $\langle A\psi,P_j\rangle=0$ and $\Gamma_{02}=\Gamma_{12}=0$.

*Step 4 (mixed sector, $k=1$).* The mixed feature $r^2\otimes v$ is residualised against $r\otimes v$ (the tangential gradient), $r^2\otimes1$ and $r\otimes1$, and $r^2\otimes A\psi$. Writing $v=\bar v+(v-\bar v)$ with $\bar v$ the $\xi_v$-mean, its residual is $e(r)\otimes(v-\bar v)$, where $e$ is the residual of $r^2$ against $r$ in $L^2(\xi_r)$; it is orthogonal to $r^2\otimes A\psi$ because $\langle v-\bar v,A\psi\rangle=0$. The cross moment with the axial cubic sector is $\langle e,r^3\rangle\langle v-\bar v,P_0\rangle=0$ because $P_0$ is constant. Hence $\Gamma_{01}=0$. Together with Step 3, $\Gamma_{jk}=0$ for all $j<k$.

*Step 5 (non-product designs).* Without the factorisation, the residualised tangential feature is no longer of the form $r^2\otimes A\psi$ and its cross moment with $r^3P_j$ does not reduce to $\langle A\psi,P_j\rangle$. Each entry of $\Gamma_{02}$ is a rational function of the design coordinates on the open set where the scaled Gram matrix is nonsingular. For the rational design
$$v\in\{(0,0),(1,0),(0,1),(-1,0),(0,-1),(\tfrac12,\tfrac12),(-\tfrac12,\tfrac12),(\tfrac12,-\tfrac12)\}$$
with ray-dependent radii $\{1,2\},\{1,2\},\{1,3\},\{1,2\},\{2,3\},\{1,2\},\{1,3\},\{2,3\}$ and the base point, exact arithmetic gives
$$\Gamma_{02}\,e_{nnn}=\left(\tfrac{22854987}{17422013588},\ \tfrac{1774220127}{17422013588},\ -\tfrac{2558549649\sqrt2}{17422013588}\right)\ne0,$$
and, for the same design,
$$\Gamma_{01}e_{nnn}=\left(-\tfrac{29345361\sqrt2}{17422013588},\ -\tfrac{1560958545\sqrt2}{17422013588}\right),\qquad
\Gamma_{12}e_{1nn}=\left(\tfrac{10374325911}{348440271760},\ \tfrac{3746447451}{348440271760},\ \tfrac{2560572171\sqrt2}{348440271760}\right),$$
all nonzero, while the same angular points with the common radii $\{1,2\}$ give exactly $0$ for all three blocks (a non-symmetric angular design, so no angular symmetry is used). Each entry is a rational function of the design coordinates whose numerator is a polynomial that is not identically zero; its zero set is therefore a proper algebraic subset of the design space. Hence $\Gamma_{01},\Gamma_{02},\Gamma_{12}\neq0$ for all non-product designs outside a set of measure zero. $\square$

**Remark 7.2 (what is not an alias).** In product designs the paths with $j\ge k$ remain, e.g. the $u_n$-correction of $f_{nnn}$ contributes a finite $O(h)$ tangential bias, the regression counterpart of the $-(\nu/2-1)f_{n\cdots n}I$ term of Lemma 6.2. These terms do not grow as $\theta\to0$ and are not alias amplification.

## 7.4 Closed form

For a specified design, the leading tangential alias coefficient is the Schur complement of design moments, $\Gamma_{02}=S^{-1}C_0$ with $S$ and $C_0$ as in 7.1 and $d_0=r^3/6$. No least-squares solve is needed; all quantities are finite weighted moments. For a general non-product design no simpler expression exists; for product designs the expression is exactly zero by Theorem 7.1(ii).

## 7.5 Validation (numbers from the archived pipeline)

For the validation design of Section 10 ($n=3$, 40 rays uniform in the disc, $h=0.05$, $\theta\in[0.0125,0.1]$):

| Path | Predicted | Non-product (random radii per ray) | Product $\{1,2\}$ | Product $\{0.3,0.9,2\}$ | Product $\{0.5,1,1.5,2\}$ |
|---|---|---|---|---|---|
| $f_{nnn}\to$ mixed | $\theta^{-1}$ / 0 | slope $-1.02$; limit $0.01090$ (moment formula $0.01090$) | slope $+1.00$ (vanishes) | $+0.99$ | $+0.99$ |
| $f_{nnn}\to$ tangential | $\theta^{-2}$ / 0 | slope $-1.95$; limit $0.00443$ (moment $0.00443$) | $0.00$ (finite, non-alias) | $0.00$ | $0.00$ |
| $f_{\alpha nn}\to$ tangential | $\theta^{-1}$ / 0 | slope $-1.01$; limit $0.07993$ (moment $0.07995$) | $+1.00$ (vanishes) | $+1.00$ | $+1.00$ |
| $f_{\alpha nn}\to$ mixed (non-divergent) | $\theta^0$ | $0.00$ | $0.00$ | $0.00$ | $0.00$ |

The moment formula and the independent regression (implementation B, physical coordinates) agree to four or five digits on every alias path, and implementations A and B agree on the fitted Hessians to $1.4\times10^{-14}$. The slope $-1.95$ rather than $-2$ on the $f_{nnn}\to$ tangential path reflects the non-divergent $O(1)$ term at the largest angles; the limit coefficient itself agrees exactly.

*Note on the earlier value 0.0233:* that value belongs to a different random design used in exploratory runs. The alias coefficient is design-dependent; the manuscript must quote the archived design's value ($0.00443$) or report both designs explicitly.

## 7.6 Relation to the stencil estimator (for Section 11, not part of the theorem)

On identical points (a product design with radii $h$ and $2h$ plus the base point, 81 evaluations), quadratic regression has **21.2 times lower** tangential variance than the 3-point stencil with least squares, at every $\theta$ tested ($0.2$ to $0.025$; exact formula, confirmed by Monte Carlo within 2%). Its tangential bias is also lower, by a factor that depends on the cubic term: $1.6$ for the archived random cubic, $3.4$ for a purely axial cubic in exploratory runs. Both estimators are free of alias amplification on this design; regression uses the same samples more efficiently.
