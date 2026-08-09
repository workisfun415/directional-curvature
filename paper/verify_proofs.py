#!/usr/bin/env python3
"""Symbolic verification of every algebraic step in the manuscript proofs.
Run:  python verify_proofs.py
All checks must print True."""
import sympy as sp

h, R = sp.symbols('h R', positive=True)
f0, f1, f2 = sp.symbols('f0 f1 f2')
ok = []

# Appendix A: probe = 2 * second divided difference on {x, x+Rh, x+h}
dd = f0/((0-R*h)*(0-h)) + f1/((R*h)*(R*h-h)) + f2/(h*(h-R*h))
ok.append(("divided-difference identity",
           sp.simplify(R*(1-R)*h**2*dd - ((1-R)*f0 - f1 + R*f2)) == 0))

# Proposition (probe expansion): coefficients of D^k f(u)
d = sp.symbols('d0:8')
F = lambda s: sum(d[k]*s**k/sp.factorial(k) for k in range(8))
q = sp.expand(2*((1-R)*F(0) + R*F(h) - F(R*h))/(R*(1-R)*h**2))
ok.append(("constant term vanishes", sp.simplify(q.coeff(d[0])) == 0))
ok.append(("first-order term vanishes", sp.simplify(q.coeff(d[1])) == 0))
for k in range(2, 6):
    pred = 2*h**(k-2)/sp.factorial(k)*sum(R**j for j in range(k-1))
    ok.append((f"coefficient of D^{k}f", sp.simplify(q.coeff(d[k]) - pred) == 0))

# Theorem (cone conditioning): block identity
eps, psi, c, tau, a, b0, b1, b2 = sp.symbols(
    'varepsilon psi c tau a b0 b1 b2', real=True)
u = sp.Matrix([eps*sp.cos(psi), eps*sp.sin(psi), sp.sqrt(1-eps**2)])
T0 = sp.Matrix([[a, b0], [b0, -a]])
T = tau*sp.eye(2) + T0
H = sp.Matrix([[T[0, 0], T[0, 1], b1], [T[1, 0], T[1, 1], b2], [b1, b2, c]])
n = sp.Matrix([sp.cos(psi), sp.sin(psi)])
rhs = (c + 2*eps*sp.sqrt(1-eps**2)*(sp.Matrix([b1, b2]).T*n)[0, 0]
       + eps**2*(tau-c) + eps**2*(n.T*T0*n)[0, 0])
ok.append(("block identity", sp.simplify(sp.expand((u.T*H*u)[0, 0] - rhs)) == 0))

# limiting moments on the cap, normalised surface measure
th, r = sp.symbols('theta r', positive=True)
dens = sp.sin(th*r)*th/(1-sp.cos(th))
m2 = sp.limit(sp.integrate(sp.sin(th*r)**2*dens, (r, 0, 1))/th**2, th, 0)
m4 = sp.limit(sp.integrate(sp.sin(th*r)**4*dens, (r, 0, 1))/th**4, th, 0)
ok.append(("<eps^2> -> theta^2/2", sp.simplify(m2 - sp.Rational(1, 2)) == 0))
ok.append(("<eps^4> -> theta^4/3", sp.simplify(m4 - sp.Rational(1, 3)) == 0))

# constants: groups (ii), (iii), (iv)
inner2 = sp.integrate((2*(sp.Matrix([b1, b2]).T*n)[0, 0])**2, (psi, 0, 2*sp.pi))/(2*sp.pi)
rad2 = sp.limit(sp.integrate(sp.sin(th*r)**2*(1-sp.sin(th*r)**2)*dens, (r, 0, 1))/th**2, th, 0)
s2 = sp.sqrt(sp.simplify(inner2*rad2/(2*(b1**2+b2**2))))
ok.append(("s2 = theta/sqrt(2)", sp.simplify(s2 - 1/sp.sqrt(2)) == 0))

inner4 = sp.integrate(((n.T*T0*n)[0, 0])**2, (psi, 0, 2*sp.pi))/(2*sp.pi)
s4 = sp.sqrt(sp.simplify(inner4*m4/(2*(a**2+b0**2))))
ok.append(("group (iv) = theta^2/sqrt(12)", sp.simplify(s4 - 1/sp.sqrt(12)) == 0))

cc, dv = sp.symbols('cc dv', real=True)
expr = cc**2 + cc*dv*th**2 + dv**2*th**4/3
cstar = sp.solve(sp.diff(expr, cc), cc)[0]
s6 = sp.sqrt(sp.simplify(expr.subs(cc, cstar).subs(dv, 1/sp.sqrt(2))))
ok.append(("s6 = theta^2/sqrt(24)", sp.simplify(s6/th**2 - 1/sp.sqrt(24)) == 0))
ok.append(("s6 < group (iv)", sp.simplify(1/sp.sqrt(24) < 1/sp.sqrt(12))))

w = max(len(k) for k, _ in ok)
for k, v in ok:
    print(f"{k:<{w}}  {v}")
print("\nall checks passed:", all(v for _, v in ok))
