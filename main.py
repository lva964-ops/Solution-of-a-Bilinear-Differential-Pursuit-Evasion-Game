"""
Differential game model: Worm/Botnet (A) vs. Active Cyber Defense (B)
2-segment network: (x1, x2) infected, (y1, y2) protected
dx/dt = (A0 + U·A1) x
dy/dt = (B0 + V·B1) y
"""

import sympy as sp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

matplotlib.rcParams["figure.dpi"] = 110

# ============================================================
# 1. Symbolic analysis
# ============================================================
def analyze_system(M0, M1, name="M", verbose=True):
    """Print eigenvalues/eigenvectors for M0 + diag(c)*M1, c in {0,1}^2."""
    M0 = sp.Matrix(M0)
    M1 = sp.Matrix(M1)
    combos = [(0, 0), (1, 0), (0, 1), (1, 1)]
    if verbose:
        print(f"=== {name} ===")
    results = {}
    for c in combos:
        V = sp.diag(*c)
        M = M0 + V * M1
        evals = M.eigenvals()
        evects = M.eigenvects()
        if verbose:
            print(f"  diag{c}: M = {M.tolist()}, evals = {evals}")
            for val, mult, vectors in evects:
                print(f"    lambda = {val}: vect = {vectors[0].T.tolist()}")
        results[c] = M
    return results


# --- Defense matrices ---
B0 = sp.Matrix([[1, 0], [0, 1]])           # baseline hygiene
B1 = sp.Matrix([[5, -2], [2, 0]])          # active remediation/quarantine
B_plus = B0 + B1                           # V = I  -> worst defense?

# --- Worm matrices ---
A0 = sp.Matrix([[sp.Rational(5, 2), sp.Rational(-3, 2)],
                [sp.Rational(1, 2), sp.Rational(1, 2)]])
A1 = sp.Matrix([[2, 0], [0, 2]])
A_plus = A0 + A1                           # U = I

print("B_+ =", B_plus.tolist(), " evals =", B_plus.eigenvals())
print("A_+ =", A_plus.tolist(), " evals =", A_plus.eigenvals())
print()
analyze_system(B0, B1, "Defense B")
print()
analyze_system(A0, A1, "Worm A")


# ============================================================
# 2. Numerical simulation of the linear ODEs
# ============================================================
def simulate(M, y0, t_end=2.0, n=400):
    """Solve  dy/dt = M y  numerically."""
    M_np = np.array(M.tolist(), dtype=float)
    sol = solve_ivp(lambda t, y: M_np @ y, [0, t_end], y0,
                    t_eval=np.linspace(0, t_end, n))
    return sol.t, sol.y          # t, array of shape (2, n)


x0 = [3.0, 1.0]   # initial infected  (segment 1, segment 2)
y0 = [3.0, 1.0]   # initial protected (segment 1, segment 2)
t_end = 2.0

# Worm: nominal (U=0) vs. worst case (U = diag(1,0), λ_max ≈ 4.303)
t_x0, X0 = simulate(A0, x0, t_end)
t_x,  X  = simulate(A0 + sp.diag(1, 0) * A1, x0, t_end)

# Defense: nominal (V=0) vs. best (V = diag(1,0), λ_max = 6) vs. both (V=I, λ_max=5)
t_y0, Y0 = simulate(B0, y0, t_end)
t_y,  Y  = simulate(B0 + sp.diag(1, 0) * B1, y0, t_end)
t_yb, Yb = simulate(B_plus, y0, t_end)


# ============================================================
# 3. Plots
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ----- Panel 1: dynamics ------------------------------------
ax = axes[0]
ax.semilogy(t_x0, X0.sum(axis=0), 'r:',  lw=2, label=r'worm, $U=0$ (nominal)')
ax.semilogy(t_x,  X.sum(axis=0),  'r-',  lw=2, label=r'worm, $U=\mathrm{diag}(1,0)$ (worst)')
ax.semilogy(t_y0, Y0.sum(axis=0), 'b:',  lw=2, label=r'defense, $V=0$ (nominal)')
ax.semilogy(t_y,  Y.sum(axis=0),  'b-',  lw=2, label=r'defense, $V=\mathrm{diag}(1,0)$ (best)')
ax.semilogy(t_yb, Yb.sum(axis=0),'b--', lw=2, label=r'defense, $V=I$ (both)')
ax.set_title('Total counts: infected vs. protected')
ax.set_xlabel('time t')
ax.set_ylabel('total nodes (log scale)')
ax.grid(True, which='both', alpha=0.3)
ax.legend(fontsize=8, loc='lower right')

# ----- Panel 2: dominant eigenvalues ------------------------
ax = axes[1]
labels = [r'$U=V=0$', r'$U=\mathrm{diag}(1,0)$', r'$U=\mathrm{diag}(0,1)$',
          r'$U=I$', r'$V=\mathrm{diag}(1,0)$', r'$V=I$']
worm_rates    = [2.0,  2.5 + np.sqrt(13)/2, 2.5,  4.0,  np.nan, np.nan]
defense_rates = [np.nan, np.nan, np.nan, np.nan,  6.0,  5.0]
x = np.arange(len(labels))
ax.bar(x - 0.2, [v if not np.isnan(v) else 0 for v in worm_rates],
       width=0.4, color='crimson', label='worm $\\lambda_{\\max}$')
ax.bar(x + 0.2, [v if not np.isnan(v) else 0 for v in defense_rates],
       width=0.4, color='royalblue', label='defense $\\lambda_{\\max}$')
ax.axhline(4.303, ls='--', color='crimson',  alpha=0.6,
           label='worm worst rate ≈ 4.303')
ax.axhline(5.0,   ls='--', color='royalblue', alpha=0.6,
           label='defense (V=I) rate = 5')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('dominant eigenvalue (growth rate)')
ax.set_title('Growth rates: defense outpaces worm')
ax.legend(fontsize=8)
ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('cyber_defense_plot.png', dpi=150, bbox_inches='tight')
plt.show()