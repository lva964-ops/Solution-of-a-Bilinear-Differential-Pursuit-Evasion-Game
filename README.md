# Bilinear Differential Pursuit–Evasion Game: Botnet vs. Active Cyber Defense

Python reference implementation of the constructive method for solving
**bilinear differential games of pursuit–evasion** with **binary controls**
`u_i, v_j ∈ {0, 1}`, applied to a concrete cybersecurity scenario:
containment of a self-replicating botnet in a two-segment corporate network
by an active cyber-defense system.

Companion code to:

> A. Chikrii, V. Malyukov, V. Lakhno, I. Malyukova.
> *Solution of a Bilinear Differential Pursuit–Evasion Game.*
> V. M. Glushkov Institute of Cybernetics, NAS of Ukraine, Kyiv.

---

## Model

Two-segment network — (1) workstation/perimeter, (2) core servers/database:
