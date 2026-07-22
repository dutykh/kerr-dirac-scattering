#!/usr/bin/env python3
# Authors: Dr. Davide Batic and Dr. Denys Dutykh
#          (Mathematics Department, Khalifa University of Science and
#           Technology, Abu Dhabi, UAE)
"""GATED quasi-bound resonance search in the mass gap (closed channel).

Locates a complex zero of the closed-channel Evans denominator

    D_dec(omega) = W(R_H_out, R_dec)(omega),

for the Schwarzschild benchmark family (M = 1, m_e = 0.2, lambda = 1), near
the hydrogenic estimate omega_1 ~ m_e (1 - (M m_e)^2/2) ~ 0.196, and tests
the parameter-free Breit-Wigner prediction of the paper: near a simple
quasi-bound pole omega_n = E_n + i gamma_n of S_HH, the phase derivative
of the unimodular horizon reflection coefficient is a Lorentzian of width
|gamma_n| and area 2 pi, with no fitted parameters.

The decaying start data include the first transport correction:
    R_dec ~ e^{-beta u} (u/ell)^{alpha_t} (v_- + b/u),
with (A_inf + beta I) b = (alpha_t I - A_1) v_-, where the solvability of
this equation along v_- REQUIRES alpha_t = M m_e^2 / beta; the script
asserts this numerically (an internal consistency check of the paper's
closed-channel Coulomb power).

GATE (all must hold, otherwise the results are marked not certified and no
manuscript figure is produced):
  G1  relative Evans residual at the root < 1e-10;
  G2  root stable across (u_+, rtol) refinements: spread in E_n and in
      gamma_n below 5 per cent of |gamma_n|;
  G3  parameter-free Breit-Wigner overlay: relative L2 deviation of the
      background-subtracted phase derivative from the pole-predicted SIGNED
      Lorentzian -2 gamma_n/((w-E_n)^2+gamma_n^2) < 10 per cent on the core
      window |omega - E_n| <= 1.5 |gamma_n|. The affine background is
      estimated on the wing windows 3.0 |gamma_n| <= |omega - E_n| <=
      4.5 |gamma_n| (chosen to stay clear of the neighbouring level at
      5.3 |gamma_1|), each wing unwrapped and differentiated separately,
      after subtracting the (small) local Lorentzian tail; no parameter is
      fitted to the resonant core.

Output: results/quasibound_search.json (+ CSV of the real-axis scan).
"""

from __future__ import annotations

import cmath
import math
from pathlib import Path

import numpy as np

import dirac_radial_common as drc

M, MASS, LAM, M_K = 1.0, 0.2, 1.0, 0.5
U_LEFT, U_MATCH = -40.0, 10.0
ELL = 1.0
SCAN = (0.1870, 0.1998, 160)
BASE = dict(u_right=320.0, rtol=1e-11, atol=1e-12)
GRID = [dict(u_right=240.0, rtol=1e-10, atol=1e-11),
        dict(u_right=320.0, rtol=1e-10, atol=1e-11),
        dict(u_right=320.0, rtol=1e-11, atol=1e-12),
        dict(u_right=400.0, rtol=1e-11, atol=1e-12)]

A1 = np.array([[0.0, LAM - 1j * MASS * M],
               [LAM + 1j * MASS * M, 0.0]])


def decaying_start(omega: complex, u_right: float) -> np.ndarray:
    """Transport-corrected decaying data e^{-bu}(u/l)^{at}(v_- + b/u)."""
    beta = cmath.sqrt(MASS * MASS - omega * omega)      # Re beta > 0
    alpha_t = M * MASS * MASS / beta
    A_inf = np.array([[-1j * omega, 1j * MASS], [-1j * MASS, 1j * omega]])
    vals, vecs = np.linalg.eig(A_inf)
    i_m = int(np.argmin(vals.real))
    v_m, v_p = vecs[:, i_m], vecs[:, 1 - i_m]
    rhs = (alpha_t * np.eye(2) - A1) @ v_m
    c = np.linalg.solve(np.column_stack([v_m, v_p]), rhs)
    # Solvability along v_-  <=>  alpha_t = M m^2 / beta  (paper's power).
    assert abs(c[0]) < 1e-9 * max(1.0, abs(c[1])), \
        f"transport solvability violated: c_minus = {c[0]:.3e}"
    b = (c[1] / (2.0 * beta)) * v_p
    scale = cmath.exp(-beta * u_right) * (u_right / ELL) ** alpha_t
    return scale * (v_m + b / u_right)


def radial_rhs(u: float, y: np.ndarray, omega: complex) -> np.ndarray:
    r = drc.schwarzschild_r_of_u(u, M)
    A = drc.radial_matrix(r, omega, MASS, LAM, M, 0.0, M_K)
    return (A @ y.reshape(2, -1)).reshape(-1)


def columns(omega: complex, u_right: float, rtol: float, atol: float,
            with_h_in: bool = False):
    def rhs(u, y):
        return radial_rhs(u, y, omega)
    cols_H = [np.exp(+1j * omega * U_LEFT) * np.array([0.0, 1.0])]
    if with_h_in:
        cols_H.append(np.exp(-1j * omega * U_LEFT) * np.array([1.0, 0.0]))
    sol_H = drc.integrate_columns(rhs, U_LEFT, U_MATCH,
                                  np.column_stack(cols_H), rtol, atol)
    Y_dec = decaying_start(omega, u_right).reshape(2, 1)
    sol_D = drc.integrate_columns(rhs, u_right, U_MATCH, Y_dec, rtol, atol)
    YH = drc.columns_at(sol_H, U_MATCH)
    YD = drc.columns_at(sol_D, U_MATCH)
    return YH, YD


def evans_dec(omega: complex, u_right: float, rtol: float, atol: float
              ) -> tuple[complex, float]:
    YH, YD = columns(omega, u_right, rtol, atol)
    D = drc.det_w(YH[:, 0], YD[:, 0])
    rel = abs(D) / (np.linalg.norm(YH[:, 0]) * np.linalg.norm(YD[:, 0]))
    return D, rel


def s_hh(omega: float, u_right: float, rtol: float, atol: float) -> complex:
    YH, YD = columns(omega, u_right, rtol, atol, with_h_in=True)
    num = drc.det_w(YH[:, 1], YD[:, 0])
    den = drc.det_w(YH[:, 0], YD[:, 0])
    return -num / den


def secant_root(z0: complex, z1: complex, u_right: float, rtol: float,
                atol: float, maxit: int = 40):
    f0, _ = evans_dec(z0, u_right, rtol, atol)
    f1, r1 = evans_dec(z1, u_right, rtol, atol)
    for _ in range(maxit):
        dz = -f1 * (z1 - z0) / (f1 - f0)
        z0, f0 = z1, f1
        z1 = z1 + dz
        f1, r1 = evans_dec(z1, u_right, rtol, atol)
        if abs(dz) < 1e-14 * max(1.0, abs(z1)) or r1 < 1e-14:
            break
    return z1, r1


def main() -> None:
    # ---- real-axis scan of the relative Evans residual ------------------
    om_lo, om_hi, n_scan = SCAN
    omegas = np.linspace(om_lo, om_hi, n_scan)
    scan = [evans_dec(w, **BASE)[1] for w in omegas]
    i_min = int(np.argmin(scan))
    om_seed = float(omegas[i_min])
    print(f"scan minimum |D|_rel = {scan[i_min]:.3e} at omega = {om_seed:.6f}")

    # ---- complex secant root from the scan minimum ----------------------
    root, resid = secant_root(om_seed + 0j, om_seed + 1e-5j, **BASE)
    print(f"root  = {root.real:.12f} {root.imag:+.3e} i   |D|_rel = {resid:.2e}")
    E_n, g_n = root.real, root.imag

    # ---- stability grid -------------------------------------------------
    grid_roots = []
    for cfg in GRID:
        r_g, res_g = secant_root(root + 0j, root + 1e-6j, **cfg)
        grid_roots.append({**cfg, "root": drc.complex_to_pair(r_g),
                           "residual": res_g})
        print(f"  u+={cfg['u_right']:5.0f} rtol={cfg['rtol']:.0e}: "
              f"E={r_g.real:.12f}  gamma={r_g.imag:+.6e}  res={res_g:.1e}")
    Es = np.array([g["root"][0] for g in grid_roots])
    Gs = np.array([g["root"][1] for g in grid_roots])
    spread_E, spread_G = float(Es.max() - Es.min()), float(Gs.max() - Gs.min())

    # ---- neighbouring level (context for the scan figure) ---------------
    root2, resid2 = secant_root(0.1990 + 0j, 0.1990 + 2e-4j, **BASE)
    print(f"neighbour root = {root2.real:.10f} {root2.imag:+.3e} i  "
          f"|D|_rel = {resid2:.1e}")

    # ---- parameter-free Breit-Wigner fingerprint ------------------------
    gate = {}
    gate["G1_residual"] = bool(resid < 1e-10)
    gate["G2_stability"] = bool(abs(g_n) > 0 and
                                spread_E < 0.05 * abs(g_n) and
                                spread_G < 0.05 * abs(g_n))
    bw = None
    if abs(g_n) > 0:
        g_a = abs(g_n)

        def lorentz(om):
            return -2.0 * g_n / ((om - E_n) ** 2 + g_n * g_n)

        om_wing_l = np.linspace(E_n - 4.5 * g_a, E_n - 3.0 * g_a, 9)
        om_wing_r = np.linspace(E_n + 3.0 * g_a, E_n + 4.5 * g_a, 9)
        om_core = np.linspace(E_n - 1.5 * g_a, E_n + 1.5 * g_a, 25)

        def dphi_on(om_grid):
            ph = np.unwrap([cmath.phase(s_hh(w, **BASE)) for w in om_grid])
            return np.gradient(ph, om_grid)

        # each wing is unwrapped and differentiated on its own grid
        dphi_w = np.concatenate([dphi_on(om_wing_l), dphi_on(om_wing_r)])
        om_wing = np.concatenate([om_wing_l, om_wing_r])
        dphi_c = dphi_on(om_core)
        # affine background from the wings, local Lorentzian tail removed
        resid_w = dphi_w - lorentz(om_wing)
        coef = np.polyfit(om_wing - E_n, resid_w, 1)
        bg_core = np.polyval(coef, om_core - E_n)
        dev = (dphi_c - bg_core) - lorentz(om_core)
        rel_l2 = float(np.linalg.norm(dev) / np.linalg.norm(lorentz(om_core)))
        bw = {"omega_core": om_core.tolist(),
              "dphi_domega_core": dphi_c.tolist(),
              "omega_wing": om_wing.tolist(),
              "dphi_domega_wing": dphi_w.tolist(),
              "background_affine": [float(coef[0]), float(coef[1])],
              "relative_l2_deviation": rel_l2}
        gate["G3_breit_wigner"] = bool(rel_l2 < 0.10)
        print(f"Breit-Wigner overlay (signed, core |w-E|<=1.5|g|): "
              f"rel L2 dev = {rel_l2:.3f}")
    else:
        gate["G3_breit_wigner"] = False
    gate["certified"] = all(gate.values())
    print(f"GATE: {gate}")

    payload = {
        "description": "gated closed-channel quasi-bound search "
                       "(Schwarzschild benchmark family)",
        "parameters": {"M": M, "mass": MASS, "lambda": LAM, "m_k": M_K,
                       "u_left": U_LEFT, "u_match": U_MATCH,
                       "hydrogenic_estimate": MASS * (1 - (M * MASS) ** 2 / 2),
                       **BASE},
        "scan": {"omega": omegas.tolist(), "relative_residual": scan},
        "root": {"E_n": E_n, "gamma_n": g_n, "residual": resid},
        "neighbour_root": {"E": root2.real, "gamma": root2.imag,
                           "residual": resid2},
        "stability_grid": grid_roots,
        "spread": {"E": spread_E, "gamma": spread_G},
        "breit_wigner": bw,
        "gate": gate,
    }
    drc.write_json(Path("results") / "quasibound_search.json", payload)
    drc.write_csv(Path("results") / "quasibound_scan.csv",
                  ["omega", "relative_evans_residual"],
                  [[f"{w:.8f}", f"{s:.6e}"] for w, s in zip(omegas, scan)])


if __name__ == "__main__":
    main()
