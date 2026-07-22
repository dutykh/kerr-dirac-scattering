#!/usr/bin/env python3
# Authors: Dr. Davide Batic and Dr. Denys Dutykh
#          (Mathematics Department, Khalifa University of Science and
#           Technology, Abu Dhabi, UAE)
"""Mass-gap horizon reflection benchmark: |S_HH| = 1.

For |omega| < m_e the spatial-infinity channel is closed. The scattering
fiber is the scalar horizon reflection coefficient (eq. (IX.16) of the paper)

    S_HH = - W(R_H_in, R_dec) / W(R_H_out, R_dec),

where R_dec is the exponentially decaying spatial-infinity solution
R_dec ~ e^{-beta u} (u/ell)^{alpha_tilde} v_inf with beta = sqrt(m^2-w^2)
and alpha_tilde = M m^2 / beta. Current conservation forces |S_HH| = 1
(Theorem X.6): this script verifies it numerically on a grid of gap
energies for the Schwarzschild and Kerr benchmark parameters, together with
the conjugation prediction that S_HH is a pure phase.

Output: results/massgap_reflection.json (+ CSV).
"""

from __future__ import annotations

import cmath
import math
from pathlib import Path

import numpy as np

import dirac_radial_common as drc

M, MASS, LAM_SCHW = 1.0, 0.2, 1.0
A_KERR, K_KERR = 0.4, 0
U_LEFT, U_RIGHT, U_MATCH = -40.0, 90.0, 10.0
RTOL, ATOL = 1e-10, 1e-11
OMEGAS = [0.05, 0.10, 0.15, 0.18]


def decaying_data(omega: float, mass: float, Mm: float, lam: float,
                  a: float, m_k: float, u_right: float, ell: float = 1.0
                  ) -> np.ndarray:
    """First-order decaying start data e^{-beta u}(u/ell)^{alpha_tilde} v_inf,
    with v_inf the stable eigenvector of the limiting radial matrix."""
    beta = math.sqrt(mass * mass - omega * omega)
    alpha_t = Mm * mass * mass / beta
    A_inf = np.array([[-1j * omega, 1j * mass], [-1j * mass, 1j * omega]])
    vals, vecs = np.linalg.eig(A_inf)
    idx = int(np.argmin(vals.real))          # eigenvalue -beta
    v_inf = vecs[:, idx]
    scale = math.exp(-beta * u_right) * (u_right / ell) ** alpha_t
    return scale * v_inf


def gap_reflection(omega: float, lam: float, a: float, k: int) -> dict:
    m_k = k + 0.5
    if a != 0.0:
        geo = drc.KerrGeometry(M, a, k, omega)
        r_of_u = geo.r_of_u
        omega_H = geo.omega_H
    else:
        r_of_u = lambda u: drc.schwarzschild_r_of_u(u, M)
        omega_H = omega

    def rhs(u, y):
        r = r_of_u(u)
        A = drc.radial_matrix(r, omega, MASS, lam, M, a, m_k)
        return (A @ y.reshape(2, -1)).reshape(-1)

    YH0 = np.column_stack([
        np.exp(-1j * omega_H * U_LEFT) * np.array([1.0, 0.0]),
        np.exp(+1j * omega_H * U_LEFT) * np.array([0.0, 1.0]),
    ])
    sol_H = drc.integrate_columns(rhs, U_LEFT, U_RIGHT, YH0, RTOL, ATOL)
    ydec = decaying_data(omega, MASS, M, lam, a, m_k, U_RIGHT)
    sol_D = drc.integrate_columns(rhs, U_RIGHT, U_LEFT,
                                  ydec.reshape(2, 1), RTOL, ATOL)

    YH = drc.columns_at(sol_H, U_MATCH)
    RD = drc.columns_at(sol_D, U_MATCH)[:, 0]
    num = drc.det_w(YH[:, 0], RD)
    den = drc.det_w(YH[:, 1], RD)
    S_HH = -num / den
    return {
        "omega": omega,
        "beta": math.sqrt(MASS ** 2 - omega ** 2),
        "S_HH": drc.complex_to_pair(S_HH),
        "abs_S_HH_minus_1": abs(abs(S_HH) - 1.0),
        "phase_S_HH": cmath.phase(S_HH),
    }


def main() -> None:
    results = {"schwarzschild": [], "kerr": []}
    lam_kerr = None
    for omega in OMEGAS:
        results["schwarzschild"].append(
            gap_reflection(omega, LAM_SCHW, 0.0, 0))
        lam_kerr, _ = drc.angular_lambda(K_KERR + 0.5, A_KERR * MASS,
                                         A_KERR * omega)
        results["kerr"].append(gap_reflection(omega, lam_kerr, A_KERR, K_KERR))

    payload = {
        "description": "mass-gap horizon reflection |S_HH| = 1 benchmark",
        "parameters": {"M": M, "mass": MASS, "lambda_schwarzschild": LAM_SCHW,
                       "a_kerr": A_KERR, "k_kerr": K_KERR,
                       "u_left": U_LEFT, "u_right": U_RIGHT,
                       "u_match": U_MATCH, "rtol": RTOL, "atol": ATOL},
        "results": results,
    }
    out = Path("results")
    drc.write_json(out / "massgap_reflection.json", payload)
    rows = [[g, r["omega"], r["abs_S_HH_minus_1"], r["phase_S_HH"]]
            for g in ("schwarzschild", "kerr") for r in results[g]]
    drc.write_csv(out / "massgap_reflection.csv",
                  ["geometry", "omega", "abs_S_HH_minus_1", "phase_S_HH"], rows)
    for g in ("schwarzschild", "kerr"):
        print(f"--- {g} ---")
        for r in results[g]:
            print(f"  omega={r['omega']:5.2f}  | |S_HH|-1 | = "
                  f"{r['abs_S_HH_minus_1']:.3e}   arg S_HH = "
                  f"{r['phase_S_HH']:+.6f}")


if __name__ == "__main__":
    main()
