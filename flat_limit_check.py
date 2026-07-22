#!/usr/bin/env python3
# Authors: Dr. Davide Batic and Dr. Denys Dutykh
#          (Mathematics Department, Khalifa University of Science and
#           Technology, Abu Dhabi, UAE)
"""Exactly solvable and oblate flat benchmarks (M = 0): certification of the
matching conventions.

Case A (a = 0, lambda = 0): the radial matrix is the CONSTANT free matrix
A_inf = [[-i w, i m], [-i m, i w]]; the Jost solutions are exact plane waves
in the diagonalizing frame and the scattering matrix is exactly
anti-diagonal with unimodular entries. The finite-matching pipeline must
reproduce this to integrator accuracy — a certification of the
normalization conventions (III.4)-(III.5) of the paper.

Case B (a = 0.4, M = 0, oblate spheroidal Minkowski, eq. (II.6) of the
paper): nu = 1/sqrt(u^2+a^2) is regular on the whole line, the angular
eigenvalue is computed from the Chandrasekhar-Page problem, and the flat
scattering matrix must be unitary and satisfy the conjugation identities,
with alpha = 0 (no Dollard phase).

Output: results/flat_limit_check.json.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import dirac_radial_common as drc

MASS, OMEGA, ELL = 0.2, 0.5, 1.0
U_LEFT, U_RIGHT, U_MATCH = -120.0, 120.0, 0.0
RTOL, ATOL = 1e-11, 1e-12
A_OBLATE, K_OBLATE = 0.4, 0


def flat_case(lam: float, a: float, k: int) -> dict:
    tr = drc.InfinityTransport(0.0, MASS, OMEGA, lam, ELL)   # M = 0: alpha = 0
    m_k = k + 0.5
    if a == 0.0 and lam == 0.0:
        def rhs(u, y):                       # constant free matrix: exact case
            A = np.array([[-1j * OMEGA, 1j * MASS],
                          [-1j * MASS, 1j * OMEGA]])
            return (A @ y.reshape(2, -1)).reshape(-1)
    else:
        def rhs(u, y):                       # oblate flat case, regular at u=0
            r = u
            A = drc.radial_matrix(r, OMEGA, MASS, lam, 0.0, a, m_k)
            return (A @ y.reshape(2, -1)).reshape(-1)

    kap = tr.kappa
    # Left-end diagonalizer: for the oblate case the effective mass term at
    # u -> -infty is  nu * i m r -> -i m  (sign r/|r|), so the frame is
    # T(-Theta) = T^{-1}; for the exact a = 0, lambda = 0 case both frames
    # diagonalize the constant matrix and T is kept for continuity.
    TL = tr.T if (a == 0.0 and lam == 0.0) else tr.Tinv
    YH0 = np.column_stack([
        TL @ (np.exp(-1j * kap * U_LEFT) * np.array([1.0, 0.0])),
        TL @ (np.exp(+1j * kap * U_LEFT) * np.array([0.0, 1.0])),
    ])
    col_out, col_in = tr.infinity_columns(U_RIGHT)
    YI0 = np.column_stack([col_out, col_in])

    sol_H = drc.integrate_columns(rhs, U_LEFT, U_RIGHT, YH0, RTOL, ATOL)
    sol_I = drc.integrate_columns(rhs, U_RIGHT, U_LEFT, YI0, RTOL, ATOL)
    S, F_in, F_out = drc.scattering_matrix_at(U_MATCH, sol_H, sol_I)
    return {
        "lambda": lam, "a": a,
        "S": drc.matrix_to_pairs(S),
        "unitary_defect_2norm": drc.unitarity_defect(S),
        "antidiagonality_defect": float(abs(S[0, 0]) + abs(S[1, 1])),
        "transmission_modulus_defect": float(abs(abs(S[0, 1]) - 1.0)),
        **drc.conjugation_checks(F_in, F_out, S),
    }


def main() -> None:
    lam_oblate, _ = drc.angular_lambda(K_OBLATE + 0.5, A_OBLATE * MASS,
                                       A_OBLATE * OMEGA)
    cases = [flat_case(0.0, 0.0, 0), flat_case(lam_oblate, A_OBLATE, K_OBLATE)]
    payload = {
        "description": "flat (M = 0) certification benchmarks: exact a=0 case "
                       "and oblate a!=0 case",
        "parameters": {"mass": MASS, "omega": OMEGA, "u_left": U_LEFT,
                       "u_right": U_RIGHT, "u_match": U_MATCH,
                       "rtol": RTOL, "atol": ATOL,
                       "a_oblate": A_OBLATE, "k_oblate": K_OBLATE,
                       "lambda_oblate": lam_oblate},
        "cases": cases,
    }
    drc.write_json(Path("results") / "flat_limit_check.json", payload)
    for c in cases:
        print(f"a={c['a']:3.1f} lambda={c['lambda']:8.6f}: "
              f"defect={c['unitary_defect_2norm']:.2e}"
              f"  antidiag={c['antidiagonality_defect']:.2e}"
              f"  |T|-1={c['transmission_modulus_defect']:.2e}"
              f"  conj={c['det_conjugation_defect']:.1e}")


if __name__ == "__main__":
    main()
