#!/usr/bin/env python3
# Authors: Dr. Davide Batic and Dr. Denys Dutykh
#          (Mathematics Department, Khalifa University of Science and
#           Technology, Abu Dhabi, UAE)
"""Minimal rotating Kerr (a != 0) massive Dirac finite-matching benchmark.

Computes one simple Chandrasekhar-Page angular eigenvalue by two-sided
shooting (adaptive bracket), then performs the same finite matching as the
Schwarzschild benchmark with the Kerr radial matrix, the horizon frequency
shift omega_H = omega + Omega_H (k + 1/2), and the Kerr tortoise map.
Diagnostics include the oriented fluxes and the conjugation identities.

Usage:
    python3 kerr_dirac_matching.py                # paper benchmark
    python3 kerr_dirac_matching.py --refined      # Phi_star start data
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import dirac_radial_common as drc


@dataclass
class Params:
    M: float = 1.0
    a: float = 0.4
    mass: float = 0.2
    omega: float = 0.5
    k: int = 0
    ell_infinity: float = 1.0
    u_left: float = -40.0
    u_match: float = 20.0
    angular_epsilon: float = 1e-6
    angular_rtol: float = 1e-10
    angular_atol: float = 1e-12
    radial_rtol: float = 1e-10
    radial_atol: float = 1e-11
    grid_points: int = 240
    lambda_bracket_left: float = 0.7
    lambda_bracket_right: float = 1.6
    refined: bool = False


def make_rhs(p: Params, geo: drc.KerrGeometry, lam: float):
    m_k = p.k + 0.5

    def rhs(u, y):
        r = geo.r_of_u(u)
        A = drc.radial_matrix(r, p.omega, p.mass, lam, p.M, p.a, m_k)
        return (A @ y.reshape(2, -1)).reshape(-1)
    return rhs


def run_once(p: Params, geo: drc.KerrGeometry, tr: drc.InfinityTransport,
             lam: float, u_right: float) -> dict:
    rhs = make_rhs(p, geo, lam)
    YH0 = np.column_stack([
        np.exp(-1j * geo.omega_H * p.u_left) * np.array([1.0, 0.0]),
        np.exp(+1j * geo.omega_H * p.u_left) * np.array([0.0, 1.0]),
    ])
    col_out, col_in = tr.infinity_columns(u_right, refined=p.refined)
    YI0 = np.column_stack([col_out, col_in])

    sol_H = drc.integrate_columns(rhs, p.u_left, u_right, YH0,
                                  p.radial_rtol, p.radial_atol)
    sol_I = drc.integrate_columns(rhs, u_right, p.u_left, YI0,
                                  p.radial_rtol, p.radial_atol)
    S, F_in, F_out = drc.scattering_matrix_at(p.u_match, sol_H, sol_I)

    grid = np.linspace(p.u_left + 5.0, u_right - 5.0, p.grid_points)
    w_out, w_in = [], []
    for u in grid:
        YH = drc.columns_at(sol_H, u)
        YI = drc.columns_at(sol_I, u)
        w_out.append(drc.det_w(YH[:, 1], YI[:, 0]))
        w_in.append(drc.det_w(YH[:, 0], YI[:, 1]))
    w_out, w_in = np.array(w_out), np.array(w_in)
    w0_out = drc.det_w(drc.columns_at(sol_H, p.u_match)[:, 1],
                       drc.columns_at(sol_I, p.u_match)[:, 0])
    w0_in = drc.det_w(drc.columns_at(sol_H, p.u_match)[:, 0],
                      drc.columns_at(sol_I, p.u_match)[:, 1])

    sub = np.linspace(p.u_left + 10.0, u_right - 10.0, 25)
    dS = 0.0
    for u in sub:
        Su, _, _ = drc.scattering_matrix_at(u, sol_H, sol_I)
        dS = max(dS, float(np.linalg.norm(Su - S)))

    YHm = drc.columns_at(sol_H, p.u_match)
    YIm = drc.columns_at(sol_I, p.u_match)
    return {
        "u_right": u_right,
        "S": drc.matrix_to_pairs(S),
        "unitary_defect_2norm": drc.unitarity_defect(S),
        "wronskian_out_relative_variation":
            float(np.max(np.abs(w_out - w0_out)) / abs(w0_out)),
        "wronskian_in_relative_variation":
            float(np.max(np.abs(w_in - w0_in)) / abs(w0_in)),
        "matching_point_variation_frobenius": dS,
        "det_Fout": drc.complex_to_pair(np.linalg.det(F_out)),
        "det_Fin": drc.complex_to_pair(np.linalg.det(F_in)),
        "q_RH_in": drc.q_flux(YHm[:, 0]),
        "q_RH_out": drc.q_flux(YHm[:, 1]),
        "q_Rinf_out": drc.q_flux(YIm[:, 0]),
        "q_Rinf_in": drc.q_flux(YIm[:, 1]),
        **drc.conjugation_checks(F_in, F_out, S),
        "nfev_horizon": int(sol_H.nfev),
        "nfev_infinity": int(sol_I.nfev),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--u-rights", default="80,120,180")
    ap.add_argument("--refined", action="store_true",
                    help="use the refined phase Phi_star in the start data")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    p = Params(refined=args.refined)
    u_rights = [float(x) for x in args.u_rights.split(",")]

    geo = drc.KerrGeometry(p.M, p.a, p.k, p.omega)
    m_k = p.k + 0.5
    mu_a, nu_a = p.a * p.mass, p.a * p.omega
    lam, shoot = drc.angular_lambda(
        m_k, mu_a, nu_a,
        bracket=(p.lambda_bracket_left, p.lambda_bracket_right),
        eps=p.angular_epsilon, rtol=p.angular_rtol, atol=p.angular_atol)
    tr = drc.InfinityTransport(p.M, p.mass, p.omega, lam, p.ell_infinity)

    runs = [run_once(p, geo, tr, lam, ur) for ur in u_rights]

    tag = "refined" if p.refined else "first_order"
    payload = {
        "description": "Kerr a != 0 massive Dirac finite-matching benchmark "
                       "for one simple angular branch",
        "start_data_order": tag,
        "parameters": {k: v for k, v in vars(p).items()},
        "geometry": {"r0": geo.r0, "r1": geo.r1, "kappa_plus": geo.kappa_plus,
                     "Omega_H": geo.Omega_H, "omega_H": geo.omega_H},
        "angular": {"m_k": m_k, "mu": mu_a, "nu": nu_a,
                    "angular_lambda": lam,
                    "shooting_determinant_at_root": shoot},
        "derived": {"kappa": tr.kappa, "alpha": tr.alpha,
                    "B1": drc.matrix_to_pairs(tr.B1),
                    "b_minus_out": drc.matrix_to_pairs(tr.b_minus.reshape(2, 1)),
                    "b_plus_in": drc.matrix_to_pairs(tr.b_plus.reshape(2, 1))},
        "runs": runs,
    }
    out = Path(args.outdir)
    suffix = "_refined" if p.refined else ""
    drc.write_json(out / f"kerr_dirac_matching{suffix}_output.json", payload)
    drc.write_csv(out / f"kerr_dirac_matching{suffix}_summary.csv",
                  ["u_right", "unitary_defect", "wronskian_out_var",
                   "wronskian_in_var", "matching_var"],
                  [[r["u_right"], r["unitary_defect_2norm"],
                    r["wronskian_out_relative_variation"],
                    r["wronskian_in_relative_variation"],
                    r["matching_point_variation_frobenius"]] for r in runs])
    print(f"lambda = {lam:.12f}  (shooting det {shoot:.2e})")
    print(f"{'u_+':>6} {'defect':>12} {'dW_out':>12} {'dW_in':>12} {'dS':>12}")
    for r in runs:
        print(f"{r['u_right']:6.0f} {r['unitary_defect_2norm']:12.4e} "
              f"{r['wronskian_out_relative_variation']:12.4e} "
              f"{r['wronskian_in_relative_variation']:12.4e} "
              f"{r['matching_point_variation_frobenius']:12.4e}")


if __name__ == "__main__":
    main()
