#!/usr/bin/env python3
# Authors: Dr. Davide Batic and Dr. Denys Dutykh
#          (Mathematics Department, Khalifa University of Science and
#           Technology, Abu Dhabi, UAE)
"""Minimal Schwarzschild (a = 0) massive Dirac finite-matching benchmark.

Reproduces the reproducibility smoke test of Section XI of the paper:
horizon and spatial-infinity Jost columns are integrated to a common matching
point, the partial scattering matrix is S = (F_out)^{-1} F_in, and the
independent diagnostics are the unitarity defect, the determinant-Wronskian
conservation, the matching-point invariance, the oriented fluxes, and the
conjugation identities of Propositions VIII.1 and IX.1.

Usage:
    python3 schwarzschild_dirac_matching.py                  # paper benchmark
    python3 schwarzschild_dirac_matching.py --refined        # Phi_star data
    python3 schwarzschild_dirac_matching.py --u-rights 80,120,180
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import dirac_radial_common as drc


@dataclass
class Params:
    M: float = 1.0
    mass: float = 0.2
    omega: float = 0.5
    angular_lambda: float = 1.0
    ell_infinity: float = 1.0
    u_left: float = -80.0
    u_match: float = 20.0
    rtol: float = 1e-10
    atol: float = 1e-11
    grid_points: int = 240
    refined: bool = False


def make_rhs(p: Params):
    def rhs(u, y):
        r = drc.schwarzschild_r_of_u(u, p.M)
        A = drc.radial_matrix(r, p.omega, p.mass, p.angular_lambda,
                              p.M, 0.0, 0.0)
        return (A @ y.reshape(2, -1)).reshape(-1)
    return rhs


def run_once(p: Params, tr: drc.InfinityTransport, u_right: float) -> dict:
    rhs = make_rhs(p)
    # Horizon columns at u_left: (R_H_in, R_H_out) ~ (e^{-i w u} e_-, e^{+i w u} e_+)
    YH0 = np.column_stack([
        np.exp(-1j * p.omega * p.u_left) * np.array([1.0, 0.0]),
        np.exp(+1j * p.omega * p.u_left) * np.array([0.0, 1.0]),
    ])
    # Infinity columns at u_right: (R_inf_out, R_inf_in), first transport order
    col_out, col_in = tr.infinity_columns(u_right, refined=p.refined)
    YI0 = np.column_stack([col_out, col_in])

    sol_H = drc.integrate_columns(rhs, p.u_left, u_right, YH0, p.rtol, p.atol)
    sol_I = drc.integrate_columns(rhs, u_right, p.u_left, YI0, p.rtol, p.atol)

    S, F_in, F_out = drc.scattering_matrix_at(p.u_match, sol_H, sol_I)

    # Wronskian conservation over the common interval
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

    # Matching-point invariance of S
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
    tr = drc.InfinityTransport(p.M, p.mass, p.omega, p.angular_lambda,
                               p.ell_infinity)
    runs = [run_once(p, tr, ur) for ur in u_rights]

    tag = "refined" if p.refined else "first_order"
    payload = {
        "description": "Schwarzschild a=0 massive Dirac finite-matching benchmark",
        "start_data_order": tag,
        "parameters": {k: v for k, v in vars(p).items()},
        "derived": {
            "kappa": tr.kappa, "alpha": tr.alpha,
            "B1": drc.matrix_to_pairs(tr.B1),
            "b_minus_out": drc.matrix_to_pairs(tr.b_minus.reshape(2, 1)),
            "b_plus_in": drc.matrix_to_pairs(tr.b_plus.reshape(2, 1)),
        },
        "runs": runs,
    }
    out = Path(args.outdir)
    suffix = "_refined" if p.refined else ""
    drc.write_json(out / f"schwarzschild_dirac_matching{suffix}_output.json", payload)
    drc.write_csv(out / f"schwarzschild_dirac_matching{suffix}_summary.csv",
                  ["u_right", "unitary_defect", "wronskian_out_var",
                   "wronskian_in_var", "matching_var"],
                  [[r["u_right"], r["unitary_defect_2norm"],
                    r["wronskian_out_relative_variation"],
                    r["wronskian_in_relative_variation"],
                    r["matching_point_variation_frobenius"]] for r in runs])
    print(f"{'u_+':>6} {'defect':>12} {'dW_out':>12} {'dW_in':>12} {'dS':>12}")
    for r in runs:
        print(f"{r['u_right']:6.0f} {r['unitary_defect_2norm']:12.4e} "
              f"{r['wronskian_out_relative_variation']:12.4e} "
              f"{r['wronskian_in_relative_variation']:12.4e} "
              f"{r['matching_point_variation_frobenius']:12.4e}")


if __name__ == "__main__":
    main()
