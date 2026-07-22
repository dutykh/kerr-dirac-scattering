#!/usr/bin/env python3
# Authors: Dr. Davide Batic and Dr. Denys Dutykh
#          (Mathematics Department, Khalifa University of Science and
#           Technology, Abu Dhabi, UAE)
"""Finite-endpoint convergence study for the Dirac finite-matching benchmarks.

For Schwarzschild and Kerr benchmark parameters, and for the two phase
conventions of the infinity start data — the Dollard phase Phi and the
refined phase Phi_star of eq. (II.10) — this script records, as functions of
the endpoint u_+:

* the unitarity defect  ||S* S - I||_2                    (gauge invariant),
* the transmission-phase drift  |arg S_12(u_+) - arg S_12(u_ref)|
  relative to a far reference endpoint                    (gauge sensitive).

It demonstrates the two-tier convergence stated in Section XI of the paper:
every channel-gauge-invariant diagnostic (defect, Wronskian conservation,
entry moduli, det/reciprocity identities) converges at the quadratic-class
rate u_+^{-2} and is IDENTICAL between the two phase conventions to machine
precision, whereas the phases of individual entries are pure channel-gauge
content: they converge only at the rate u_+^{-1} log u_+ and shift between
the conventions at exactly that order. Feeds figure fig06 of the paper.

Output: results/convergence_study.json (+ CSV).
"""

from __future__ import annotations

import cmath
import json
from pathlib import Path

import numpy as np

import dirac_radial_common as drc
import schwarzschild_dirac_matching as schw
import kerr_dirac_matching as kerr


U_RIGHTS = [60.0, 80.0, 110.0, 150.0, 200.0, 270.0, 360.0]
U_REF = 480.0


def phase(z: complex) -> float:
    return cmath.phase(z)


def schwarzschild_series(refined: bool) -> dict:
    p = schw.Params(refined=refined)
    tr = drc.InfinityTransport(p.M, p.mass, p.omega, p.angular_lambda,
                               p.ell_infinity)
    rows = []
    ref = schw.run_once(p, tr, U_REF)
    ref_phase = phase(complex(*ref["S"][0][1]))
    for ur in U_RIGHTS:
        r = schw.run_once(p, tr, ur)
        rows.append({
            "u_right": ur,
            "unitary_defect": r["unitary_defect_2norm"],
            "phase_drift_S12": abs(phase(complex(*r["S"][0][1])) - ref_phase),
        })
    return {"refined": refined, "u_ref": U_REF, "rows": rows}


def kerr_series(refined: bool) -> dict:
    p = kerr.Params(refined=refined)
    geo = drc.KerrGeometry(p.M, p.a, p.k, p.omega)
    m_k = p.k + 0.5
    lam, _ = drc.angular_lambda(m_k, p.a * p.mass, p.a * p.omega,
                                bracket=(p.lambda_bracket_left,
                                         p.lambda_bracket_right),
                                eps=p.angular_epsilon,
                                rtol=p.angular_rtol, atol=p.angular_atol)
    tr = drc.InfinityTransport(p.M, p.mass, p.omega, lam, p.ell_infinity)
    rows = []
    ref = kerr.run_once(p, geo, tr, lam, U_REF)
    ref_phase = phase(complex(*ref["S"][0][1]))
    for ur in U_RIGHTS:
        r = kerr.run_once(p, geo, tr, lam, ur)
        rows.append({
            "u_right": ur,
            "unitary_defect": r["unitary_defect_2norm"],
            "phase_drift_S12": abs(phase(complex(*r["S"][0][1])) - ref_phase),
        })
    return {"refined": refined, "u_ref": U_REF, "rows": rows}


def main() -> None:
    payload = {
        "description": "finite-endpoint convergence of the matching benchmarks",
        "u_rights": U_RIGHTS,
        "schwarzschild": {
            "first_order": schwarzschild_series(False),
            "refined": schwarzschild_series(True),
        },
        "kerr": {
            "first_order": kerr_series(False),
            "refined": kerr_series(True),
        },
    }
    out = Path("results")
    drc.write_json(out / "convergence_study.json", payload)
    rows = []
    for geom in ("schwarzschild", "kerr"):
        for tag in ("first_order", "refined"):
            for r in payload[geom][tag]["rows"]:
                rows.append([geom, tag, r["u_right"], r["unitary_defect"],
                             r["phase_drift_S12"]])
    drc.write_csv(out / "convergence_study.csv",
                  ["geometry", "order", "u_right", "unitary_defect",
                   "phase_drift_S12"], rows)
    for geom in ("schwarzschild", "kerr"):
        print(f"--- {geom} ---")
        print(f"{'u_+':>6} {'defect(1st)':>12} {'phase(1st)':>12} "
              f"{'defect(ref)':>12} {'phase(ref)':>12}")
        for r1, r2 in zip(payload[geom]["first_order"]["rows"],
                          payload[geom]["refined"]["rows"]):
            print(f"{r1['u_right']:6.0f} {r1['unitary_defect']:12.4e} "
                  f"{r1['phase_drift_S12']:12.4e} {r2['unitary_defect']:12.4e} "
                  f"{r2['phase_drift_S12']:12.4e}")


if __name__ == "__main__":
    main()
