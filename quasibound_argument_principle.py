#!/usr/bin/env python3
"""Argument-principle certification for the Schwarzschild quasi-bound Evans zeros.

The script evaluates the same finite-endpoint closed-channel Evans determinant
used in the manuscript and computes its winding number on positively oriented
rectangular contours.  The stable numerical winding count estimates the number of zeros
inside each contour, provided the determinant is nonzero on the boundary.

This is a reproducibility/certification calculation, not a proof uniform in the
finite endpoints.  The calculation is repeated at two endpoint/tolerance
settings and at two contour resolutions.
"""
from __future__ import annotations

import cmath
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import lambertw

M = 1.0
MASS = 0.2
LAM = 1.0
M_K = 0.5
U_LEFT = -40.0
U_MATCH = 10.0
ELL = 1.0
A1 = np.array(
    [[0.0, LAM - 1j * MASS * M], [LAM + 1j * MASS * M, 0.0]],
    dtype=np.complex128,
)


@dataclass(frozen=True)
class SolverConfig:
    u_right: float
    rtol: float
    atol: float


@dataclass(frozen=True)
class Rectangle:
    name: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float


CONFIGS = [
    SolverConfig(u_right=240.0, rtol=1.0e-10, atol=1.0e-11),
    SolverConfig(u_right=320.0, rtol=1.0e-11, atol=1.0e-12),
]

# The union window is deliberately bounded away from the threshold omega=m_e.
# It certifies the two reported near-threshold zeros, not a global enumeration
# of the whole mass gap or of the infinite threshold-accumulating sequence.
RECTANGLES = [
    Rectangle("level_1_box", 0.1935, 0.1964, 0.0, 0.00135),
    Rectangle("level_2_box", 0.1978, 0.19918, 0.0, 0.00032),
    Rectangle("two_level_window", 0.1870, 0.19918, 0.0, 0.00135),
]


def lambertw_exp(x: float) -> float:
    """Return W_0(exp(x)) on the real branch without overflow."""
    if x < 700.0:
        return float(np.real(lambertw(math.exp(x))))
    w = x - math.log(x)
    for _ in range(8):
        w -= (w + math.log(w) - x) / (1.0 + 1.0 / w)
    return w


def schwarzschild_r_of_u(u: float) -> float:
    return 2.0 * M * (1.0 + lambertw_exp(u / (2.0 * M) - 1.0))


def radial_matrix(u: float, omega: complex) -> np.ndarray:
    r = schwarzschild_r_of_u(u)
    delta = r * (r - 2.0 * M)
    nu = math.sqrt(delta) / (r * r)
    return np.array(
        [
            [-1j * omega, nu * (LAM + 1j * MASS * r)],
            [nu * (LAM - 1j * MASS * r), 1j * omega],
        ],
        dtype=np.complex128,
    )


def closed_beta(omega: complex) -> complex:
    beta = cmath.sqrt(MASS * MASS - omega * omega)
    if beta.real < 0.0:
        beta = -beta
    return beta


def decaying_start(omega: complex, u_right: float) -> np.ndarray:
    beta = closed_beta(omega)
    if beta.real <= 0.0:
        raise ValueError(f"decaying sheet requires Re beta > 0, got {beta}")
    alpha_t = M * MASS * MASS / beta
    # Analytic eigenvector gauge on the chosen beta sheet.  Using
    # numpy.linalg.eig here would introduce arbitrary phase jumps along a
    # contour and would invalidate an argument-principle winding count.
    v_minus = np.array([-1j * MASS, beta - 1j * omega], dtype=np.complex128)
    v_plus = np.array([+1j * MASS, beta + 1j * omega], dtype=np.complex128)
    rhs = (alpha_t * np.eye(2) - A1) @ v_minus
    coeff = np.linalg.solve(np.column_stack([v_minus, v_plus]), rhs)
    if abs(coeff[0]) >= 1.0e-8 * max(1.0, abs(coeff[1])):
        raise RuntimeError(f"closed-channel transport solvability failed: {coeff[0]}")
    b_dec = (coeff[1] / (2.0 * beta)) * v_plus
    scale = cmath.exp(-beta * u_right) * (u_right / ELL) ** alpha_t
    return scale * (v_minus + b_dec / u_right)


def integrate_column(omega: complex, cfg: SolverConfig) -> tuple[np.ndarray, np.ndarray]:
    def rhs(u: float, y: np.ndarray) -> np.ndarray:
        return radial_matrix(u, omega) @ y

    y_h = cmath.exp(1j * omega * U_LEFT) * np.array([0.0, 1.0], dtype=np.complex128)
    y_d = decaying_start(omega, cfg.u_right)
    sol_h = solve_ivp(
        rhs,
        (U_LEFT, U_MATCH),
        y_h,
        method="DOP853",
        rtol=cfg.rtol,
        atol=cfg.atol,
    )
    sol_d = solve_ivp(
        rhs,
        (cfg.u_right, U_MATCH),
        y_d,
        method="DOP853",
        rtol=cfg.rtol,
        atol=cfg.atol,
    )
    if not sol_h.success or not sol_d.success:
        raise RuntimeError(f"ODE failure: {sol_h.message}; {sol_d.message}")
    return sol_h.y[:, -1], sol_d.y[:, -1]


def evans_dec(omega: complex, cfg: SolverConfig) -> tuple[complex, float]:
    y_h, y_d = integrate_column(omega, cfg)
    determinant = complex(y_h[0] * y_d[1] - y_h[1] * y_d[0])
    relative = abs(determinant) / (np.linalg.norm(y_h) * np.linalg.norm(y_d))
    return determinant, float(relative)


def rectangle_contour(rect: Rectangle, points_per_edge: int) -> np.ndarray:
    """Counterclockwise contour, without duplicating corner points."""
    n = int(points_per_edge)
    if n < 4:
        raise ValueError("points_per_edge must be at least four")
    bottom = np.linspace(rect.x_min, rect.x_max, n, endpoint=False) + 1j * rect.y_min
    right = rect.x_max + 1j * np.linspace(rect.y_min, rect.y_max, n, endpoint=False)
    top = np.linspace(rect.x_max, rect.x_min, n, endpoint=False) + 1j * rect.y_max
    left = rect.x_min + 1j * np.linspace(rect.y_max, rect.y_min, n, endpoint=False)
    return np.concatenate([bottom, right, top, left]).astype(np.complex128)


def winding_count(values: np.ndarray) -> tuple[int, float, float]:
    """Return integer winding, raw winding, and max adjacent phase increment."""
    if np.any(values == 0):
        raise RuntimeError("Evans determinant vanished at a contour sample")
    closed = np.concatenate([values, values[:1]])
    ratios = closed[1:] / closed[:-1]
    increments = np.angle(ratios)
    raw = float(np.sum(increments) / (2.0 * math.pi))
    integer = int(np.rint(raw))
    return integer, raw, float(np.max(np.abs(increments)))


def certify_rectangle(rect: Rectangle, cfg: SolverConfig, points_per_edge: int) -> dict:
    contour = rectangle_contour(rect, points_per_edge)
    values: list[complex] = []
    relative: list[float] = []
    for z in contour:
        d, rel = evans_dec(complex(z), cfg)
        values.append(d)
        relative.append(rel)
    arr = np.asarray(values, dtype=np.complex128)
    count, raw, max_step = winding_count(arr)
    return {
        "rectangle": asdict(rect),
        "solver": asdict(cfg),
        "points_per_edge": points_per_edge,
        "contour_samples": int(len(contour)),
        "zero_count": count,
        "raw_winding": raw,
        "integer_distance": abs(raw - count),
        "min_abs_evans": float(np.min(np.abs(arr))),
        "min_relative_evans": float(np.min(relative)),
        "max_phase_increment": max_step,
        "max_phase_increment_over_pi": max_step / math.pi,
    }


def main() -> None:
    results: list[dict] = []
    for cfg in CONFIGS:
        for rect in RECTANGLES:
            for n in (24, 48):
                result = certify_rectangle(rect, cfg, n)
                results.append(result)
                print(
                    f"{rect.name:18s} u+={cfg.u_right:3.0f} n={n:2d}: "
                    f"N={result['zero_count']:2d}, raw={result['raw_winding']:+.12f}, "
                    f"min rel={result['min_relative_evans']:.3e}, "
                    f"max darg/pi={result['max_phase_increment_over_pi']:.3f}"
                )

    # Gate: counts must be resolution- and solver-stable, boundary must stay
    # separated from zero, and no adjacent phase increment may approach pi.
    grouped: dict[str, list[dict]] = {}
    for item in results:
        grouped.setdefault(item["rectangle"]["name"], []).append(item)
    expected = {"level_1_box": 1, "level_2_box": 1, "two_level_window": 2}
    gates = {}
    for name, items in grouped.items():
        counts = {item["zero_count"] for item in items}
        gates[name] = {
            "stable_integer_count": counts == {expected[name]},
            "min_relative_evans": min(item["min_relative_evans"] for item in items),
            "max_integer_distance": max(item["integer_distance"] for item in items),
            "max_phase_increment_over_pi": max(
                item["max_phase_increment_over_pi"] for item in items
            ),
        }
        gates[name]["certified"] = bool(
            gates[name]["stable_integer_count"]
            and gates[name]["min_relative_evans"] > 1.0e-5
            and gates[name]["max_integer_distance"] < 1.0e-6
            and gates[name]["max_phase_increment_over_pi"] < 0.80
        )
    certified = all(g["certified"] for g in gates.values())
    payload = {
        "description": "argument-principle count for the two reported Schwarzschild quasi-bound Evans zeros",
        "scope": (
            "Counts zeros only in the stated rectangles and near-threshold window; "
            "does not enumerate the whole mass gap or the threshold-accumulating sequence."
        ),
        "results": results,
        "gates": gates,
        "numerically_certified": certified,
    }
    out = Path("results") / "quasibound_argument_principle.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"numerically_certified={certified}; wrote {out}")
    if not certified:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
