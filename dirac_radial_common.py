#!/usr/bin/env python3
# Authors: Dr. Davide Batic and Dr. Denys Dutykh
#          (Mathematics Department, Khalifa University of Science and
#           Technology, Abu Dhabi, UAE)
"""Shared machinery for the massive Dirac finite-matching benchmarks on
Schwarzschild and Kerr backgrounds.

Companion of the paper

    D. Batic & D. Dutykh, "Two-Channel Dollard Scattering and Partial-Wave
    Resonances for Massive Dirac Fields on Kerr".

Contents
--------
* tortoise maps (Schwarzschild via Lambert W, Kerr via Brent inversion);
* the separated radial matrix A_{kj}(u, omega) in the paper's gauge
  (eq. (VIII.5)):  A = [[-i*Omega, nu*(lam + i*m*r)],
                       [nu*(lam - i*m*r),  i*Omega]],
  with Omega = omega + a*m_k/(r^2+a^2) and nu = sqrt(Delta)/(r^2+a^2);
* the spatial-infinity diagonalizer T(omega), the 1/u matrix B1 = T^{-1}A1 T
  with A1 = [[0, lam - i*m*M], [lam + i*m*M, 0]], and the first transport
  coefficients b_- = (0, B1[1,0]/(-2i*kappa)), b_+ = (B1[0,1]/(2i*kappa), 0)
  (Lemma X.4 of the paper);
* the Dollard phase Phi(u) = kappa*u + alpha*log(u/ell) and the refined phase
  Phi_star(u) = Phi(u) - (2*M*alpha/u)*log(u/ell)  (eq. (II.10)); starting the
  infinity Jost columns with Phi_star removes the universal pure-phase
  log(u)/u error of the unrefined data and restores the quadratic-class
  convergence rate for the phases of the scattering entries;
* the Chandrasekhar-Page angular shooting problem (Kerr) with an adaptive
  eigenvalue bracket;
* finite matching: S = (F_out)^{-1} F_in, determinant Wronskians, unitarity
  defect, oriented flux diagnostics q = Re(v* J0 v), and the conjugation
  identities det F_in = -conj(det F_out), S_12 = S_21 = -1/det F_out
  (Propositions VIII.1 and IX.1 of the paper) used as internal checks;
* JSON / CSV writers.

This module is a smoke-test / reproducibility tool, not a production code.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import lambertw

J0 = np.diag([1.0, -1.0])

# ----------------------------------------------------------------------------
# JSON helpers
# ----------------------------------------------------------------------------

def complex_to_pair(z: complex) -> list[float]:
    return [float(np.real(z)), float(np.imag(z))]


def matrix_to_pairs(m: np.ndarray) -> list[list[list[float]]]:
    return [[complex_to_pair(m[i, j]) for j in range(m.shape[1])]
            for i in range(m.shape[0])]


# ----------------------------------------------------------------------------
# Tortoise coordinates
# ----------------------------------------------------------------------------

def lambertw_exp(x: float) -> float:
    """w = W0(exp(x)), robust for large x (solves w + log w = x)."""
    if x < 700.0:
        return float(np.real(lambertw(math.exp(x))))
    w = x - math.log(x)
    for _ in range(8):
        w = w - (w + math.log(w) - x) / (1.0 + 1.0 / w)
    return w


def schwarzschild_r_of_u(u: float, M: float) -> float:
    """Invert u = r + 2M log(r/(2M) - 1) via Lambert W."""
    return 2.0 * M * (1.0 + lambertw_exp(u / (2.0 * M) - 1.0))


def kerr_horizons(M: float, a: float) -> tuple[float, float]:
    if M <= 0.0 or abs(a) >= M:
        raise ValueError("need M > 0 and |a| < M")
    s = math.sqrt(M * M - a * a)
    return M - s, M + s          # r0 (Cauchy), r1 (event)


@dataclass
class KerrGeometry:
    M: float
    a: float
    k: int
    omega: float
    r0: float = field(init=False)
    r1: float = field(init=False)
    kappa_plus: float = field(init=False)
    Omega_H: float = field(init=False)
    omega_H: float = field(init=False)
    A_t: float = field(init=False)
    B_t: float = field(init=False)

    def __post_init__(self) -> None:
        self.r0, self.r1 = kerr_horizons(self.M, self.a)
        self.kappa_plus = (self.r1 - self.r0) / (2.0 * (self.r1 ** 2 + self.a ** 2))
        self.Omega_H = self.a / (self.r1 ** 2 + self.a ** 2)
        self.omega_H = self.omega + self.Omega_H * (self.k + 0.5)
        self.A_t = (self.r1 ** 2 + self.a ** 2) / (self.r1 - self.r0)
        self.B_t = -(self.r0 ** 2 + self.a ** 2) / (self.r1 - self.r0)

    def u_of_r(self, r: float) -> float:
        return r + self.A_t * math.log(r - self.r1) + self.B_t * math.log(r - self.r0)

    def r_of_u(self, u: float) -> float:
        """Brent inversion of the Kerr tortoise map (double precision limits
        the usable range to u >= about -40 for moderate spins)."""
        lo = self.r1 + 1e-13
        hi = self.r1 + max(10.0, abs(u)) + 10.0
        while self.u_of_r(hi) < u:
            hi *= 2.0
        return brentq(lambda r: self.u_of_r(r) - u, lo, hi,
                      xtol=1e-13, rtol=1e-13)


# ----------------------------------------------------------------------------
# Radial system
# ----------------------------------------------------------------------------

def radial_matrix(r: float, omega: float, mass: float, lam: float,
                  M: float, a: float, m_k: float) -> np.ndarray:
    """A_{kj}(u, omega) of eq. (VIII.5), evaluated at r = r(u)."""
    Delta = r * r - 2.0 * M * r + a * a
    nu = math.sqrt(Delta) / (r * r + a * a)
    Omega = omega + a * m_k / (r * r + a * a)
    return np.array([[-1j * Omega, nu * (lam + 1j * mass * r)],
                     [nu * (lam - 1j * mass * r), 1j * Omega]])


# ----------------------------------------------------------------------------
# Spatial-infinity diagonalizer, transport data, Dollard phases
# ----------------------------------------------------------------------------

@dataclass
class InfinityTransport:
    M: float
    mass: float
    omega: float
    lam: float
    ell: float = 1.0
    kappa: float = field(init=False)
    alpha: float = field(init=False)
    T: np.ndarray = field(init=False)
    Tinv: np.ndarray = field(init=False)
    B1: np.ndarray = field(init=False)
    b_minus: np.ndarray = field(init=False)
    b_plus: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if abs(self.omega) <= self.mass:
            raise ValueError("open channel requires |omega| > mass")
        self.kappa = math.copysign(1.0, self.omega) * math.sqrt(
            self.omega ** 2 - self.mass ** 2)
        self.alpha = self.M * self.mass ** 2 / self.kappa
        theta = 0.25 * math.log((self.omega + self.mass) / (self.omega - self.mass))
        c, s = math.cosh(theta), math.sinh(theta)
        self.T = np.array([[c, s], [s, c]])
        self.Tinv = np.array([[c, -s], [-s, c]])
        A1 = np.array([[0.0, self.lam - 1j * self.mass * self.M],
                       [self.lam + 1j * self.mass * self.M, 0.0]])
        self.B1 = self.Tinv @ A1 @ self.T
        self.b_minus = np.array([0.0, self.B1[1, 0] / (-2j * self.kappa)])
        self.b_plus = np.array([self.B1[0, 1] / (2j * self.kappa), 0.0])

    def phi(self, u: float) -> float:
        """Unrefined Dollard phase Phi(u) = kappa u + alpha log(u/ell)."""
        return self.kappa * u + self.alpha * math.log(u / self.ell)

    def phi_star(self, u: float) -> float:
        """Refined phase Phi_star = Phi - (2 M alpha / u) log(u/ell)
        (eq. (II.10) of the paper): the exact r-variable Coulomb phase
        transported to the tortoise variable."""
        return self.phi(u) - (2.0 * self.M * self.alpha / u) * math.log(u / self.ell)

    def infinity_columns(self, u_right: float, refined: bool = False
                         ) -> tuple[np.ndarray, np.ndarray]:
        """First-order Jost starting data at u = u_right for the outgoing
        (e^{-i Phi} e_-) and incoming (e^{+i Phi} e_+) columns; with
        refined=True the phase Phi_star is used instead of Phi."""
        ph = self.phi_star(u_right) if refined else self.phi(u_right)
        e_minus = np.array([1.0, 0.0]) + self.b_minus / u_right
        e_plus = np.array([0.0, 1.0]) + self.b_plus / u_right
        col_out = self.T @ (np.exp(-1j * ph) * e_minus)
        col_in = self.T @ (np.exp(+1j * ph) * e_plus)
        return col_out, col_in


# ----------------------------------------------------------------------------
# Chandrasekhar-Page angular problem (Kerr; Batic-Schmid gauge)
# ----------------------------------------------------------------------------

def angular_rhs(theta: float, y: np.ndarray, lam: float, varkappa: float,
                mu: float, nu: float) -> np.ndarray:
    s, c = math.sin(theta), math.cos(theta)
    S1, S2 = y
    return np.array([(-varkappa / s - nu * s) * S1 + (mu * c - lam) * S2,
                     (lam + mu * c) * S1 + (varkappa / s + nu * s) * S2])


def angular_det(lam: float, varkappa: float, mu: float, nu: float,
                eps: float = 1e-6, rtol: float = 1e-10,
                atol: float = 1e-12) -> float:
    """Two-sided shooting determinant at theta = pi/2 with pole-regular data."""
    den = 2.0 * varkappa + 1.0
    yL = np.array([((mu - lam) / den) * eps ** (varkappa + 1.0), eps ** varkappa])
    yR = np.array([eps ** varkappa, ((mu - lam) / den) * eps ** (varkappa + 1.0)])
    solL = solve_ivp(angular_rhs, (eps, math.pi / 2.0), yL,
                     args=(lam, varkappa, mu, nu), rtol=rtol, atol=atol,
                     method="DOP853")
    solR = solve_ivp(angular_rhs, (math.pi - eps, math.pi / 2.0), yR,
                     args=(lam, varkappa, mu, nu), rtol=rtol, atol=atol,
                     method="DOP853")
    a, b = solL.y[:, -1], solR.y[:, -1]
    return float(a[0] * b[1] - a[1] * b[0])


def angular_lambda(varkappa: float, mu: float, nu: float,
                   bracket: tuple[float, float] = (0.7, 1.6),
                   eps: float = 1e-6, rtol: float = 1e-10,
                   atol: float = 1e-12) -> tuple[float, float]:
    """Simple Chandrasekhar-Page eigenvalue in (or near) the given bracket.
    If the bracket does not straddle a sign change, it is widened by scanning
    (adaptive bracket; see the repository README)."""
    lo, hi = bracket
    f = lambda lam: angular_det(lam, varkappa, mu, nu, eps, rtol, atol)
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0.0:
        grid = np.linspace(max(0.05, lo - 2.0), hi + 2.0, 61)
        vals = [f(x) for x in grid]
        for x0, x1, v0, v1 in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
            if v0 * v1 <= 0.0:
                lo, hi, flo, fhi = x0, x1, v0, v1
                break
        else:
            raise RuntimeError("no angular eigenvalue bracket found")
    lam = brentq(f, lo, hi, xtol=1e-12, rtol=1e-12)
    return float(lam), f(lam)


# ----------------------------------------------------------------------------
# Matching machinery
# ----------------------------------------------------------------------------

def integrate_columns(rhs, u_from: float, u_to: float, cols: np.ndarray,
                      rtol: float, atol: float, dense: bool = True):
    y0 = np.asarray(cols, dtype=complex).reshape(-1)
    sol = solve_ivp(rhs, (u_from, u_to), y0, method="DOP853",
                    dense_output=dense, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"radial integration failed: {sol.message}")
    return sol


def columns_at(sol, u: float) -> np.ndarray:
    return sol.sol(u).reshape(2, -1)


def det_w(Y: np.ndarray, Z: np.ndarray) -> complex:
    """Alternating (determinant) Wronskian W(Y, Z) = Y_- Z_+ - Y_+ Z_-."""
    return Y[0] * Z[1] - Y[1] * Z[0]


def q_flux(v: np.ndarray) -> float:
    """Oriented radial charge flux q(v) = Re(v* J0 v)."""
    return float(np.real(np.conj(v) @ (J0 @ v)))


def scattering_matrix_at(u: float, sol_H, sol_I) -> np.ndarray:
    """S = (F_out)^{-1} F_in from the four Jost columns at the point u.
    Columns of sol_H: (R_H_in, R_H_out); of sol_I: (R_inf_out, R_inf_in)."""
    YH = columns_at(sol_H, u)
    YI = columns_at(sol_I, u)
    F_in = np.column_stack([YH[:, 0], YI[:, 1]])
    F_out = np.column_stack([YH[:, 1], YI[:, 0]])
    return np.linalg.solve(F_out, F_in), F_in, F_out


def unitarity_defect(S: np.ndarray) -> float:
    return float(np.linalg.norm(np.conj(S.T) @ S - np.eye(2), 2))


def conjugation_checks(F_in: np.ndarray, F_out: np.ndarray,
                       S: np.ndarray) -> dict[str, float]:
    """Internal checks from the paper's conjugation identities:
    det F_in = -conj(det F_out)  (Prop. VIII.1)  and
    S_12 = S_21 = -1/det F_out   (Prop. IX.1)."""
    dFi, dFo = np.linalg.det(F_in), np.linalg.det(F_out)
    return {
        "det_conjugation_defect": float(abs(dFi + np.conj(dFo))),
        "reciprocity_defect": float(abs(S[0, 1] - S[1, 0])),
        "transmission_identity_defect": float(abs(S[0, 1] + 1.0 / dFo)),
    }


# ----------------------------------------------------------------------------
# Output writers
# ----------------------------------------------------------------------------

def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1))
    print(f"  wrote {path}")


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"  wrote {path}")
