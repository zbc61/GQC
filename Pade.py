# -*- coding: utf-8 -*-
"""
Author: Beichen Zheng

Chiba Case C probability table with NRA error diagnostics.

This script builds groupwise total and partial probability tables,
evaluates the PT approximation against an NRA reference curve,
and writes plots plus summary files.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter, LogFormatterMathtext
import matplotlib.ticker as mticker

plt.rcParams["axes.grid"] = False

from scipy.linalg import solve, svd, hankel
from scipy.linalg import LinAlgWarning
from scipy.optimize import linear_sum_assignment

from numpy.polynomial.polynomial import Polynomial


XLAB_G = r"Energy-group index $g$"
XLAB_SIG0 = r"Background (dilution) cross section, $\sigma_0$ (b)"
YLAB_ERR_PCM = r"Relative error $e_g(\sigma_0)$ (pcm)"
YLAB_ERR_REL = r"Relative error (dimensionless)"
TITLE_FAMILY = r"Pad\'e--Stieltjes error"

WARN_AS_ERROR = False
RECORD_WARNINGS = True

IMAG_TOL = 1e-12
NEG_TOL = -1e-14
MIN_FINITE_ERR_POINTS = 3

ERR_QUANTILES = (0.50, 0.90, 0.95, 0.99)
SCATTER_S = 10

if WARN_AS_ERROR:
    warnings.filterwarnings("error", category=LinAlgWarning)


def _max_abs_imag(x: np.ndarray) -> float:
    x = np.asarray(x)
    if np.iscomplexobj(x):
        return float(np.nanmax(np.abs(np.imag(x)))) if x.size else 0.0
    return 0.0


def _min_real(x: np.ndarray) -> float:
    x = np.asarray(x)
    if x.size == 0:
        return float("nan")
    return float(np.nanmin(np.real(x)))


def is_complex_flag(x: np.ndarray, tol: float = IMAG_TOL) -> tuple[bool, float]:
    mi = _max_abs_imag(x)
    return (np.isfinite(mi) and mi > tol), float(mi)


def warn_if_complex(x: np.ndarray, *, name: str, group_id: int, tol: float = IMAG_TOL) -> bool:
    flag, mi = is_complex_flag(x, tol=tol)
    if flag:
        print(f"[WARNING][Group {group_id}] {name}: max|Im| = {mi:.3e}")
        return True
    return False


def format_array_maybe_real(x: np.ndarray, *, group_id: int, name: str, tol: float = IMAG_TOL) -> str:
    x = np.asarray(x)
    if not np.iscomplexobj(x):
        return np.array2string(x, precision=10, suppress_small=False)

    mi = _max_abs_imag(x)
    if np.isfinite(mi) and mi <= tol:
        xr = np.real(x)
        return np.array2string(xr, precision=10, suppress_small=False)

    warn_if_complex(x, name=name, group_id=group_id, tol=tol)
    return np.array2string(x, precision=10, suppress_small=False)


def as_real_for_plot(y: np.ndarray, *, name: str, group_id: int, tol: float = IMAG_TOL) -> np.ndarray:
    y = np.asarray(y)
    if not np.iscomplexobj(y):
        return y
    mi = _max_abs_imag(y)
    if np.isfinite(mi) and mi > tol:
        print(f"[WARNING][Group {group_id}] {name} is complex: max|Im| = {mi:.3e}. Plot uses Re({name}).")
    return np.real(y)


def read_cross_sections(file_path: Path) -> np.ndarray:
    """Read (E, sigma) pairs from a text file."""
    file_path = Path(file_path)
    rows = []
    try:
        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                ln = line.strip()
                if not ln or ln.startswith("#") or ln.startswith("!"):
                    continue
                parts = ln.split()
                if len(parts) < 2:
                    continue
                try:
                    e = float(parts[0])
                    s = float(parts[1])
                    if np.isfinite(e) and np.isfinite(s):
                        rows.append((e, s))
                except Exception:
                    continue
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")

    if not rows:
        return np.empty((0, 2), dtype=float)

    arr = np.asarray(rows, dtype=float)
    arr = arr[np.argsort(arr[:, 0])]
    return arr


def read_energy_structure_1col(path: Path) -> np.ndarray:
    """Read energy-group boundaries from a one-column text file."""
    path = Path(path)
    vals = []
    for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#") or ln.startswith("!"):
            continue
        tok = ln.split()[0]
        try:
            vals.append(float(tok))
        except Exception:
            continue

    edges = np.asarray(vals, dtype=float)
    edges = edges[np.isfinite(edges)]
    edges = np.unique(np.sort(edges))
    if edges.size < 2:
        raise ValueError(f"Need at least 2 distinct numeric boundaries in {path}")
    return edges


def chiba_case_c_orders(N: int) -> np.ndarray:
    """Return Case C affine moment orders for a given N."""
    if N < 1:
        raise ValueError("N must be >= 1")
    if N == 1:
        return np.array([-1.0, 0.0], dtype=float)
    b = 1.0 / (N - 1)
    orders = -1.0 + b * np.arange(2 * N, dtype=float)
    orders[np.isclose(orders, 0.0)] = 0.0
    return orders


def _cond2_via_svd(A: np.ndarray) -> tuple[float, float]:
    s = svd(A, compute_uv=False)
    if s.size == 0:
        return float("inf"), 0.0
    smax = float(np.max(s))
    smin = float(np.min(s))
    if (not np.isfinite(smax)) or smax == 0.0:
        return float("inf"), 0.0
    if (not np.isfinite(smin)) or smin == 0.0:
        return float("inf"), 0.0
    cond2 = smax / smin
    return float(cond2), float(1.0 / cond2)


def _infer_affine_params(orders: np.ndarray) -> tuple[float, float]:
    orders = np.asarray(orders, dtype=float).ravel()
    if orders.size < 2:
        raise ValueError("Need at least 2 orders.")
    a = float(orders[0])
    b = float(orders[1] - orders[0])
    k = np.arange(orders.size, dtype=float)
    if not np.allclose(orders, a + b * k, atol=1e-12, rtol=0.0):
        raise ValueError("orders is not affine.")
    if b == 0.0:
        raise ValueError("b=0 is invalid.")
    return a, b


def solve_with_warning_tag(A: np.ndarray, b: np.ndarray, *, tag: str, group_id: int):
    if not RECORD_WARNINGS:
        return solve(A, b)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", LinAlgWarning)
        x = solve(A, b)
        for wi in w:
            if issubclass(wi.category, LinAlgWarning):
                print(f"[LinAlgWarning][Group {group_id}][{tag}] {wi.message}  @ {wi.filename}:{wi.lineno}")
        return x


def lstsq_fallback(A: np.ndarray, b: np.ndarray, *, rcond: float = 1e-12):
    return np.linalg.lstsq(A, b, rcond=rcond)[0]


def _order_key(x: float) -> str:
    return f"{float(x):.12g}"


def _safe_rel_residual(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    """Return ||Ax-b||_2 / ||b||_2."""
    A = np.asarray(A)
    x = np.asarray(x)
    b = np.asarray(b)
    r = A @ x - b
    nb = float(np.linalg.norm(b))
    nr = float(np.linalg.norm(r))
    if nb == 0.0:
        return float("inf") if nr > 0 else 0.0
    return float(nr / nb)


def _min_pairwise_distance(z: np.ndarray) -> float:
    z = np.asarray(z).ravel()
    n = z.size
    if n <= 1:
        return float("nan")
    D = np.abs(z.reshape(-1, 1) - z.reshape(1, -1))
    D = D + np.eye(n) * np.inf
    return float(np.min(D))


def _root_sensitivity_metrics(Qcoef: np.ndarray, rho: np.ndarray) -> tuple[float, float]:
    """Return min |Q'(rho_i)| and max 1/|Q'(rho_i)|."""
    rho = np.asarray(rho).ravel()
    if rho.size == 0:
        return float("nan"), float("nan")
    Qpoly = Polynomial(Qcoef)
    Qp = Qpoly.deriv()
    vals = Qp(rho)
    absvals = np.abs(vals)
    finite = np.isfinite(absvals) & (absvals > 0)
    if not np.any(finite):
        return float("nan"), float("inf")
    min_abs = float(np.min(absvals[finite]))
    max_inv = float(np.max(1.0 / absvals[finite]))
    return min_abs, max_inv


def _match_by_hungarian(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Match b to a by minimizing sum |a_i-b_j|."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    n = a.size
    if b.size != n:
        raise ValueError("matching requires equal lengths")
    C = np.abs(a.reshape(-1, 1) - b.reshape(1, -1))
    row, col = linear_sum_assignment(C)
    idx = col[np.argsort(row)]
    return idx


def _is_finite_all(x: np.ndarray) -> bool:
    x = np.asarray(x)
    if np.iscomplexobj(x):
        return bool(np.all(np.isfinite(np.real(x))) and np.all(np.isfinite(np.imag(x))))
    return bool(np.all(np.isfinite(x)))


def compute_error_pcm_curve(
    ref_curve: np.ndarray,
    pt_curve: np.ndarray,
    *,
    group_id: int,
) -> np.ndarray:
    """Compute error curve in pcm."""
    ref_curve = np.asarray(ref_curve, dtype=float)
    pt_plot = as_real_for_plot(pt_curve, name="PT_eff_x", group_id=group_id, tol=IMAG_TOL)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        err_pcm = (pt_plot - ref_curve) / ref_curve * 1e5
    return np.asarray(err_pcm, dtype=float)


def error_stats_from_err_pcm(
    err_pcm: np.ndarray,
    *,
    quantiles: tuple[float, ...] = ERR_QUANTILES,
    min_finite: int = MIN_FINITE_ERR_POINTS,
) -> tuple[bool, dict]:
    """Return max, RMS, and |error| quantiles on finite points."""
    e = np.asarray(err_pcm, dtype=float)
    fin = np.isfinite(e)
    nfin = int(np.sum(fin))
    if nfin < int(min_finite):
        return False, {
            "max_abs_err_pcm": float("nan"),
            "rms_err_pcm": float("nan"),
            "q_abs_err_pcm": np.full(len(quantiles), np.nan, dtype=float),
            "n_finite": nfin,
        }

    ef = e[fin]
    abs_ef = np.abs(ef)

    max_abs = float(np.max(abs_ef)) if abs_ef.size else float("nan")
    rms = float(np.sqrt(np.mean(ef * ef))) if ef.size else float("nan")

    try:
        q = np.quantile(abs_ef, quantiles) if abs_ef.size else np.full(len(quantiles), np.nan, dtype=float)
        q = np.asarray(q, dtype=float).ravel()
    except Exception:
        q = np.full(len(quantiles), np.nan, dtype=float)

    ok = np.isfinite(max_abs) and np.isfinite(rms)
    return ok, {
        "max_abs_err_pcm": max_abs,
        "rms_err_pcm": rms,
        "q_abs_err_pcm": q,
        "n_finite": nfin,
    }


def _err_pcm_stats_if_usable(
    ref_curve: np.ndarray,
    pt_curve: np.ndarray,
    *,
    group_id: int,
    min_finite: int = MIN_FINITE_ERR_POINTS,
) -> tuple[bool, np.ndarray, dict]:
    if not _is_finite_all(ref_curve):
        return False, np.asarray([]), {}

    err_pcm = compute_error_pcm_curve(ref_curve, pt_curve, group_id=group_id)
    ok, stats = error_stats_from_err_pcm(err_pcm, min_finite=min_finite)
    return ok, err_pcm, stats


def compute_group_moments_average(
    XS_t: np.ndarray,
    edges_asc: np.ndarray,
    orders: np.ndarray,
    *,
    XS_x: np.ndarray | None = None,
    side: str = "right",
    assume_sorted: bool = True,
):
    """
    Compute group-averaged moments <sigma_t^n> using trapezoid integration
    on the union grid of in-group nodes.
    """
    XS_t = np.asarray(XS_t, dtype=float)
    orders = np.asarray(orders, dtype=float).ravel()
    edges = np.asarray(edges_asc, dtype=float).ravel()

    if XS_t.ndim != 2 or XS_t.shape[1] < 2:
        raise ValueError("XS_t must be (n,2) array [E, sigma_t].")

    if XS_x is not None:
        XS_x = np.asarray(XS_x, dtype=float)
        if XS_x.ndim != 2 or XS_x.shape[1] < 2:
            raise ValueError("XS_x must be (n,2) array [E, sigma_x] if provided.")

    if not assume_sorted:
        XS_t = XS_t[np.argsort(XS_t[:, 0])]
        if XS_x is not None:
            XS_x = XS_x[np.argsort(XS_x[:, 0])]

    Et = XS_t[:, 0].astype(float)
    St = XS_t[:, 1].astype(float)

    if XS_x is not None:
        Ex = XS_x[:, 0].astype(float)
    else:
        Ex = None

    edges = edges[np.isfinite(edges)]
    edges = np.unique(np.sort(edges))
    if edges.size < 2:
        raise ValueError("Energy structure must have >=2 boundaries.")

    G = edges.size - 1
    K = orders.size

    denom = (edges[1:] - edges[:-1]).astype(float)
    moments_num = np.zeros((G, K), dtype=float)

    g_low_pts = np.searchsorted(edges, Et, side=side) - 1
    mask_pts = (g_low_pts >= 0) & (g_low_pts < G)
    counts_low = np.bincount(g_low_pts[mask_pts], minlength=G)

    for g in range(G):
        eL, eR = float(edges[g]), float(edges[g + 1])
        if eR <= eL:
            continue

        mask_t = (Et >= eL) & (Et <= eR)
        if Ex is not None:
            mask_x = (Ex >= eL) & (Ex <= eR)
            nodes = np.concatenate(([eL, eR], Et[mask_t], Ex[mask_x]))
        else:
            nodes = np.concatenate(([eL, eR], Et[mask_t]))

        nodes = np.unique(nodes)
        nodes = nodes[(nodes >= eL) & (nodes <= eR)]
        if nodes.size < 2:
            continue

        st = np.interp(nodes, Et, St)

        dx = nodes[1:] - nodes[:-1]
        stL, stR = st[:-1], st[1:]

        ok_base = (dx > 0) & (stL > 0) & (stR > 0)
        if not np.any(ok_base):
            continue

        dx_ok = dx[ok_base]
        stL_ok = stL[ok_base]
        stR_ok = stR[ok_base]

        for k, n in enumerate(orders):
            fL = stL_ok ** n
            fR = stR_ok ** n
            moments_num[g, k] = float(np.sum(0.5 * (fL + fR) * dx_ok))

    with np.errstate(divide="ignore", invalid="ignore"):
        moments_low = np.where(denom[:, None] > 0, moments_num / denom[:, None], 0.0)

    return moments_low[::-1, :], counts_low[::-1], edges[::-1], denom[::-1]


def crop_edges_to_overlap_with_chiba(edges_asc: np.ndarray) -> np.ndarray:
    """Crop ES=2 boundaries to the Chiba overlap range."""
    edges = np.asarray(edges_asc, dtype=float).ravel()
    edges = edges[np.isfinite(edges)]
    edges = np.unique(np.sort(edges))
    if edges.size < 2:
        raise ValueError("Energy structure must have >=2 valid boundaries.")

    E_hi = 9.1188e3
    E_lo = 5.0435

    if edges[0] > E_lo or edges[-1] < E_hi:
        raise ValueError(
            f"Energy structure does not cover Chiba range: "
            f"[{E_lo}, {E_hi}]. edges=[{edges[0]}, {edges[-1]}]"
        )

    hi_candidates = edges[edges >= E_hi]
    lo_candidates = edges[edges <= E_lo]
    if hi_candidates.size == 0 or lo_candidates.size == 0:
        raise ValueError("Cannot bracket Chiba range with given energy boundaries.")

    hi_b = float(np.min(hi_candidates))
    lo_b = float(np.max(lo_candidates))
    if lo_b >= hi_b:
        raise ValueError(f"Invalid bracketing: lo_b={lo_b}, hi_b={hi_b}")

    edges_sub = edges[(edges >= lo_b) & (edges <= hi_b)]
    if edges_sub.size < 2:
        raise ValueError("Cropped energy structure has <2 boundaries.")

    return edges_sub


def compute_group_mixed_moments_average(
    XS_t: np.ndarray,
    XS_x: np.ndarray,
    edges_asc: np.ndarray,
    orders: np.ndarray,
):
    """Compute group-averaged mixed moments <sigma_x * sigma_t^n>."""
    XS_t = np.asarray(XS_t, dtype=float)
    XS_x = np.asarray(XS_x, dtype=float)
    orders = np.asarray(orders, dtype=float).ravel()
    edges = np.asarray(edges_asc, dtype=float).ravel()

    edges = edges[np.isfinite(edges)]
    edges = np.unique(np.sort(edges))
    if edges.size < 2:
        raise ValueError("Energy structure must have >=2 boundaries.")

    Et, St = XS_t[:, 0], XS_t[:, 1]
    Ex, Sx = XS_x[:, 0], XS_x[:, 1]
    if Et.size < 2 or Ex.size < 2:
        raise ValueError("Need >=2 points for both total and partial XS.")

    G = edges.size - 1
    K = orders.size
    numer = np.zeros((G, K), dtype=float)
    denom = (edges[1:] - edges[:-1]).astype(float)

    for g in range(G):
        eL, eR = edges[g], edges[g + 1]
        if eR <= eL:
            continue

        mask_t = (Et >= eL) & (Et <= eR)
        mask_x = (Ex >= eL) & (Ex <= eR)

        nodes = np.concatenate(([eL, eR], Et[mask_t], Ex[mask_x]))
        nodes = np.unique(nodes)
        nodes = nodes[(nodes >= eL) & (nodes <= eR)]
        if nodes.size < 2:
            continue

        st = np.interp(nodes, Et, St)
        sx = np.interp(nodes, Ex, Sx)

        dx = nodes[1:] - nodes[:-1]
        stL, stR = st[:-1], st[1:]
        sxL, sxR = sx[:-1], sx[1:]

        ok_base = (stL > 0) & (stR > 0)
        if not np.any(ok_base):
            continue

        for k, n in enumerate(orders):
            fL = np.zeros_like(dx)
            fR = np.zeros_like(dx)
            fL[ok_base] = sxL[ok_base] * (stL[ok_base] ** n)
            fR[ok_base] = sxR[ok_base] * (stR[ok_base] ** n)
            numer[g, k] = np.sum(0.5 * (fL + fR) * dx)

    with np.errstate(divide="ignore", invalid="ignore"):
        mixed_low = np.where(denom[:, None] > 0, numer / denom[:, None], 0.0)

    return mixed_low[::-1]


def group_union_node_count(XS_t: np.ndarray, XS_x: np.ndarray, edges_asc: np.ndarray, g_low: int) -> int:
    edges = np.asarray(edges_asc, dtype=float)
    Et = XS_t[:, 0]
    Ex = XS_x[:, 0]
    eL = float(edges[g_low])
    eR = float(edges[g_low + 1])
    if eR <= eL:
        return 0

    mask_t = (Et >= eL) & (Et <= eR)
    mask_x = (Ex >= eL) & (Ex <= eR)
    nodes = np.concatenate(([eL, eR], Et[mask_t], Ex[mask_x]))
    nodes = np.unique(nodes)
    nodes = nodes[(nodes >= eL) & (nodes <= eR)]
    return int(nodes.size)


def compute_counts_and_union_nodes_hi2lo(
    XS_t: np.ndarray,
    XS_x: np.ndarray,
    edges_asc: np.ndarray,
    *,
    side: str = "right",
) -> tuple[np.ndarray, np.ndarray]:
    """Return total-XS point counts and union-node counts in high->low order."""
    XS_t = np.asarray(XS_t, dtype=float)
    XS_x = np.asarray(XS_x, dtype=float)
    edges = np.asarray(edges_asc, dtype=float).ravel()
    edges = edges[np.isfinite(edges)]
    edges = np.unique(np.sort(edges))
    if edges.size < 2:
        raise ValueError("Energy structure must have >=2 boundaries.")

    Et = XS_t[:, 0]
    Ex = XS_x[:, 0]
    G = edges.size - 1

    g_low_pts = np.searchsorted(edges, Et, side=side) - 1
    mask_pts = (g_low_pts >= 0) & (g_low_pts < G)
    counts_low = np.bincount(g_low_pts[mask_pts], minlength=G).astype(int)

    M_nodes_low = np.zeros(G, dtype=int)
    for g_low in range(G):
        eL = float(edges[g_low])
        eR = float(edges[g_low + 1])
        if eR <= eL:
            M_nodes_low[g_low] = 0
            continue

        mask_t = (Et >= eL) & (Et <= eR)
        mask_x = (Ex >= eL) & (Ex <= eR)

        nodes = np.concatenate(([eL, eR], Et[mask_t], Ex[mask_x]))
        nodes = np.unique(nodes)
        nodes = nodes[(nodes >= eL) & (nodes <= eR)]
        M_nodes_low[g_low] = int(nodes.size)

    return counts_low[::-1], M_nodes_low[::-1]


def solve_affine_pade_probability_table_hankel(
    m_2N: np.ndarray,
    orders_r_2N: np.ndarray,
    *,
    group_id: int,
    diagnostics: bool = True,
):
    """Solve the total table from affine moments via the Hankel route."""
    m = np.asarray(m_2N, dtype=float).ravel()
    orders_r = np.asarray(orders_r_2N, dtype=float).ravel()

    N = m.size // 2
    if m.size != 2 * N:
        raise ValueError("m_2N must have length 2N.")
    if orders_r.size != 2 * N:
        raise ValueError("orders_r_2N must have length 2N.")

    a, b = _infer_affine_params(orders_r)

    H = hankel(m[:N], m[N - 1: 2 * N - 1])
    svals_H = svd(H, compute_uv=False)
    condH, rcondH = _cond2_via_svd(H)

    rhs = -m[N: 2 * N]

    try:
        q = solve_with_warning_tag(H, rhs, tag="solve(Hankel_H, rhs)", group_id=group_id)
    except Exception:
        q = lstsq_fallback(H, rhs, rcond=1e-12)

    Qcoef = np.concatenate([q, [1.0]])
    rho = Polynomial(Qcoef).roots()

    min_abs_Qp, max_inv_abs_Qp = _root_sensitivity_metrics(Qcoef, rho)
    min_root_sep = _min_pairwise_distance(rho)

    rho_scale = float(np.nanmax(np.abs(rho))) if rho.size else float("nan")
    if np.isfinite(rho_scale) and np.isfinite(min_root_sep) and (min_root_sep > 0.0):
        root_cluster_index = rho_scale / min_root_sep
    elif np.isfinite(rho_scale) and (min_root_sep == 0.0):
        root_cluster_index = float("inf")
    else:
        root_cluster_index = float("nan")

    V = np.vander(rho, N, increasing=True).T
    condV, rcondV = _cond2_via_svd(V)

    try:
        pi = solve_with_warning_tag(V, m[:N], tag="solve(Vander_pi, c0..cN-1)", group_id=group_id)
    except Exception:
        pi = lstsq_fallback(V, m[:N], rcond=1e-12)

    res_pi = _safe_rel_residual(V, pi, m[:N])

    sigma = np.power(rho, 1.0 / b)
    p = pi / np.power(sigma, a)

    info = None
    if diagnostics:
        info = {
            "N": int(N),
            "a": float(a),
            "b": float(b),
            "cond_hankel": float(condH),
            "rcond_hankel": float(rcondH),
            "svals_hankel": np.asarray(svals_H, dtype=float),
            "min_abs_Qp_at_roots": float(min_abs_Qp),
            "max_inv_abs_Qp_at_roots": float(max_inv_abs_Qp),
            "min_root_sep_rho": float(min_root_sep),
            "root_cluster_index_rho": float(root_cluster_index),
            "cond_vandermonde_pi": float(condV),
            "rcond_vandermonde_pi": float(rcondV),
            "res_vandermonde_pi_rel": float(res_pi),
        }
    return sigma, p, rho, info


def solve_partial_ubar_via_rho_chiba(
    m_x_N: np.ndarray,
    rho: np.ndarray,
    *,
    group_id: int,
):
    """Solve ubar from mixed moments on the rho grid."""
    m_x = np.asarray(m_x_N, dtype=float).ravel()
    rho = np.asarray(rho).ravel()
    N = rho.size
    if m_x.size != N:
        raise ValueError("m_x_N must have length N (= number of rho roots).")

    V = np.vander(rho, N, increasing=True).T
    condV, rcondV = _cond2_via_svd(V)

    try:
        ubar = solve_with_warning_tag(V, m_x, tag="solve(Vander_ubar, mx)", group_id=group_id)
    except Exception:
        ubar = lstsq_fallback(V, m_x, rcond=1e-12)

    res_ubar = _safe_rel_residual(V, ubar, m_x)

    return ubar, {
        "cond_vandermonde_ubar": float(condV),
        "rcond_vandermonde_ubar": float(rcondV),
        "res_vandermonde_ubar_rel": float(res_ubar),
    }


def nra_effective_x_reference(
    XS_t: np.ndarray,
    XS_x: np.ndarray,
    edges_asc: np.ndarray,
    g_low: int,
    sigma0_grid: np.ndarray,
) -> np.ndarray:
    """Compute the NRA reference effective cross section."""
    edges = np.asarray(edges_asc, dtype=float)
    Et, Rt = XS_t[:, 0], XS_t[:, 1]
    Ex, Rx = XS_x[:, 0], XS_x[:, 1]

    eL = edges[g_low]
    eR = edges[g_low + 1]
    if eR <= eL:
        return np.full_like(sigma0_grid, np.nan, dtype=float)

    mask_t = (Et >= eL) & (Et <= eR)
    mask_x = (Ex >= eL) & (Ex <= eR)
    nodes = np.concatenate(([eL, eR], Et[mask_t], Ex[mask_x]))
    nodes = np.unique(nodes)
    nodes = nodes[(nodes >= eL) & (nodes <= eR)]
    if nodes.size < 2:
        return np.full_like(sigma0_grid, np.nan, dtype=float)

    rt = np.interp(nodes, Et, Rt)
    rx = np.interp(nodes, Ex, Rx)

    dx = nodes[1:] - nodes[:-1]
    rtL, rtR = rt[:-1], rt[1:]
    rxL, rxR = rx[:-1], rx[1:]

    out = np.zeros_like(sigma0_grid, dtype=float)
    for j, s0 in enumerate(sigma0_grid):
        dL = 1.0 / (rtL + s0)
        dR = 1.0 / (rtR + s0)
        denom = np.sum(0.5 * (dL + dR) * dx)

        nL = rxL / (rtL + s0)
        nR = rxR / (rtR + s0)
        numer = np.sum(0.5 * (nL + nR) * dx)

        out[j] = numer / denom if denom != 0.0 else np.nan

    return out


def nra_effective_x_probability_table_chiba_ubar(
    sigma_t_nodes: np.ndarray,
    p: np.ndarray,
    ubar: np.ndarray,
    a: float,
    sigma0_grid: np.ndarray,
) -> np.ndarray:
    """Evaluate PT effective cross section from (sigma_t, p, ubar)."""
    r = np.asarray(sigma_t_nodes).ravel()
    p = np.asarray(p).ravel()
    ubar = np.asarray(ubar).ravel()

    out_dtype = np.complex128 if (np.iscomplexobj(r) or np.iscomplexobj(p) or np.iscomplexobj(ubar)) else float
    out = np.zeros_like(sigma0_grid, dtype=out_dtype)

    ra = np.power(r, a)
    for j, s0 in enumerate(sigma0_grid):
        den = np.sum(p / (r + s0))
        num = np.sum((ubar / ra) / (r + s0))
        out[j] = num / den if den != 0 else np.nan
    return out


def save_group_error_plot_pcm_scatter(
    outdir: Path,
    group_id: int,
    e_high: float,
    e_low: float,
    sigma0_grid: np.ndarray,
    err_pcm: np.ndarray,
    *,
    N_g: int,
    b_g: float,
    M_nodes: int,
    rms_pcm: float,
    q_abs_pcm: np.ndarray,
) -> None:
    """Save the per-group error scatter plot."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(sigma0_grid, err_pcm, s=SCATTER_S)
    ax.set_xscale("log")

    ax.set_xlabel(XLAB_SIG0)
    ax.set_ylabel(YLAB_ERR_PCM)
    ax.set_title(rf"Group $g={group_id}$")

    qtxt = ", ".join([f"q{qq:.2f}={vv:.2g}" for qq, vv in zip(ERR_QUANTILES, q_abs_pcm)])
    ax.text(
        0.98, 0.98,
        rf"$E\in[{e_low:.4g},{e_high:.4g}]$"
        "\n" rf"$N_g={N_g}$, $b={b_g:.4g}$, $M={M_nodes}$"
        "\n" rf"RMS={rms_pcm:.3g} pcm; $|e|$:{qtxt}",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=9,
    )

    ax.axhline(0.0, linewidth=0.8, alpha=0.6)

    fig.savefig(outdir / f"group_{group_id:03d}_err.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def group_rt_minmax(XS_t: np.ndarray, edges_asc: np.ndarray, g_low: int) -> tuple[float, float]:
    edges = np.asarray(edges_asc, dtype=float)
    eL, eR = edges[g_low], edges[g_low + 1]
    Et, Rt = XS_t[:, 0], XS_t[:, 1]
    mask = (Et >= eL) & (Et <= eR) & np.isfinite(Rt)
    if not np.any(mask):
        return float("nan"), float("nan")
    return float(np.min(Rt[mask])), float(np.max(Rt[mask]))


_LOGFMT_10K = LogFormatterMathtext(base=10.0)


def _decimal_log_formatter(y, _pos):
    if not np.isfinite(y) or y <= 0:
        return ""
    return _LOGFMT_10K(y)


def _nice_decade_ylim(y: np.ndarray) -> tuple[float, float] | None:
    y = np.asarray(y, dtype=float).ravel()
    ok = np.isfinite(y) & (y > 0)
    if not np.any(ok):
        return None
    y_min = float(np.min(y[ok]))
    y_max = float(np.max(y[ok]))
    ymin_dec = 10.0 ** np.floor(np.log10(y_min))
    ymax_dec = 10.0 ** np.ceil(np.log10(y_max))
    if not (np.isfinite(ymin_dec) and np.isfinite(ymax_dec) and ymin_dec > 0 and ymax_dec > 0):
        return None
    return ymin_dec, ymax_dec


def _apply_group_ticks_1_50(ax, G: int) -> None:
    """Use x ticks at 1, 50, 100, ... with a small margin."""
    G = int(G)
    if G < 1:
        return

    ax.set_xlim(1 - 5, G + 5)

    majors = [1]
    majors.extend(list(range(50, G + 1, 50)))
    majors = sorted(set([m for m in majors if 1 <= m <= G]))
    ax.xaxis.set_major_locator(mticker.FixedLocator(majors))

    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())


def save_q95_error_by_group_plot_log(
    outdir: Path,
    group_ids: np.ndarray,
    q_abs_err_pcm: np.ndarray,
    flag_complex: np.ndarray,
    flag_negative: np.ndarray,
    *,
    q_target: float = 0.95,
    quantiles: tuple[float, ...] = ERR_QUANTILES,
    filename: str = "q95_err_by_group.png",
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gid = np.asarray(group_ids, dtype=int).ravel()
    q_pcm = np.asarray(q_abs_err_pcm, dtype=float)
    fc = np.asarray(flag_complex, dtype=bool).ravel()
    fn = np.asarray(flag_negative, dtype=bool).ravel()

    if q_pcm.ndim != 2 or q_pcm.shape[0] != gid.size:
        raise ValueError("q_abs_err_pcm must be (G,Q) and match group_ids length.")
    if not (gid.size == fc.size == fn.size):
        raise ValueError("group_ids/flag_complex/flag_negative must have same length.")

    try:
        j_q = list(quantiles).index(q_target)
    except ValueError:
        j_q = int(np.argmin(np.abs(np.asarray(quantiles, dtype=float) - float(q_target))))

    q_target_pcm = q_pcm[:, j_q]
    q_rel = q_target_pcm * 1e-5

    ok = np.isfinite(q_rel) & (q_rel > 0)
    if not np.any(ok):
        print("[Q95Plot] No positive finite q95 values.")
        return

    both = ok & fc & fn
    comp_only = ok & fc & (~fn)
    neg_only = ok & fn & (~fc)
    normal = ok & (~fc) & (~fn)

    ylim = _nice_decade_ylim(q_rel)
    if ylim is None:
        print("[Q95Plot] Invalid y-limits.")
        return

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    if np.any(normal):
        ax.scatter(gid[normal], q_rel[normal], s=SCATTER_S, c="tab:blue", label="real & non-negative")
    if np.any(neg_only):
        ax.scatter(gid[neg_only], q_rel[neg_only], s=SCATTER_S, c="tab:orange", label="negative")
    if np.any(comp_only):
        ax.scatter(gid[comp_only], q_rel[comp_only], s=SCATTER_S, c="tab:red", label="complex")
    if np.any(both):
        ax.scatter(gid[both], q_rel[both], s=SCATTER_S, c="tab:purple", label="complex & negative")

    ax.set_xlabel(XLAB_G)
    _apply_group_ticks_1_50(ax, G=gid.size)
    ax.set_ylabel(rf"$q_{{{quantiles[j_q]:.2f}}}(|e_g|)$")
    ax.set_title(rf"{TITLE_FAMILY}: $q_{{{quantiles[j_q]:.2f}}}(|e_g|)$ by group")

    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=12))
    ax.yaxis.set_major_formatter(FuncFormatter(_decimal_log_formatter))
    ax.legend(loc="best", fontsize=9)

    fig.savefig(outdir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_max_error_by_group_plot_log(
    outdir: Path,
    group_ids: np.ndarray,
    max_err_pcm: np.ndarray,
    flag_complex: np.ndarray,
    flag_negative: np.ndarray,
    *,
    filename: str = "max_err_by_group.png",
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gid = np.asarray(group_ids, dtype=int).ravel()
    mx_pcm = np.asarray(max_err_pcm, dtype=float).ravel()
    fc = np.asarray(flag_complex, dtype=bool).ravel()
    fn = np.asarray(flag_negative, dtype=bool).ravel()

    if not (gid.size == mx_pcm.size == fc.size == fn.size):
        raise ValueError("group_ids/max_err_pcm/flag_complex/flag_negative must have same length.")

    ok = np.isfinite(mx_pcm) & (mx_pcm > 0)
    if not np.any(ok):
        print("[MaxPlot] No positive finite max error values.")
        return

    mx_rel = mx_pcm * 1e-5

    both = ok & fc & fn
    comp_only = ok & fc & (~fn)
    neg_only = ok & fn & (~fc)
    normal = ok & (~fc) & (~fn)

    ylim = _nice_decade_ylim(mx_rel)
    if ylim is None:
        print("[MaxPlot] Invalid y-limits.")
        return

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    if np.any(normal):
        ax.scatter(gid[normal], mx_rel[normal], s=SCATTER_S, c="tab:blue", label="real & non-negative")
    if np.any(neg_only):
        ax.scatter(gid[neg_only], mx_rel[neg_only], s=SCATTER_S, c="tab:orange", label="negative")
    if np.any(comp_only):
        ax.scatter(gid[comp_only], mx_rel[comp_only], s=SCATTER_S, c="tab:red", label="complex")
    if np.any(both):
        ax.scatter(gid[both], mx_rel[both], s=SCATTER_S, c="tab:purple", label="complex & negative")

    ax.set_xlabel(XLAB_G)
    _apply_group_ticks_1_50(ax, G=gid.size)
    ax.set_ylabel(r"$\|e_g\|_\infty=\max_{\sigma_0\in S}|e_g(\sigma_0)|$")
    ax.set_title(rf"{TITLE_FAMILY}: $\|e_g\|_\infty$ by group")

    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=12))
    ax.yaxis.set_major_formatter(FuncFormatter(_decimal_log_formatter))
    ax.legend(loc="best", fontsize=9)

    fig.savefig(outdir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_rms_error_by_group_plot_log(
    outdir: Path,
    group_ids: np.ndarray,
    rms_err_pcm: np.ndarray,
    flag_complex: np.ndarray,
    flag_negative: np.ndarray,
    *,
    filename: str = "rms_err_by_group.png",
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gid = np.asarray(group_ids, dtype=int).ravel()
    rms_pcm = np.asarray(rms_err_pcm, dtype=float).ravel()
    fc = np.asarray(flag_complex, dtype=bool).ravel()
    fn = np.asarray(flag_negative, dtype=bool).ravel()

    if not (gid.size == rms_pcm.size == fc.size == fn.size):
        raise ValueError("group_ids/rms_err_pcm/flag_complex/flag_negative must have same length.")

    ok = np.isfinite(rms_pcm) & (rms_pcm > 0)
    if not np.any(ok):
        print("[RMSPlot] No positive finite RMS values.")
        return

    rms_rel = rms_pcm * 1e-5

    both = ok & fc & fn
    comp_only = ok & fc & (~fn)
    neg_only = ok & fn & (~fc)
    normal = ok & (~fc) & (~fn)

    ylim = _nice_decade_ylim(rms_rel)
    if ylim is None:
        print("[RMSPlot] Invalid y-limits.")
        return

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    if np.any(normal):
        ax.scatter(gid[normal], rms_rel[normal], s=SCATTER_S, c="tab:blue", label="real & non-negative")
    if np.any(neg_only):
        ax.scatter(gid[neg_only], rms_rel[neg_only], s=SCATTER_S, c="tab:orange", label="negative")
    if np.any(comp_only):
        ax.scatter(gid[comp_only], rms_rel[comp_only], s=SCATTER_S, c="tab:red", label="complex")
    if np.any(both):
        ax.scatter(gid[both], rms_rel[both], s=SCATTER_S, c="tab:purple", label="complex & negative")

    ax.set_xlabel(XLAB_G)
    _apply_group_ticks_1_50(ax, G=gid.size)
    ax.set_ylabel(r"$\mathrm{RMS}(e_g)$")
    ax.set_title(rf"{TITLE_FAMILY}: RMS$(e_g)$ by group")

    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=12))
    ax.yaxis.set_major_formatter(FuncFormatter(_decimal_log_formatter))
    ax.legend(loc="best", fontsize=9)

    fig.savefig(outdir / filename, dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_quantile_error_by_group_plot_log(
    outdir: Path,
    group_ids: np.ndarray,
    q_abs_err_pcm: np.ndarray,
    *,
    quantiles: tuple[float, ...] = ERR_QUANTILES,
    filename: str = "qerr_by_group.png",
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gid = np.asarray(group_ids, dtype=int).ravel()
    Q = len(quantiles)

    q_pcm = np.asarray(q_abs_err_pcm, dtype=float)
    if q_pcm.ndim != 2 or q_pcm.shape[0] != gid.size or q_pcm.shape[1] != Q:
        raise ValueError(f"q_abs_err_pcm must be (G,{Q}) and match group_ids length.")

    q_rel = q_pcm * 1e-5
    ylim = _nice_decade_ylim(q_rel[np.isfinite(q_rel)])
    if ylim is None:
        print("[QPlot] No positive finite quantile values.")
        return

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    markers = ["o", "s", "D", "^", "v", "P", "X"]
    for j, qq in enumerate(quantiles):
        y = q_rel[:, j]
        ok = np.isfinite(y) & (y > 0)
        if np.any(ok):
            ax.scatter(
                gid[ok], y[ok],
                s=SCATTER_S,
                marker=markers[j % len(markers)],
                label=rf"$q_{{{qq:.2f}}}(|e_g|)$"
            )

    ax.set_xlabel(XLAB_G)
    _apply_group_ticks_1_50(ax, G=gid.size)
    ax.set_ylabel(r"$q_\alpha(|e_g|)$ (dimensionless)")
    ax.set_title(rf"{TITLE_FAMILY}: error quantiles by group")
    ax.set_yscale("log")
    ax.set_ylim(*ylim)

    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=12))
    ax.yaxis.set_major_formatter(FuncFormatter(_decimal_log_formatter))
    ax.legend(loc="best", fontsize=9)

    fig.savefig(outdir / filename, dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_condH_by_group_plot_log(
    outdir: Path,
    group_ids: np.ndarray,
    condH: np.ndarray,
    *,
    filename: str = "condH_by_group.png",
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gid = np.asarray(group_ids, dtype=int).ravel()
    ch = np.asarray(condH, dtype=float).ravel()

    if gid.size != ch.size:
        raise ValueError("group_ids and condH must have same length.")

    ok = np.isfinite(ch) & (ch > 0)
    if not np.any(ok):
        print("[CondHPlot] No positive finite cond(H) values.")
        return

    ylim = _nice_decade_ylim(ch[ok])
    if ylim is None:
        print("[CondHPlot] Invalid y-limits.")
        return

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(gid[ok], ch[ok], s=SCATTER_S)
    ax.set_xlabel(XLAB_G)
    _apply_group_ticks_1_50(ax, G=gid.size)
    ax.set_ylabel(r"Hankel condition number $\kappa_2(H_g)$")
    ax.set_title(r"Hankel matrix condition number by group")
    ax.set_yscale("log")
    ax.set_ylim(*ylim)

    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=12))
    ax.yaxis.set_major_formatter(FuncFormatter(_decimal_log_formatter))

    fig.savefig(outdir / filename, dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_condH_by_group_txt(
    outdir: Path,
    group_ids: np.ndarray,
    condH: np.ndarray,
    *,
    filename: str = "condH_by_group.txt",
):
    """Save per-group Hankel condition numbers."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gid = np.asarray(group_ids, dtype=int).ravel()
    ch = np.asarray(condH, dtype=float).ravel()

    if gid.size != ch.size:
        raise ValueError("group_ids and condH must have same length.")

    txt_path = outdir / filename
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("g  condH\n")
        for g, v in zip(gid, ch):
            if np.isfinite(v):
                f.write(f"{int(g):4d}  {float(v):.16e}\n")
            else:
                f.write(f"{int(g):4d}  nan\n")


def _select_representative_groups_by_cond(
    condH: np.ndarray,
    *,
    n_reps: int = 3,
    mode: str = "logquantile",
) -> list[int]:
    ch = np.asarray(condH, dtype=float).ravel()
    ok = np.isfinite(ch) & (ch > 0)
    if not np.any(ok):
        return []

    idx_ok = np.where(ok)[0]
    ch_ok = ch[ok]

    n_reps = int(n_reps)
    if n_reps <= 0:
        return []
    n_reps = min(n_reps, idx_ok.size)

    selected: list[int] = []

    if mode == "rank":
        order = np.argsort(ch_ok)
        idx_sorted = idx_ok[order]
        pos = np.linspace(0, idx_sorted.size - 1, n_reps)
        picks = np.unique(np.round(pos).astype(int))
        for p in picks:
            selected.append(int(idx_sorted[p] + 1))

    elif mode == "quantile":
        qs = np.linspace(0.0, 1.0, n_reps)
        targets = np.quantile(ch_ok, qs)
        for t in targets:
            j = int(np.argmin(np.abs(ch_ok - t)))
            selected.append(int(idx_ok[j] + 1))

    elif mode == "logquantile":
        logc = np.log10(ch_ok)
        qs = np.linspace(0.0, 1.0, n_reps)
        targets = np.quantile(logc, qs)
        for t in targets:
            j = int(np.argmin(np.abs(logc - t)))
            selected.append(int(idx_ok[j] + 1))

    else:
        raise ValueError(f"Unknown mode={mode}. Use 'logquantile', 'quantile', or 'rank'.")

    out: list[int] = []
    for g in selected:
        if g not in out:
            out.append(g)

    if len(out) < n_reps:
        order = np.argsort(ch_ok)
        idx_sorted = idx_ok[order]
        pos = np.linspace(0, idx_sorted.size - 1, n_reps * 3)
        for p in np.round(pos).astype(int):
            g = int(idx_sorted[np.clip(p, 0, idx_sorted.size - 1)] + 1)
            if g not in out:
                out.append(g)
            if len(out) >= n_reps:
                break

    return out[:n_reps]


def save_hankel_svals_examples(
    outdir: Path,
    svals_by_group: np.ndarray,
    N_used_by_group: np.ndarray,
    condH_by_group: np.ndarray,
    *,
    filename: str = "hankel_svals_examples.png",
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    reps = _select_representative_groups_by_cond(condH_by_group)
    if len(reps) == 0:
        print("[HankelSvals] No representative groups selected.")
        return

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for g1 in reps:
        i = g1 - 1
        N = int(N_used_by_group[i]) if np.isfinite(N_used_by_group[i]) else 0
        if N <= 0:
            continue
        s = svals_by_group[i, :N]
        ok = np.isfinite(s) & (s > 0)
        if not np.any(ok):
            continue
        x = np.arange(1, N + 1)
        ax.plot(
            x[ok], s[ok],
            marker="o", linewidth=1.2,
            label=rf"$g={g1}$, $N={N}$, $\kappa(H)={condH_by_group[i]:.2e}$"
        )

    ax.set_yscale("log")
    ax.set_xlabel(r"Singular-value index $i$")
    ax.set_ylabel(r"$s_i(H)$")
    ax.set_title(r"Hankel singular-value spectra (representative groups)")
    ax.legend(loc="best", fontsize=9)

    fig.savefig(outdir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_hankel_svals_heatmap(
    outdir: Path,
    svals_by_group: np.ndarray,
    *,
    filename: str = "hankel_svals_heatmap.png",
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    S = np.asarray(svals_by_group, dtype=float)
    if S.ndim != 2:
        return
    with np.errstate(divide="ignore", invalid="ignore"):
        L = np.log10(S)
    L[~np.isfinite(L)] = np.nan

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    G, Nmax = L.shape

    im = ax.imshow(
        L,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=(0.5, Nmax + 0.5, 0.5, G + 0.5),
    )

    ax.set_xlabel(r"Singular-value index")
    ax.set_ylabel(XLAB_G)

    ax.set_xlim(1, Nmax)
    ax.set_xticks(np.arange(1, Nmax + 1))

    ax.set_ylim(1, G)
    maj_y = [1] + list(range(50, G + 1, 50))
    ax.set_yticks(maj_y)

    ax.set_title(r"$\log_{10}(s_i(H))$ heatmap (all groups)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\log_{10}(s_i(H))$")

    fig.savefig(outdir / filename, dpi=250, bbox_inches="tight")
    plt.close(fig)


def save_root_sensitivity_by_group_plot(
    outdir: Path,
    group_ids: np.ndarray,
    max_inv_abs_Qp: np.ndarray,
    *,
    filename: str = "root_sensitivity_by_group.png",
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gid = np.asarray(group_ids, dtype=int).ravel()
    y = np.asarray(max_inv_abs_Qp, dtype=float).ravel()
    if gid.size != y.size:
        raise ValueError("group_ids and max_inv_abs_Qp size mismatch")

    ok = np.isfinite(y) & (y > 0)
    if not np.any(ok):
        print("[RootSens] No positive finite values.")
        return

    ylim = _nice_decade_ylim(y[ok])
    if ylim is None:
        print("[RootSens] Invalid y-limits.")
        return

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(gid[ok], y[ok], s=SCATTER_S)

    ax.set_xlabel(XLAB_G)
    _apply_group_ticks_1_50(ax, G=gid.size)
    ax.set_ylabel(r"$\max_i 1/|Q'(\rho_i)|$")
    ax.set_title(r"Root sensitivity indicator by group")
    ax.set_yscale("log")
    ax.set_ylim(*ylim)

    fig.savefig(outdir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_root_separation_by_group_plot(
    outdir: Path,
    group_ids: np.ndarray,
    min_root_sep: np.ndarray,
    *,
    filename: str = "root_separation_by_group.png",
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gid = np.asarray(group_ids, dtype=int).ravel()
    y = np.asarray(min_root_sep, dtype=float).ravel()
    if gid.size != y.size:
        raise ValueError("group_ids and min_root_sep size mismatch")

    ok = np.isfinite(y) & (y > 0)
    if not np.any(ok):
        print("[RootSep] No positive finite values.")
        return

    ylim = _nice_decade_ylim(y[ok])
    if ylim is None:
        print("[RootSep] Invalid y-limits.")
        return

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(gid[ok], y[ok], s=SCATTER_S)

    ax.set_xlabel(XLAB_G)
    _apply_group_ticks_1_50(ax, G=gid.size)
    ax.set_ylabel(r"$\min_{i\neq j}|\rho_i-\rho_j|$")
    ax.set_title(r"Minimum root separation by group")
    ax.set_yscale("log")
    ax.set_ylim(*ylim)

    fig.savefig(outdir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_root_clustering_by_group_plot(
    outdir: Path,
    group_ids: np.ndarray,
    root_cluster_index: np.ndarray,
    *,
    filename: str = "root_clustering_by_group.png",
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gid = np.asarray(group_ids, dtype=int).ravel()
    y = np.asarray(root_cluster_index, dtype=float).ravel()
    if gid.size != y.size:
        raise ValueError("group_ids and root_cluster_index size mismatch")

    ok = np.isfinite(y) & (y > 0)
    if not np.any(ok):
        print("[RootCluster] No positive finite values.")
        return

    ylim = _nice_decade_ylim(y[ok])
    if ylim is None:
        print("[RootCluster] Invalid y-limits.")
        return

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(gid[ok], y[ok], s=SCATTER_S)

    ax.set_xlabel(XLAB_G)
    _apply_group_ticks_1_50(ax, G=gid.size)
    ax.set_ylabel(r"$\max_i|\rho_i| \,/\, \min_{i\neq j}|\rho_i-\rho_j|$")
    ax.set_title(r"Root clustering indicator by group")
    ax.set_yscale("log")
    ax.set_ylim(*ylim)

    fig.savefig(outdir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_partial_recon_cond_by_group_plot(
    outdir: Path,
    group_ids: np.ndarray,
    condV_pi: np.ndarray,
    condV_ubar: np.ndarray,
    *,
    filename: str = "partial_recon_cond_by_group.png",
):
    """Plot one reconstruction conditioning curve, preferring V_ubar."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gid = np.asarray(group_ids, dtype=int).ravel()
    cpi = np.asarray(condV_pi, dtype=float).ravel()
    cub = np.asarray(condV_ubar, dtype=float).ravel()

    if not (gid.size == cpi.size == cub.size):
        raise ValueError("size mismatch in partial recon cond arrays")

    ok_ub = np.isfinite(cub) & (cub > 0)
    ok_pi = np.isfinite(cpi) & (cpi > 0)

    if np.any(ok_ub):
        y = cub
        ok = ok_ub
        ylab = r"$\kappa_2(V_{\bar{u}})$"
        title = r"Partial-level reconstruction conditioning by group"
    elif np.any(ok_pi):
        y = cpi
        ok = ok_pi
        ylab = r"$\kappa_2(V_{\pi})$"
        title = r"Partial-level reconstruction conditioning by group (fallback: $V_\pi$)"
    else:
        print("[PartialCond] No finite positive values in either series.")
        return

    ylim = _nice_decade_ylim(y[ok])
    if ylim is None:
        print("[PartialCond] Invalid y-limits.")
        return

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(gid[ok], y[ok], s=SCATTER_S)

    ax.set_xlabel(XLAB_G)
    _apply_group_ticks_1_50(ax, G=gid.size)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.set_yscale("log")
    ax.set_ylim(*ylim)

    fig.savefig(outdir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_partial_recon_res_by_group_plot(
    outdir: Path,
    group_ids: np.ndarray,
    res_pi: np.ndarray,
    res_ubar: np.ndarray,
    *,
    filename: str = "partial_recon_res_by_group.png",
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gid = np.asarray(group_ids, dtype=int).ravel()
    rpi = np.asarray(res_pi, dtype=float).ravel()
    rub = np.asarray(res_ubar, dtype=float).ravel()

    if not (gid.size == rpi.size == rub.size):
        raise ValueError("size mismatch in partial recon residual arrays")

    ok1 = np.isfinite(rpi) & (rpi > 0)
    ok2 = np.isfinite(rub) & (rub > 0)
    if (not np.any(ok1)) and (not np.any(ok2)):
        print("[PartialRes] No finite positive values.")
        return

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    y_all = []
    if np.any(ok1):
        y_all.append(rpi[ok1])
        ax.scatter(gid[ok1], rpi[ok1], s=SCATTER_S, marker="o",
                   label=r"$\|V_{\pi}\pi - m\|_2/\|m\|_2$")
    if np.any(ok2):
        y_all.append(rub[ok2])
        ax.scatter(gid[ok2], rub[ok2], s=SCATTER_S, marker="D",
                   label=r"$\|V_{\bar{u}}\bar{u} - m^x\|_2/\|m^x\|_2$")

    y_all = np.concatenate(y_all) if len(y_all) else np.array([], dtype=float)
    ylim = _nice_decade_ylim(y_all)
    if ylim is None:
        print("[PartialRes] Invalid y-limits.")
        return

    ax.set_xlabel(XLAB_G)
    _apply_group_ticks_1_50(ax, G=gid.size)
    ax.set_ylabel(r"Relative residual")
    ax.set_title(r"Partial-level reconstruction residuals by group")
    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ax.legend(loc="best", fontsize=9)

    fig.savefig(outdir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_group_perturbation_response(
    outdir: Path,
    group_id: int,
    m2N_base: np.ndarray,
    mx_base: np.ndarray,
    orders_r_2N: np.ndarray,
    *,
    n_trials: int = 20,
    eps_list: np.ndarray | None = None,
    filename_prefix: str = "group",
) -> None:
    """Save median perturbation-response curves for one representative group."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    m2N_base = np.asarray(m2N_base, dtype=float).ravel()
    mx_base = np.asarray(mx_base, dtype=float).ravel()
    orders_r_2N = np.asarray(orders_r_2N, dtype=float).ravel()

    if eps_list is None:
        eps_list = np.array([1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6], dtype=float)

    try:
        _sigma0, p0, rho0, info0 = solve_affine_pade_probability_table_hankel(
            m2N_base, orders_r_2N, group_id=group_id, diagnostics=True
        )
        N = int(info0["N"])
        _ubar0, _ = solve_partial_ubar_via_rho_chiba(mx_base[:N], rho0, group_id=group_id)
    except Exception as ex:
        print(f"[Perturb][Group {group_id}] baseline failed: {type(ex).__name__}: {ex}")
        return

    rng = np.random.default_rng(seed=12345 + int(group_id))

    med_rel_rho = []
    med_rel_p = []

    n0 = float(np.linalg.norm(rho0))
    n1 = float(np.linalg.norm(p0))
    if n0 <= 0.0:
        n0 = 1.0
    if n1 <= 0.0:
        n1 = 1.0

    for eps in eps_list:
        rel_rho_trials = []
        rel_p_trials = []

        for _ in range(int(n_trials)):
            xi1 = rng.standard_normal(size=m2N_base.size)
            xi2 = rng.standard_normal(size=mx_base.size)

            m2N = m2N_base * (1.0 + float(eps) * xi1)
            mx = mx_base * (1.0 + float(eps) * xi2)

            try:
                _sigma1, p1, rho1, _info1 = solve_affine_pade_probability_table_hankel(
                    m2N, orders_r_2N, group_id=group_id, diagnostics=False
                )
                _ubar1, _ = solve_partial_ubar_via_rho_chiba(mx[:N], rho1, group_id=group_id)

                if (not _is_finite_all(rho1)) or (not _is_finite_all(p1)):
                    raise RuntimeError("non-finite rho/p")

                idx = _match_by_hungarian(rho0, rho1)
                rho1m = rho1[idx]
                p1m = p1[idx]

                dr = float(np.linalg.norm(rho1m - rho0))
                dp = float(np.linalg.norm(p1m - p0))

                rel_rho = dr / n0
                rel_pv = dp / n1

                if not (np.isfinite(rel_rho) and np.isfinite(rel_pv)):
                    raise RuntimeError("non-finite rel metrics")

                rel_rho_trials.append(rel_rho)
                rel_p_trials.append(rel_pv)

            except Exception:
                pass

        if len(rel_rho_trials) == 0:
            med_rel_rho.append(np.nan)
        else:
            med_rel_rho.append(float(np.median(np.asarray(rel_rho_trials, dtype=float))))

        if len(rel_p_trials) == 0:
            med_rel_p.append(np.nan)
        else:
            med_rel_p.append(float(np.median(np.asarray(rel_p_trials, dtype=float))))

    med_rel_rho = np.asarray(med_rel_rho, dtype=float)
    med_rel_p = np.asarray(med_rel_p, dtype=float)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ok1 = np.isfinite(med_rel_rho) & (med_rel_rho > 0)
    ok2 = np.isfinite(med_rel_p) & (med_rel_p > 0)

    if np.any(ok1):
        ax.plot(eps_list[ok1], med_rel_rho[ok1], marker="o", linewidth=1.2, label=r"median $\|\Delta \rho\|/\|\rho\|$")
    if np.any(ok2):
        ax.plot(eps_list[ok2], med_rel_p[ok2], marker="D", linewidth=1.2, label=r"median $\|\Delta p\|/\|p\|$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Relative moment perturbation amplitude $\epsilon$")
    ax.set_ylabel(r"Median relative response")
    ax.set_title(rf"Perturbation-response (Group $g={group_id}$)")
    ax.legend(loc="best", fontsize=9)

    fig.savefig(outdir / f"{filename_prefix}_{group_id:03d}_perturb_response.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _group_median_perturb_response(
    *,
    group_id: int,
    m2N_base: np.ndarray,
    mx_base: np.ndarray,
    orders_r_2N: np.ndarray,
    eps_list: np.ndarray,
    n_trials: int,
    min_success: int = 5,
    seed_base: int = 24680,
) -> tuple[np.ndarray, np.ndarray]:
    """Return median rho and p perturbation responses for one group."""
    m2N_base = np.asarray(m2N_base, dtype=float).ravel()
    mx_base = np.asarray(mx_base, dtype=float).ravel()
    orders_r_2N = np.asarray(orders_r_2N, dtype=float).ravel()
    eps_list = np.asarray(eps_list, dtype=float).ravel()

    try:
        _sigma0, p0, rho0, info0 = solve_affine_pade_probability_table_hankel(
            m2N_base, orders_r_2N, group_id=group_id, diagnostics=True
        )
        N = int(info0["N"])
        _ubar0, _ = solve_partial_ubar_via_rho_chiba(mx_base[:N], rho0, group_id=group_id)
    except Exception:
        L = eps_list.size
        nan = np.full(L, np.nan, dtype=float)
        return nan, nan

    rng = np.random.default_rng(seed=seed_base + int(group_id) * 1000)

    resp_rho = np.full(eps_list.size, np.nan, dtype=float)
    resp_p = np.full(eps_list.size, np.nan, dtype=float)

    n0 = float(np.linalg.norm(rho0))
    n1 = float(np.linalg.norm(p0))
    if n0 <= 0.0:
        n0 = 1.0
    if n1 <= 0.0:
        n1 = 1.0

    for j, eps in enumerate(eps_list):
        if not (np.isfinite(eps) and eps > 0):
            continue

        rel_rho_trials = []
        rel_p_trials = []

        for _ in range(int(n_trials)):
            xi1 = rng.standard_normal(size=m2N_base.size)
            xi2 = rng.standard_normal(size=mx_base.size)

            m2N = m2N_base * (1.0 + float(eps) * xi1)
            mx = mx_base * (1.0 + float(eps) * xi2)

            try:
                _sigma1, p1, rho1, _ = solve_affine_pade_probability_table_hankel(
                    m2N, orders_r_2N, group_id=group_id, diagnostics=False
                )
                _ubar1, _ = solve_partial_ubar_via_rho_chiba(mx[:N], rho1, group_id=group_id)

                if (not _is_finite_all(rho1)) or (not _is_finite_all(p1)):
                    continue

                idx = _match_by_hungarian(rho0, rho1)
                rho1m = rho1[idx]
                p1m = p1[idx]

                dr = float(np.linalg.norm(rho1m - rho0))
                dp = float(np.linalg.norm(p1m - p0))

                rel_rho = dr / n0
                rel_pv = dp / n1

                if np.isfinite(rel_rho):
                    rel_rho_trials.append(rel_rho)
                if np.isfinite(rel_pv):
                    rel_p_trials.append(rel_pv)

            except Exception:
                pass

        if len(rel_rho_trials) >= int(min_success):
            resp_rho[j] = float(np.median(np.asarray(rel_rho_trials, dtype=float)))

        if len(rel_p_trials) >= int(min_success):
            resp_p[j] = float(np.median(np.asarray(rel_p_trials, dtype=float)))

    return resp_rho, resp_p


def save_global_perturb_response_heatmap(
    outdir: Path,
    group_ids: np.ndarray,
    eps_list: np.ndarray,
    resp_mat: np.ndarray,
    *,
    title: str,
    filename: str,
    cbar_label: str,
    reverse_cmap: bool = True,
    nan_color: str = "0.85",
    clip_percentiles: tuple[float, float] | None = (5.0, 95.0),
) -> None:
    """Save a global perturbation-response heatmap."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gid = np.asarray(group_ids, dtype=int).ravel()
    eps = np.asarray(eps_list, dtype=float).ravel()
    R = np.asarray(resp_mat, dtype=float)

    if R.ndim != 2 or R.shape[0] != gid.size or R.shape[1] != eps.size:
        raise ValueError("resp_mat must be (G, len(eps_list)) and match group_ids.")

    Z = np.ma.masked_invalid(R)

    L = int(eps.size)
    G = int(gid.size)

    cmap_name = "viridis_r" if reverse_cmap else "viridis"
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color=nan_color)

    vmin = None
    vmax = None
    if clip_percentiles is not None:
        lo, hi = clip_percentiles
        vals = R[np.isfinite(R)]
        if vals.size > 0:
            vmin = float(np.percentile(vals, lo))
            vmax = float(np.percentile(vals, hi))
            if not (np.isfinite(vmin) and np.isfinite(vmax)) or (vmin == vmax):
                vmin, vmax = None, None

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    im = ax.imshow(
        Z,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=(0.5, L + 0.5, 0.5, G + 0.5),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xlabel(r"Moment perturbation amplitude $\epsilon$")
    ax.set_ylabel(XLAB_G)
    ax.set_title(title)

    ax.set_xticks(np.arange(1, L + 1))
    ax.set_xticklabels([f"{e:.0e}" for e in eps], rotation=0)

    yt = [1] + list(range(50, G + 1, 50))
    ax.set_yticks([t for t in yt if 1 <= t <= G])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    fig.savefig(outdir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    HERE = Path(__file__).resolve().parent
    RUN_TAG = Path(__file__).stem

    ES = 2
    file_total = HERE / "U238_TOT.txt"
    if ES == 1:
        file_edges = HERE / "Energy_structure1.txt"
    if ES == 2:
        file_edges = HERE / "Energy_structure2.txt"
    file_partial = HERE / "U238_(N,G).txt"

    XS_t = read_cross_sections(file_total)
    XS_x = read_cross_sections(file_partial)
    edges_asc_full = read_energy_structure_1col(file_edges)

    if ES == 2:
        edges_asc = crop_edges_to_overlap_with_chiba(edges_asc_full)
        print(f"[EnergyStructure] ES=2: full={edges_asc_full.size}, cropped={edges_asc.size}")
        print(f"[EnergyStructure] range=[{edges_asc[0]:.6g}, {edges_asc[-1]:.6g}]")
    else:
        edges_asc = edges_asc_full
        print(f"[EnergyStructure] ES=1: edges={edges_asc.size}")
        print(f"[EnergyStructure] range=[{edges_asc[0]:.6g}, {edges_asc[-1]:.6g}]")

    if XS_t.size == 0 or XS_x.size == 0:
        raise RuntimeError("Empty XS data. Check your input files.")

    N_MAX = 30
    SIGMA0_GRID = np.logspace(-1, 6, 200)

    orders_pool = []
    for Ng in range(1, N_MAX + 1):
        orders_pool.extend(chiba_case_c_orders(Ng).tolist())
    orders_pool = np.unique(np.asarray(orders_pool, dtype=float))
    orders_pool = np.sort(orders_pool)

    order_to_idx = {_order_key(x): i for i, x in enumerate(orders_pool)}

    outdir = HERE / f"{RUN_TAG}_Pade_Nmax{N_MAX}_ES{ES}_CHIBA"
    outdir.mkdir(parents=True, exist_ok=True)

    moments_pool, _counts_unused, edges_desc, denom_desc = compute_group_moments_average(
        XS_t, edges_asc, orders_pool, XS_x=XS_x, side="right", assume_sorted=True
    )
    mixed_pool = compute_group_mixed_moments_average(XS_t, XS_x, edges_asc, orders_pool)

    G = moments_pool.shape[0]

    counts_hi2lo, M_nodes_hi2lo = compute_counts_and_union_nodes_hi2lo(
        XS_t, XS_x, edges_asc, side="right"
    )

    print("\nEt points per group (high->low): min/mean/max =",
          int(np.min(counts_hi2lo)), float(np.mean(counts_hi2lo)), int(np.max(counts_hi2lo)))
    print("Union nodes per group (high->low): min/mean/max =",
          int(np.min(M_nodes_hi2lo)), float(np.mean(M_nodes_hi2lo)), int(np.max(M_nodes_hi2lo)))

    max_err_by_group_pcm = np.full(G, np.nan, dtype=float)
    rms_err_by_group_pcm = np.full(G, np.nan, dtype=float)
    q_abs_err_by_group_pcm = np.full((G, len(ERR_QUANTILES)), np.nan, dtype=float)

    flag_complex = np.zeros(G, dtype=bool)
    flag_negative = np.zeros(G, dtype=bool)

    max_im_sigma = np.full(G, np.nan, dtype=float)
    max_im_p = np.full(G, np.nan, dtype=float)
    max_im_rho = np.full(G, np.nan, dtype=float)
    max_im_ubar = np.full(G, np.nan, dtype=float)
    max_im_pt = np.full(G, np.nan, dtype=float)
    min_re_sigma = np.full(G, np.nan, dtype=float)
    min_re_p = np.full(G, np.nan, dtype=float)

    condH_by_group = np.full(G, np.nan, dtype=float)
    rcondH_by_group = np.full(G, np.nan, dtype=float)

    N_used_by_group = np.full(G, np.nan, dtype=float)
    hankel_svals_by_group = np.full((G, N_MAX), np.nan, dtype=float)

    root_max_inv_abs_Qp = np.full(G, np.nan, dtype=float)
    root_min_sep_rho = np.full(G, np.nan, dtype=float)
    root_cluster_index_rho = np.full(G, np.nan, dtype=float)

    condV_pi_by_group = np.full(G, np.nan, dtype=float)
    condV_ubar_by_group = np.full(G, np.nan, dtype=float)
    resV_pi_by_group = np.full(G, np.nan, dtype=float)
    resV_ubar_by_group = np.full(G, np.nan, dtype=float)

    m2N_best_list = [None] * G
    mx_best_list = [None] * G
    orders2N_best_list = [None] * G

    print("\n=== SETTINGS ===")
    print(f"N_MAX = {N_MAX} (adaptive per group, N_g<=floor(M/2), fallback down to N=1)")
    print("orders pool size =", int(orders_pool.size))
    print("sigma0 grid:", f"[{SIGMA0_GRID[0]:.3e} ... {SIGMA0_GRID[-1]:.3e}], n={SIGMA0_GRID.size}")
    print("Output folder:", outdir)
    print("\nGroup index g in plots/summary starts at 1.")

    summary_path = outdir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as sf:
        qcols = "  ".join([f"q{q:.2f}_abs_err_pcm" for q in ERR_QUANTILES])
        sf.write(
            "g  E_high  E_low  M_nodes  N_g  b_g  rt_min  rt_max  "
            "max_abs_err_pcm  rms_err_pcm  "
            f"{qcols}  "
            "condH  condV_pi  condV_ubar  "
            "resV_pi  resV_ubar  "
            "root_maxInvAbsQp  root_minSepRho  root_clusterIndexRho  "
            "FLAG_COMPLEX  FLAG_NEG  "
            "maxIm_sigma  maxIm_p  maxIm_rho  maxIm_ubar  maxIm_PT  "
            "minRe_sigma  minRe_p  n_finite_err\n"
        )

    for e_hi2lo in range(G):
        gid1 = e_hi2lo + 1

        e_high = float(edges_desc[e_hi2lo])
        e_low = float(edges_desc[e_hi2lo + 1])

        g_low = (G - 1) - e_hi2lo

        M_nodes = group_union_node_count(XS_t, XS_x, edges_asc, g_low)
        N_init = min(N_MAX, M_nodes // 2)

        rtmin, rtmax = group_rt_minmax(XS_t, edges_asc, g_low)

        if N_init < 1:
            print(f"\n[Group {gid1}] skipped: M_nodes={M_nodes}, floor(M/2)={M_nodes//2}")
            continue

        best = None
        last_err_msg = ""

        for Ng_try in range(int(N_init), 0, -1):
            try:
                orders_r_2N_try = chiba_case_c_orders(Ng_try)
                orders_r_N_try = orders_r_2N_try[:Ng_try]
                a_try, b_try = _infer_affine_params(orders_r_2N_try)

                idx2N_try = [order_to_idx[_order_key(x)] for x in orders_r_2N_try]
                idxN_try = [order_to_idx[_order_key(x)] for x in orders_r_N_try]

                m2N_try = moments_pool[e_hi2lo, idx2N_try]
                mx_try = mixed_pool[e_hi2lo, idxN_try]

                if (not np.all(np.isfinite(m2N_try))) or (not np.all(np.isfinite(mx_try))):
                    last_err_msg = "non-finite pooled moments"
                    continue

                sigma_try, p_try, rho_try, info_try = solve_affine_pade_probability_table_hankel(
                    m2N_try, orders_r_2N_try, group_id=gid1, diagnostics=True
                )

                if (not _is_finite_all(sigma_try)) or (not _is_finite_all(p_try)) or (not _is_finite_all(rho_try)):
                    last_err_msg = "non-finite sigma/p/rho"
                    continue

                ubar_try, info_ubar_try = solve_partial_ubar_via_rho_chiba(mx_try, rho_try, group_id=gid1)
                if not _is_finite_all(ubar_try):
                    last_err_msg = "non-finite ubar"
                    continue

                ref_try = nra_effective_x_reference(XS_t, XS_x, edges_asc, g_low, SIGMA0_GRID)
                if not _is_finite_all(ref_try):
                    last_err_msg = "non-finite reference curve"
                    continue

                pt_try = nra_effective_x_probability_table_chiba_ubar(sigma_try, p_try, ubar_try, a_try, SIGMA0_GRID)

                ok_err, err_pcm_try, stats_try = _err_pcm_stats_if_usable(
                    ref_try, pt_try, group_id=gid1, min_finite=MIN_FINITE_ERR_POINTS
                )
                if not ok_err:
                    last_err_msg = f"error curve not usable (finite={stats_try.get('n_finite', 0)})"
                    continue

                best = {
                    "N_used": int(Ng_try),
                    "a": float(a_try),
                    "b": float(b_try),
                    "sigma": sigma_try,
                    "p": p_try,
                    "rho": rho_try,
                    "ubar": ubar_try,
                    "info": info_try,
                    "info_ubar": info_ubar_try,
                    "ref": ref_try,
                    "pt": pt_try,
                    "err_pcm": err_pcm_try,
                    "err_stats": stats_try,
                    "m2N": m2N_try,
                    "mx": mx_try,
                    "orders2N": orders_r_2N_try,
                }
                break

            except Exception as ex:
                last_err_msg = f"{type(ex).__name__}: {ex}"
                continue

        if best is None:
            print(f"\n[Group {gid1}] infeasible: N_init={N_init}, failed down to N=1. Last: {last_err_msg}")
            with summary_path.open("a", encoding="utf-8") as sf:
                qnans = "  ".join([f"{float('nan'):.6g}" for _ in ERR_QUANTILES])
                sf.write(
                    f"{gid1:4d}  {e_high:.8g}  {e_low:.8g}  "
                    f"{M_nodes:6d}  {N_init:3d}  {float('nan'):.8g}  "
                    f"{rtmin:.8g}  {rtmax:.8g}  "
                    f"{float('nan'):.6g}  {float('nan'):.6g}  {qnans}  "
                    f"{float('nan'):.6g}  {float('nan'):.6g}  {float('nan'):.6g}  "
                    f"{float('nan'):.6g}  {float('nan'):.6g}  "
                    f"{float('nan'):.6g}  {float('nan'):.6g}  {float('nan'):.6g}  "
                    f"{0:11d}  {0:8d}  "
                    f"{float('nan'):.6g}  {float('nan'):.6g}  {float('nan'):.6g}  {float('nan'):.6g}  {float('nan'):.6g}  "
                    f"{float('nan'):.6g}  {float('nan'):.6g}  {0:12d}\n"
                )
            continue

        N_used = best["N_used"]
        a_total = best["a"]
        b_total = best["b"]
        sigma_t_nodes = best["sigma"]
        p = best["p"]
        rho = best["rho"]
        ubar = best["ubar"]
        info = best["info"]
        info_ubar = best["info_ubar"]
        pt = best["pt"]
        err_pcm = best["err_pcm"]
        estats = best["err_stats"]

        N_used_by_group[e_hi2lo] = float(N_used)
        if info is not None and "svals_hankel" in info:
            sH = np.asarray(info["svals_hankel"], dtype=float).ravel()
            nn = min(N_MAX, sH.size)
            hankel_svals_by_group[e_hi2lo, :nn] = sH[:nn]

        root_max_inv_abs_Qp[e_hi2lo] = float(info.get("max_inv_abs_Qp_at_roots", np.nan))
        root_min_sep_rho[e_hi2lo] = float(info.get("min_root_sep_rho", np.nan))
        root_cluster_index_rho[e_hi2lo] = float(info.get("root_cluster_index_rho", np.nan))

        condV_pi_by_group[e_hi2lo] = float(info.get("cond_vandermonde_pi", np.nan))
        resV_pi_by_group[e_hi2lo] = float(info.get("res_vandermonde_pi_rel", np.nan))
        condV_ubar_by_group[e_hi2lo] = float(info_ubar.get("cond_vandermonde_ubar", np.nan))
        resV_ubar_by_group[e_hi2lo] = float(info_ubar.get("res_vandermonde_ubar_rel", np.nan))

        m2N_best_list[e_hi2lo] = np.asarray(best["m2N"], dtype=float).copy()
        mx_best_list[e_hi2lo] = np.asarray(best["mx"], dtype=float).copy()
        orders2N_best_list[e_hi2lo] = np.asarray(best["orders2N"], dtype=float).copy()

        max_abs_err_pcm = float(estats["max_abs_err_pcm"])
        rms_err_pcm = float(estats["rms_err_pcm"])
        q_abs_pcm = np.asarray(estats["q_abs_err_pcm"], dtype=float).ravel()
        n_finite_err = int(estats.get("n_finite", 0))

        warn_if_complex(sigma_t_nodes, name="sigma_t_nodes", group_id=gid1, tol=IMAG_TOL)
        warn_if_complex(p,            name="p",            group_id=gid1, tol=IMAG_TOL)
        warn_if_complex(rho,          name="rho(z_roots)", group_id=gid1, tol=IMAG_TOL)
        warn_if_complex(ubar,         name="ubar",         group_id=gid1, tol=IMAG_TOL)
        warn_if_complex(pt,           name="PT_eff_x",     group_id=gid1, tol=IMAG_TOL)

        c_sigma, mi_sigma = is_complex_flag(sigma_t_nodes, tol=IMAG_TOL)
        c_p,     mi_p     = is_complex_flag(p,            tol=IMAG_TOL)
        c_rho,   mi_rho   = is_complex_flag(rho,          tol=IMAG_TOL)
        c_ubar,  mi_ubar  = is_complex_flag(ubar,         tol=IMAG_TOL)
        c_pt,    mi_pt    = is_complex_flag(pt,           tol=IMAG_TOL)

        min_sigma = _min_real(sigma_t_nodes)
        min_pv = _min_real(p)

        neg_flag = (np.isfinite(min_sigma) and (min_sigma < NEG_TOL)) or (np.isfinite(min_pv) and (min_pv < NEG_TOL))
        comp_flag = c_sigma or c_p or c_rho or c_ubar or c_pt

        flag_negative[e_hi2lo] = bool(neg_flag)
        flag_complex[e_hi2lo] = bool(comp_flag)

        max_im_sigma[e_hi2lo] = mi_sigma
        max_im_p[e_hi2lo] = mi_p
        max_im_rho[e_hi2lo] = mi_rho
        max_im_ubar[e_hi2lo] = mi_ubar
        max_im_pt[e_hi2lo] = mi_pt
        min_re_sigma[e_hi2lo] = min_sigma
        min_re_p[e_hi2lo] = min_pv

        save_group_error_plot_pcm_scatter(
            outdir=outdir,
            group_id=gid1,
            e_high=e_high,
            e_low=e_low,
            sigma0_grid=SIGMA0_GRID,
            err_pcm=err_pcm,
            N_g=N_used,
            b_g=b_total,
            M_nodes=M_nodes,
            rms_pcm=rms_err_pcm,
            q_abs_pcm=q_abs_pcm,
        )

        max_err_by_group_pcm[e_hi2lo] = max_abs_err_pcm
        rms_err_by_group_pcm[e_hi2lo] = rms_err_pcm
        if q_abs_pcm.size == len(ERR_QUANTILES):
            q_abs_err_by_group_pcm[e_hi2lo, :] = q_abs_pcm

        condH = float(info["cond_hankel"])
        condVpi = float(info["cond_vandermonde_pi"])
        condVubar = float(info_ubar["cond_vandermonde_ubar"])
        root_cluster_val = float(info.get("root_cluster_index_rho", np.nan))

        condH_by_group[e_hi2lo] = condH
        rcondH_by_group[e_hi2lo] = float(info["rcond_hankel"])

        qtxt = ", ".join([f"q{qq:.2f}={vv:.3g}" for qq, vv in zip(ERR_QUANTILES, q_abs_pcm)])
        print(f"\n[Group {gid1}] E=[{e_high:g}->{e_low:g}]  M_nodes={M_nodes}  N_init={N_init}  N_used={N_used}  b={b_total:g}")
        print(f"  rt_min={rtmin:.6g}, rt_max={rtmax:.6g}")
        print(f"  cond(H)       = {condH:.3e} (rcond={info['rcond_hankel']:.3e})")
        print(f"  cond(V_pi)    = {condVpi:.3e} (rcond={info['rcond_vandermonde_pi']:.3e})")
        print(f"  cond(V_ubar)  = {condVubar:.3e} (rcond={info_ubar['rcond_vandermonde_ubar']:.3e})")
        print(f"  res(V*pi)     = {info['res_vandermonde_pi_rel']:.3e}")
        print(f"  res(V*ubar)   = {info_ubar['res_vandermonde_ubar_rel']:.3e}")
        print(f"  max 1/|Q'|    = {info['max_inv_abs_Qp_at_roots']:.3e}")
        print(f"  min |Δrho|    = {info['min_root_sep_rho']:.3e}")
        print(f"  cluster index = {root_cluster_val:.3e}")
        print(f"  flags: complex={int(comp_flag)}  negative={int(neg_flag)}  (minRe_sigma={min_sigma:.6g}, minRe_p={min_pv:.6g})")
        print(f"  finite sigma0 points = {n_finite_err}")
        print(f"  max|err| = {max_abs_err_pcm:.3f} pcm")
        print(f"  RMS(err) = {rms_err_pcm:.3f} pcm")
        print(f"  |err| quantiles: {qtxt}")

        print("  sigma_t:", format_array_maybe_real(sigma_t_nodes, group_id=gid1, name="sigma_t_nodes", tol=IMAG_TOL))
        print("  p      :", format_array_maybe_real(p,            group_id=gid1, name="p",            tol=IMAG_TOL))
        print("  ubar   :", format_array_maybe_real(ubar,         group_id=gid1, name="ubar",         tol=IMAG_TOL))

        with np.errstate(divide="ignore", invalid="ignore"):
            psigx = ubar / np.power(sigma_t_nodes, a_total)
            sigx_print = psigx / p
        print("  p*sigma_x:", format_array_maybe_real(psigx,      group_id=gid1, name="p*sigma_x", tol=IMAG_TOL))
        print("  sigma_x  :", format_array_maybe_real(sigx_print, group_id=gid1, name="sigma_x",   tol=IMAG_TOL))

        with summary_path.open("a", encoding="utf-8") as sf:
            qvals = (
                "  ".join([f"{vv:.6g}" for vv in q_abs_pcm.tolist()])
                if q_abs_pcm.size == len(ERR_QUANTILES)
                else "  ".join([f"{float('nan'):.6g}" for _ in ERR_QUANTILES])
            )
            sf.write(
                f"{gid1:4d}  {e_high:.8g}  {e_low:.8g}  "
                f"{M_nodes:6d}  {N_used:3d}  {b_total:.8g}  "
                f"{rtmin:.8g}  {rtmax:.8g}  "
                f"{max_abs_err_pcm:.6g}  {rms_err_pcm:.6g}  {qvals}  "
                f"{condH:.6g}  {condVpi:.6g}  {condVubar:.6g}  "
                f"{info['res_vandermonde_pi_rel']:.6g}  {info_ubar['res_vandermonde_ubar_rel']:.6g}  "
                f"{info['max_inv_abs_Qp_at_roots']:.6g}  {info['min_root_sep_rho']:.6g}  {root_cluster_val:.6g}  "
                f"{int(comp_flag):11d}  {int(neg_flag):8d}  "
                f"{mi_sigma:.6g}  {mi_p:.6g}  {mi_rho:.6g}  {mi_ubar:.6g}  {mi_pt:.6g}  "
                f"{min_sigma:.6g}  {min_pv:.6g}  {n_finite_err:12d}\n"
            )

    group_ids_1based = np.arange(1, G + 1, dtype=int)

    save_max_error_by_group_plot_log(
        outdir=outdir,
        group_ids=group_ids_1based,
        max_err_pcm=max_err_by_group_pcm,
        flag_complex=flag_complex,
        flag_negative=flag_negative,
        filename="max_err_by_group.png",
    )

    save_q95_error_by_group_plot_log(
        outdir=outdir,
        group_ids=group_ids_1based,
        q_abs_err_pcm=q_abs_err_by_group_pcm,
        flag_complex=flag_complex,
        flag_negative=flag_negative,
        q_target=0.95,
        quantiles=ERR_QUANTILES,
        filename="q95_err_by_group.png",
    )

    save_rms_error_by_group_plot_log(
        outdir=outdir,
        group_ids=group_ids_1based,
        rms_err_pcm=rms_err_by_group_pcm,
        flag_complex=flag_complex,
        flag_negative=flag_negative,
        filename="rms_err_by_group.png",
    )

    save_quantile_error_by_group_plot_log(
        outdir=outdir,
        group_ids=group_ids_1based,
        q_abs_err_pcm=q_abs_err_by_group_pcm,
        quantiles=ERR_QUANTILES,
        filename="qerr_by_group.png",
    )

    save_condH_by_group_plot_log(
        outdir=outdir,
        group_ids=group_ids_1based,
        condH=condH_by_group,
        filename="condH_by_group.png",
    )

    save_condH_by_group_txt(
        outdir=outdir,
        group_ids=group_ids_1based,
        condH=condH_by_group,
        filename="condH_by_group.txt",
    )

    save_hankel_svals_examples(
        outdir=outdir,
        svals_by_group=hankel_svals_by_group,
        N_used_by_group=N_used_by_group,
        condH_by_group=condH_by_group,
        filename="hankel_svals_examples.png",
    )
    save_hankel_svals_heatmap(
        outdir=outdir,
        svals_by_group=hankel_svals_by_group,
        filename="hankel_svals_heatmap.png",
    )

    save_root_sensitivity_by_group_plot(
        outdir=outdir,
        group_ids=group_ids_1based,
        max_inv_abs_Qp=root_max_inv_abs_Qp,
        filename="root_sensitivity_by_group.png",
    )
    save_root_separation_by_group_plot(
        outdir=outdir,
        group_ids=group_ids_1based,
        min_root_sep=root_min_sep_rho,
        filename="root_separation_by_group.png",
    )
    save_root_clustering_by_group_plot(
        outdir=outdir,
        group_ids=group_ids_1based,
        root_cluster_index=root_cluster_index_rho,
        filename="root_clustering_by_group.png",
    )

    save_partial_recon_cond_by_group_plot(
        outdir=outdir,
        group_ids=group_ids_1based,
        condV_pi=condV_pi_by_group,
        condV_ubar=condV_ubar_by_group,
        filename="partial_recon_cond_by_group.png",
    )

    save_partial_recon_res_by_group_plot(
        outdir=outdir,
        group_ids=group_ids_1based,
        res_pi=resV_pi_by_group,
        res_ubar=resV_ubar_by_group,
        filename="partial_recon_res_by_group.png",
    )

    N_REP_PERTURB = 12
    rep_groups = _select_representative_groups_by_cond(
        condH_by_group,
        n_reps=N_REP_PERTURB,
        mode="logquantile",
    )
    if len(rep_groups) > 0:
        print("\nRepresentative perturbation groups:", rep_groups)
        for g1 in rep_groups:
            i = g1 - 1
            m2N = m2N_best_list[i]
            mx = mx_best_list[i]
            ord2N = orders2N_best_list[i]
            if (m2N is None) or (mx is None) or (ord2N is None):
                continue
            save_group_perturbation_response(
                outdir=outdir,
                group_id=g1,
                m2N_base=m2N,
                mx_base=mx,
                orders_r_2N=ord2N,
                n_trials=20,
                eps_list=np.array([1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6], dtype=float),
                filename_prefix="group",
            )

    HEAT_EPS_LIST = np.array([1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6], dtype=float)
    HEAT_N_TRIALS = 10
    HEAT_MIN_SUCCESS = 5
    HEAT_SEED_BASE = 13579

    resp_rho_all = np.full((G, HEAT_EPS_LIST.size), np.nan, dtype=float)
    resp_p_all = np.full((G, HEAT_EPS_LIST.size), np.nan, dtype=float)

    print("\n[GlobalHeatmap] Computing perturbation responses for all groups...")
    for g1 in range(1, G + 1):
        i = g1 - 1
        m2N = m2N_best_list[i]
        mx = mx_best_list[i]
        ord2N = orders2N_best_list[i]
        if (m2N is None) or (mx is None) or (ord2N is None):
            continue

        rr, rp = _group_median_perturb_response(
            group_id=g1,
            m2N_base=m2N,
            mx_base=mx,
            orders_r_2N=ord2N,
            eps_list=HEAT_EPS_LIST,
            n_trials=HEAT_N_TRIALS,
            min_success=HEAT_MIN_SUCCESS,
            seed_base=HEAT_SEED_BASE,
        )
        resp_rho_all[i, :] = rr
        resp_p_all[i, :] = rp

    save_global_perturb_response_heatmap(
        outdir=outdir,
        group_ids=group_ids_1based,
        eps_list=HEAT_EPS_LIST,
        resp_mat=resp_rho_all,
        title=r"Global perturbation response: median$(\|\Delta \rho\|/\|\rho\|)$",
        filename="perturb_resp_rho_heatmap.png",
        cbar_label=r"$\mathrm{median}(\|\Delta\rho\|/\|\rho\|)$",
    )
    save_global_perturb_response_heatmap(
        outdir=outdir,
        group_ids=group_ids_1based,
        eps_list=HEAT_EPS_LIST,
        resp_mat=resp_p_all,
        title=r"Global perturbation response: median$(\|\Delta p\|/\|p\|)$",
        filename="perturb_resp_p_heatmap.png",
        cbar_label=r"$\mathrm{median}(\|\Delta p\|/\|p\|)$",
    )

    print("[GlobalHeatmap] Saved: perturb_resp_rho_heatmap.png, perturb_resp_p_heatmap.png")
    print("\nDone. Outputs saved to:", outdir)
    print("Summary:", summary_path)
    print("Saved: max_err_by_group.png, rms_err_by_group.png, qerr_by_group.png, q95_err_by_group.png, condH_by_group.png")
    print("Saved: condH_by_group.txt")
    print("Saved: hankel_svals_examples.png, hankel_svals_heatmap.png")
    print("Saved: root_sensitivity_by_group.png, root_separation_by_group.png, root_clustering_by_group.png")
    print("Saved: partial_recon_cond_by_group.png, partial_recon_res_by_group.png")
    print("Saved: group_XXX_perturb_response.png")
    print("Saved: perturb_resp_rho_heatmap.png, perturb_resp_p_heatmap.png")