# -*- coding: utf-8 -*-
"""
Lanczos probability-table construction and NRA error diagnostics.
Author: Beichen Zheng

Case C is used throughout:
- a = -1
- b = 1 / (N - 1)
- ubar = p * sigma_x * sigma_t**a

Modes:
- realization: "ana", "trap"
- sigma_x reconstruction: "eigsolve", "positive_hat", "fullcorr"
"""

from __future__ import annotations

import os

# ------------------------------------------------------------------
# Avoid oversubscription; set BLAS/OpenMP thread caps before numpy/scipy.
# ------------------------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from pathlib import Path
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context, freeze_support
import math
import re
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter

try:
    from scipy.linalg import eigh_tridiagonal
except Exception:
    eigh_tridiagonal = None

try:
    from scipy.optimize import Bounds, LinearConstraint, minimize
except Exception:
    Bounds = None
    LinearConstraint = None
    minimize = None

# ============================================================
# Global controls
# ============================================================
SERIES_X_TOL = 1e-6
IMAG_TOL = 0
NEG_TOL = 0  # mark as negative if Re(x) < NEG_TOL (ignore tiny roundoff)

POINT_SIZE_GROUP = 3
POINT_SIZE_SUMMARY = 3
POINT_SIZE_MOMENT = 3

COLOR_RELERR = "#3E5C76"

WARN_LOG: list[str] = []
# ------------------------------------------------------------
# Selective reorth controls
# ------------------------------------------------------------
# 达到这个维度后才开始做 Ritz 锁定检查
SELECTIVE_LOCK_MIN_K_DEFAULT = 6

# beta_probe / spectral_scale 的阈值；residual 偏大时只周期性强制检查。
SELECTIVE_LOCK_BETA_REL_TRIGGER_DEFAULT = 5e-3

# residual 仍偏大时，也每隔若干次 probe 强制检查一次。
SELECTIVE_LOCK_FORCE_EVERY_DEFAULT = 8

def log_warning(msg: str) -> None:
    print(msg)
    WARN_LOG.append(msg)

# ============================================================
# Output folder naming helpers
# ============================================================
def _safe_stem(s: str) -> str:
    s = str(s).strip()
    if not s:
        return "run"
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_") or "run"

def build_outdir_name(
    *,
    script_stem: str,
    input_tag: str,
    ES: int,
    N_MAX: int,
    N_DENSE: int,
    M_MIN: int,
    REALIZATION_MODE: str,
    SIGMA_X_METHOD: str,
    Z_AFFINE_NORMALIZE: bool,
    Z_NORM_METHOD: str,
    GL_NQ: int,
    QP_RETENTION_COUNT: int | None = None,
) -> str:
    stem = _safe_stem(script_stem)
    input_tag_safe = _safe_stem(input_tag)

    rmode = str(REALIZATION_MODE).strip().lower()
    if rmode not in ("ana", "trap"):
        rmode = "unknown"

    sx_method = str(SIGMA_X_METHOD).strip().lower()
    if sx_method == "positive_hat":
        sx_tag = "hat"
    elif sx_method in ("fullcorr", "active_set", "admissible_correction", "full_first", "qp", "constrained_qp"):
        sx_tag = "corr"
    else:
        sx_tag = _safe_stem(sx_method)

    if Z_AFFINE_NORMALIZE:
        z_tag = "Z" + _safe_stem(Z_NORM_METHOD)
    else:
        z_tag = "Znone"

    quad_tag = f"_GL{int(GL_NQ)}" if rmode == "ana" else ""

    ret_tag = ""
    if QP_RETENTION_COUNT is not None:
        ret_tag = f"_Ret{int(QP_RETENTION_COUNT)}"

    return (
        f"{stem}_{input_tag_safe}_ES{int(ES)}_Nmax{int(N_MAX)}_"
        f"R{rmode}{quad_tag}_Sx{sx_tag}{ret_tag}_{z_tag}_Mmin{int(M_MIN)}_reorth"
    )

# ============================================================
# Realization-mode labels
# ============================================================
def get_realization_labels(realization_mode: str) -> dict[str, str]:
    r = str(realization_mode).strip().lower()
    if r == "ana":
        return {
            "mode": "ana",
            "ref_label": "analytic reference",
            "direct_label": "GL-direct surrogate",
            "discrete_label": "piecewise Gauss-Legendre discrete realization",
            "short_label": "ana",
        }
    if r == "trap":
        return {
            "mode": "trap",
            "ref_label": "trapezoidal reference",
            "direct_label": "trap-direct surrogate",
            "discrete_label": "nodal trapezoidal discrete realization",
            "short_label": "trap",
        }
    raise ValueError("REALIZATION_MODE must be 'ana' or 'trap'.")

# ============================================================
# Complex / negativity helpers
# ============================================================
def _max_abs_imag(x: np.ndarray) -> float:
    x = np.asarray(x)
    if np.iscomplexobj(x):
        return float(np.nanmax(np.abs(np.imag(x)))) if x.size else 0.0
    return 0.0

def _min_real(x: np.ndarray) -> float:
    x = np.asarray(x)
    if x.size == 0:
        return float("nan")
    xr = np.real(x)
    m = np.isfinite(xr)
    if not np.any(m):
        return float("nan")
    return float(np.nanmin(xr[m]))

def classify_complex_negative(
    x: np.ndarray,
    *,
    imag_tol: float = IMAG_TOL,
    neg_tol: float = NEG_TOL,
) -> tuple[bool, bool, float, float]:
    """
    Return:
      is_complex, is_negative, min_real, max_abs_imag

    Negative means:
      min Re(x) < neg_tol
    Complex means:
      max |Im(x)| > imag_tol
    """
    x = np.asarray(x)
    max_im = _max_abs_imag(x)
    is_complex = bool(np.isfinite(max_im) and (max_im > imag_tol))

    min_real = _min_real(x)
    is_negative = bool(np.isfinite(min_real) and (min_real < neg_tol))

    return is_complex, is_negative, float(min_real), float(max_im)

def warn_if_complex(x: np.ndarray, *, name: str, group_no: int, tol: float = IMAG_TOL) -> bool:
    mi = _max_abs_imag(x)
    if np.isfinite(mi) and mi > tol:
        log_warning(f"[WARNING][Group {group_no}] {name} has non-zero imaginary part: max|Im| = {mi:.3e}")
        return True
    return False

def as_real_for_plot(y: np.ndarray, *, name: str, group_no: int, tol: float = IMAG_TOL) -> np.ndarray:
    y = np.asarray(y)
    if not np.iscomplexobj(y):
        return y
    mi = _max_abs_imag(y)
    if np.isfinite(mi) and mi > tol:
        log_warning(f"[WARNING][Group {group_no}] {name} is complex: max|Im| = {mi:.3e}. Plot uses Re({name}).")
    return np.real(y)

def _cond2(A: np.ndarray) -> tuple[float, float]:
    try:
        s = np.linalg.svd(A, compute_uv=False)
    except Exception:
        return float("inf"), 0.0
    if s.size == 0:
        return float("inf"), 0.0
    smax = float(np.max(s))
    smin = float(np.min(s))
    if (not np.isfinite(smax)) or smax == 0.0:
        return float("inf"), 0.0
    if (not np.isfinite(smin)) or smin == 0.0:
        return float("inf"), 0.0
    cond = smax / smin
    return float(cond), float(1.0 / cond)

# ============================================================
# I/O
# ============================================================
def read_cross_sections(file_path: Path) -> np.ndarray:
    """
    Read (E, sigma) from a whitespace-separated text file.
    Skip lines starting with # or !. Use first two columns as floats.
    Returns sorted by energy ascending.
    """
    file_path = Path(file_path)
    rows = []
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
            except Exception:
                continue
            if np.isfinite(e) and np.isfinite(s):
                rows.append((e, s))

    if not rows:
        return np.empty((0, 2), dtype=float)

    arr = np.asarray(rows, dtype=float)
    arr = arr[np.argsort(arr[:, 0])]
    return arr

def read_energy_structure_1col(path: Path) -> np.ndarray:
    """
    Read energy group boundaries from a text file (single column).
    Returns ascending unique boundaries.
    """
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

# ============================================================
# Crop ES=2 to Chiba overlap
# ============================================================
def crop_edges_to_overlap_with_chiba(edges_asc: np.ndarray) -> np.ndarray:
    """
    Crop an ascending energy-boundary array to the overlap with Chiba group range.

    Chiba range:
        E_hi = 9.1188e3
        E_lo = 5.0435
    """
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

# ============================================================
# Union-grid densification (linear-in-energy)
# ============================================================
def densify_union_nodes_linear(nodes: np.ndarray, target_count: int) -> np.ndarray:
    """
    Densify a sorted unique node array to reach at least target_count nodes by
    subdividing each interval with LINEAR-in-energy points.
    """
    nodes = np.asarray(nodes, dtype=float).ravel()
    nodes = nodes[np.isfinite(nodes)]
    nodes = np.unique(np.sort(nodes))
    if nodes.size < 2:
        return nodes
    target_count = int(target_count)
    if target_count <= nodes.size:
        return nodes

    span = float(nodes[-1] - nodes[0])
    if not np.isfinite(span) or span <= 0.0:
        return nodes

    h_max = span / float(target_count - 1)
    if not np.isfinite(h_max) or h_max <= 0.0:
        return nodes

    extra = []
    for j in range(nodes.size - 1):
        xL = float(nodes[j])
        xR = float(nodes[j + 1])
        dx = xR - xL
        if dx <= 0.0:
            continue
        n = int(np.ceil(dx / h_max))
        if n <= 1:
            continue
        pts = np.linspace(xL, xR, n + 1, endpoint=True)[1:-1]
        if pts.size:
            extra.append(pts)

    if not extra:
        return nodes

    nodes2 = np.concatenate([nodes] + extra)
    nodes2 = nodes2[np.isfinite(nodes2)]
    nodes2 = np.unique(np.sort(nodes2))
    return nodes2

# ============================================================
# Discrete samples: Scheme A (piecewise Gauss-Legendre on union segments)
# ============================================================
_GL_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}

def _leggauss_cached(nq: int) -> tuple[np.ndarray, np.ndarray]:
    nq = int(nq)
    if nq not in _GL_CACHE:
        x, w = np.polynomial.legendre.leggauss(nq)
        _GL_CACHE[nq] = (x.astype(float), w.astype(float))
    return _GL_CACHE[nq]

def build_group_discrete_samples_gl_nodes(
    XS_t: np.ndarray,
    XS_x: np.ndarray,
    edges_asc: np.ndarray,
    g_low: int,
    *,
    densify_gauss_N: int | None = None,
    gl_nq: int = 4,
    max_densify_rounds: int = 3,
):
    """
    Scheme A:
    Build discrete samples at Gauss-Legendre points within each UNION segment [E_j,E_{j+1}],
    assuming rt(E), rx(E) are LINEAR in E on that segment.

    Output:
        rt_node, rx_node, w_base, width, cover_dx, skip_dx, skip_seg

    where w_base are energy quadrature weights normalized by group width:
        <f>_Eavg ≈ sum_i w_base[i] * f(sample_i),
        w_base_i = wE_i / width,  wE_i are Gauss-Legendre weights on energy.
    """
    gl_nq = int(gl_nq)
    if gl_nq < 1:
        raise ValueError("gl_nq must be >= 1")

    edges = np.asarray(edges_asc, dtype=float)
    Et, Rt = XS_t[:, 0], XS_t[:, 1]
    Ex, Rx = XS_x[:, 0], XS_x[:, 1]

    eL = float(edges[g_low])
    eR = float(edges[g_low + 1])
    width = eR - eL
    if width <= 0:
        return (np.empty(0), np.empty(0), np.empty(0), width, 0.0, 0.0, 0)

    mask_t = (Et >= eL) & (Et <= eR)
    mask_x = (Ex >= eL) & (Ex <= eR)

    nodes = np.concatenate(([eL, eR], Et[mask_t], Ex[mask_x]))
    nodes = np.unique(nodes)
    nodes = nodes[(nodes >= eL) & (nodes <= eR)]
    nodes = np.unique(np.sort(nodes))
    if nodes.size < 2:
        return (np.empty(0), np.empty(0), np.empty(0), width, 0.0, 0.0, 0)

    target_atoms = None
    if densify_gauss_N is not None:
        Nd = int(densify_gauss_N)
        if Nd >= 2:
            target_atoms = int(2 * Nd)

    def _assemble(nodes_use: np.ndarray):
        rt_u = np.interp(nodes_use, Et, Rt)
        rx_u = np.interp(nodes_use, Ex, Rx)

        xi, wi = _leggauss_cached(gl_nq)

        rt_list = []
        rx_list = []
        w_list = []

        skip_seg = 0
        skip_dx = 0.0
        cover_dx = 0.0

        for j in range(nodes_use.size - 1):
            EL = float(nodes_use[j])
            ER = float(nodes_use[j + 1])
            dx = ER - EL
            if dx <= 0.0 or (not np.isfinite(dx)):
                continue

            rtL = float(rt_u[j])
            rtR = float(rt_u[j + 1])
            rxL = float(rx_u[j])
            rxR = float(rx_u[j + 1])

            if (not np.isfinite(rtL + rtR)) or (rtL <= 0.0) or (rtR <= 0.0):
                skip_seg += 1
                skip_dx += dx
                continue

            cover_dx += dx

            mid = 0.5 * (EL + ER)
            half = 0.5 * dx
            Epts = mid + half * xi
            wE = half * wi

            t = (Epts - EL) / dx
            rt_pts = rtL + (rtR - rtL) * t
            rx_pts = rxL + (rxR - rxL) * t

            w_base = wE / width

            rt_list.append(rt_pts)
            rx_list.append(rx_pts)
            w_list.append(w_base)

        if not rt_list:
            return (np.empty(0), np.empty(0), np.empty(0), cover_dx, skip_dx, skip_seg)

        rt_node = np.concatenate(rt_list).astype(float)
        rx_node = np.concatenate(rx_list).astype(float)
        w_base = np.concatenate(w_list).astype(float)

        m = np.isfinite(rt_node) & np.isfinite(rx_node) & np.isfinite(w_base) & (w_base > 0.0)
        rt_node = rt_node[m]
        rx_node = rx_node[m]
        w_base = w_base[m]

        return (rt_node, rx_node, w_base, cover_dx, skip_dx, skip_seg)

    nodes_use = nodes
    if target_atoms is not None:
        for _round in range(int(max_densify_rounds)):
            rt_node, rx_node, w_base, cover_dx, skip_dx, skip_seg = _assemble(nodes_use)
            M = int(rt_node.size)
            if M >= target_atoms or nodes_use.size >= 5000:
                return rt_node, rx_node, w_base, width, cover_dx, skip_dx, skip_seg

            seg_need = int(np.ceil(target_atoms / float(gl_nq)))
            target_nodes = max(int(nodes_use.size), int(seg_need + 1))
            target_nodes = max(target_nodes, int(nodes_use.size * 2 - 1))

            nodes_use = densify_union_nodes_linear(nodes_use, target_nodes)

        rt_node, rx_node, w_base, cover_dx, skip_dx, skip_seg = _assemble(nodes_use)
        return rt_node, rx_node, w_base, width, cover_dx, skip_dx, skip_seg

    rt_node, rx_node, w_base, cover_dx, skip_dx, skip_seg = _assemble(nodes_use)
    return rt_node, rx_node, w_base, width, cover_dx, skip_dx, skip_seg

# ============================================================
# Discrete samples: trap nodal union-grid realization
# ============================================================
def build_group_discrete_samples_trapz_nodes(
    XS_t: np.ndarray,
    XS_x: np.ndarray,
    edges_asc: np.ndarray,
    g_low: int,
    *,
    densify_gauss_N: int | None = None,
):
    """
    Build discrete samples at UNION NODES inside group, with weights w_base such that:
        <f>_Eavg ≈ sum_i w_base[i] * f(node_i)
    where w_base are trapezoid weights normalized by group width ΔE.

    If densify_gauss_N is provided and the raw union nodes are too sparse,
    densify the union grid by linear-in-energy interpolation so that
    node_count >= 2*densify_gauss_N (to support Gauss order densify_gauss_N under HARD: N<=floor(M/2)).

    Strict skip on any segment whose total XS endpoints are non-positive.
    """
    edges = np.asarray(edges_asc, dtype=float)
    Et, Rt = XS_t[:, 0], XS_t[:, 1]
    Ex, Rx = XS_x[:, 0], XS_x[:, 1]

    eL = float(edges[g_low])
    eR = float(edges[g_low + 1])
    width = eR - eL
    if width <= 0:
        return (np.empty(0), np.empty(0), np.empty(0), width, 0.0, 0.0, 0)

    mask_t = (Et >= eL) & (Et <= eR)
    mask_x = (Ex >= eL) & (Ex <= eR)

    nodes = np.concatenate(([eL, eR], Et[mask_t], Ex[mask_x]))
    nodes = np.unique(nodes)
    nodes = nodes[(nodes >= eL) & (nodes <= eR)]
    nodes = np.unique(np.sort(nodes))

    if nodes.size < 2:
        return (np.empty(0), np.empty(0), np.empty(0), width, 0.0, 0.0, 0)

    if densify_gauss_N is not None:
        Nd = int(densify_gauss_N)
        if Nd >= 2:
            target_nodes = int(max(nodes.size, 2 * Nd))
            if nodes.size < target_nodes:
                nodes = densify_union_nodes_linear(nodes, target_nodes)

    if nodes.size < 2:
        return (np.empty(0), np.empty(0), np.empty(0), width, 0.0, 0.0, 0)

    rt = np.interp(nodes, Et, Rt)
    rx = np.interp(nodes, Ex, Rx)

    w = np.zeros_like(nodes, dtype=float)

    skip_seg = 0
    skip_dx = 0.0
    cover_dx = 0.0

    for j in range(nodes.size - 1):
        xL = float(nodes[j])
        xR = float(nodes[j + 1])
        dx = xR - xL
        if dx <= 0:
            continue
        cover_dx += dx

        rtL = float(rt[j])
        rtR = float(rt[j + 1])

        if (not np.isfinite(rtL + rtR)) or (rtL <= 0.0) or (rtR <= 0.0):
            skip_seg += 1
            skip_dx += dx
            continue

        add = dx / (2.0 * width)
        w[j] += add
        w[j + 1] += add

    mask = (w > 0) & np.isfinite(rt) & np.isfinite(rx)
    rt_node = rt[mask].astype(float)
    rx_node = rx[mask].astype(float)
    w_base = w[mask].astype(float)

    if rt_node.size == 0:
        return (np.empty(0), np.empty(0), np.empty(0), width, cover_dx, skip_dx, skip_seg)

    return rt_node, rx_node, w_base, width, cover_dx, skip_dx, skip_seg

# ============================================================
# Realization dispatchers
# ============================================================
def build_group_discrete_samples(
    *,
    realization_mode: str,
    XS_t: np.ndarray,
    XS_x: np.ndarray,
    edges_asc: np.ndarray,
    g_low: int,
    densify_gauss_N: int | None,
    gl_nq: int,
):
    r = str(realization_mode).strip().lower()
    if r == "ana":
        return build_group_discrete_samples_gl_nodes(
            XS_t, XS_x, edges_asc, g_low,
            densify_gauss_N=densify_gauss_N,
            gl_nq=gl_nq,
        )
    if r == "trap":
        return build_group_discrete_samples_trapz_nodes(
            XS_t, XS_x, edges_asc, g_low,
            densify_gauss_N=densify_gauss_N,
        )
    raise ValueError("Unknown realization_mode. Use 'ana' or 'trap'.")

# ============================================================
# Diagnostics: orthogonality & tridiagonalization residual
# ============================================================
def tridiag_matrix(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    alpha = np.asarray(alpha, dtype=float).ravel()
    beta = np.asarray(beta, dtype=float).ravel()
    N = alpha.size
    T = np.diag(alpha)
    if N >= 2:
        T += np.diag(beta, 1) + np.diag(beta, -1)
    return T

def diagnostics_lanczos(z_work: np.ndarray, Q: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> tuple[float, float, float]:
    """
    Returns:
      orth_inf = ||Q^T Q - I||_inf
      orth_off = max_{i!=j} |(Q^T Q)_{ij}|
      tri_res  = ||A Q - Q T||_F / ||A||_F  with A=diag(z_work)
    """
    z = np.asarray(z_work, dtype=float).ravel()
    Q = np.asarray(Q, dtype=float)
    N = Q.shape[1] if Q.ndim == 2 else 0
    if N == 0 or z.size == 0:
        return float("nan"), float("nan"), float("nan")

    Gm = Q.T @ Q
    I = np.eye(N, dtype=float)
    D = Gm - I
    orth_inf = float(np.linalg.norm(D, ord=np.inf))

    if N == 1:
        orth_off = 0.0
    else:
        off = D.copy()
        np.fill_diagonal(off, 0.0)
        orth_off = float(np.max(np.abs(off)))

    T = tridiag_matrix(alpha, beta)
    AQ = z[:, None] * Q
    QT = Q @ T
    num = float(np.linalg.norm(AQ - QT, ord="fro"))
    den = float(np.linalg.norm(z))
    tri_res = num / den if den > 0 else float("nan")
    return orth_inf, orth_off, tri_res

# ============================================================
# Optional affine normalization of z (recommended)
# ============================================================
def affine_normalize_z(z: np.ndarray, w_norm: np.ndarray, method: str = "minmax") -> tuple[np.ndarray, float, float]:
    """
    Returns z_work, c, s such that z = s*z_work + c.
    """
    z = np.asarray(z, dtype=float).ravel()
    w = np.asarray(w_norm, dtype=float).ravel()
    if z.size == 0:
        return z, 0.0, 1.0

    if method == "wmean_wstd":
        c = float(np.dot(w, z))
        v = float(np.dot(w, (z - c) ** 2))
        s = float(np.sqrt(v)) if v > 0 else 1.0
    else:
        zmin = float(np.min(z))
        zmax = float(np.max(z))
        c = 0.5 * (zmin + zmax)
        s = 0.5 * (zmax - zmin)
        if not np.isfinite(s) or s == 0.0:
            s = 1.0

    z_work = (z - c) / s
    return z_work, c, s

# ============================================================
# Lanczos + Golub-Welsch with NONE / SELECTIVE / FULL reorth
# ============================================================
def lanczos_tridiag_from_diag(
    z: np.ndarray,
    w_norm: np.ndarray,
    N: int,
    *,
    reorth_mode: str = "none",
    sel_tol: float = 1e-10,
    sel_check_every: int = 1,
    full_passes: int = 2,
    return_profile: bool = False,
    lock_check_min_k: int = SELECTIVE_LOCK_MIN_K_DEFAULT,
    lock_beta_rel_trigger: float = SELECTIVE_LOCK_BETA_REL_TRIGGER_DEFAULT,
    lock_force_every: int = SELECTIVE_LOCK_FORCE_EVERY_DEFAULT,
):
    """
    Symmetric Lanczos on A=diag(z) with start vector v0=sqrt(w_norm), normalized.

    Optional internal timing/profile output.
    In selective mode, a cheap gate is used before the current tridiagonal eigendecomposition.

    return_profile=False:
        return alpha, beta, Q

    return_profile=True:
        return alpha, beta, Q, profile
    """
    t_func0 = perf_counter()

    z = np.asarray(z, dtype=float).ravel()
    w = np.asarray(w_norm, dtype=float).ravel()
    if z.size != w.size:
        raise ValueError("z and w_norm size mismatch.")
    if z.size == 0:
        raise ValueError("Empty z.")
    if np.any(w < 0) or not np.isfinite(w.sum()):
        raise ValueError("Invalid weights.")
    sw = float(w.sum())
    if sw <= 0:
        raise ValueError("sum(w_norm) must be > 0.")
    w = w / sw

    reorth_mode = str(reorth_mode).lower().strip()
    if reorth_mode not in ("none", "selective", "full"):
        raise ValueError("reorth_mode must be one of: 'none', 'selective', 'full'.")

    if sel_check_every < 1:
        sel_check_every = 1
    if full_passes < 1:
        full_passes = 1

    lock_check_min_k = max(int(lock_check_min_k), 2)
    if not np.isfinite(lock_beta_rel_trigger) or lock_beta_rel_trigger <= 0.0:
        lock_beta_rel_trigger = SELECTIVE_LOCK_BETA_REL_TRIGGER_DEFAULT
    if lock_force_every < 1:
        lock_force_every = 1

    M = z.size
    if N < 1:
        raise ValueError("N must be >= 1.")

    Q = np.zeros((M, N), dtype=float)

    Y_lock = np.empty((M, 0), dtype=float)
    theta_lock: list[float] = []

    LOCK_ADD_TOL = 1e-12
    THETA_DUP_ABS_TOL = max(1e-12, 10.0 * float(sel_tol))
    THETA_DUP_REL_TOL = max(1e-10, 10.0 * float(sel_tol))

    eps = float(np.finfo(float).eps)
    SEMIORTH_TRIGGER = max(np.sqrt(eps), 10.0 * float(sel_tol))
    SEMIORTH_ACCEPT = 0.1 * SEMIORTH_TRIGGER

    spectral_scale = float(max(np.max(np.abs(z)), 1.0))

    profile = {
        "mode": reorth_mode,
        "time_total_internal": 0.0,
        "time_lock_probe_norm": 0.0,
        "time_lock_eigh": 0.0,
        "time_lock_accept": 0.0,
        "time_lock_apply": 0.0,
        "time_semiorth_measure": 0.0,
        "time_q_fullreorth": 0.0,
        "n_lock_probe": 0,
        "n_lock_eigh": 0,
        "n_lock_added": 0,
        "n_lock_skip_smallk": 0,
        "n_lock_skip_beta": 0,
        "n_semiorth_trigger": 0,
        "n_q_fullreorth": 0,
    }

    def _orth_against_basis(x: np.ndarray, B: np.ndarray, passes: int) -> np.ndarray:
        if B.size == 0:
            return x
        y = x
        for _ in range(int(passes)):
            coeff = B.T @ y
            y = y - B @ coeff
        return y

    def _full_reorth_against_Q(x: np.ndarray, Qbasis: np.ndarray) -> np.ndarray:
        if Qbasis.size == 0:
            return x
        return _orth_against_basis(x, Qbasis, full_passes)

    def _append_locked_vectors(Y_new: np.ndarray, theta_new: np.ndarray) -> int:
        nonlocal Y_lock, theta_lock

        if Y_new.size == 0:
            return 0

        accepted_cols: list[np.ndarray] = []
        accepted_thetas: list[float] = []

        for jcol in range(Y_new.shape[1]):
            th = float(theta_new[jcol])

            is_dup = False
            for th_old in theta_lock:
                if abs(th - th_old) <= THETA_DUP_ABS_TOL + THETA_DUP_REL_TOL * max(abs(th), abs(th_old), 1.0):
                    is_dup = True
                    break
            if not is_dup:
                for th_old in accepted_thetas:
                    if abs(th - th_old) <= THETA_DUP_ABS_TOL + THETA_DUP_REL_TOL * max(abs(th), abs(th_old), 1.0):
                        is_dup = True
                        break
            if is_dup:
                continue

            y = Y_new[:, jcol].astype(float, copy=True)

            if Y_lock.shape[1] > 0:
                y = _orth_against_basis(y, Y_lock, full_passes)

            if accepted_cols:
                Btmp = np.column_stack(accepted_cols)
                y = _orth_against_basis(y, Btmp, full_passes)

            ny = float(np.linalg.norm(y))
            if np.isfinite(ny) and ny > LOCK_ADD_TOL:
                accepted_cols.append(y / ny)
                accepted_thetas.append(th)

        if accepted_cols:
            Y_add = np.column_stack(accepted_cols)
            if Y_lock.shape[1] == 0:
                Y_lock = Y_add
            else:
                Y_lock = np.column_stack([Y_lock, Y_add])
            theta_lock.extend(accepted_thetas)
            return int(len(accepted_cols))

        return 0

    def _eigh_current_tridiag(alpha_cur: np.ndarray, beta_cur: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ncur = alpha_cur.size
        if ncur == 1:
            lam = alpha_cur.copy()
            S = np.array([[1.0]], dtype=float)
            return lam, S

        if eigh_tridiagonal is not None:
            lam, S = eigh_tridiagonal(alpha_cur, beta_cur)
        else:
            Tcur = np.diag(alpha_cur) + np.diag(beta_cur, 1) + np.diag(beta_cur, -1)
            lam, S = np.linalg.eigh(Tcur)
        return lam, S

    def _maybe_lock_good_ritz(k_now: int, r_now: np.ndarray) -> None:
        nonlocal Y_lock

        if reorth_mode != "selective":
            return

        ncur = k_now + 1
        if ncur < 2:
            return
        if (ncur % sel_check_every) != 0:
            return

        profile["n_lock_probe"] += 1

        if ncur < lock_check_min_k:
            profile["n_lock_skip_smallk"] += 1
            return

        t0 = perf_counter()
        beta_probe = float(np.linalg.norm(r_now))
        profile["time_lock_probe_norm"] += perf_counter() - t0

        if not np.isfinite(beta_probe) or beta_probe <= 0.0:
            return

        do_force = ((profile["n_lock_probe"] % lock_force_every) == 0)

        # residual 还不小的时候通常跳过 eig，只做周期性强制检查。
        if (not do_force) and (beta_probe > lock_beta_rel_trigger * spectral_scale):
            profile["n_lock_skip_beta"] += 1
            return

        alpha_cur = alpha[:ncur].copy()
        beta_cur = beta[:k_now].copy()

        t1 = perf_counter()
        theta_cur, Scur = _eigh_current_tridiag(alpha_cur, beta_cur)
        profile["time_lock_eigh"] += perf_counter() - t1
        profile["n_lock_eigh"] += 1

        ritz_res = beta_probe * np.abs(Scur[-1, :])
        good_mask = (ritz_res <= float(sel_tol) * np.maximum(1.0, np.abs(theta_cur)))

        if np.any(good_mask):
            t2 = perf_counter()
            Y_new = Q[:, :ncur] @ Scur[:, good_mask]
            theta_new = theta_cur[good_mask]
            n_added = _append_locked_vectors(Y_new, theta_new)
            profile["time_lock_accept"] += perf_counter() - t2
            profile["n_lock_added"] += int(n_added)

    def _semiorth_measure(r_now: np.ndarray, Qbasis: np.ndarray) -> float:
        if Qbasis.size == 0:
            return 0.0
        nr = float(np.linalg.norm(r_now))
        if (not np.isfinite(nr)) or nr == 0.0:
            return 0.0
        overlaps = Qbasis.T @ r_now
        if overlaps.size == 0:
            return 0.0
        return float(np.max(np.abs(overlaps)) / nr)

    v_prev = np.zeros_like(z)
    v = np.sqrt(w)
    nv = float(np.linalg.norm(v))
    if nv == 0.0 or not np.isfinite(nv):
        raise ValueError("Degenerate starting vector.")
    v = v / nv

    Q[:, 0] = v

    alpha = np.zeros(N, dtype=float)
    beta = np.zeros(max(N - 1, 0), dtype=float)

    beta_prev = 0.0
    for k in range(N):
        v = Q[:, k]
        Av = z * v
        alpha[k] = float(np.dot(v, Av))
        r = Av - alpha[k] * v - beta_prev * v_prev

        if k < N - 1:
            V = Q[:, : (k + 1)]

            if reorth_mode == "full":
                t = perf_counter()
                r = _full_reorth_against_Q(r, V)
                profile["time_q_fullreorth"] += perf_counter() - t
                profile["n_q_fullreorth"] += 1

            elif reorth_mode == "selective":
                _maybe_lock_good_ritz(k, r)

                if Y_lock.shape[1] > 0:
                    t = perf_counter()
                    r = _orth_against_basis(r, Y_lock, full_passes)
                    profile["time_lock_apply"] += perf_counter() - t

                t = perf_counter()
                mu = _semiorth_measure(r, V)
                profile["time_semiorth_measure"] += perf_counter() - t

                if mu > SEMIORTH_TRIGGER:
                    profile["n_semiorth_trigger"] += 1

                    t = perf_counter()
                    r = _full_reorth_against_Q(r, V)
                    profile["time_q_fullreorth"] += perf_counter() - t
                    profile["n_q_fullreorth"] += 1

                    if Y_lock.shape[1] > 0:
                        t = perf_counter()
                        r = _orth_against_basis(r, Y_lock, full_passes)
                        profile["time_lock_apply"] += perf_counter() - t

                    t = perf_counter()
                    mu2 = _semiorth_measure(r, V)
                    profile["time_semiorth_measure"] += perf_counter() - t

                    if mu2 > SEMIORTH_ACCEPT:
                        t = perf_counter()
                        r = _full_reorth_against_Q(r, V)
                        profile["time_q_fullreorth"] += perf_counter() - t
                        profile["n_q_fullreorth"] += 1

                        if Y_lock.shape[1] > 0:
                            t = perf_counter()
                            r = _orth_against_basis(r, Y_lock, full_passes)
                            profile["time_lock_apply"] += perf_counter() - t

            b = float(np.linalg.norm(r))
            if not np.isfinite(b) or b == 0.0:
                raise ValueError(f"Lanczos breakdown at k={k}, beta={b}")
            beta[k] = b
            v_prev = v
            beta_prev = b
            Q[:, k + 1] = r / b

    profile["time_total_internal"] = perf_counter() - t_func0

    if return_profile:
        return alpha, beta, Q, profile
    return alpha, beta, Q

def golub_welsch(alpha: np.ndarray, beta: np.ndarray):
    alpha = np.asarray(alpha, dtype=float).ravel()
    beta = np.asarray(beta, dtype=float).ravel()
    N = alpha.size
    if beta.size != N - 1:
        raise ValueError("beta must have length N-1.")

    if N == 1:
        lam = alpha.copy()
        Q = np.array([[1.0]], dtype=float)
    else:
        if eigh_tridiagonal is not None:
            lam, Q = eigh_tridiagonal(alpha, beta)
        else:
            J = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)
            lam, Q = np.linalg.eigh(J)

    q0 = Q[0, :].copy()
    p_norm = q0 * q0
    p_norm = p_norm / p_norm.sum()
    return lam, Q, p_norm

def solve_sigma_x_nodes_from_eigbasis(Q: np.ndarray, b: np.ndarray):
    Q = np.asarray(Q, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    N = Q.shape[0]
    if Q.shape != (N, N) or b.size != N:
        raise ValueError("Q must be NxN and b must have length N.")
    q0 = Q[0, :].copy()
    M = Q * q0
    condM, rcondM = _cond2(M)

    try:
        s = np.linalg.solve(M, b)
    except np.linalg.LinAlgError:
        s = np.linalg.lstsq(M, b, rcond=1e-12)[0]

    return s, condM, rcondM

# ============================================================
# Positive projection on Gauss nodes: local hat functions
# ============================================================
def _strictly_increasing_copy(x: np.ndarray, *, rtol: float = 1e-12, atol: float = 1e-14) -> np.ndarray:
    """
    Return a strictly increasing copy of x by adding only the minimal forward
    perturbation needed to resolve ties / reversals.

    This is only a geometric safeguard for building local hat functions.
    It does NOT modify the physical sigma_t_nodes used elsewhere.
    """
    y = np.asarray(x, dtype=float).ravel().copy()
    if y.size <= 1:
        return y
    for i in range(1, y.size):
        min_gap = atol + rtol * max(abs(y[i - 1]), abs(y[i]), 1.0)
        if not np.isfinite(min_gap) or min_gap <= 0.0:
            min_gap = 1e-14
        if y[i] <= y[i - 1]:
            y[i] = y[i - 1] + min_gap
        elif (y[i] - y[i - 1]) < min_gap:
            y[i] = y[i - 1] + min_gap
    return y

def project_u_num_positive_hat(
    sigma_t_nodes: np.ndarray,
    rt_node: np.ndarray,
    rx_node: np.ndarray,
    w_base: np.ndarray,
) -> np.ndarray:
    """
    Positivity-oriented local hat projection on the Gauss sigma_t nodes.

    We build piecewise-linear hat functions phi_i centered at the Gauss nodes
    sigma_t_nodes (using their real parts only for geometry), then project the
    numerator mass

        u_i = p_i * sigma_x_i  ~=  sum_j w_base[j] * phi_i(rt_node[j]) * rx_node[j].

    Notes:
    - This is a positive averaging / Markov-type projection on sigma_t-space.
    - If rx_node >= 0 and w_base >= 0, then u_i >= 0 by construction.
    - sigma_t_nodes themselves are NOT altered; only their real parts are used to
      define the local hat partition.
    - The output order matches the original sigma_t_nodes order.
    """
    centers = np.asarray(np.real(sigma_t_nodes), dtype=float).ravel()
    x = np.asarray(rt_node, dtype=float).ravel()
    y = np.asarray(rx_node, dtype=float).ravel()
    w = np.asarray(w_base, dtype=float).ravel()

    if not (centers.size > 0 and x.size == y.size == w.size):
        raise ValueError("Size mismatch in positive hat projection inputs.")
    if not np.all(np.isfinite(centers)):
        raise ValueError("Non-finite Gauss sigma_t nodes cannot define hat functions.")

    N = centers.size
    order = np.argsort(centers)
    centers_sorted = _strictly_increasing_copy(centers[order])

    u_sorted = np.zeros(N, dtype=float)

    if N == 1:
        u_sorted[0] = float(np.sum(w * y))
    else:
        idx = np.searchsorted(centers_sorted, x, side="right")

        m_left = (idx == 0)
        if np.any(m_left):
            u_sorted[0] += float(np.sum(w[m_left] * y[m_left]))

        m_right = (idx >= N)
        if np.any(m_right):
            u_sorted[-1] += float(np.sum(w[m_right] * y[m_right]))

        m_mid = (~m_left) & (~m_right)
        if np.any(m_mid):
            kk = idx[m_mid] - 1
            xx = x[m_mid]
            yy = y[m_mid]
            ww = w[m_mid]

            cL = centers_sorted[kk]
            cR = centers_sorted[kk + 1]
            denom = cR - cL

            t = (xx - cL) / denom
            t = np.clip(t, 0.0, 1.0)

            contrib_L = ww * yy * (1.0 - t)
            contrib_R = ww * yy * t

            np.add.at(u_sorted, kk, contrib_L)
            np.add.at(u_sorted, kk + 1, contrib_R)

    u_num = np.empty(N, dtype=float)
    u_num[order] = u_sorted
    return u_num

def solve_sigma_x_nodes_positive_hat(
    *,
    sigma_t_nodes: np.ndarray,
    p: np.ndarray,
    rt_node: np.ndarray,
    rx_node: np.ndarray,
    w_base: np.ndarray,
    a: float,
):
    """
    Reconstruct sigma_x_nodes by positive local-hat projection on the Gauss sigma_t nodes.

    Steps:
      (1) project numerator mass u_i = p_i * sigma_x_i using local hat functions;
      (2) recover sigma_x_i = u_i / p_i for reporting;
      (3) set ubar_i = u_i * sigma_t_i^a so that the PT numerator uses the
          projected positive numerator mass directly.
    """
    p = np.asarray(p)
    sigma_t_nodes = np.asarray(sigma_t_nodes)
    u_num = project_u_num_positive_hat(
        sigma_t_nodes=sigma_t_nodes,
        rt_node=rt_node,
        rx_node=rx_node,
        w_base=w_base,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_x_nodes = u_num / p

    ra = np.power(sigma_t_nodes, a)
    ubar = u_num * ra

    condM = float("nan")
    rcondM = float("nan")
    return sigma_x_nodes, ubar, condM, rcondM

def _coerce_real_vector_for_qp(
    x: np.ndarray,
    *,
    name: str,
    imag_tol: float = 1e-10,
    positive: bool = False,
) -> np.ndarray:
    arr = np.asarray(x)
    if np.iscomplexobj(arr):
        max_im = _max_abs_imag(arr)
        if np.isfinite(max_im) and (max_im > imag_tol):
            raise ValueError(
                f"QP sigma_x reconstruction requires {name} to be real within tol={imag_tol:g}; "
                f"got max|Im|={max_im:.3e}."
            )
        arr = np.real(arr)
    arr = np.asarray(arr, dtype=float).ravel()
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"QP sigma_x reconstruction requires finite {name}.")
    if positive and np.any(arr <= 0.0):
        raise ValueError(f"QP sigma_x reconstruction requires strictly positive {name}.")
    return arr

def build_full_matching_system_from_eigbasis(
    *,
    Qk: np.ndarray,
    Qeig: np.ndarray,
    w_norm: np.ndarray,
    rx_node: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Build the original full-matching linear system M s = b from the existing
    orthogonal-basis / eigenbasis reconstruction.

    The original code solves
        (Q * diag(q0)) s = b,
    where b = Qk^T (sqrt(w_norm) * rx_node).
    """
    t = np.sqrt(w_norm) * rx_node
    bvec = (Qk.T @ t).astype(float)
    q0 = np.asarray(Qeig, dtype=float)[0, :].copy()
    Mmat = np.asarray(Qeig, dtype=float) * q0
    condM, rcondM = _cond2(Mmat)
    return Mmat, bvec, condM, rcondM

def build_positive_seed_and_local_bounds(
    *,
    sigma_t_nodes: np.ndarray,
    p: np.ndarray,
    rt_node: np.ndarray,
    rx_node: np.ndarray,
    w_base: np.ndarray,
    imag_tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the positive-hat seed s_seed together with the NEW admissible bounds
    used by the full-first correction on the SAME fixed sigma_t nodes.

    Updated policy in this version:
      - keep only the nonnegativity lower bound,
      - remove the old group-global support upper bound,
      - remove the old global linear inequality elsewhere in the solver.

    Therefore the admissible set is now simply

        s_i >= 0,   i = 1, ..., N,

    represented here as
        ell_i     = 0,
        u_upper_i = +inf.

    Returns (s_seed, u_num_seed, ell, u_upper) in the ORIGINAL sigma_t-node order.
    """
    centers = _coerce_real_vector_for_qp(
        sigma_t_nodes, name="sigma_t_nodes", imag_tol=imag_tol, positive=True
    )
    p_real = _coerce_real_vector_for_qp(
        p, name="p", imag_tol=imag_tol, positive=True
    )
    x = np.asarray(rt_node, dtype=float).ravel()
    y = np.asarray(rx_node, dtype=float).ravel()
    w = np.asarray(w_base, dtype=float).ravel()

    if not (x.size == y.size == w.size):
        raise ValueError("rt_node/rx_node/w_base size mismatch in QP seed builder.")
    if x.size == 0:
        raise ValueError("Empty rt_node/rx_node/w_base in QP seed builder.")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)) or np.any(~np.isfinite(w)):
        raise ValueError("Non-finite values in QP seed builder inputs.")

    N = centers.size
    order = np.argsort(centers)
    centers_sorted = _strictly_increasing_copy(centers[order])
    p_sorted = p_real[order]

    u_num_sorted = np.zeros(N, dtype=float)

    for xj, yj, wj in zip(x, y, w):
        idx = int(np.searchsorted(centers_sorted, xj, side="right"))

        if idx == 0:
            u_num_sorted[0] += wj * yj
            continue

        if idx >= N:
            u_num_sorted[-1] += wj * yj
            continue

        k = idx - 1
        cL = centers_sorted[k]
        cR = centers_sorted[k + 1]
        denom = cR - cL
        t = float(np.clip((xj - cL) / denom, 0.0, 1.0))
        phiL = 1.0 - t
        phiR = t

        if phiL > 0.0:
            u_num_sorted[k] += wj * yj * phiL

        if phiR > 0.0:
            u_num_sorted[k + 1] += wj * yj * phiR

    with np.errstate(divide="ignore", invalid="ignore"):
        s_seed_sorted = u_num_sorted / p_sorted

    s_seed = np.empty(N, dtype=float)
    u_num_seed = np.empty(N, dtype=float)
    ell = np.zeros(N, dtype=float)
    u_upper = np.full(N, np.inf, dtype=float)

    s_seed[order] = s_seed_sorted
    u_num_seed[order] = u_num_sorted

    if np.any(~np.isfinite(s_seed)):
        raise ValueError("Non-finite QP seed encountered after hat projection.")
    if np.any(~np.isfinite(ell)):
        raise ValueError("Non-finite lower bounds encountered in QP seed builder.")
    if np.any(ell < 0.0):
        raise ValueError("Invalid nonnegativity lower bounds encountered in QP seed builder.")

    return s_seed, u_num_seed, ell, u_upper

def build_qp_retained_coefficients(
    *,
    sigma_t_nodes: np.ndarray,
    p: np.ndarray,
    a: float,
    retained_mode: str = "minus1_and_0",
    imag_tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the retained hard-constraint rows for the ORIGINAL mixed moments.

    In this code, the compressed weights p_i belong to the transformed measure
        d\\mu \\sim \\sigma_t^a dE
    with Case C using a = -1. Therefore an original mixed moment of order n is
    represented on the compressed rule as

        sum_i p_i * sigma_x_i * sigma_t_i^(n-a).

    Supported retained modes:
      - "zero_only"      : retain the original mixed moment of order n = 0 only
      - "minus1_and_0"   : retain the original mixed moments of orders n = -1 and n = 0

    Returns
    -------
    E, hard_orders
        E is the hard-constraint matrix and hard_orders stores the corresponding
        original mixed-moment orders.
    """
    mode = str(retained_mode).strip().lower()
    sigma_t_real = _coerce_real_vector_for_qp(
        sigma_t_nodes, name="sigma_t_nodes", imag_tol=imag_tol, positive=True
    )
    p_real = _coerce_real_vector_for_qp(
        p, name="p", imag_tol=imag_tol, positive=True
    )

    if mode in ("zero_only", "0_only", "zero", "infinite_dilution", "m0_only"):
        row_0 = p_real.copy()
        E = row_0[None, :]
        hard_orders = np.asarray([0.0], dtype=float)
        return E, hard_orders

    if mode in ("minus1_and_0", "m-1_0", "casec_mixed_moments", "default"):
        row_m1 = p_real / sigma_t_real
        row_0 = p_real.copy()
        E = np.vstack([row_m1, row_0])
        hard_orders = np.asarray([-1.0, 0.0], dtype=float)
        return E, hard_orders

    raise ValueError(
        "Unknown QP_RETAINED_MODE. Use 'zero_only' or 'minus1_and_0'."
    )

def build_qp_hard_moment_rhs_from_discrete(
    *,
    rt_node: np.ndarray,
    rx_node: np.ndarray,
    w_base: np.ndarray,
    moment_orders: tuple[float, ...] = (-1.0, 0.0),
) -> np.ndarray:
    """
    Build the hard-constraint right-hand side from the ORIGINAL discrete
    realization, not from the seed.

    For original mixed moment order n, the target is
        m_n = sum_j w_base[j] * rx_node[j] * rt_node[j]^n.

    With the requested hard constraints, this returns
        [m_{-1}, m_0]^T.
    """
    rt = np.asarray(rt_node, dtype=float).ravel()
    rx = np.asarray(rx_node, dtype=float).ravel()
    w = np.asarray(w_base, dtype=float).ravel()

    if not (rt.size == rx.size == w.size):
        raise ValueError("rt_node/rx_node/w_base size mismatch in QP hard-moment RHS builder.")
    if rt.size == 0:
        raise ValueError("Empty rt_node/rx_node/w_base in QP hard-moment RHS builder.")
    if np.any(~np.isfinite(rt)) or np.any(~np.isfinite(rx)) or np.any(~np.isfinite(w)):
        raise ValueError("Non-finite values in QP hard-moment RHS builder.")
    if np.any(rt <= 0.0):
        raise ValueError("QP hard-moment RHS builder requires strictly positive rt_node.")

    rhs = []
    for n in tuple(moment_orders):
        rhs.append(float(np.sum(w * rx * np.power(rt, float(n)))))
    return np.asarray(rhs, dtype=float)

def build_qp_objective_matrices(
    *,
    M_match: np.ndarray,
    b_match: np.ndarray,
    weight_mode: str = "identity",
    p: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Kept for compatibility.

    Returns H and g for the soft full-matching objective
        0.5 * ||W^{1/2}(M s - b)||_2^2 = 0.5 s^T H s - g^T s + const.
    """
    M_match = np.asarray(M_match, dtype=float)
    b_match = np.asarray(b_match, dtype=float).ravel()
    if M_match.ndim != 2:
        raise ValueError("M_match must be 2D.")
    if M_match.shape[0] != b_match.size:
        raise ValueError("M_match/b_match size mismatch in objective builder.")

    mode = str(weight_mode).strip().lower()
    nrow = M_match.shape[0]

    if mode == "identity":
        w_diag = np.ones(nrow, dtype=float)
    elif mode in ("probability", "p"):
        if p is None:
            raise ValueError("objective weight_mode='probability' requires p.")
        p_arr = np.asarray(p, dtype=float).ravel()
        if p_arr.size != nrow:
            raise ValueError("p size mismatch for objective weight_mode='probability'.")
        w_diag = np.maximum(p_arr, 1.0e-300)
    else:
        raise ValueError("Unknown objective weight mode. Use 'identity' or 'probability'.")

    WM = w_diag[:, None] * M_match
    H = M_match.T @ WM
    g = M_match.T @ (w_diag * b_match)
    H = 0.5 * (H + H.T)
    return H, g

def build_correction_objective_operator(
    *,
    M_match: np.ndarray,
    weight_mode: str = "identity",
    p: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Legacy helper for the old full-first correction objective

        min_d 0.5 * ||W^{1/2} M d||_2^2.

    Kept for backward compatibility with older notes / experiments.
    """
    M_match = np.asarray(M_match, dtype=float)
    if M_match.ndim != 2:
        raise ValueError("M_match must be 2D.")

    mode = str(weight_mode).strip().lower()
    nrow = M_match.shape[0]

    if mode == "identity":
        w_sqrt = np.ones(nrow, dtype=float)
    elif mode in ("probability", "p"):
        if p is None:
            raise ValueError("weight_mode='probability' requires p.")
        p_arr = np.asarray(p, dtype=float).ravel()
        if p_arr.size != nrow:
            raise ValueError("p size mismatch for weight_mode='probability'.")
        w_sqrt = np.sqrt(np.maximum(p_arr, 1.0e-300))
    else:
        raise ValueError("Unknown objective weight mode. Use 'identity' or 'probability'.")

    A_obj = w_sqrt[:, None] * M_match
    H = A_obj.T @ A_obj
    H = 0.5 * (H + H.T)
    return A_obj, H

def build_soft_moment_objective_operator(
    *,
    sigma_t_nodes: np.ndarray,
    p: np.ndarray,
    rt_node: np.ndarray,
    rx_node: np.ndarray,
    w_base: np.ndarray,
    weight_mode: str = "identity",
    hard_orders: tuple[float, ...] = (-1.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the soft-objective operator for the admissible reconstruction:

        min 0.5 * ||A_obj s - b_obj||_2^2

    where the objective penalizes all Chiba-consistent mixed moments EXCEPT the
    orders listed in hard_orders. Those retained orders are intended to be
    enforced separately as hard equalities.

    Each soft row is normalized as

        alpha_k = max( ||C_k||_2 * s_ref, |c_k|, eps_abs ),

    with s_ref taken from the original 0th mixed moment scale. This keeps the
    residuals dimensionless and prevents tiny target moments from dominating the
    objective.

    Returns
    -------
    A_obj, b_obj, H, g, soft_orders
        A_obj, b_obj define the normalized least-squares objective;
        H = A_obj^T A_obj and g = A_obj^T b_obj;
        soft_orders stores all mixed-moment orders used in the penalty.
    """
    st = np.asarray(sigma_t_nodes, dtype=float).ravel()
    pp = np.asarray(p, dtype=float).ravel()
    rt = np.asarray(rt_node, dtype=float).ravel()
    rx = np.asarray(rx_node, dtype=float).ravel()
    wb = np.asarray(w_base, dtype=float).ravel()

    if not (st.size == pp.size):
        raise ValueError("sigma_t_nodes/p size mismatch in soft moment objective builder.")
    if not (rt.size == rx.size == wb.size):
        raise ValueError("rt_node/rx_node/w_base size mismatch in soft moment objective builder.")
    if st.size == 0:
        raise ValueError("Empty sigma_t_nodes in soft moment objective builder.")
    if np.any(st <= 0.0):
        raise ValueError("soft moment objective builder requires strictly positive sigma_t_nodes.")
    if np.any(rt <= 0.0):
        raise ValueError("soft moment objective builder requires strictly positive rt_node.")

    N = int(st.size)
    all_orders = np.linspace(-1.0, 0.0, N, dtype=float)
    hard_orders_arr = np.asarray(tuple(float(v) for v in hard_orders), dtype=float)
    if hard_orders_arr.size == 0:
        soft_orders = all_orders.copy()
    else:
        keep = np.ones(all_orders.size, dtype=bool)
        for j, n in enumerate(all_orders):
            if np.any(np.isclose(n, hard_orders_arr, rtol=0.0, atol=1.0e-12)):
                keep[j] = False
        soft_orders = all_orders[keep]

    if soft_orders.size == 0:
        A_obj = np.zeros((0, N), dtype=float)
        b_obj = np.zeros(0, dtype=float)
        H = np.zeros((N, N), dtype=float)
        g = np.zeros(N, dtype=float)
        return A_obj, b_obj, H, g, soft_orders

    C = np.empty((soft_orders.size, N), dtype=float)
    c = np.empty(soft_orders.size, dtype=float)
    for k, n in enumerate(soft_orders):
        C[k, :] = pp * np.power(st, float(n))
        c[k] = float(np.sum(wb * rx * np.power(rt, float(n))))

    m0_ref = float(np.sum(wb * rx))
    s_ref = max(abs(m0_ref), 1.0e-12 * max(1.0, float(np.max(st))))
    eps_abs = 1.0e-12 * max(1.0, s_ref)

    row_norm = np.linalg.norm(C, axis=1)
    alpha = np.maximum.reduce([
        row_norm * s_ref,
        np.abs(c),
        np.full_like(c, eps_abs),
    ])

    Cn = C / alpha[:, None]
    cn = c / alpha

    mode = str(weight_mode).strip().lower()
    if mode in ("identity", "probability", "p"):
        w_sqrt = np.ones(Cn.shape[0], dtype=float)
    else:
        raise ValueError(
            "Unknown objective weight mode for soft mixed moments. Use 'identity' or 'probability'."
        )

    A_obj = w_sqrt[:, None] * Cn
    b_obj = w_sqrt * cn
    H = A_obj.T @ A_obj
    g = A_obj.T @ b_obj
    H = 0.5 * (H + H.T)
    return A_obj, b_obj, H, g, soft_orders

def _svd_nullspace(A: np.ndarray, *, rtol: float = 1e-12) -> tuple[np.ndarray, int]:
    """
    Return a basis Z for ker(A) together with rank(A).
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError("A must be 2D in _svd_nullspace.")

    n = A.shape[1]
    if n == 0:
        return np.zeros((0, 0), dtype=float), 0
    if A.shape[0] == 0:
        return np.eye(n, dtype=float), 0

    U, s, Vh = np.linalg.svd(A, full_matrices=True)
    if s.size == 0:
        return np.eye(n, dtype=float), 0

    s0 = float(s[0]) if np.isfinite(s[0]) and s[0] > 0.0 else 1.0
    tol = float(rtol) * max(A.shape) * s0
    rank = int(np.sum(s > tol))
    Z = Vh[rank:, :].T.copy()
    return Z, rank

def _solve_equality_particular(
    B: np.ndarray,
    rhs: np.ndarray,
    *,
    eq_tol: float,
) -> tuple[np.ndarray, float]:
    """
    Solve B x = rhs in least-squares form and require the residual to be small.
    """
    B = np.asarray(B, dtype=float)
    rhs = np.asarray(rhs, dtype=float).ravel()
    if B.ndim != 2:
        raise ValueError("B must be 2D in _solve_equality_particular.")
    if B.shape[0] != rhs.size:
        raise ValueError("B/rhs size mismatch in _solve_equality_particular.")

    n = B.shape[1]
    if n == 0:
        res = float(np.linalg.norm(rhs))
        scale = max(1.0, float(np.linalg.norm(rhs)))
        if res > float(eq_tol) * scale:
            raise ValueError("No free variables remain but the equality constraints are inconsistent.")
        return np.zeros(0, dtype=float), res

    x = np.linalg.lstsq(B, rhs, rcond=None)[0]
    res = float(np.linalg.norm(B @ x - rhs))
    scale = max(1.0, float(np.linalg.norm(rhs)))
    if res > float(eq_tol) * scale:
        raise ValueError("Current active set cannot satisfy the retained equalities.")
    return np.asarray(x, dtype=float).ravel(), res

def _pick_active_index_to_release(
    *,
    s_full: np.ndarray,
    ell: np.ndarray,
    u: np.ndarray,
    active_lower: list[int],
    active_upper: list[int],
) -> int:
    """
    Heuristic release rule used only when the current active set makes the
    equality constraints inconsistent. We release the active bound with the
    smallest absolute correction relative to s_full.
    """
    candidates: list[tuple[float, int]] = []
    for i in active_lower:
        candidates.append((abs(float(ell[i] - s_full[i])), int(i)))
    for i in active_upper:
        candidates.append((abs(float(u[i] - s_full[i])), int(i)))
    if not candidates:
        raise RuntimeError("No active bound is available to release.")
    candidates.sort(key=lambda t: (t[0], t[1]))
    return int(candidates[0][1])

def _compute_reduced_gradient(
    *,
    H: np.ndarray,
    x: np.ndarray,
    E: np.ndarray,
    free_idx: np.ndarray,
    g: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the reduced Lagrangian gradient
        red = (H x - g) + E^T lambda,
    with lambda determined from the free variables by least squares. If no free
    variables remain, use all coordinates as the fitting set.
    """
    H = np.asarray(H, dtype=float)
    x = np.asarray(x, dtype=float).ravel()
    E = np.asarray(E, dtype=float)
    free_idx = np.asarray(free_idx, dtype=int).ravel()

    if g is None:
        g_arr = np.zeros(x.size, dtype=float)
    else:
        g_arr = np.asarray(g, dtype=float).ravel()
        if g_arr.size != x.size:
            raise ValueError("g size mismatch in _compute_reduced_gradient.")

    grad = H @ x - g_arr
    n = x.size

    if free_idx.size > 0:
        ref = free_idx
    else:
        ref = np.arange(n, dtype=int)

    A = E[:, ref].T
    b = -grad[ref]
    if A.size == 0:
        lam = np.zeros(E.shape[0], dtype=float)
    else:
        lam = np.linalg.lstsq(A, b, rcond=None)[0]
    red = grad + E.T @ lam
    return red, lam

def _solve_full_first_admissible_correction(
    *,
    A_obj: np.ndarray,
    H: np.ndarray,
    E: np.ndarray,
    s_full: np.ndarray,
    ell: np.ndarray,
    u: np.ndarray,
    eq_rhs: np.ndarray | None = None,
    maxiter: int,
    eq_tol: float,
    bound_tol: float,
    multiplier_tol: float,
) -> np.ndarray:
    """
    Full-first admissible correction from the plan PDF.

    Solve
        min_d 0.5 ||A_obj d||_2^2
        s.t.  E d = 0,
              ell <= s_full + d <= u,
    by an active-set loop, and use null-space elimination only inside each free
    subproblem.

    Parameters
    ----------
    eq_rhs : ndarray or None
        Right-hand side in E d = eq_rhs. In the exact full-first scheme this is
        zero. For finite-precision robustness we may optionally pass
        eq_rhs = r_hard - E s_full when the computed full solution does not
        exactly satisfy the retained equalities.
    """
    A_obj = np.asarray(A_obj, dtype=float)
    H = np.asarray(H, dtype=float)
    E = np.asarray(E, dtype=float)
    s_full = np.asarray(s_full, dtype=float).ravel()
    ell = np.asarray(ell, dtype=float).ravel()
    u = np.asarray(u, dtype=float).ravel()

    n = s_full.size
    if not (ell.size == u.size == n):
        raise ValueError("s_full/ell/u size mismatch in admissible correction.")

    if eq_rhs is None:
        eq_rhs_arr = np.zeros(E.shape[0], dtype=float)
    else:
        eq_rhs_arr = np.asarray(eq_rhs, dtype=float).ravel()
        if eq_rhs_arr.size != E.shape[0]:
            raise ValueError("eq_rhs size mismatch in admissible correction.")

    finite_lower = np.isfinite(ell)
    finite_upper = np.isfinite(u)

    L = {int(i) for i in np.where(finite_lower & (s_full < ell - bound_tol))[0]}
    U = {int(i) for i in np.where(finite_upper & (s_full > u + bound_tol))[0]}

    if L & U:
        raise RuntimeError("A variable cannot be active at both lower and upper bounds.")

    for _it in range(int(maxiter)):
        while True:
            L_list = sorted(L)
            U_list = sorted(U)
            A_list = L_list + U_list
            F_list = [i for i in range(n) if (i not in L and i not in U)]

            A_idx = np.asarray(A_list, dtype=int)
            F_idx = np.asarray(F_list, dtype=int)

            d = np.zeros(n, dtype=float)
            if L_list:
                d[np.asarray(L_list, dtype=int)] = ell[np.asarray(L_list, dtype=int)] - s_full[np.asarray(L_list, dtype=int)]
            if U_list:
                d[np.asarray(U_list, dtype=int)] = u[np.asarray(U_list, dtype=int)] - s_full[np.asarray(U_list, dtype=int)]

            dA = d[A_idx] if A_idx.size else np.zeros(0, dtype=float)
            rhs = eq_rhs_arr - (E[:, A_idx] @ dA if A_idx.size else 0.0)
            rhs = np.asarray(rhs, dtype=float).ravel()
            B = E[:, F_idx] if F_idx.size else np.zeros((E.shape[0], 0), dtype=float)

            try:
                dF0, _ = _solve_equality_particular(B, rhs, eq_tol=eq_tol)
                break
            except ValueError:
                if A_idx.size == 0:
                    raise RuntimeError(
                        "No feasible admissible correction exists for the current box bounds."
                    )
                rel = _pick_active_index_to_release(
                    s_full=s_full,
                    ell=ell,
                    u=u,
                    active_lower=L_list,
                    active_upper=U_list,
                )
                if rel in L:
                    L.remove(rel)
                elif rel in U:
                    U.remove(rel)
                else:
                    raise RuntimeError("Internal active-set bookkeeping error.")

        if F_idx.size:
            A_F = A_obj[:, F_idx]
            a_fix = (A_obj[:, A_idx] @ dA) if A_idx.size else np.zeros(A_obj.shape[0], dtype=float)
            Z_F, _rank_B = _svd_nullspace(B, rtol=max(eq_tol, 1.0e-14))

            if Z_F.shape[1] == 0:
                dF = dF0
            else:
                rhs_ls = -(A_F @ dF0 + a_fix)
                y = np.linalg.lstsq(A_F @ Z_F, rhs_ls, rcond=None)[0]
                dF = dF0 + Z_F @ y

            d[F_idx] = dF
        else:
            res = float(np.linalg.norm(rhs))
            scale = max(1.0, float(np.linalg.norm(rhs)))
            if res > float(eq_tol) * scale:
                raise RuntimeError("Active set overconstrains the equalities and leaves no feasible free variables.")

        s = s_full + d

        add_lower = [int(i) for i in F_idx if finite_lower[i] and (s[i] < ell[i] - bound_tol)]
        add_upper = [int(i) for i in F_idx if finite_upper[i] and (s[i] > u[i] + bound_tol)]
        if add_lower or add_upper:
            for i in add_lower:
                U.discard(i)
                L.add(i)
            for i in add_upper:
                L.discard(i)
                U.add(i)
            continue

        red, _lam = _compute_reduced_gradient(H=H, x=d, E=E, free_idx=F_idx)
        release_lower = [int(i) for i in L if red[i] < -multiplier_tol]
        release_upper = [int(i) for i in U if red[i] > multiplier_tol]
        if release_lower or release_upper:
            for i in release_lower:
                L.discard(i)
            for i in release_upper:
                U.discard(i)
            continue

        eq_res = float(np.linalg.norm(E @ d - eq_rhs_arr))
        if eq_res > float(eq_tol) * max(1.0, float(np.linalg.norm(d)), float(np.linalg.norm(eq_rhs_arr))):
            raise RuntimeError("Returned correction violates the retained equalities beyond tolerance.")

        return s

    raise RuntimeError("Full-first admissible correction did not converge within maxiter.")

def _solve_soft_moment_admissible_qp(
    *,
    A_obj: np.ndarray,
    b_obj: np.ndarray,
    H: np.ndarray,
    g: np.ndarray,
    E: np.ndarray,
    r_hard: np.ndarray,
    ell: np.ndarray,
    u: np.ndarray,
    x_ref: np.ndarray,
    maxiter: int,
    eq_tol: float,
    bound_tol: float,
    multiplier_tol: float,
) -> np.ndarray:
    """
    Solve the NEW admissible reconstruction problem

        min_s 0.5 * ||A_obj s - b_obj||_2^2
        s.t.  E s = r_hard,
              ell <= s <= u,

    by an active-set loop with null-space elimination on each free subproblem.

    Compared with the old full-first correction, there is no distinguished
    full-matching center here: the objective comes only from the interior
    (N-2) soft mixed-moment residuals.
    """
    A_obj = np.asarray(A_obj, dtype=float)
    b_obj = np.asarray(b_obj, dtype=float).ravel()
    H = np.asarray(H, dtype=float)
    g = np.asarray(g, dtype=float).ravel()
    E = np.asarray(E, dtype=float)
    r_hard = np.asarray(r_hard, dtype=float).ravel()
    ell = np.asarray(ell, dtype=float).ravel()
    u = np.asarray(u, dtype=float).ravel()
    x_ref = np.asarray(x_ref, dtype=float).ravel()

    n = ell.size
    if not (u.size == x_ref.size == n):
        raise ValueError("ell/u/x_ref size mismatch in soft admissible solver.")
    if H.shape != (n, n):
        raise ValueError("H shape mismatch in soft admissible solver.")
    if g.size != n:
        raise ValueError("g size mismatch in soft admissible solver.")
    if E.shape[1] != n:
        raise ValueError("E shape mismatch in soft admissible solver.")
    if r_hard.size != E.shape[0]:
        raise ValueError("r_hard size mismatch in soft admissible solver.")
    if A_obj.shape[1] != n or b_obj.size != A_obj.shape[0]:
        raise ValueError("A_obj/b_obj size mismatch in soft admissible solver.")

    finite_lower = np.isfinite(ell)
    finite_upper = np.isfinite(u)

    L: set[int] = set()
    U: set[int] = set()
    if L & U:
        raise RuntimeError("A variable cannot be active at both lower and upper bounds.")

    for _it in range(int(maxiter)):
        while True:
            L_list = sorted(L)
            U_list = sorted(U)
            A_list = L_list + U_list
            F_list = [i for i in range(n) if (i not in L and i not in U)]

            A_idx = np.asarray(A_list, dtype=int)
            F_idx = np.asarray(F_list, dtype=int)

            s = np.zeros(n, dtype=float)
            if L_list:
                s[np.asarray(L_list, dtype=int)] = ell[np.asarray(L_list, dtype=int)]
            if U_list:
                s[np.asarray(U_list, dtype=int)] = u[np.asarray(U_list, dtype=int)]

            sA = s[A_idx] if A_idx.size else np.zeros(0, dtype=float)
            rhs = r_hard - (E[:, A_idx] @ sA if A_idx.size else 0.0)
            rhs = np.asarray(rhs, dtype=float).ravel()
            B = E[:, F_idx] if F_idx.size else np.zeros((E.shape[0], 0), dtype=float)

            try:
                sF0, _ = _solve_equality_particular(B, rhs, eq_tol=eq_tol)
                break
            except ValueError:
                if A_idx.size == 0:
                    raise RuntimeError(
                        "No feasible nonnegative solution exists for the retained hard moments."
                    )
                rel = _pick_active_index_to_release(
                    s_full=x_ref,
                    ell=ell,
                    u=u,
                    active_lower=L_list,
                    active_upper=U_list,
                )
                if rel in L:
                    L.remove(rel)
                elif rel in U:
                    U.remove(rel)
                else:
                    raise RuntimeError("Internal active-set bookkeeping error.")

        if F_idx.size:
            A_F = A_obj[:, F_idx]
            a_fix = (A_obj[:, A_idx] @ sA) if A_idx.size else np.zeros(A_obj.shape[0], dtype=float)
            Z_F, _rank_B = _svd_nullspace(B, rtol=max(eq_tol, 1.0e-14))

            if Z_F.shape[1] == 0:
                sF = sF0
            else:
                rhs_ls = -(A_F @ sF0 + a_fix - b_obj)
                y = np.linalg.lstsq(A_F @ Z_F, rhs_ls, rcond=None)[0]
                sF = sF0 + Z_F @ y

            s[F_idx] = sF
        else:
            res = float(np.linalg.norm(rhs))
            scale = max(1.0, float(np.linalg.norm(rhs)))
            if res > float(eq_tol) * scale:
                raise RuntimeError("Active set overconstrains the hard equalities and leaves no feasible free variables.")

        add_lower = [int(i) for i in F_idx if finite_lower[i] and (s[i] < ell[i] - bound_tol)]
        add_upper = [int(i) for i in F_idx if finite_upper[i] and (s[i] > u[i] + bound_tol)]
        if add_lower or add_upper:
            for i in add_lower:
                U.discard(i)
                L.add(i)
            for i in add_upper:
                L.discard(i)
                U.add(i)
            continue

        red, _lam = _compute_reduced_gradient(H=H, x=s, E=E, free_idx=F_idx, g=g)
        release_lower = [int(i) for i in L if red[i] < -multiplier_tol]
        release_upper = [int(i) for i in U if red[i] > multiplier_tol]
        if release_lower or release_upper:
            for i in release_lower:
                L.discard(i)
            for i in release_upper:
                U.discard(i)
            continue

        eq_res = float(np.linalg.norm(E @ s - r_hard))
        if eq_res > float(eq_tol) * max(1.0, float(np.linalg.norm(s)), float(np.linalg.norm(r_hard))):
            raise RuntimeError("Returned solution violates the retained hard moments beyond tolerance.")

        return s

    raise RuntimeError("Soft mixed-moment admissible solver did not converge within maxiter.")

def solve_sigma_x_nodes_constrained_qp(
    *,
    Qk: np.ndarray,
    Qeig: np.ndarray,
    w_norm: np.ndarray,
    rt_node: np.ndarray,
    rx_node: np.ndarray,
    sigma_t_nodes: np.ndarray,
    p: np.ndarray,
    w_base: np.ndarray,
    a: float,
    qp_retained_mode: str = "minus1_and_0",
    qp_objective_weight_mode: str = "identity",
    qp_gamma_mode: str = "weighted_total",
    qp_relax_gamma_to_seed: bool = True,
    qp_solver: str = "active-set",
    qp_maxiter: int = 500,
    qp_gtol: float = 1e-10,
    qp_xtol: float = 1e-12,
    qp_barrier_tol: float = 1e-12,
    qp_real_imag_tol: float = 1e-10,
):
    """
    Kept for compatibility.

    Current behavior:
      - compute the original full-matching solution s_full first;
      - if s_full is already componentwise nonnegative, accept it directly;
      - otherwise, enforce admissibility under the retained HARD mixed moments
        (n=-1 and n=0) and nonnegativity, while penalizing ONLY the remaining
        interior (N-2) mixed-moment residuals via a normalized least-squares
        objective.

    So, when correction is needed, we solve

        min_s 0.5 * ||A_obj s - b_obj||_2^2
        s.t.  E s = r_hard,
              s >= 0,

    where E encodes the retained n=-1 and n=0 moments, and A_obj/b_obj encode
    the normalized residuals of the remaining interior mixed moments.

    The legacy QP_* option names are retained only to minimize intrusion into the
    rest of the script:
      - qp_objective_weight_mode controls optional row weights on the soft moments,
      - qp_maxiter is the active-set iteration cap,
      - qp_gtol / qp_xtol / qp_barrier_tol provide numerical tolerances,
      - qp_gamma_mode / qp_relax_gamma_to_seed / qp_solver are now ignored.
    """
    sigma_t_real = _coerce_real_vector_for_qp(
        sigma_t_nodes, name="sigma_t_nodes", imag_tol=qp_real_imag_tol, positive=True
    )
    p_real = _coerce_real_vector_for_qp(
        p, name="p", imag_tol=qp_real_imag_tol, positive=True
    )

    M_match, b_match, condM, rcondM = build_full_matching_system_from_eigbasis(
        Qk=Qk,
        Qeig=Qeig,
        w_norm=w_norm,
        rx_node=rx_node,
    )

    s_full, _condM_full, _rcondM_full = solve_sigma_x_nodes_from_eigbasis(Q=Qeig, b=b_match)
    s_full = np.asarray(s_full, dtype=float).ravel()

    s_seed, _u_num_seed, ell, u_upper = build_positive_seed_and_local_bounds(
        sigma_t_nodes=sigma_t_real,
        p=p_real,
        rt_node=rt_node,
        rx_node=rx_node,
        w_base=w_base,
        imag_tol=qp_real_imag_tol,
    )

    bound_tol = max(float(qp_xtol), 1.0e-14)
    within_nonnegative = np.all(s_full >= ell - bound_tol)

    # Diagnostics info is kept with the same keys used elsewhere in the script.
    correction_info = {
        "full_solution_used_directly": bool(within_nonnegative),
        "constraint_switch_due_to_violation": bool(not within_nonnegative),
        "full_solution_min": float(np.min(s_full)) if s_full.size else float("nan"),
        "full_solution_sigma_x": np.asarray(s_full, dtype=float).copy(),
        "n_negative_full_entries": int(np.sum(s_full < ell - bound_tol)),
        "used_nonnegativity_only": True,
        "used_global_constraint": False,
        "soft_moment_only": bool(not within_nonnegative),
        "n_soft_moment_constraints": 0,
        "n_hard_moment_constraints": 0,
    }

    if within_nonnegative:
        sigma_x_nodes = s_full
    else:
        E, hard_orders = build_qp_retained_coefficients(
            sigma_t_nodes=sigma_t_real,
            p=p_real,
            a=a,
            retained_mode=qp_retained_mode,
            imag_tol=qp_real_imag_tol,
        )
        r_hard = build_qp_hard_moment_rhs_from_discrete(
            rt_node=rt_node,
            rx_node=rx_node,
            w_base=w_base,
            moment_orders=tuple(float(v) for v in hard_orders),
        )

        A_obj, b_obj, H, g, soft_orders = build_soft_moment_objective_operator(
            sigma_t_nodes=sigma_t_real,
            p=p_real,
            rt_node=rt_node,
            rx_node=rx_node,
            w_base=w_base,
            weight_mode=qp_objective_weight_mode,
            hard_orders=tuple(float(v) for v in hard_orders),
        )

        sigma_x_nodes = _solve_soft_moment_admissible_qp(
            A_obj=A_obj,
            b_obj=b_obj,
            H=H,
            g=g,
            E=E,
            r_hard=r_hard,
            ell=ell,
            u=u_upper,
            x_ref=s_seed,
            maxiter=int(qp_maxiter),
            eq_tol=max(float(qp_gtol), float(qp_barrier_tol), 1.0e-14),
            bound_tol=bound_tol,
            multiplier_tol=max(float(qp_gtol), 1.0e-14),
        )

        correction_info["n_soft_moment_constraints"] = int(soft_orders.size)
        correction_info["n_hard_moment_constraints"] = int(E.shape[0])

    sigma_x_nodes = np.asarray(sigma_x_nodes, dtype=float).ravel()
    if sigma_x_nodes.size != sigma_t_real.size:
        raise RuntimeError("sigma_x reconstruction returned wrong vector length.")

    u_num = p_real * sigma_x_nodes
    ubar = u_num * np.power(sigma_t_real, a)
    return sigma_x_nodes, ubar, condM, rcondM, correction_info

def reconstruct_sigma_x_nodes(
    *,
    method: str,
    Qk: np.ndarray,
    Qeig: np.ndarray,
    w_norm: np.ndarray,
    rt_node: np.ndarray,
    rx_node: np.ndarray,
    sigma_t_nodes: np.ndarray,
    p: np.ndarray,
    p_norm: np.ndarray,
    m0: float,
    w_base: np.ndarray,
    a: float,
    qp_retained_mode: str = "minus1_and_0",
    qp_objective_weight_mode: str = "identity",
    qp_gamma_mode: str = "weighted_total",
    qp_relax_gamma_to_seed: bool = True,
    qp_solver: str = "active-set",
    qp_maxiter: int = 500,
    qp_gtol: float = 1e-10,
    qp_xtol: float = 1e-12,
    qp_barrier_tol: float = 1e-12,
    qp_real_imag_tol: float = 1e-10,
):
    """
    Dispatch partial-level reconstruction method.

    method:
      - "eigsolve"     : original mixed-moment/eigenbasis full matching
      - "positive_hat" : positive local-hat projection on Gauss sigma_t nodes
      - "fullcorr"     : full-first admissible correction (方案.pdf)

    Aliases:
      - "active_set", "admissible_correction", "full_first"
      - "qp", "constrained_qp"  -> mapped to the same full-first correction
    """
    m = str(method).strip().lower()

    default_info = {
        "full_solution_used_directly": False,
        "constraint_switch_due_to_violation": False,
        "full_solution_min": float("nan"),
        "full_solution_sigma_x": None,
        "n_negative_full_entries": 0,
        "used_nonnegativity_only": False,
        "used_global_constraint": False,
    }

    if m == "eigsolve":
        t = np.sqrt(w_norm) * rx_node
        bvec = (Qk.T @ t).astype(float)
        sigma_x_nodes, condM, rcondM = solve_sigma_x_nodes_from_eigbasis(Q=Qeig, b=bvec)
        pi_nodes = m0 * p_norm
        ubar = pi_nodes * sigma_x_nodes
        return sigma_x_nodes, ubar, condM, rcondM, dict(default_info)

    if m == "positive_hat":
        sigma_x_nodes, ubar, condM, rcondM = solve_sigma_x_nodes_positive_hat(
            sigma_t_nodes=sigma_t_nodes,
            p=p,
            rt_node=rt_node,
            rx_node=rx_node,
            w_base=w_base,
            a=a,
        )
        info = dict(default_info)
        info["used_nonnegativity_only"] = True
        return sigma_x_nodes, ubar, condM, rcondM, info

    if m in ("fullcorr", "active_set", "admissible_correction", "full_first", "qp", "constrained_qp"):
        return solve_sigma_x_nodes_constrained_qp(
            Qk=Qk,
            Qeig=Qeig,
            w_norm=w_norm,
            rt_node=rt_node,
            rx_node=rx_node,
            sigma_t_nodes=sigma_t_nodes,
            p=p,
            w_base=w_base,
            a=a,
            qp_retained_mode=qp_retained_mode,
            qp_objective_weight_mode=qp_objective_weight_mode,
            qp_gamma_mode=qp_gamma_mode,
            qp_relax_gamma_to_seed=qp_relax_gamma_to_seed,
            qp_solver=qp_solver,
            qp_maxiter=qp_maxiter,
            qp_gtol=qp_gtol,
            qp_xtol=qp_xtol,
            qp_barrier_tol=qp_barrier_tol,
            qp_real_imag_tol=qp_real_imag_tol,
        )

    raise ValueError(
        "Unknown SIGMA_X_METHOD. Use 'eigsolve', 'positive_hat', or 'fullcorr'."
    )

# ============================================================
# NRA references
# ============================================================
def nra_effective_x_reference_ana(
    XS_t,
    XS_x,
    edges_asc,
    g_low,
    sigma0_grid,
    *,
    densify_gauss_N: int | None = None,
):
    """
    Stable piecewise-analytic NRA reference integral on the (optionally densified) union grid.
    """
    def _log1p_over_x(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        ax = np.abs(x)
        out = np.empty_like(x, dtype=float)
        small = ax < SERIES_X_TOL
        xs = x[small]
        out[small] = 1.0 - xs/2.0 + xs*xs/3.0 - xs**3/4.0 + xs**4/5.0
        out[~small] = np.log1p(x[~small]) / x[~small]
        return out

    def _x_minus_log1p_over_x2(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        ax = np.abs(x)
        out = np.empty_like(x, dtype=float)
        small = ax < SERIES_X_TOL
        xs = x[small]
        out[small] = 0.5 - xs/3.0 + xs*xs/4.0 - xs**3/5.0 + xs**4/6.0
        out[~small] = (x[~small] - np.log1p(x[~small])) / (x[~small] ** 2)
        return out

    def _B_over_A(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        ax = np.abs(x)
        out = np.empty_like(x, dtype=float)
        small = ax < SERIES_X_TOL
        xs = x[small]
        out[small] = (
            0.5
            - xs / 12.0
            + xs * xs / 24.0
            - 19.0 * xs**3 / 720.0
            + 3.0 * xs**4 / 160.0
        )
        if np.any(~small):
            A = _log1p_over_x(x[~small])
            B = _x_minus_log1p_over_x2(x[~small])
            out[~small] = B / A
        return out

    edges = np.asarray(edges_asc, dtype=float)
    Et, Rt = XS_t[:, 0], XS_t[:, 1]
    Ex, Rx = XS_x[:, 0], XS_x[:, 1]

    eL = float(edges[g_low])
    eR = float(edges[g_low + 1])
    if eR <= eL:
        return np.full_like(sigma0_grid, np.nan, dtype=float)

    mask_t = (Et >= eL) & (Et <= eR)
    mask_x = (Ex >= eL) & (Ex <= eR)
    nodes = np.concatenate(([eL, eR], Et[mask_t], Ex[mask_x]))
    nodes = np.unique(nodes)
    nodes = nodes[(nodes >= eL) & (nodes <= eR)]
    nodes = np.unique(np.sort(nodes))
    if nodes.size < 2:
        return np.full_like(sigma0_grid, np.nan, dtype=float)

    if densify_gauss_N is not None:
        Nd = int(densify_gauss_N)
        if Nd >= 2:
            target_nodes = int(max(nodes.size, 2 * Nd))
            if nodes.size < target_nodes:
                nodes = densify_union_nodes_linear(nodes, target_nodes)

    if nodes.size < 2:
        return np.full_like(sigma0_grid, np.nan, dtype=float)

    rt = np.interp(nodes, Et, Rt)
    rx = np.interp(nodes, Ex, Rx)

    dx = nodes[1:] - nodes[:-1]
    rtL, rtR = rt[:-1], rt[1:]
    rxL, rxR = rx[:-1], rx[1:]

    ok = (
        (dx > 0.0)
        & np.isfinite(rtL) & np.isfinite(rtR) & np.isfinite(rxL) & np.isfinite(rxR)
        & (rtL > 0.0) & (rtR > 0.0)
    )
    if not np.any(ok):
        return np.full_like(sigma0_grid, np.nan, dtype=float)

    dx = dx[ok]
    rtL = rtL[ok]
    rtR = rtR[ok]
    rxL = rxL[ok]
    rxR = rxR[ok]

    dt = rtR - rtL
    e = rxR - rxL

    out = np.zeros_like(sigma0_grid, dtype=float)

    for j, s0 in enumerate(sigma0_grid):
        if not (np.isfinite(s0) and s0 > 0.0):
            out[j] = np.nan
            continue

        c = rtL + s0
        x = dt / c

        A = _log1p_over_x(x)
        D = (dx / c) * A

        H = _B_over_A(x)
        r_eff = rxL + e * H

        denom = math.fsum(D.tolist())
        numer = math.fsum((D * r_eff).tolist())

        out[j] = numer / denom if denom != 0.0 else np.nan

    return out

def nra_effective_x_reference_trap(
    XS_t,
    XS_x,
    edges_asc,
    g_low,
    sigma0_grid,
    *,
    densify_gauss_N: int | None = None,
):
    """
    Composite-trapezoid NRA reference integral on the (optionally densified) union grid.
    """
    edges = np.asarray(edges_asc, dtype=float)
    Et, Rt = XS_t[:, 0], XS_t[:, 1]
    Ex, Rx = XS_x[:, 0], XS_x[:, 1]

    eL = float(edges[g_low])
    eR = float(edges[g_low + 1])
    if eR <= eL:
        return np.full_like(sigma0_grid, np.nan, dtype=float)

    mask_t = (Et >= eL) & (Et <= eR)
    mask_x = (Ex >= eL) & (Ex <= eR)
    nodes = np.concatenate(([eL, eR], Et[mask_t], Ex[mask_x]))
    nodes = np.unique(nodes)
    nodes = nodes[(nodes >= eL) & (nodes <= eR)]
    nodes = np.unique(np.sort(nodes))
    if nodes.size < 2:
        return np.full_like(sigma0_grid, np.nan, dtype=float)

    if densify_gauss_N is not None:
        Nd = int(densify_gauss_N)
        if Nd >= 2:
            target_nodes = int(max(nodes.size, 2 * Nd))
            if nodes.size < target_nodes:
                nodes = densify_union_nodes_linear(nodes, target_nodes)

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

def nra_effective_x_reference_dispatch(
    *,
    realization_mode: str,
    XS_t,
    XS_x,
    edges_asc,
    g_low,
    sigma0_grid,
    densify_gauss_N: int | None = None,
):
    r = str(realization_mode).strip().lower()
    if r == "ana":
        return nra_effective_x_reference_ana(
            XS_t, XS_x, edges_asc, g_low, sigma0_grid,
            densify_gauss_N=densify_gauss_N,
        )
    if r == "trap":
        return nra_effective_x_reference_trap(
            XS_t, XS_x, edges_asc, g_low, sigma0_grid,
            densify_gauss_N=densify_gauss_N,
        )
    raise ValueError("Unknown realization_mode. Use 'ana' or 'trap'.")

def nra_effective_x_probability_table_chiba_ubar(sigma_t_nodes, p, ubar, a, sigma0_grid):
    """
    Numerically stable PT evaluation for large sigma0.
    """
    r = np.asarray(sigma_t_nodes).ravel()
    p = np.asarray(p).ravel()
    ubar = np.asarray(ubar).ravel()

    out_dtype = np.result_type(r, p, ubar, np.float64)
    out = np.empty_like(sigma0_grid, dtype=out_dtype)

    ra = np.power(r, a)
    lvl = ubar / ra

    for j, s0 in enumerate(sigma0_grid):
        if not (np.isfinite(s0) and s0 > 0.0):
            out[j] = np.nan
            continue

        t = 1.0 / (1.0 + r / s0)

        den = np.sum(p * t)
        num = np.sum(lvl * t)

        out[j] = num / den if den != 0 else np.nan

    return out

# ============================================================
# Direct discrete surrogate (generic for ana and trap)
# ============================================================
def nra_effective_x_direct_discrete(rt_node: np.ndarray, rx_node: np.ndarray, w_base: np.ndarray, sigma0_grid: np.ndarray) -> np.ndarray:
    """
    Direct discrete surrogate on the SAME discrete atoms:

        direct(s0) = sum_i w_i * rx_i/(rt_i+s0) / sum_i w_i * 1/(rt_i+s0).

    Works for both:
      - ana  : GL discrete atoms on union segments
      - trap : nodal trapezoidal atoms on the union grid
    """
    rt = np.asarray(rt_node, dtype=float).ravel()
    rx = np.asarray(rx_node, dtype=float).ravel()
    w = np.asarray(w_base, dtype=float).ravel()
    sgrid = np.asarray(sigma0_grid, dtype=float).ravel()

    if rt.size == 0 or rx.size == 0 or w.size == 0:
        return np.full_like(sgrid, np.nan, dtype=float)
    if not (rt.size == rx.size == w.size):
        raise ValueError("rt_node/rx_node/w_base must have same size in direct discrete surrogate.")

    out = np.full_like(sgrid, np.nan, dtype=float)

    m = np.isfinite(rt) & np.isfinite(rx) & np.isfinite(w) & (w > 0.0) & (rt > 0.0)
    rt = rt[m]
    rx = rx[m]
    w = w[m]
    if rt.size == 0:
        return out

    for j, s0 in enumerate(sgrid):
        if not (np.isfinite(s0) and s0 > 0.0):
            out[j] = np.nan
            continue

        t = 1.0 / (1.0 + rt / s0)
        den = float(np.sum(w * t))
        if den == 0.0 or (not np.isfinite(den)):
            out[j] = np.nan
            continue
        num = float(np.sum(w * rx * t))
        out[j] = num / den

    return out

# ============================================================
# REF export helper
# ============================================================
def append_ref_curve_to_txt(
    txt_path: Path,
    group_no: int,
    g_low: int,
    e_high: float,
    e_low: float,
    sigma0_grid: np.ndarray,
    ref_curve: np.ndarray,
) -> None:
    txt_path = Path(txt_path)
    sgrid = np.asarray(sigma0_grid, dtype=float).ravel()
    refv = np.asarray(ref_curve, dtype=float).ravel()
    n = min(sgrid.size, refv.size)

    with txt_path.open("a", encoding="utf-8") as f:
        for k in range(n):
            f.write(
                f"{group_no:6d}  {g_low:5d}  {e_high:.8g}  {e_low:.8g}  "
                f"{sgrid[k]:.17e}  {refv[k]:.17e}\n"
            )

# ============================================================
# Error metrics and summary statistics (DIMENSIONLESS)
# ============================================================
def compute_error_rel(ref_curve: np.ndarray, pt_curve: np.ndarray) -> np.ndarray:
    """
    Relative error:
        err_rel = (PT - REF)/REF
    Always uses Re(PT) if complex.
    """
    pt_plot = np.real(pt_curve) if np.iscomplexobj(pt_curve) else pt_curve
    with np.errstate(divide="ignore", invalid="ignore"):
        err_rel = (pt_plot - ref_curve) / ref_curve
    return np.asarray(err_rel, dtype=float)

def compute_incremental_error_rel_refden(
    ref_curve: np.ndarray,
    hi_curve: np.ndarray,
    lo_curve: np.ndarray,
) -> np.ndarray:
    """
    Incremental relative error with UNIFIED REFERENCE denominator:

        err_rel = (hi_curve - lo_curve)/ref_curve
    """
    hi_plot = np.real(hi_curve) if np.iscomplexobj(hi_curve) else hi_curve
    lo_plot = np.real(lo_curve) if np.iscomplexobj(lo_curve) else lo_curve

    with np.errstate(divide="ignore", invalid="ignore"):
        err_rel = (hi_plot - lo_plot) / ref_curve
    return np.asarray(err_rel, dtype=float)

def _finite_percentile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=float).ravel()
    m = np.isfinite(x)
    if not np.any(m):
        return float("nan")
    return float(np.percentile(x[m], q))

def compute_abs_error_stats_rel(ref_curve: np.ndarray, pt_curve: np.ndarray) -> tuple[float, float]:
    """
    Returns (max_abs_err_rel, q95_abs_err_rel) over the sigma0 grid, ignoring NaNs/Infs.
    """
    err_rel = compute_error_rel(ref_curve, pt_curve)
    abs_err = np.abs(err_rel)
    max_abs = float(np.nanmax(abs_err)) if np.any(np.isfinite(abs_err)) else float("nan")
    q95_abs = _finite_percentile(abs_err, 95.0)
    return max_abs, q95_abs

def compute_abs_incremental_error_stats_rel_refden(
    ref_curve: np.ndarray,
    hi_curve: np.ndarray,
    lo_curve: np.ndarray,
) -> tuple[float, float]:
    err_rel = compute_incremental_error_rel_refden(ref_curve, hi_curve, lo_curve)
    abs_err = np.abs(err_rel)
    max_abs = float(np.nanmax(abs_err)) if np.any(np.isfinite(abs_err)) else float("nan")
    q95_abs = _finite_percentile(abs_err, 95.0)
    return max_abs, q95_abs

def compute_q95_abs_error_rel_trunc(ref_curve: np.ndarray, pt_curve: np.ndarray, mask: np.ndarray) -> float:
    err_rel = compute_error_rel(ref_curve, pt_curve)
    err_rel = np.asarray(err_rel, dtype=float)
    m = np.asarray(mask, dtype=bool).ravel()
    if m.size != err_rel.size:
        raise ValueError("mask size mismatch for truncated stats.")
    x = np.abs(err_rel[m])
    return _finite_percentile(x, 95.0)

def append_group_sigma0_error_data_txt(
    txt_path: Path,
    group_no: int,
    e_high: float,
    e_low: float,
    sigma0_grid: np.ndarray,
    ref_curve: np.ndarray,
    pt_curve_for_error: np.ndarray,
    err_rel: np.ndarray,
    *,
    N_g: int,
    b_g: float,
    sigma_x_method: str,
    realization_mode: str,
    tag: str = "",
) -> None:
    """
    Append per-group sigma_0-dependent error data to a single txt file.

    The recorded PT curve is exactly the real-valued curve used in the
    signed-relative-error plot.
    """
    txt_path = Path(txt_path)
    txt_path.parent.mkdir(parents=True, exist_ok=True)

    sigma0 = np.asarray(sigma0_grid, dtype=float).ravel()
    ref = np.asarray(ref_curve, dtype=float).ravel()
    pt = np.asarray(pt_curve_for_error, dtype=float).ravel()
    err = np.asarray(err_rel, dtype=float).ravel()

    if not (sigma0.size == ref.size == pt.size == err.size):
        raise ValueError("sigma0/ref/pt/err size mismatch in sigma0 error txt export.")

    mode = "a" if txt_path.exists() else "w"
    with txt_path.open(mode, encoding="utf-8") as f:
        if mode == "w":
            f.write("# Per-group sigma0-dependent effective-cross-section error data\n")
            f.write("# Error definition: signed_rel_err = (PT_plot_used - REF) / REF\n")
            f.write("# PT_plot_used is the real-valued PT curve actually used in the scatter plot.\n")
            f.write("# Columns per data row:\n")
            f.write("# sigma0  ref_eff_x  pt_eff_x_used_for_plot  signed_rel_err  abs_rel_err\n")

        f.write(f"\n# ---- group {int(group_no)} ----\n")
        f.write(
            f"# E_high={float(e_high):.17e}  E_low={float(e_low):.17e}  "
            f"N_g={int(N_g)}  b_g={float(b_g):.17e}  "
            f"realization={realization_mode}  sigma_x_method={sigma_x_method}"
            + (f"  tag={tag}" if tag else "")
            + "\n"
        )

        for s0, r, p, e in zip(sigma0, ref, pt, err):
            ae = abs(e) if np.isfinite(e) else np.nan
            s0_str = f"{s0:.17e}" if np.isfinite(s0) else "nan"
            r_str = f"{r:.17e}" if np.isfinite(r) else "nan"
            p_str = f"{p:.17e}" if np.isfinite(p) else "nan"
            e_str = f"{e:.17e}" if np.isfinite(e) else "nan"
            ae_str = f"{ae:.17e}" if np.isfinite(ae) else "nan"
            f.write(f"{s0_str}  {r_str}  {p_str}  {e_str}  {ae_str}\n")

# ============================================================
# Moment diagnostics
# ============================================================
def compute_moment_errors(
    rt_node: np.ndarray,
    rx_node: np.ndarray,
    w_base: np.ndarray,
    sigma_t_nodes: np.ndarray,
    sigma_x_nodes: np.ndarray,
    p: np.ndarray,
    *,
    n_samples: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compare the original mixed moments
        m(n) = sum_j w_base[j] * rx_node[j] * rt_node[j]^n
    and their compressed approximation
        m_N(n) = sum_i p[i] * sigma_x_nodes[i] * sigma_t_nodes[i]^n
    on a truly sampled grid of n-values in [-1, 0].

    This is not an interpolation between two retained moments; each sample point
    is evaluated directly from the discrete realization and the compressed table.
    """
    rt = np.asarray(rt_node, dtype=float).ravel()
    rx = np.asarray(rx_node, dtype=float).ravel()
    w = np.asarray(w_base, dtype=float).ravel()
    st = np.asarray(np.real(sigma_t_nodes), dtype=float).ravel()
    sx = np.asarray(np.real(sigma_x_nodes), dtype=float).ravel()
    pp = np.asarray(np.real(p), dtype=float).ravel()

    if not (rt.size == rx.size == w.size):
        raise ValueError("rt_node, rx_node, and w_base size mismatch.")
    if not (st.size == sx.size == pp.size):
        raise ValueError("sigma_t_nodes, sigma_x_nodes, and p size mismatch.")
    if rt.size == 0 or st.size == 0:
        raise ValueError("Empty inputs in mixed-moment diagnostics.")
    if np.any(~np.isfinite(rt)) or np.any(~np.isfinite(rx)) or np.any(~np.isfinite(w)):
        raise ValueError("Non-finite rt_node/rx_node/w_base in mixed-moment diagnostics.")
    if np.any(~np.isfinite(st)) or np.any(~np.isfinite(sx)) or np.any(~np.isfinite(pp)):
        raise ValueError("Non-finite sigma_t_nodes/sigma_x_nodes/p in mixed-moment diagnostics.")
    if np.any(rt <= 0.0):
        raise ValueError("rt_node must be strictly positive for mixed-moment diagnostics on [-1,0].")
    if np.any(st <= 0.0):
        raise ValueError("sigma_t_nodes must be strictly positive for mixed-moment diagnostics on [-1,0].")

    n_samples = int(n_samples)
    if n_samples < 2:
        raise ValueError("n_samples must be >= 2.")

    k_arr = np.linspace(-1.0, 0.0, n_samples, dtype=float)

    target = np.full(k_arr.size, np.nan, dtype=float)
    approx = np.full(k_arr.size, np.nan, dtype=float)
    rel_err = np.full(k_arr.size, np.nan, dtype=float)

    for idx, n in enumerate(k_arr):
        rt_pow = np.power(rt, float(n))
        st_pow = np.power(st, float(n))

        mt = float(np.sum(w * rx * rt_pow))
        ma = float(np.sum(pp * sx * st_pow))

        target[idx] = mt
        approx[idx] = ma

        abs_err = abs(ma - mt)
        scale = max(abs(mt), 1.0e-300)
        rel_err[idx] = abs_err / scale

    return k_arr, target, approx, rel_err

def append_moment_errors_to_txt(
    txt_path: Path,
    *,
    group_no: int,
    g_low: int,
    e_high: float,
    e_low: float,
    N_g: int,
    mode_tag: str,
    k_arr: np.ndarray,
    target: np.ndarray,
    approx: np.ndarray,
    rel_err: np.ndarray,
) -> None:
    txt_path = Path(txt_path)
    k_arr = np.asarray(k_arr, dtype=float).ravel()
    target = np.asarray(target, dtype=float).ravel()
    approx = np.asarray(approx, dtype=float).ravel()
    rel_err = np.asarray(rel_err, dtype=float).ravel()

    n = min(k_arr.size, target.size, approx.size, rel_err.size)
    with txt_path.open("a", encoding="utf-8") as f:
        for i in range(n):
            abs_err = abs(float(approx[i]) - float(target[i])) if (np.isfinite(approx[i]) and np.isfinite(target[i])) else float("nan")
            f.write(
                f"{group_no:6d}  {g_low:5d}  {e_high:.8g}  {e_low:.8g}  "
                f"{N_g:4d}  {mode_tag:9s}  {k_arr[i]:9.6f}  "
                f"{target[i]:.17e}  {approx[i]:.17e}  {abs_err:.17e}  {rel_err[i]:.17e}\n"
            )

def save_group_moment_error_plot(
    outdir: Path,
    *,
    group_no: int,
    e_high: float,
    e_low: float,
    N_g: int,
    mode_tag: str,
    k_arr: np.ndarray,
    rel_err: np.ndarray,
) -> tuple[float, float]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    k_arr = np.asarray(k_arr, dtype=float).ravel()
    rel_err = np.asarray(rel_err, dtype=float).ravel()

    max_rel = float(np.nanmax(rel_err)) if np.any(np.isfinite(rel_err)) else float("nan")
    q95_rel = _finite_percentile(rel_err, 95.0)

    m = np.isfinite(k_arr) & np.isfinite(rel_err) & (rel_err > 0.0)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    if np.any(m):
        x = k_arr[m]
        y = rel_err[m]
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        ax.plot(x, y, linewidth=0.9, zorder=1)
        ax.scatter(x, y, s=POINT_SIZE_MOMENT, zorder=2)
        ax.set_yscale("log")
        ax.set_xlim(float(np.min(x)), float(np.max(x)))
    else:
        ax.scatter(k_arr, np.zeros_like(k_arr, dtype=float), s=POINT_SIZE_MOMENT)

    if np.isfinite(np.min(k_arr)) and np.isfinite(np.max(k_arr)) and (np.max(k_arr) > np.min(k_arr)):
        ax.set_xticks(np.linspace(float(np.min(k_arr)), float(np.max(k_arr)), 6))

    ax.set_xlabel(r"Mixed-moment order $n$")
    ax.set_ylabel(r"Relative mixed-moment error")
    ax.set_title(
        f"Mixed-moment reproduction error on sampled n-grid: "
        f"group g={group_no}, E∈[{e_high:.6g},{e_low:.6g}], "
        f"N_g={N_g}, mode={mode_tag}"
    )

    fig.savefig(outdir / f"group_{group_no:03d}_moment_relerr.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    return max_rel, q95_rel

def save_moment_summary_plot_log(
    outdir: Path,
    group_nos: np.ndarray,
    stat_err: np.ndarray,
    *,
    filename: str,
    ylabel_tex: str,
    title: str,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gno = np.asarray(group_nos, dtype=int).ravel()
    y = np.asarray(stat_err, dtype=float).ravel()

    if gno.size != y.size:
        raise ValueError("group_nos and stat_err must have same length.")

    ok = np.isfinite(y) & (y > 0.0)
    if not np.any(ok):
        print(f"[MomentSummary] No positive finite values for {filename}. Skip saving.")
        return

    y_min = float(np.min(y[ok]))
    y_max = float(np.max(y[ok]))
    ymin = 10.0 ** np.floor(np.log10(y_min))
    ymax = 10.0 ** np.ceil(np.log10(y_max))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(gno[ok], y[ok], s=POINT_SIZE_SUMMARY)
    ax.set_xlabel(r"Energy-group index $g$ (high $\rightarrow$ low)")
    ax.set_ylabel(ylabel_tex)
    ax.set_title(title)
    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=12))
    ax.yaxis.set_major_formatter(FuncFormatter(_decimal_log_formatter))

    fig.savefig(outdir / filename, dpi=260, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# Plotting utilities
# ============================================================
def _decimal_log_formatter(y, _pos):
    if not np.isfinite(y) or y <= 0:
        return ""
    if 1e-6 <= y < 1e3:
        return f"{y:g}"
    return f"{y:.0e}"

def save_group_error_plot_rel_scatter(
    outdir: Path,
    group_no: int,
    e_high: float,
    e_low: float,
    sigma0_grid: np.ndarray,
    ref_curve: np.ndarray,
    pt_curve: np.ndarray,
    *,
    N_g: int,
    b_g: float,
    sigma_x_method: str,
    realization_mode: str,
    tag: str = "",
) -> tuple[float, float]:
    """
    Saves per-group scatter of signed relative error:
        (PT - REF)/REF

    Also appends the underlying sigma_0-dependent data to a txt file in the
    same output folder.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pt_plot = as_real_for_plot(pt_curve, name="PT_eff_x", group_no=group_no, tol=IMAG_TOL)
    with np.errstate(divide="ignore", invalid="ignore"):
        err_rel = (pt_plot - ref_curve) / ref_curve

    max_abs = float(np.nanmax(np.abs(err_rel))) if np.any(np.isfinite(err_rel)) else float("nan")
    q95_abs = _finite_percentile(np.abs(err_rel), 95.0)

    suffix = f"_{tag}" if tag else ""
    append_group_sigma0_error_data_txt(
        txt_path=outdir / f"sigma0_error_by_group{suffix}.txt",
        group_no=group_no,
        e_high=e_high,
        e_low=e_low,
        sigma0_grid=sigma0_grid,
        ref_curve=ref_curve,
        pt_curve_for_error=pt_plot,
        err_rel=err_rel,
        N_g=N_g,
        b_g=b_g,
        sigma_x_method=sigma_x_method,
        realization_mode=realization_mode,
        tag=tag,
    )

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(sigma0_grid, err_rel, s=POINT_SIZE_GROUP, c=COLOR_RELERR)
    ax.set_xscale("log")
    ax.set_xlabel("Background cross section")
    ax.set_ylabel("Signed relative effective-cross-section error")
    ax.set_title(
        r"Lanczos--Golub--Welsch Gauss compression: "
        f"group $g={group_no}$, $E\\in[{e_high:.6g},{e_low:.6g}]$, "
        f"$N_g={N_g}$, $b={b_g:.6g}$, "
        f"realization={realization_mode}, "
        f"$\\sigma_x$ method={sigma_x_method}"
        + (f" [{tag}]" if tag else "")
    )

    fig.savefig(outdir / f"group_{group_no:03d}_err{suffix}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    return max_abs, q95_abs

def save_group_abs_error_plot_barn(
    outdir: Path,
    group_no: int,
    e_high: float,
    e_low: float,
    sigma0_grid: np.ndarray,
    ref_curve: np.ndarray,
    pt_curve: np.ndarray,
    *,
    realization_mode: str,
    tag: str = "",
):
    """
    Per-group absolute error plot:
        |PT(σ0) - REF(σ0)|   (barn)
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pt_plot = as_real_for_plot(pt_curve, name="PT_eff_x", group_no=group_no, tol=IMAG_TOL)
    diff = np.asarray(pt_plot - ref_curve, dtype=float)
    y = np.abs(diff)

    m = np.isfinite(y) & (y > 0.0) & np.isfinite(sigma0_grid) & (sigma0_grid > 0.0)
    if not np.any(m):
        return

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(sigma0_grid[m], y[m], s=POINT_SIZE_GROUP)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Background (dilution) cross section $\sigma_0$ (barn)")
    ax.set_ylabel(r"Absolute error $|PT(\sigma_0)-REF(\sigma_0)|$ (barn)")
    ax.set_title(
        r"Absolute error vs $\sigma_0$: "
        f"group $g={group_no}$, $E\\in[{e_high:.6g},{e_low:.6g}]$, "
        f"realization={realization_mode}"
        + (f" [{tag}]" if tag else "")
    )

    suffix = f"_{tag}" if tag else ""
    fig.savefig(outdir / f"group_{group_no:03d}_abs_err{suffix}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

def save_group_error_plot_rel_scatter_uncompressed(
    outdir: Path,
    group_no: int,
    e_high: float,
    e_low: float,
    sigma0_grid: np.ndarray,
    ref_curve: np.ndarray,
    direct_curve: np.ndarray,
    *,
    M_atoms: int,
    realization_mode: str,
    direct_label: str,
    ref_label: str,
    gl_nq: int,
) -> tuple[float, float]:
    """
    Uncompressed baseline plot:
      signed relative error = (DIRECT - REF)/REF
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    y = np.asarray(direct_curve, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        err_rel = (y - ref_curve) / ref_curve

    max_abs = float(np.nanmax(np.abs(err_rel))) if np.any(np.isfinite(err_rel)) else float("nan")
    q95_abs = _finite_percentile(np.abs(err_rel), 95.0)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(sigma0_grid, err_rel, s=POINT_SIZE_GROUP, c=COLOR_RELERR)
    ax.set_xscale("log")
    ax.set_xlabel("Background cross section")
    ax.set_ylabel("Signed relative effective-cross-section error")

    extra = f", GL_NQ={gl_nq}" if str(realization_mode).lower() == "ana" else ""
    ax.set_title(
        f"UNCOMPRESSED {direct_label} vs {ref_label}: "
        f"group g={group_no}, E∈[{e_high:.6g},{e_low:.6g}], "
        f"M={M_atoms}{extra}"
    )

    fig.savefig(outdir / f"group_{group_no:03d}_err_uncompressed.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    return max_abs, q95_abs

def save_group_error_plot_rel_scatter_compression(
    outdir: Path,
    group_no: int,
    e_high: float,
    e_low: float,
    sigma0_grid: np.ndarray,
    ref_curve: np.ndarray,
    direct_curve: np.ndarray,
    pt_curve: np.ndarray,
    *,
    mode_tag: str,
    direct_label: str,
    ref_label: str,
) -> tuple[float, float]:
    """
    Compression-induced error plot with unified reference denominator:
      signed relative error = (PT - DIRECT)/REF
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pt_plot = as_real_for_plot(pt_curve, name="PT_eff_x", group_no=group_no, tol=IMAG_TOL)
    direct_plot = np.asarray(direct_curve, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        err_rel = (pt_plot - direct_plot) / ref_curve

    max_abs = float(np.nanmax(np.abs(err_rel))) if np.any(np.isfinite(err_rel)) else float("nan")
    q95_abs = _finite_percentile(np.abs(err_rel), 95.0)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(sigma0_grid, err_rel, s=POINT_SIZE_GROUP, c=COLOR_RELERR)
    ax.set_xscale("log")
    ax.set_xlabel("Background cross section")
    ax.set_ylabel("Signed relative effective-cross-section error")
    ax.set_title(
        f"COMPRESSION-induced error with reference denominator: "
        f"(PT(mode={mode_tag}) - {direct_label}) / {ref_label}, "
        f"group g={group_no}, E∈[{e_high:.6g},{e_low:.6g}]"
    )

    fig.savefig(outdir / f"group_{group_no:03d}_err_compression.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    return max_abs, q95_abs

def save_error_stat_by_group_plot_log(
    outdir: Path,
    group_nos: np.ndarray,
    stat_err_rel: np.ndarray,
    flag_complex: np.ndarray,
    flag_negative: np.ndarray,
    flag_constraint_switch: np.ndarray | None = None,
    *,
    Nmax: int,
    ES: int,
    realization_mode: str,
    sigma_x_method: str,
    filename: str,
    ylabel_tex: str,
    title: str,
):
    """
    Plot a per-group statistic of |relative error| on log-scale.

    
    Groups whose sigma_x reconstruction switched away from the raw full solution
    because the partial levels violated the admissibility constraint are
    highlighted with a distinct marker overlay.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gno = np.asarray(group_nos, dtype=int).ravel()
    y = np.asarray(stat_err_rel, dtype=float).ravel()
    fc = np.asarray(flag_complex, dtype=bool).ravel()
    fn = np.asarray(flag_negative, dtype=bool).ravel()
    if flag_constraint_switch is None:
        fs = np.zeros_like(gno, dtype=bool)
    else:
        fs = np.asarray(flag_constraint_switch, dtype=bool).ravel()

    if not (gno.size == y.size == fc.size == fn.size == fs.size):
        raise ValueError("group_nos/stat_err_rel/flag_complex/flag_negative/flag_constraint_switch must have same length.")

    ok = np.isfinite(y) & (y > 0)
    if not np.any(ok):
        print(f"[SummaryPlot] No positive finite values for {filename}. Skip saving.")
        return

    both = ok & fc & fn
    comp_only = ok & fc & (~fn)
    neg_only = ok & fn & (~fc)
    normal = ok & (~fc) & (~fn)
    switched = ok & fs

    y_min = float(np.min(y[ok]))
    y_max = float(np.max(y[ok]))
    ymin = 10.0 ** np.floor(np.log10(y_min))
    ymax = 10.0 ** np.ceil(np.log10(y_max))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    if np.any(normal):
        ax.scatter(gno[normal], y[normal], s=POINT_SIZE_SUMMARY, c="tab:blue", label="real & nonnegative")
    if np.any(neg_only):
        ax.scatter(gno[neg_only], y[neg_only], s=POINT_SIZE_SUMMARY, c="tab:orange", label="negative detected")
    if np.any(comp_only):
        ax.scatter(gno[comp_only], y[comp_only], s=POINT_SIZE_SUMMARY, c="tab:red", label="complex detected")
    if np.any(both):
        ax.scatter(gno[both], y[both], s=POINT_SIZE_SUMMARY, c="tab:purple", label="complex & negative")

    if np.any(switched):
        ax.scatter(
            gno[switched],
            y[switched],
            s=max(16, 5 * POINT_SIZE_SUMMARY),
            facecolors="none",
            edgecolors="#C44E52",
            linewidths=1.0,
            marker="o",
            label=r"$\sigma_x$ constraint-triggered switch",
        )

    ax.set_xlabel(r"Energy-group index $g$ (high $\rightarrow$ low)")
    ax.set_ylabel(ylabel_tex)
    ax.set_title(
        f"{title}  (Case C, adaptive $N_g\\leq {Nmax}$; ES={ES}; "
        f"realization={realization_mode}; $\\sigma_x$ method={sigma_x_method})"
    )

    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=12))
    ax.yaxis.set_major_formatter(FuncFormatter(_decimal_log_formatter))
    ax.legend(loc="best", fontsize=9)

    fig.savefig(outdir / filename, dpi=260, bbox_inches="tight")
    plt.close(fig)

def save_q95_by_group_plot_log_uncompressed(
    outdir: Path,
    group_nos: np.ndarray,
    q95_err_rel: np.ndarray,
    *,
    ES: int,
    realization_mode: str,
    direct_label: str,
    ref_label: str,
    gl_nq: int,
    filename: str = "q95_err_by_group_uncompressed.png",
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gno = np.asarray(group_nos, dtype=int).ravel()
    q95_rel = np.asarray(q95_err_rel, dtype=float).ravel()

    if gno.size != q95_rel.size:
        raise ValueError("group_nos and q95_err_rel must have same length.")

    ok = np.isfinite(q95_rel) & (q95_rel > 0.0)
    if not np.any(ok):
        print(f"[UncompressedSummary] No positive finite q95 values for {filename}. Skip saving.")
        return

    y_min = float(np.min(q95_rel[ok]))
    y_max = float(np.max(q95_rel[ok]))
    ymin = 10.0 ** np.floor(np.log10(y_min))
    ymax = 10.0 ** np.ceil(np.log10(y_max))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(gno[ok], q95_rel[ok], s=POINT_SIZE_SUMMARY)

    extra = f", GL_NQ={gl_nq}" if str(realization_mode).lower() == "ana" else ""
    ax.set_xlabel(r"Energy-group index $g$ (high $\rightarrow$ low)")
    ax.set_ylabel(r"Effective cross-section relative error $E_{g,\mathrm{unc}}^{0.95}$")
    ax.set_title(
        f"UNCOMPRESSED {direct_label} vs {ref_label}: "
        f"Q0.95 by group  (ES={ES}, realization={realization_mode}{extra})"
    )

    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=12))
    ax.yaxis.set_major_formatter(FuncFormatter(_decimal_log_formatter))

    fig.savefig(outdir / filename, dpi=260, bbox_inches="tight")
    plt.close(fig)

def save_q95_by_group_plot_log_compression(
    outdir: Path,
    group_nos: np.ndarray,
    q95_err_rel: np.ndarray,
    *,
    ES: int,
    realization_mode: str,
    direct_label: str,
    ref_label: str,
    mode_tag: str,
    gl_nq: int,
    filename: str = "q95_err_by_group_compression.png",
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gno = np.asarray(group_nos, dtype=int).ravel()
    q95_rel = np.asarray(q95_err_rel, dtype=float).ravel()

    if gno.size != q95_rel.size:
        raise ValueError("group_nos and q95_err_rel must have same length.")

    ok = np.isfinite(q95_rel) & (q95_rel > 0.0)
    if not np.any(ok):
        print(f"[CompressionSummary] No positive finite q95 values for {filename}. Skip saving.")
        return

    y_min = float(np.min(q95_rel[ok]))
    y_max = float(np.max(q95_rel[ok]))
    ymin = 10.0 ** np.floor(np.log10(y_min))
    ymax = 10.0 ** np.ceil(np.log10(y_max))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(gno[ok], q95_rel[ok], s=POINT_SIZE_SUMMARY)

    extra = f", GL_NQ={gl_nq}" if str(realization_mode).lower() == "ana" else ""
    ax.set_xlabel(r"Energy-group index $g$ (high $\rightarrow$ low)")
    ax.set_ylabel(r"Compression-induced relative error $E_{g,\mathrm{comp,ref}}^{0.95}$")
    ax.set_title(
        f"COMPRESSION-induced with reference denominator: "
        f"(PT mode={mode_tag} - {direct_label}) / {ref_label}, "
        f"Q0.95 by group  (ES={ES}, realization={realization_mode}{extra})"
    )

    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=12))
    ax.yaxis.set_major_formatter(FuncFormatter(_decimal_log_formatter))

    fig.savefig(outdir / filename, dpi=260, bbox_inches="tight")
    plt.close(fig)

def _linear_axis_limits(values: np.ndarray, *, include_zero: bool = False) -> tuple[float, float]:
    vals = np.asarray(values, dtype=float).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (-1.0, 1.0)

    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if include_zero:
        vmin = min(vmin, 0.0)
        vmax = max(vmax, 0.0)

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return (-1.0, 1.0)

    if vmin == vmax:
        pad = 0.05 * max(1.0, abs(vmin))
        return (vmin - pad, vmax + pad)

    span = vmax - vmin
    pad = 0.06 * span
    return (vmin - pad, vmax + pad)

def _relative_l2_correction_size(corrected: np.ndarray, full: np.ndarray, *, tiny: float = 1.0e-300) -> float:
    """
    Relative L2 correction size ||corrected - full||_2 / max(||full||_2, tiny).
    Returns NaN if the input sizes do not match or contain non-finite values.
    """
    xc = np.asarray(corrected, dtype=float).ravel()
    xf = np.asarray(full, dtype=float).ravel()
    if xc.size == 0 or xf.size == 0 or xc.size != xf.size:
        return float("nan")
    if np.any(~np.isfinite(xc)) or np.any(~np.isfinite(xf)):
        return float("nan")
    num = float(np.linalg.norm(xc - xf))
    den = float(np.linalg.norm(xf))
    return num / max(den, tiny)

def save_full_reconstruction_admissibility_figure(
    outdir: Path,
    group_nos: np.ndarray,
    min_sigma_x_full: np.ndarray,
    violating_mask: np.ndarray,
    *,
    filename_base: str = "figure1_full_reconstruction_admissibility",
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gno = np.asarray(group_nos, dtype=int).ravel()
    min_full = np.asarray(min_sigma_x_full, dtype=float).ravel()
    viol = np.asarray(violating_mask, dtype=bool).ravel()

    if not (gno.size == min_full.size == viol.size):
        raise ValueError("group_nos, min_sigma_x_full, and violating_mask must have the same length.")

    y = np.maximum(0.0, -min_full)
    ok = np.isfinite(y)
    if not np.any(ok):
        print(f"[PaperFig1] No finite full-reconstruction violation magnitudes. Skip saving {filename_base}.")
        return

    normal = ok & (~viol)
    bad = ok & viol

    fig = plt.figure(figsize=(7.0, 4.2))
    ax = fig.add_subplot(1, 1, 1)

    if np.any(normal):
        ax.scatter(
            gno[normal],
            y[normal],
            s=max(12, POINT_SIZE_SUMMARY * 6),
            c="#B0B0B0",
            marker="o",
            linewidths=0.0,
            label="Non-violating groups",
            zorder=2,
        )
    if np.any(bad):
        ax.scatter(
            gno[bad],
            y[bad],
            s=max(24, POINT_SIZE_SUMMARY * 12),
            c="#C44E52",
            marker="o",
            linewidths=0.4,
            edgecolors="black",
            label="Violating groups",
            zorder=3,
        )

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, zorder=1)
    ax.set_xlabel("Energy group index")
    ax.set_ylabel("Lower-bound violation magnitude of the full reconstruction")
    ax.set_title("Figure 1. Lower-bound violation magnitude of the full reconstruction across energy groups.")
    ax.set_xlim(float(np.min(gno)) - 1.0, float(np.max(gno)) + 1.0)
    ax.set_ylim(*_linear_axis_limits(y[ok], include_zero=True))
    ax.legend(loc="best", frameon=False, fontsize=9)

    fig.savefig(outdir / f"{filename_base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / f"{filename_base}.pdf", bbox_inches="tight")
    plt.close(fig)

def save_violating_groups_before_after_figure(
    outdir: Path,
    violating_group_nos: np.ndarray,
    correction_size_rel: np.ndarray,
    err_full: np.ndarray,
    err_corrected: np.ndarray,
    *,
    filename_base: str = "figure2_violating_groups_before_after",
    error_metric_label: str = "Groupwise 95th-percentile relative error",
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gno = np.asarray(violating_group_nos, dtype=int).ravel()
    corr_rel = np.asarray(correction_size_rel, dtype=float).ravel()
    e_full = np.asarray(err_full, dtype=float).ravel()
    e_corr = np.asarray(err_corrected, dtype=float).ravel()

    if not (gno.size == corr_rel.size == e_full.size == e_corr.size):
        raise ValueError("Violating-group arrays must all have the same length.")

    fig = plt.figure(figsize=(10.0, 4.2))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    if gno.size == 0:
        for ax in (ax1, ax2):
            ax.axis("off")
            ax.text(0.5, 0.5, "No violating groups detected.", ha="center", va="center")
    else:
        x = np.arange(gno.size, dtype=float)
        dx = 0.16

        ax1.scatter(x, corr_rel, s=38, c="#4C72B0", marker="o", zorder=3)
        ax1.plot(x, corr_rel, color="0.55", linewidth=0.9, zorder=1)
        ax1.axhline(0.0, color="black", linestyle="--", linewidth=1.0, zorder=0)
        ax1.set_xticks(x)
        ax1.set_xticklabels([str(v) for v in gno])
        ax1.set_xlabel("Violating energy group")
        ax1.set_ylabel("Relative correction size")
        ax1.set_title("(a) Relative correction size with respect to the full reconstruction.")
        corr_ok = corr_rel[np.isfinite(corr_rel)]
        if corr_ok.size > 0:
            ax1.set_ylim(*_linear_axis_limits(corr_ok, include_zero=True))

        for i in range(gno.size):
            if np.isfinite(e_full[i]) and np.isfinite(e_corr[i]):
                ax2.plot([x[i] - dx, x[i] + dx], [e_full[i], e_corr[i]], color="0.55", linewidth=0.9, zorder=1)
        ax2.scatter(x - dx, e_full, s=34, c="#C44E52", marker="o", label="Full reconstruction", zorder=3)
        ax2.scatter(x + dx, e_corr, s=34, c="#4C72B0", marker="o", label="Corrected reconstruction", zorder=3)
        ax2.set_xticks(x)
        ax2.set_xticklabels([str(v) for v in gno])
        ax2.set_xlabel("Violating energy group")
        ax2.set_ylabel(error_metric_label)
        ax2.set_title("(b) Effective-cross-section error before and after correction.")

        err_pos = np.concatenate([
            e_full[np.isfinite(e_full) & (e_full > 0.0)],
            e_corr[np.isfinite(e_corr) & (e_corr > 0.0)],
        ])
        if err_pos.size > 0:
            emin = float(np.min(err_pos))
            emax = float(np.max(err_pos))
            if emin == emax:
                ax2.set_ylim(emin / 1.4, emax * 1.4)
            else:
                ax2.set_ylim(10.0 ** np.floor(np.log10(emin)), 10.0 ** np.ceil(np.log10(emax)))
            ax2.set_yscale("log")
            ax2.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=10))
            ax2.yaxis.set_major_formatter(FuncFormatter(_decimal_log_formatter))

        handles, labels = ax2.get_legend_handles_labels()
        if handles:
            ax2.legend(handles, labels, loc="best", frameon=False, fontsize=9)

    fig.savefig(outdir / f"{filename_base}.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / f"{filename_base}.pdf", bbox_inches="tight")
    plt.close(fig)

    
def save_paper_figure_captions(
    outdir: Path,
    *,
    figure1_filename_base: str = "figure1_full_reconstruction_admissibility",
    figure2_filename_base: str = "figure2_violating_groups_before_after",
    error_metric_name: str = "groupwise 95th-percentile relative error",
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cap_path = outdir / "paper_figure_captions.txt"
    with cap_path.open("w", encoding="utf-8") as f:
        f.write("Figure 1. Lower-bound violation magnitude of the full reconstruction across energy groups.\n")
        f.write(
            "Each point represents one energy group. The plotted quantity is the negative part of the groupwise minimum subgroup reaction-channel level under the direct full reconstruction, that is, the lower-bound violation magnitude. "
            "Most groups remain at zero, while only a few groups show a positive violation magnitude and therefore trigger the correction step. "
            "This figure diagnoses admissibility loss rather than observable error.\n\n"
        )
        f.write("Figure 2. Before-and-after comparison for the violating groups only.\n")
        f.write(
            "(a) Relative correction size with respect to the full reconstruction. The correction is activated only for groups where the direct full reconstruction violates admissibility. "
            "Panel (a) reports the size of the admissibility-restoring modification relative to the full reconstruction, rather than the corrected minimum subgroup level itself.\n"
        )
        f.write(
            f"(b) Effective-cross-section error before and after correction. Panel (b) shows the corresponding change in {error_metric_name}. "
            "The correction is designed to restore admissibility under the retained hard constraints, rather than to minimize observable error directly. "
            "It restores admissibility and may alter the error.\n\n"
        )
        f.write("Files:\n")
        f.write(f"- {figure1_filename_base}.png / .pdf\n")
        f.write(f"- {figure2_filename_base}.png / .pdf\n")

def save_violating_groups_diagnostics_txt(
    outdir: Path,
    violating_group_nos: np.ndarray,
    min_sigma_x_full: np.ndarray,
    min_sigma_x_corrected: np.ndarray,
    err_full: np.ndarray,
    err_corrected: np.ndarray,
    correction_norm: np.ndarray,
    *,
    total_subgroup_levels_by_group: dict[int, np.ndarray] | None = None,
    subgroup_probabilities_by_group: dict[int, np.ndarray] | None = None,
    reaction_channel_levels_full_by_group: dict[int, np.ndarray] | None = None,
    reaction_channel_levels_corrected_by_group: dict[int, np.ndarray] | None = None,
    filename: str = "violating_groups_diagnostics.txt",
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gno = np.asarray(violating_group_nos, dtype=int).ravel()
    y_full = np.asarray(min_sigma_x_full, dtype=float).ravel()
    y_corr = np.asarray(min_sigma_x_corrected, dtype=float).ravel()
    e_full = np.asarray(err_full, dtype=float).ravel()
    e_corr = np.asarray(err_corrected, dtype=float).ravel()
    cnorm = np.asarray(correction_norm, dtype=float).ravel()

    if not (gno.size == y_full.size == y_corr.size == e_full.size == e_corr.size == cnorm.size):
        raise ValueError("Violating-group diagnostic arrays must all have the same length.")

    total_subgroup_levels_by_group = dict(total_subgroup_levels_by_group or {})
    subgroup_probabilities_by_group = dict(subgroup_probabilities_by_group or {})
    reaction_channel_levels_full_by_group = dict(reaction_channel_levels_full_by_group or {})
    reaction_channel_levels_corrected_by_group = dict(reaction_channel_levels_corrected_by_group or {})

    def _fmt_value(val) -> str:
        vv = np.asarray(val)
        if np.iscomplexobj(vv):
            vr = float(np.real(val))
            vi = float(np.imag(val))
            if abs(vi) <= 1e-14:
                return f"{vr:.16e}"
            sign = "+" if vi >= 0.0 else "-"
            return f"{vr:.16e}{sign}{abs(vi):.16e}j"
        return f"{float(val):.16e}"

    def _write_named_vector(fh, name: str, group_index: int, values: np.ndarray | None) -> None:
        if values is None:
            return
        arr = np.asarray(values).ravel()
        fh.write(f"{name}\n")
        line = f"{int(group_index):12d}"
        for v in arr:
            line += f"  {_fmt_value(v)}"
        fh.write(line + "\n")

    path = outdir / filename
    with path.open("w", encoding="utf-8") as f:
        f.write("# Violating-group diagnostics for admissibility-restoring correction\n")
        f.write("# Scalar summary\n")
        f.write("# group_index  min_subgroup_level_before_correction  min_subgroup_level_after_correction  effective_xs_error_before_correction  effective_xs_error_after_correction  correction_norm\n")
        for i in range(gno.size):
            f.write(
                f"{int(gno[i]):12d}  {y_full[i]:.16e}  {y_corr[i]:.16e}  {e_full[i]:.16e}  {e_corr[i]:.16e}  {cnorm[i]:.16e}\n"
            )

        has_detailed = (
            len(total_subgroup_levels_by_group) > 0
            or len(subgroup_probabilities_by_group) > 0
            or len(reaction_channel_levels_full_by_group) > 0
            or len(reaction_channel_levels_corrected_by_group) > 0
        )

        if has_detailed:
            f.write("\n# Detailed subgroup data for violating groups\n")
            f.write("# For each variable, the first line is the variable name and the next line gives: group_index followed by the N subgroup values.\n")

            for g in gno:
                gg = int(g)
                f.write(f"\n# ---- group {gg} ----\n")
                _write_named_vector(f, "total_subgroup_levels", gg, total_subgroup_levels_by_group.get(gg))
                _write_named_vector(f, "subgroup_probabilities", gg, subgroup_probabilities_by_group.get(gg))
                _write_named_vector(f, "reaction_channel_levels_full_matching", gg, reaction_channel_levels_full_by_group.get(gg))
                _write_named_vector(f, "reaction_channel_levels_after_optimization", gg, reaction_channel_levels_corrected_by_group.get(gg))

def save_q95_by_group_txt(
    outdir: Path,
    group_nos: np.ndarray,
    q95_err_rel: np.ndarray,
    *,
    filename: str,
    title: str,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gno = np.asarray(group_nos, dtype=int).ravel()
    q95_rel = np.asarray(q95_err_rel, dtype=float).ravel()

    if gno.size != q95_rel.size:
        raise ValueError("group_nos and q95_err_rel must have same length.")

    txt_path = outdir / filename
    with txt_path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n")
        f.write("# Columns:\n")
        f.write("# GroupNo  q95_abs_relerr\n")
        for g, v_rel in zip(gno, q95_rel):
            if np.isfinite(v_rel):
                f.write(f"{g:6d}  {v_rel:.17e}\n")
            else:
                f.write(f"{g:6d}  nan\n")

def save_orth_loss_plot(
    outdir: Path,
    group_nos: np.ndarray,
    orth_off_dict: dict[str, np.ndarray],
    *,
    Nmax: int,
    ES: int,
    realization_mode: str,
    sigma_x_method: str,
    filename: str = "orth_loss_by_group.png",
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gno = np.asarray(group_nos, dtype=int).ravel()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    any_ok = False
    for mode, arr in orth_off_dict.items():
        y = np.asarray(arr, dtype=float).ravel()
        if y.size != gno.size:
            continue
        ok = np.isfinite(y) & (y > 0)
        if np.any(ok):
            any_ok = True
            ax.plot(gno[ok], y[ok], linestyle="None", marker="o", markersize=3, label=mode)

    if not any_ok:
        print(f"[OrthPlot] No positive finite orth_off values for {filename}. Skip saving.")
        plt.close(fig)
        return

    ax.set_xlabel(r"Energy-group index $g$ (high $\rightarrow$ low)")
    ax.set_ylabel(r"$\max_{i\neq j}\,|q_i^{T} q_j|$")
    ax.set_title(
        f"Loss of orthogonality in Lanczos basis vectors  "
        f"(Case C, $N_g\\leq {Nmax}$; ES={ES}; "
        f"realization={realization_mode}; $\\sigma_x$ method={sigma_x_method})"
    )
    ax.set_yscale("log")
    ax.legend(loc="best", fontsize=9)

    fig.savefig(outdir / filename, dpi=260, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# Adaptive N rule (Case C) with HARD M/2 constraint
# ============================================================
def effective_support_neff(w_norm: np.ndarray) -> float:
    """
    Neff = 1 / sum_j w_j^2  (weights must be nonnegative and sum to 1)
    """
    w = np.asarray(w_norm, dtype=float).ravel()
    m = np.isfinite(w) & (w >= 0.0)
    w = w[m]
    s = float(w.sum())
    if s <= 0.0:
        return float("nan")
    w = w / s
    return float(1.0 / np.sum(w * w))

def choose_adaptive_N_case_c_neff(
    *,
    M: int,
    Nmax: int,
    w_norm: np.ndarray,
    c_eff: float = 0.9,
    N_min: int = 2,
) -> tuple[int, float, int, int]:
    """
    Return:
      N_g, Neff, N_cap_M, N_cap_eff

    HARD: N <= floor(M/2)
    SOFT: N <= floor(c_eff * Neff)
    """
    if M < 2:
        return 0, float("nan"), 0, 0

    N_cap_M = int(M // 2)
    if N_cap_M < 1:
        return 0, float("nan"), N_cap_M, 0

    Neff = effective_support_neff(w_norm)
    if np.isfinite(Neff):
        N_cap_eff = int(np.floor(c_eff * Neff))
    else:
        N_cap_eff = N_cap_M

    N_g = min(int(Nmax), N_cap_M, N_cap_eff)
    N_g = max(int(N_min), N_g)
    N_g = min(N_g, int(Nmax), N_cap_M)

    return int(N_g), float(Neff), int(N_cap_M), int(N_cap_eff)

def normalize_input_cases(input_cases, base_dir: Path) -> list[dict]:
    """
    Normalize user-provided input cases.

    Supported item forms:
      1) {"total": "...", "partial": "..."}
      2) {"name": "...", "total": "...", "partial": "..."}
      3) ("total.txt", "partial.txt")
      4) ["total.txt", "partial.txt"]

    Returns a list of dicts:
      {
        "name": ...,
        "total": Path(...),
        "partial": Path(...),
        "input_tag": partial_file_stem
      }
    """
    out = []
    base_dir = Path(base_dir)

    for i, item in enumerate(input_cases, start=1):
        case_name = ""

        if isinstance(item, dict):
            if ("total" not in item) or ("partial" not in item):
                raise ValueError(
                    f"INPUT_CASES[{i}] dict must contain keys 'total' and 'partial'."
                )
            file_total = Path(item["total"])
            file_partial = Path(item["partial"])
            case_name = str(item.get("name", "")).strip()

        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            file_total = Path(item[0])
            file_partial = Path(item[1])

        else:
            raise ValueError(
                f"Unsupported INPUT_CASES[{i}] format. "
                f"Use dict with keys total/partial, or a 2-item tuple/list."
            )

        if not file_total.is_absolute():
            file_total = base_dir / file_total
        if not file_partial.is_absolute():
            file_partial = base_dir / file_partial

        input_tag = file_partial.stem
        if not case_name:
            case_name = input_tag

        out.append({
            "name": case_name,
            "total": file_total,
            "partial": file_partial,
            "input_tag": input_tag,
        })

    return out

# ============================================================
# Single-job runner (must be TOP-LEVEL for Windows multiprocessing)
# ============================================================
def run_single_job(
    *,
    file_total: Path,
    file_partial: Path,
    case_name: str,
    input_tag: str,
    N_MAX_current: int,
    config: dict,
) -> None:
    WARN_LOG.clear()

    HERE = Path(config["HERE"])
    SCRIPT_STEM = str(config["SCRIPT_STEM"])

    ES = int(config["ES"])
    N_DENSE_OVERRIDE = config["N_DENSE_OVERRIDE"]
    REALIZATION_MODE = str(config["REALIZATION_MODE"])
    SIGMA_X_METHOD = str(config["SIGMA_X_METHOD"])
    QP_RETENTION_COUNT = int(config["QP_RETENTION_COUNT"])
    if QP_RETENTION_COUNT == 1:
        QP_RETAINED_MODE = "zero_only"
    elif QP_RETENTION_COUNT == 2:
        QP_RETAINED_MODE = "minus1_and_0"
    else:
        raise ValueError("QP_RETENTION_COUNT must be 1 or 2.")
    QP_OBJECTIVE_WEIGHT_MODE = str(config["QP_OBJECTIVE_WEIGHT_MODE"])
    QP_GAMMA_MODE = str(config["QP_GAMMA_MODE"])
    QP_RELAX_GAMMA_TO_SEED = bool(config["QP_RELAX_GAMMA_TO_SEED"])
    QP_SOLVER = str(config["QP_SOLVER"])
    QP_MAXITER = int(config["QP_MAXITER"])
    QP_GTOL = float(config["QP_GTOL"])
    QP_XTOL = float(config["QP_XTOL"])
    QP_BARRIER_TOL = float(config["QP_BARRIER_TOL"])
    QP_REAL_IMAG_TOL = float(config["QP_REAL_IMAG_TOL"])
    GL_NQ = int(config["GL_NQ"])
    M_MIN = int(config["M_MIN"])
    SIGMA0_LOG10_MIN = float(config["SIGMA0_LOG10_MIN"])
    SIGMA0_LOG10_MAX = float(config["SIGMA0_LOG10_MAX"])
    SIGMA0_N = int(config["SIGMA0_N"])
    SIGMA0_TRUNC_LOG10_MAX = float(config["SIGMA0_TRUNC_LOG10_MAX"])
    EXTRA_PLOT_SUBFOLDER = str(config["EXTRA_PLOT_SUBFOLDER"])
    MOMENT_FOLDER_NAME = str(config["MOMENT_FOLDER_NAME"])
    REF_DENSIFY_GAUSS_N = config["REF_DENSIFY_GAUSS_N"]
    Z_AFFINE_NORMALIZE = bool(config["Z_AFFINE_NORMALIZE"])
    Z_NORM_METHOD = str(config["Z_NORM_METHOD"])
    RUN_MODES = tuple(config["RUN_MODES"])
    OUTPUT_MODE = str(config["OUTPUT_MODE"])
    DEBUG_GROUPS = config["DEBUG_GROUPS"]
    DEBUG_GROUPS = set(DEBUG_GROUPS) if DEBUG_GROUPS is not None else None
    DEBUG_TAIL_N = int(config["DEBUG_TAIL_N"])
    SEL_REORTH_TOL = float(config["SEL_REORTH_TOL"])
    SEL_CHECK_EVERY = int(config["SEL_CHECK_EVERY"])
    FULL_REORTH_PASSES = int(config["FULL_REORTH_PASSES"])
    SKIP_FULL_IF_MAXERR_LEQ = config["SKIP_FULL_IF_MAXERR_LEQ"]
    SELECTIVE_LOCK_MIN_K = int(config["SELECTIVE_LOCK_MIN_K"])
    SELECTIVE_LOCK_BETA_REL_TRIGGER = float(config["SELECTIVE_LOCK_BETA_REL_TRIGGER"])
    SELECTIVE_LOCK_FORCE_EVERY = int(config["SELECTIVE_LOCK_FORCE_EVERY"])

    file_edges_es1 = Path(config["file_edges_es1"])
    file_edges_es2 = Path(config["file_edges_es2"])

    N_MAX = int(N_MAX_current)
    N_DENSE = N_MAX if (N_DENSE_OVERRIDE is None) else int(N_DENSE_OVERRIDE)

    if N_MAX < 1:
        raise ValueError("N_MAX must be >= 1.")
    if N_DENSE < 2:
        raise ValueError("N_DENSE must be >= 2.")

    labels = get_realization_labels(REALIZATION_MODE)
    REF_LABEL = labels["ref_label"]
    DIRECT_LABEL = labels["direct_label"]
    DISCRETE_LABEL = labels["discrete_label"]

    DENSIFY_GAUSS_N = int(min(N_DENSE, N_MAX))

    if isinstance(REF_DENSIFY_GAUSS_N, str):
        tag = REF_DENSIFY_GAUSS_N.strip().lower()
        if tag == "auto":
            REF_DENSIFY_USE = DENSIFY_GAUSS_N if REALIZATION_MODE == "trap" else None
        elif tag == "none":
            REF_DENSIFY_USE = None
        else:
            REF_DENSIFY_USE = int(tag)
    else:
        REF_DENSIFY_USE = REF_DENSIFY_GAUSS_N

    file_edges = file_edges_es1 if ES == 1 else file_edges_es2

    XS_t = read_cross_sections(file_total)
    XS_x = read_cross_sections(file_partial)
    if XS_t.size == 0 or XS_x.size == 0:
        raise RuntimeError(
            f"Empty XS data. Check input files:\n"
            f"  total   = {file_total}\n"
            f"  partial = {file_partial}"
        )

    edges_asc_full = read_energy_structure_1col(file_edges)

    if ES == 2:
        edges_asc = crop_edges_to_overlap_with_chiba(edges_asc_full)
        print(
            f"[EnergyStructure] ES=2 cropped: full edges={edges_asc_full.size}, "
            f"cropped edges={edges_asc.size}"
        )
        print(
            f"[EnergyStructure] cropped range: "
            f"[{edges_asc[0]:.6g}, {edges_asc[-1]:.6g}]"
        )
    else:
        edges_asc = edges_asc_full
        print(f"[EnergyStructure] ES=1 full: edges={edges_asc.size}")
        print(
            f"[EnergyStructure] full range: "
            f"[{edges_asc[0]:.6g}, {edges_asc[-1]:.6g}]"
        )

    a = -1.0
    sigma0_grid = np.logspace(
        SIGMA0_LOG10_MIN,
        SIGMA0_LOG10_MAX,
        int(SIGMA0_N),
    )

    sigma0_trunc_max = 10.0 ** float(SIGMA0_TRUNC_LOG10_MAX)
    mask_trunc = (sigma0_grid <= sigma0_trunc_max)
    if not np.any(mask_trunc):
        mask_trunc = np.zeros_like(sigma0_grid, dtype=bool)
        mask_trunc[0] = True

    outdir_name = build_outdir_name(
        script_stem=SCRIPT_STEM,
        input_tag=input_tag,
        ES=ES,
        N_MAX=N_MAX,
        N_DENSE=N_DENSE,
        M_MIN=M_MIN,
        REALIZATION_MODE=REALIZATION_MODE,
        SIGMA_X_METHOD=SIGMA_X_METHOD,
        Z_AFFINE_NORMALIZE=Z_AFFINE_NORMALIZE,
        Z_NORM_METHOD=Z_NORM_METHOD,
        GL_NQ=GL_NQ,
        QP_RETENTION_COUNT=QP_RETENTION_COUNT,
    )
    outdir = HERE / outdir_name
    outdir.mkdir(parents=True, exist_ok=True)

    extra_dir = outdir / EXTRA_PLOT_SUBFOLDER
    extra_dir.mkdir(parents=True, exist_ok=True)

    moment_dir = outdir / MOMENT_FOLDER_NAME
    moment_dir.mkdir(parents=True, exist_ok=True)

    UNCOMPRESSED_FOLDER_NAME = (
        f"{_safe_stem(SCRIPT_STEM)}_uncompressed_error_plots_"
        f"ES{ES}_{REALIZATION_MODE}_sx{_safe_stem(SIGMA_X_METHOD)}"
    )
    COMPRESSION_FOLDER_NAME = (
        f"{_safe_stem(SCRIPT_STEM)}_compression_error_plots_"
        f"ES{ES}_{REALIZATION_MODE}_sx{_safe_stem(SIGMA_X_METHOD)}_mode{_safe_stem(OUTPUT_MODE)}"
    )

    uncomp_dir = outdir / UNCOMPRESSED_FOLDER_NAME
    uncomp_dir.mkdir(parents=True, exist_ok=True)

    comp_dir = outdir / COMPRESSION_FOLDER_NAME
    comp_dir.mkdir(parents=True, exist_ok=True)

    G = edges_asc.size - 1

    modes = list(RUN_MODES)
    max_err_rel = {m: np.full(G, np.nan, dtype=float) for m in modes}
    q95_err_rel = {m: np.full(G, np.nan, dtype=float) for m in modes}
    q95_err_rel_trunc = {m: np.full(G, np.nan, dtype=float) for m in modes}

    time_sec = {m: np.full(G, np.nan, dtype=float) for m in modes}
    orth_inf = {m: np.full(G, np.nan, dtype=float) for m in modes}
    orth_off = {m: np.full(G, np.nan, dtype=float) for m in modes}
    tri_res = {m: np.full(G, np.nan, dtype=float) for m in modes}

    flag_complex = {m: np.zeros(G, dtype=bool) for m in modes}
    flag_negative = {m: np.zeros(G, dtype=bool) for m in modes}
    min_re_sigma_t = {m: np.full(G, np.nan, dtype=float) for m in modes}
    min_re_p = {m: np.full(G, np.nan, dtype=float) for m in modes}
    min_re_sigma_x = {m: np.full(G, np.nan, dtype=float) for m in modes}

    mode_success = {m: np.zeros(G, dtype=bool) for m in modes}

    flag_sigma_t_complex = {m: np.zeros(G, dtype=bool) for m in modes}
    flag_sigma_t_negative = {m: np.zeros(G, dtype=bool) for m in modes}

    flag_p_complex = {m: np.zeros(G, dtype=bool) for m in modes}
    flag_p_negative = {m: np.zeros(G, dtype=bool) for m in modes}

    flag_eff_complex = {m: np.zeros(G, dtype=bool) for m in modes}
    flag_eff_negative = {m: np.zeros(G, dtype=bool) for m in modes}
    flag_sigma_x_constraint_switch = {m: np.zeros(G, dtype=bool) for m in modes}

    q95_err_rel_uncompressed = np.full(G, np.nan, dtype=float)
    q95_err_rel_compression = np.full(G, np.nan, dtype=float)

    full_recon_min_sigma_x = np.full(G, np.nan, dtype=float)
    full_recon_q95_err_rel = np.full(G, np.nan, dtype=float)
    full_recon_max_err_rel = np.full(G, np.nan, dtype=float)
    correction_norm_abs = np.full(G, np.nan, dtype=float)
    correction_norm_rel = np.full(G, np.nan, dtype=float)

    violating_total_subgroup_levels: dict[int, np.ndarray] = {}
    violating_subgroup_probabilities: dict[int, np.ndarray] = {}
    violating_reaction_channel_levels_full: dict[int, np.ndarray] = {}
    violating_reaction_channel_levels_corrected: dict[int, np.ndarray] = {}

    moment_max_relerr = np.full(G, np.nan, dtype=float)
    moment_q95_relerr = np.full(G, np.nan, dtype=float)

    print("\n=== SETTINGS ===")
    print(f"Case name = {case_name}")
    print(f"Input total   = {file_total}")
    print(f"Input partial = {file_partial}")
    print(f"Input tag     = {input_tag}")
    print(f"Script = {SCRIPT_STEM}")
    print(f"ES = {ES}")
    print(f"Case C: a = {a:g}, b = 1/(N-1) with adaptive N_g (N_g==1 uses b=1)")
    print(f"REALIZATION_MODE = {REALIZATION_MODE}  -> REF={REF_LABEL}; discrete={DISCRETE_LABEL}")
    print(f"SIGMA_X_METHOD = {SIGMA_X_METHOD}")
    print(f"N_MAX = {N_MAX}, N_DENSE = {N_DENSE}, M_MIN = {M_MIN}  (HARD: N_g<=floor(M/2))")
    print(f"DENSIFY_GAUSS_N = {DENSIFY_GAUSS_N}")
    print(f"GL_NQ = {GL_NQ}  (used only when REALIZATION_MODE='ana')")
    print(f"REF_DENSIFY_GAUSS_N = {REF_DENSIFY_GAUSS_N}  -> actual use = {REF_DENSIFY_USE}")
    print(f"Z_AFFINE_NORMALIZE = {Z_AFFINE_NORMALIZE} (method={Z_NORM_METHOD})")
    print(f"RUN_MODES = {RUN_MODES}, OUTPUT_MODE = {OUTPUT_MODE}")
    print(
        f"Selective (classical Parlett--Scott style): "
        f"lock tol={SEL_REORTH_TOL:g}, check_every={SEL_CHECK_EVERY}, "
        f"full_passes={FULL_REORTH_PASSES}"
    )
    print(
        f"Selective lock gating: "
        f"lock_min_k={SELECTIVE_LOCK_MIN_K}, "
        f"beta_rel_trigger={SELECTIVE_LOCK_BETA_REL_TRIGGER:g}, "
        f"force_every={SELECTIVE_LOCK_FORCE_EVERY}"
    )
    if SKIP_FULL_IF_MAXERR_LEQ is not None:
        print(f"Skip FULL if maxerr(output mode) <= {SKIP_FULL_IF_MAXERR_LEQ}")
    print(f"Groups total G = {G}  (Group index g = 1..{G}, high->low)")
    print("sigma0 grid:", f"[{sigma0_grid[0]:.3e} ... {sigma0_grid[-1]:.3e}], n={sigma0_grid.size}")
    print(
        f"Trunc for q95(|e|): sigma0 <= {sigma0_trunc_max:.3e} "
        f"(log10_max={SIGMA0_TRUNC_LOG10_MAX:g})"
    )
    print("Output folder:", outdir)
    print("Extra plots folder:", extra_dir)
    print("Moment diagnostics folder:", moment_dir)
    print("UNCOMPRESSED plots folder (under outdir):", uncomp_dir)
    print("COMPRESSION plots folder (under outdir):", comp_dir)

    summary_path = outdir / "summary.txt"
    complex_log_path = outdir / "complex_warnings.txt"
    ref_txt_path = outdir / "ref_values_by_group.txt"
    reorth_txt_path = outdir / "reorth_mode_results.txt"
    moment_txt_path = moment_dir / "moment_errors.txt"
    reorth_timing_path = outdir / "reorth_timing_details.txt"

    with summary_path.open("w", encoding="utf-8") as sf:
        sf.write(
            "GroupNo  g_low  E_high  E_low  width  M_nodes  Neff  N_g  b_g  cover_frac  skip_frac  "
            "realization  sigma_x_method  "
            "m0=<sigma^a>  z_shift  z_scale  "
            "mode  elapsed_s  "
            "cond_partialM  rcond_partialM  sum(p)  minRe(p)  maxRe(p)  "
            "max_abs_err_rel  q95_abs_err_rel  "
            "orth_inf  orth_off  tri_res  "
            "FLAG_COMPLEX  FLAG_NEG  FLAG_SIGMA_X_SWITCH  minRe_sigma_t  minRe_p  minRe_sigma_x  "
            "maxIm_sigma_t  maxIm_p  maxIm_sigma_x  maxIm_ubar  maxIm_PT  eff_x_neg\n"
        )

    with ref_txt_path.open("w", encoding="utf-8") as rf:
        rf.write("# Reference effective cross section values\n")
        rf.write(f"# realization = {REALIZATION_MODE}\n")
        rf.write("# Columns:\n")
        rf.write("# GroupNo  g_low  E_high  E_low  sigma0  ref\n")

    with reorth_txt_path.open("w", encoding="utf-8") as rf:
        rf.write("# Per-group results for all reorthogonalization modes\n")
        rf.write(f"# realization = {REALIZATION_MODE}\n")
        rf.write(f"# sigma_x_method = {SIGMA_X_METHOD}\n")
        rf.write("# Columns:\n")
        rf.write(
            "# GroupNo  g_low  E_high  E_low  N_g  mode  status  "
            "elapsed_s  orth_inf  orth_off  tri_res  "
            "max_abs_err_rel  q95_abs_err_rel  q95_trunc_abs_err_rel  "
            "FLAG_COMPLEX  FLAG_NEG  FLAG_SIGMA_X_SWITCH\n"
        )
        
    with reorth_timing_path.open("w", encoding="utf-8") as tf:
        tf.write("# Internal Lanczos / reorth timing breakdown by group and mode\n")
        tf.write(f"# realization = {REALIZATION_MODE}\n")
        tf.write(f"# sigma_x_method = {SIGMA_X_METHOD}\n")
        tf.write("# Columns:\n")
        tf.write(
            "# GroupNo  g_low  E_high  E_low  N_g  mode  status  elapsed_s  "
            "lanczos_internal_s  "
            "n_lock_probe  n_lock_eigh  n_lock_added  n_lock_skip_smallk  n_lock_skip_beta  "
            "n_semiorth_trigger  n_q_fullreorth  "
            "time_lock_probe_norm  time_lock_eigh  time_lock_accept  time_lock_apply  "
            "time_semiorth_measure  time_q_fullreorth\n"
        )

    with moment_txt_path.open("w", encoding="utf-8") as mf:
        mf.write("# Detailed moment-reproduction diagnostics for OUTPUT_MODE result\n")
        mf.write(f"# realization = {REALIZATION_MODE}\n")
        mf.write(f"# sigma_x_method = {SIGMA_X_METHOD}\n")
        mf.write("# Compared moments are for the normalized working measure in z_work:\n")
        mf.write("#   target_k = sum_j w_norm[j] * z_work[j]^k\n")
        mf.write("#   gauss_k  = sum_i p_norm[i] * lam_work[i]^k\n")
        mf.write("# for k = 0,1,...,2N_g-1.\n")
        mf.write("# Columns:\n")
        mf.write("# GroupNo  g_low  E_high  E_low  N_g  mode  k  target_moment  gauss_moment  abs_err  rel_err\n")

    for idx_hi2lo in range(G):
        group_no = idx_hi2lo + 1
        g_low = (G - 1) - idx_hi2lo
        e_low = float(edges_asc[g_low])
        e_high = float(edges_asc[g_low + 1])
        width = e_high - e_low

        ref = nra_effective_x_reference_dispatch(
            realization_mode=REALIZATION_MODE,
            XS_t=XS_t,
            XS_x=XS_x,
            edges_asc=edges_asc,
            g_low=g_low,
            sigma0_grid=sigma0_grid,
            densify_gauss_N=REF_DENSIFY_USE,
        )
        append_ref_curve_to_txt(
            ref_txt_path,
            group_no=group_no,
            g_low=g_low,
            e_high=e_high,
            e_low=e_low,
            sigma0_grid=sigma0_grid,
            ref_curve=ref,
        )

        rt_node, rx_node, w_base, width, cover_dx, skip_dx, skip_seg = build_group_discrete_samples(
            realization_mode=REALIZATION_MODE,
            XS_t=XS_t,
            XS_x=XS_x,
            edges_asc=edges_asc,
            g_low=g_low,
            densify_gauss_N=DENSIFY_GAUSS_N,
            gl_nq=GL_NQ,
        )

        M = int(rt_node.size)
        cover_frac = (cover_dx / width) if (np.isfinite(width) and width > 0) else float("nan")
        skip_frac = (skip_dx / width) if (np.isfinite(width) and width > 0) else float("nan")

        if M < M_MIN:
            print(f"\n[Group {group_no}] skipped: too few discrete atoms M={M} (need M>={M_MIN}).")
            continue
        if np.any(rt_node <= 0) or not np.all(np.isfinite(rt_node)):
            print(f"\n[Group {group_no}] skipped: non-positive/invalid rt_node present.")
            continue

        direct_curve = nra_effective_x_direct_discrete(rt_node, rx_node, w_base, sigma0_grid)
        _max_u, _q95_u = compute_abs_error_stats_rel(ref, direct_curve)
        q95_err_rel_uncompressed[idx_hi2lo] = float(_q95_u)

        save_group_error_plot_rel_scatter_uncompressed(
            outdir=uncomp_dir,
            group_no=group_no,
            e_high=e_high,
            e_low=e_low,
            sigma0_grid=sigma0_grid,
            ref_curve=ref,
            direct_curve=direct_curve,
            M_atoms=M,
            realization_mode=REALIZATION_MODE,
            direct_label=DIRECT_LABEL,
            ref_label=REF_LABEL,
            gl_nq=GL_NQ,
        )

        pi = w_base * np.power(rt_node, a)
        m0 = float(np.sum(pi))
        if not np.isfinite(m0) or m0 <= 0.0:
            print(f"\n[Group {group_no}] skipped: invalid m0={m0}")
            continue
        w_norm = pi / m0

        N_g, Neff, N_cap_M, N_cap_eff = choose_adaptive_N_case_c_neff(
            M=M, Nmax=N_MAX, w_norm=w_norm, c_eff=0.9, N_min=2
        )
        if N_g < 2:
            print(f"\n[Group {group_no}] skipped: no valid N_g>=2 (M={M}, Neff={Neff})")
            continue

        b_g = 1.0 / (N_g - 1)

        z = np.power(rt_node, b_g)

        z_shift = 0.0
        z_scale = 1.0
        z_work = z.copy()
        if Z_AFFINE_NORMALIZE:
            z_work, z_shift, z_scale = affine_normalize_z(z, w_norm, method=Z_NORM_METHOD)

        if (DEBUG_GROUPS is None) or (group_no in DEBUG_GROUPS):
            print("\n" + "="*70)
            print(f"[DEBUG][Group {group_no}] sigma0_max = {sigma0_grid[-1]:.6e}")
            print(f"[DEBUG][Group {group_no}] REF(s0_max) = {ref[-1]:.17e}")
            print(f"[DEBUG][Group {group_no}] REF tail (last {DEBUG_TAIL_N}):")
            for s0, r in zip(sigma0_grid[-DEBUG_TAIL_N:], ref[-DEBUG_TAIL_N:]):
                print(f"  s0={s0:.6e}  REF={r:.17e}")

        def run_once(mode: str):
            t0 = perf_counter()

            alpha, beta, Qk, profile = lanczos_tridiag_from_diag(
                z=z_work,
                w_norm=w_norm,
                N=N_g,
                reorth_mode=mode,
                sel_tol=SEL_REORTH_TOL,
                sel_check_every=SEL_CHECK_EVERY,
                full_passes=FULL_REORTH_PASSES,
                return_profile=True,
                lock_check_min_k=SELECTIVE_LOCK_MIN_K,
                lock_beta_rel_trigger=SELECTIVE_LOCK_BETA_REL_TRIGGER,
                lock_force_every=SELECTIVE_LOCK_FORCE_EVERY,
            )

            oi, oo, tr = diagnostics_lanczos(z_work, Qk, alpha, beta)

            lam_work, Qeig, p_norm = golub_welsch(alpha, beta)

            lam_z = (z_scale * lam_work + z_shift) if Z_AFFINE_NORMALIZE else lam_work
            lam_z_c = np.asarray(lam_z, dtype=np.complex128)
            sigma_t_nodes = np.power(lam_z_c, 1.0 / b_g)

            pi_nodes = m0 * p_norm
            p = pi_nodes / np.power(sigma_t_nodes, a)

            sigma_x_nodes, ubar, condM, rcondM, sigma_x_info = reconstruct_sigma_x_nodes(
                method=SIGMA_X_METHOD,
                Qk=Qk,
                Qeig=Qeig,
                w_norm=w_norm,
                rt_node=rt_node,
                rx_node=rx_node,
                sigma_t_nodes=sigma_t_nodes,
                p=p,
                p_norm=p_norm,
                m0=m0,
                w_base=w_base,
                a=a,
                qp_retained_mode=QP_RETAINED_MODE,
                qp_objective_weight_mode=QP_OBJECTIVE_WEIGHT_MODE,
                qp_gamma_mode=QP_GAMMA_MODE,
                qp_relax_gamma_to_seed=QP_RELAX_GAMMA_TO_SEED,
                qp_solver=QP_SOLVER,
                qp_maxiter=QP_MAXITER,
                qp_gtol=QP_GTOL,
                qp_xtol=QP_XTOL,
                qp_barrier_tol=QP_BARRIER_TOL,
                qp_real_imag_tol=QP_REAL_IMAG_TOL,
            )

            pt = nra_effective_x_probability_table_chiba_ubar(
                sigma_t_nodes, p, ubar, a, sigma0_grid
            )
            maxe, q95e = compute_abs_error_stats_rel(ref, pt)
            q95e_tr = compute_q95_abs_error_rel_trunc(ref, pt, mask_trunc)

            full_sigma_x = sigma_x_info.get("full_solution_sigma_x", None)
            full_pt = None
            full_maxe = float("nan")
            full_q95e = float("nan")
            if full_sigma_x is not None:
                full_sigma_x = np.asarray(full_sigma_x, dtype=float).ravel()
                if full_sigma_x.size == sigma_t_nodes.size:
                    ubar_full = p * full_sigma_x * np.power(sigma_t_nodes, a)
                    full_pt = nra_effective_x_probability_table_chiba_ubar(
                        sigma_t_nodes, p, ubar_full, a, sigma0_grid
                    )
                    full_maxe, full_q95e = compute_abs_error_stats_rel(ref, full_pt)

            elapsed = perf_counter() - t0
            return {
                "orth_inf": oi, "orth_off": oo, "tri_res": tr,
                "sigma_t_nodes": sigma_t_nodes, "p": p, "sigma_x_nodes": sigma_x_nodes,
                "ubar": ubar, "pt": pt,
                "condM": condM, "rcondM": rcondM,
                "maxerr_rel": maxe, "q95_rel": q95e, "q95_rel_trunc": q95e_tr,
                "elapsed": elapsed,
                "lam_work": lam_work,
                "p_norm": p_norm,
                "profile": profile,
                "sigma_x_info": sigma_x_info,
                "full_pt": full_pt,
                "full_maxerr_rel": full_maxe,
                "full_q95_rel": full_q95e,
            }
        results = {}
        for mode in modes:
            if mode == "full" and (SKIP_FULL_IF_MAXERR_LEQ is not None) and (OUTPUT_MODE in results):
                max_out = results[OUTPUT_MODE]["maxerr_rel"]
                if np.isfinite(max_out) and (max_out <= SKIP_FULL_IF_MAXERR_LEQ):
                    continue
            try:
                results[mode] = run_once(mode)
            except Exception as e:
                print(f"\n[Group {group_no}] mode={mode} FAILED: {e}")
                continue

        with reorth_txt_path.open("a", encoding="utf-8") as rf:
            for mode_name in modes:
                if mode_name in results:
                    resm = results[mode_name]

                    sigma_t_nodes_m = resm["sigma_t_nodes"]
                    p_m = resm["p"]
                    sigma_x_nodes_m = resm["sigma_x_nodes"]
                    ubar_m = resm["ubar"]
                    pt_m = resm["pt"]

                    mi_sigma_t_m = _max_abs_imag(sigma_t_nodes_m)
                    mi_p_m = _max_abs_imag(p_m)
                    mi_sigma_x_m = _max_abs_imag(sigma_x_nodes_m)
                    mi_ubar_m = _max_abs_imag(ubar_m)
                    mi_pt_m = _max_abs_imag(pt_m)
                    comp_flag_m = (
                        (mi_sigma_t_m > IMAG_TOL)
                        or (mi_p_m > IMAG_TOL)
                        or (mi_sigma_x_m > IMAG_TOL)
                        or (mi_ubar_m > IMAG_TOL)
                        or (mi_pt_m > IMAG_TOL)
                    )

                    min_sigma_m = _min_real(sigma_t_nodes_m)
                    min_p_m = _min_real(p_m)
                    min_sigma_x_m = _min_real(sigma_x_nodes_m)
                    neg_flag_m = (
                        (np.isfinite(min_sigma_m) and (min_sigma_m < NEG_TOL))
                        or (np.isfinite(min_p_m) and (min_p_m < NEG_TOL))
                        or (np.isfinite(min_sigma_x_m) and (min_sigma_x_m < NEG_TOL))
                    )

                    sigma_x_switch_m = bool(resm.get("sigma_x_info", {}).get("constraint_switch_due_to_violation", False))
                    rf.write(
                        f"{group_no:6d}  {g_low:5d}  {e_high:.8g}  {e_low:.8g}  {N_g:4d}  "
                        f"{mode_name:9s}  {'OK':6s}  "
                        f"{resm['elapsed']:.6g}  {resm['orth_inf']:.6g}  {resm['orth_off']:.6g}  {resm['tri_res']:.6g}  "
                        f"{resm['maxerr_rel']:.6g}  {resm['q95_rel']:.6g}  {resm['q95_rel_trunc']:.6g}  "
                        f"{int(comp_flag_m):11d}  {int(neg_flag_m):8d}  {int(sigma_x_switch_m):19d}\n"
                    )
                else:
                    rf.write(
                        f"{group_no:6d}  {g_low:5d}  {e_high:.8g}  {e_low:.8g}  {N_g:4d}  "
                        f"{mode_name:9s}  {'FAIL':6s}  "
                        f"nan  nan  nan  nan  nan  nan  nan  "
                        f"{-1:11d}  {-1:8d}  {-1:19d}\n"
                    )

        with reorth_timing_path.open("a", encoding="utf-8") as tf:
            for mode_name in modes:
                if mode_name in results:
                    resm = results[mode_name]
                    prof = resm.get("profile", {})

                    tf.write(
                        f"{group_no:6d}  {g_low:5d}  {e_high:.8g}  {e_low:.8g}  {N_g:4d}  "
                        f"{mode_name:9s}  {'OK':6s}  "
                        f"{resm['elapsed']:.6g}  "
                        f"{prof.get('time_total_internal', float('nan')):.6g}  "
                        f"{int(prof.get('n_lock_probe', 0)):6d}  "
                        f"{int(prof.get('n_lock_eigh', 0)):6d}  "
                        f"{int(prof.get('n_lock_added', 0)):6d}  "
                        f"{int(prof.get('n_lock_skip_smallk', 0)):6d}  "
                        f"{int(prof.get('n_lock_skip_beta', 0)):6d}  "
                        f"{int(prof.get('n_semiorth_trigger', 0)):6d}  "
                        f"{int(prof.get('n_q_fullreorth', 0)):6d}  "
                        f"{prof.get('time_lock_probe_norm', float('nan')):.6g}  "
                        f"{prof.get('time_lock_eigh', float('nan')):.6g}  "
                        f"{prof.get('time_lock_accept', float('nan')):.6g}  "
                        f"{prof.get('time_lock_apply', float('nan')):.6g}  "
                        f"{prof.get('time_semiorth_measure', float('nan')):.6g}  "
                        f"{prof.get('time_q_fullreorth', float('nan')):.6g}\n"
                    )
                else:
                    tf.write(
                        f"{group_no:6d}  {g_low:5d}  {e_high:.8g}  {e_low:.8g}  {N_g:4d}  "
                        f"{mode_name:9s}  {'FAIL':6s}  "
                        f"nan  nan  "
                        f"-1  -1  -1  -1  -1  -1  -1  "
                        f"nan  nan  nan  nan  nan  nan\n"
                    )
        if OUTPUT_MODE not in results:
            print(f"\n[Group {group_no}] FAILED: OUTPUT_MODE={OUTPUT_MODE} not available.")
            continue

        for mode, res in results.items():
            sigma_t_nodes = res["sigma_t_nodes"]
            p = res["p"]
            sigma_x_nodes = res["sigma_x_nodes"]
            ubar = res["ubar"]
            pt = res["pt"]

            sigma_t_complex_flag, sigma_t_neg_flag, min_sigma, mi_sigma_t = classify_complex_negative(
                sigma_t_nodes, imag_tol=IMAG_TOL, neg_tol=NEG_TOL
            )
            p_complex_flag, p_neg_flag, min_pv, mi_p = classify_complex_negative(
                p, imag_tol=IMAG_TOL, neg_tol=NEG_TOL
            )
            sigma_x_complex_flag, sigma_x_neg_flag, min_sigma_x_v, mi_sigma_x = classify_complex_negative(
                sigma_x_nodes, imag_tol=IMAG_TOL, neg_tol=NEG_TOL
            )
            ubar_complex_flag, _ubar_neg_flag_unused, _min_ubar_unused, mi_ubar = classify_complex_negative(
                ubar, imag_tol=IMAG_TOL, neg_tol=NEG_TOL
            )
            eff_complex_flag, eff_neg_flag, _min_eff_unused, mi_pt = classify_complex_negative(
                pt, imag_tol=IMAG_TOL, neg_tol=NEG_TOL
            )

            comp_flag = (
                sigma_t_complex_flag
                or p_complex_flag
                or sigma_x_complex_flag
                or ubar_complex_flag
                or eff_complex_flag
            )
            neg_flag = (
                sigma_t_neg_flag
                or p_neg_flag
                or sigma_x_neg_flag
            )

            mode_success[mode][idx_hi2lo] = True

            flag_sigma_t_complex[mode][idx_hi2lo] = bool(sigma_t_complex_flag)
            flag_sigma_t_negative[mode][idx_hi2lo] = bool(sigma_t_neg_flag)

            flag_p_complex[mode][idx_hi2lo] = bool(p_complex_flag)
            flag_p_negative[mode][idx_hi2lo] = bool(p_neg_flag)

            flag_eff_complex[mode][idx_hi2lo] = bool(eff_complex_flag)
            flag_eff_negative[mode][idx_hi2lo] = bool(eff_neg_flag)

            flag_complex[mode][idx_hi2lo] = bool(comp_flag)
            flag_negative[mode][idx_hi2lo] = bool(neg_flag)
            flag_sigma_x_constraint_switch[mode][idx_hi2lo] = bool(
                res.get("sigma_x_info", {}).get("constraint_switch_due_to_violation", False)
            )
            min_re_sigma_t[mode][idx_hi2lo] = float(min_sigma)
            min_re_p[mode][idx_hi2lo] = float(min_pv)
            min_re_sigma_x[mode][idx_hi2lo] = float(min_sigma_x_v)

            max_err_rel[mode][idx_hi2lo] = float(res["maxerr_rel"])
            q95_err_rel[mode][idx_hi2lo] = float(res["q95_rel"])
            q95_err_rel_trunc[mode][idx_hi2lo] = float(res["q95_rel_trunc"])

            time_sec[mode][idx_hi2lo] = float(res["elapsed"])
            orth_inf[mode][idx_hi2lo] = float(res["orth_inf"])
            orth_off[mode][idx_hi2lo] = float(res["orth_off"])
            tri_res[mode][idx_hi2lo] = float(res["tri_res"])

            sum_p = np.sum(p)
            min_p_print = float(np.min(np.real(p))) if np.iscomplexobj(p) else float(np.min(p))
            max_p_print = float(np.max(np.real(p))) if np.iscomplexobj(p) else float(np.max(p))

            with summary_path.open("a", encoding="utf-8") as sf:
                sf.write(
                    f"{group_no:6d}  {g_low:5d}  {e_high:.8g}  {e_low:.8g}  {width:.8g}  {M:d}  "
                    f"{Neff:.6g}  {N_g:d}  {b_g:.12g}  {cover_frac:.8g}  {skip_frac:.8g}  "
                    f"{REALIZATION_MODE:11s}  {SIGMA_X_METHOD:13s}  "
                    f"{m0:.12g}  {z_shift:.6g}  {z_scale:.6g}  "
                    f"{mode:9s}  {res['elapsed']:.6g}  "
                    f"{res['condM']:.6g}  {res['rcondM']:.6g}  {sum_p:.12g}  {min_p_print:.6g}  {max_p_print:.6g}  "
                    f"{res['maxerr_rel']:.6g}  {res['q95_rel']:.6g}  "
                    f"{res['orth_inf']:.6g}  {res['orth_off']:.6g}  {res['tri_res']:.6g}  "
                    f"{int(comp_flag):11d}  {int(neg_flag):8d}  "
                    f"{int(res.get('sigma_x_info', {}).get('constraint_switch_due_to_violation', False)):19d}  "
                    f"{min_sigma:.6g}  {min_pv:.6g}  {min_sigma_x_v:.6g}  "
                    f"{mi_sigma_t:.6g}  {mi_p:.6g}  {mi_sigma_x:.6g}  {mi_ubar:.6g}  {mi_pt:.6g}  {int(eff_neg_flag):9d}\n"
                )

        res_used = results[OUTPUT_MODE]
        sigma_t_nodes = res_used["sigma_t_nodes"]
        p = res_used["p"]
        sigma_x_nodes = res_used["sigma_x_nodes"]
        ubar = res_used["ubar"]
        pt = res_used["pt"]

        _max_c, _q95_c = compute_abs_incremental_error_stats_rel_refden(ref, pt, direct_curve)
        q95_err_rel_compression[idx_hi2lo] = float(_q95_c)

        full_recon_min_sigma_x[idx_hi2lo] = float(res_used.get("sigma_x_info", {}).get("full_solution_min", float("nan")))
        full_recon_q95_err_rel[idx_hi2lo] = float(res_used.get("full_q95_rel", float("nan")))
        full_recon_max_err_rel[idx_hi2lo] = float(res_used.get("full_maxerr_rel", float("nan")))

        full_sigma_x_used = res_used.get("sigma_x_info", {}).get("full_solution_sigma_x", None)
        corr_sigma_x_used = res_used.get("sigma_x_nodes", None)
        if full_sigma_x_used is not None and corr_sigma_x_used is not None:
            full_sigma_x_arr = np.asarray(full_sigma_x_used, dtype=float).ravel()
            corr_sigma_x_arr = np.asarray(corr_sigma_x_used, dtype=float).ravel()
            if full_sigma_x_arr.size == corr_sigma_x_arr.size and full_sigma_x_arr.size > 0:
                correction_norm_abs[idx_hi2lo] = float(np.linalg.norm(corr_sigma_x_arr - full_sigma_x_arr))
                correction_norm_rel[idx_hi2lo] = _relative_l2_correction_size(corr_sigma_x_arr, full_sigma_x_arr)

        if bool(res_used.get("sigma_x_info", {}).get("constraint_switch_due_to_violation", False)):
            violating_total_subgroup_levels[int(group_no)] = np.asarray(sigma_t_nodes).ravel().copy()
            violating_subgroup_probabilities[int(group_no)] = np.asarray(p).ravel().copy()
            if full_sigma_x_used is not None:
                violating_reaction_channel_levels_full[int(group_no)] = np.asarray(full_sigma_x_used).ravel().copy()
            if corr_sigma_x_used is not None:
                violating_reaction_channel_levels_corrected[int(group_no)] = np.asarray(corr_sigma_x_used).ravel().copy()

        save_group_error_plot_rel_scatter_compression(
            outdir=comp_dir,
            group_no=group_no,
            e_high=e_high,
            e_low=e_low,
            sigma0_grid=sigma0_grid,
            ref_curve=ref,
            direct_curve=direct_curve,
            pt_curve=pt,
            mode_tag=OUTPUT_MODE,
            direct_label=DIRECT_LABEL,
            ref_label=REF_LABEL,
        )

        k_arr, m_target, m_approx, m_relerr = compute_moment_errors(
            rt_node=rt_node,
            rx_node=rx_node,
            w_base=w_base,
            sigma_t_nodes=sigma_t_nodes,
            sigma_x_nodes=sigma_x_nodes,
            p=p,
            n_samples=100,
        )
        _mmax, _mq95 = save_group_moment_error_plot(
            outdir=moment_dir,
            group_no=group_no,
            e_high=e_high,
            e_low=e_low,
            N_g=N_g,
            mode_tag=OUTPUT_MODE,
            k_arr=k_arr,
            rel_err=m_relerr,
        )
        moment_max_relerr[idx_hi2lo] = float(_mmax)
        moment_q95_relerr[idx_hi2lo] = float(_mq95)

        append_moment_errors_to_txt(
            moment_txt_path,
            group_no=group_no,
            g_low=g_low,
            e_high=e_high,
            e_low=e_low,
            N_g=N_g,
            mode_tag=OUTPUT_MODE,
            k_arr=k_arr,
            target=m_target,
            approx=m_approx,
            rel_err=m_relerr,
        )

        err_tot = compute_error_rel(ref, pt)
        err_unc = compute_error_rel(ref, direct_curve)
        err_comp = compute_incremental_error_rel_refden(ref, pt, direct_curve)
        decomp_resid = err_tot - err_unc - err_comp
        decomp_resid_max = np.nanmax(np.abs(decomp_resid)) if np.any(np.isfinite(decomp_resid)) else np.nan

        if (DEBUG_GROUPS is None) or (group_no in DEBUG_GROUPS):
            pt_real = np.real(pt) if np.iscomplexobj(pt) else pt
            ref_s0max = float(ref[-1])
            pt_s0max = float(pt_real[-1])
            diff = pt_s0max - ref_s0max
            rel_err = diff / ref_s0max if np.isfinite(ref_s0max) and ref_s0max != 0.0 else np.nan

            print(f"[DEBUG][Group {group_no}] PT(s0_max)  = {pt_s0max:.17e}")
            print(f"[DEBUG][Group {group_no}] PT-REF     = {diff:+.17e}  (barn)")
            print(f"[DEBUG][Group {group_no}] rel_err    = {rel_err:+.6e}")

            print(f"[DEBUG][Group {group_no}] tail compare (last {DEBUG_TAIL_N}):")
            for s0, r, ppp in zip(sigma0_grid[-DEBUG_TAIL_N:], ref[-DEBUG_TAIL_N:], pt_real[-DEBUG_TAIL_N:]):
                d = float(ppp - r)
                e_rel = d / float(r) if np.isfinite(r) and r != 0.0 else np.nan
                print(f"  s0={s0:.6e}  PT-REF={d:+.3e}  err={e_rel:+.3e}")
            print(f"[DEBUG][Group {group_no}] DECOMP check: max|tot - unc - comp| = {decomp_resid_max:.3e}")
            print("="*70)

        warn_if_complex(sigma_t_nodes, name="sigma_t_nodes", group_no=group_no)
        warn_if_complex(p, name="p", group_no=group_no)
        warn_if_complex(sigma_x_nodes, name="sigma_x_nodes", group_no=group_no)
        warn_if_complex(ubar, name="ubar", group_no=group_no)
        warn_if_complex(pt, name="PT_eff_x", group_no=group_no)

        max_abs_err_rel = res_used["maxerr_rel"]
        q95_abs_err_rel = res_used["q95_rel"]

        save_group_error_plot_rel_scatter(
            outdir=outdir,
            group_no=group_no,
            e_high=e_high,
            e_low=e_low,
            sigma0_grid=sigma0_grid,
            ref_curve=ref,
            pt_curve=pt,
            N_g=N_g,
            b_g=b_g,
            sigma_x_method=SIGMA_X_METHOD,
            realization_mode=REALIZATION_MODE,
            tag=OUTPUT_MODE,
        )

        save_group_abs_error_plot_barn(
            outdir=extra_dir,
            group_no=group_no,
            e_high=e_high,
            e_low=e_low,
            sigma0_grid=sigma0_grid,
            ref_curve=ref,
            pt_curve=pt,
            realization_mode=REALIZATION_MODE,
            tag=OUTPUT_MODE,
        )

        print(f"\n[Group {group_no}] E=[{e_high:g}->{e_low:g}] width={width:.6g} (g_low={g_low})")
        print(f"  atoms M={M}, adaptive N_g={N_g}, b={b_g:g}, cover_frac={cover_frac:.6g}, skip_frac={skip_frac:.6g}")
        print(f"  realization={REALIZATION_MODE}, sigma_x_method={SIGMA_X_METHOD}")
        print(
            f"  sigma_x switch by admissibility violation = "
            f"{bool(res_used.get('sigma_x_info', {}).get('constraint_switch_due_to_violation', False))}"
        )
        print(
            f"  OUTPUT_MODE={OUTPUT_MODE}: max|e|={max_abs_err_rel:.3e}, q95(|e|)={q95_abs_err_rel:.3e}, "
            f"q95_trunc(|e|)={res_used['q95_rel_trunc']:.3e}, "
            f"orth_off={results[OUTPUT_MODE]['orth_off']:.3e}, tri_res={results[OUTPUT_MODE]['tri_res']:.3e}, "
            f"time={results[OUTPUT_MODE]['elapsed']:.3g}s"
        )
        print(f"  UNCOMPRESSED({DIRECT_LABEL}): q95(|e|)={_q95_u:.3e}")
        print(f"  COMPRESSION((PT-DIRECT)/REF): q95(|e|)={_q95_c:.3e}")
        print(f"  MOMENTS(z_work):             max_rel={_mmax:.3e}, q95_rel={_mq95:.3e}")
        print(f"  DECOMP check:                max|tot-unc-comp|={decomp_resid_max:.3e}")
        prof_used = res_used.get("profile", {})
        print(
            f"  TIMING({OUTPUT_MODE}): "
            f"lanczos={prof_used.get('time_total_internal', float('nan')):.3g}s, "
            f"lock_eigh={prof_used.get('time_lock_eigh', float('nan')):.3g}s "
            f"({int(prof_used.get('n_lock_eigh', 0))} calls), "
            f"lock_apply={prof_used.get('time_lock_apply', float('nan')):.3g}s, "
            f"semiorth={prof_used.get('time_semiorth_measure', float('nan')):.3g}s, "
            f"q_fullreorth={prof_used.get('time_q_fullreorth', float('nan')):.3g}s "
            f"({int(prof_used.get('n_q_fullreorth', 0))} calls), "
            f"locks_added={int(prof_used.get('n_lock_added', 0))}"
        )
        if "full" in results:
            print(
                f"  FULL: max|e|={results['full']['maxerr_rel']:.3e}, "
                f"orth_off={results['full']['orth_off']:.3e}, time={results['full']['elapsed']:.3g}s"
            )
            prof_full = results["full"].get("profile", {})
            print(
                f"  TIMING(full): lanczos={prof_full.get('time_total_internal', float('nan')):.3g}s, "
                f"q_fullreorth={prof_full.get('time_q_fullreorth', float('nan')):.3g}s "
                f"({int(prof_full.get('n_q_fullreorth', 0))} calls)"
            )

    group_nos = np.arange(1, G + 1, dtype=int)

    save_error_stat_by_group_plot_log(
        outdir=outdir,
        group_nos=group_nos,
        stat_err_rel=max_err_rel[OUTPUT_MODE],
        flag_complex=flag_complex[OUTPUT_MODE],
        flag_negative=flag_negative[OUTPUT_MODE],
        flag_constraint_switch=flag_sigma_x_constraint_switch[OUTPUT_MODE],
        Nmax=N_MAX,
        ES=ES,
        realization_mode=REALIZATION_MODE,
        sigma_x_method=SIGMA_X_METHOD,
        filename="max_err_by_group.png",
        ylabel_tex=r"$\max_{\sigma_0}\,|\varepsilon_g(\sigma_0)|$",
        title=f"Per-group worst-case relative error (mode={OUTPUT_MODE})",
    )
    save_error_stat_by_group_plot_log(
        outdir=outdir,
        group_nos=group_nos,
        stat_err_rel=q95_err_rel[OUTPUT_MODE],
        flag_complex=flag_complex[OUTPUT_MODE],
        flag_negative=flag_negative[OUTPUT_MODE],
        flag_constraint_switch=flag_sigma_x_constraint_switch[OUTPUT_MODE],
        Nmax=N_MAX,
        ES=ES,
        realization_mode=REALIZATION_MODE,
        sigma_x_method=SIGMA_X_METHOD,
        filename="q95_err_by_group.png",
        ylabel_tex=r"Effective cross-section relative error $E_{g,\mathrm{tot}}^{0.95}$",
        title=f"Per-group 95th-percentile relative error (mode={OUTPUT_MODE})",
    )

    if "full" in modes:
        save_error_stat_by_group_plot_log(
            outdir=outdir,
            group_nos=group_nos,
            stat_err_rel=max_err_rel["full"],
            flag_complex=flag_complex["full"],
            flag_negative=flag_negative["full"],
            flag_constraint_switch=flag_sigma_x_constraint_switch["full"],
            Nmax=N_MAX,
            ES=ES,
            realization_mode=REALIZATION_MODE,
            sigma_x_method=SIGMA_X_METHOD,
            filename="max_err_by_group_full.png",
            ylabel_tex=r"$\max_{\sigma_0}\,|\varepsilon_g(\sigma_0)|$",
            title="Per-group worst-case relative error (mode=full reorth)",
        )

    orth_dict = {}
    if "none" in modes:
        orth_dict["none"] = orth_off["none"]
    if "selective" in modes:
        orth_dict["selective"] = orth_off["selective"]
    if "full" in modes:
        orth_dict["full"] = orth_off["full"]
    save_orth_loss_plot(
        outdir=outdir,
        group_nos=group_nos,
        orth_off_dict=orth_dict,
        Nmax=N_MAX,
        ES=ES,
        realization_mode=REALIZATION_MODE,
        sigma_x_method=SIGMA_X_METHOD,
        filename="orth_loss_by_group.png",
    )

    save_error_stat_by_group_plot_log(
        outdir=extra_dir,
        group_nos=group_nos,
        stat_err_rel=q95_err_rel_trunc[OUTPUT_MODE],
        flag_complex=flag_complex[OUTPUT_MODE],
        flag_negative=flag_negative[OUTPUT_MODE],
        flag_constraint_switch=flag_sigma_x_constraint_switch[OUTPUT_MODE],
        Nmax=N_MAX,
        ES=ES,
        realization_mode=REALIZATION_MODE,
        sigma_x_method=SIGMA_X_METHOD,
        filename="q95_err_by_group_trunc.png",
        ylabel_tex=rf"Effective cross-section relative error $E_{{g,\mathrm{{tot}}}}^{{0.95}}$  (trunc: $\sigma_0\leq {sigma0_trunc_max:.0e}$)",
        title=f"Per-group 95th-percentile relative error (truncated; mode={OUTPUT_MODE})",
    )

    save_q95_by_group_plot_log_uncompressed(
        outdir=uncomp_dir,
        group_nos=group_nos,
        q95_err_rel=q95_err_rel_uncompressed,
        ES=ES,
        realization_mode=REALIZATION_MODE,
        direct_label=DIRECT_LABEL,
        ref_label=REF_LABEL,
        gl_nq=GL_NQ,
        filename="q95_err_by_group_uncompressed.png",
    )

    save_q95_by_group_txt(
        outdir=uncomp_dir,
        group_nos=group_nos,
        q95_err_rel=q95_err_rel_uncompressed,
        filename="q95_err_by_group_uncompressed.txt",
        title=f"UNCOMPRESSED {DIRECT_LABEL} vs {REF_LABEL}: per-group Q0.95 data",
    )

    save_q95_by_group_plot_log_compression(
        outdir=comp_dir,
        group_nos=group_nos,
        q95_err_rel=q95_err_rel_compression,
        ES=ES,
        realization_mode=REALIZATION_MODE,
        direct_label=DIRECT_LABEL,
        ref_label=REF_LABEL,
        mode_tag=OUTPUT_MODE,
        gl_nq=GL_NQ,
        filename="q95_err_by_group_compression.png",
    )

    save_q95_by_group_txt(
        outdir=comp_dir,
        group_nos=group_nos,
        q95_err_rel=q95_err_rel_compression,
        filename="q95_err_by_group_compression.txt",
        title=f"COMPRESSION-induced with reference denominator: "
              f"(PT mode={OUTPUT_MODE} - {DIRECT_LABEL}) / {REF_LABEL}, per-group Q0.95 data",
    )

    violating_mask_output = np.asarray(flag_sigma_x_constraint_switch[OUTPUT_MODE], dtype=bool)
    save_full_reconstruction_admissibility_figure(
        outdir=outdir,
        group_nos=group_nos,
        min_sigma_x_full=full_recon_min_sigma_x,
        violating_mask=violating_mask_output,
        filename_base="figure1_full_reconstruction_admissibility",
    )

    save_violating_groups_before_after_figure(
        outdir=outdir,
        violating_group_nos=group_nos[violating_mask_output],
        correction_size_rel=correction_norm_rel[violating_mask_output],
        err_full=full_recon_q95_err_rel[violating_mask_output],
        err_corrected=q95_err_rel[OUTPUT_MODE][violating_mask_output],
        filename_base="figure2_violating_groups_before_after",
        error_metric_label="Groupwise 95th-percentile relative error",
    )

    save_violating_groups_diagnostics_txt(
        outdir=outdir,
        violating_group_nos=group_nos[violating_mask_output],
        min_sigma_x_full=full_recon_min_sigma_x[violating_mask_output],
        min_sigma_x_corrected=min_re_sigma_x[OUTPUT_MODE][violating_mask_output],
        err_full=full_recon_q95_err_rel[violating_mask_output],
        err_corrected=q95_err_rel[OUTPUT_MODE][violating_mask_output],
        correction_norm=correction_norm_abs[violating_mask_output],
        total_subgroup_levels_by_group=violating_total_subgroup_levels,
        subgroup_probabilities_by_group=violating_subgroup_probabilities,
        reaction_channel_levels_full_by_group=violating_reaction_channel_levels_full,
        reaction_channel_levels_corrected_by_group=violating_reaction_channel_levels_corrected,
        filename="violating_groups_diagnostics.txt",
    )

    save_paper_figure_captions(
        outdir=outdir,
        figure1_filename_base="figure1_full_reconstruction_admissibility",
        figure2_filename_base="figure2_violating_groups_before_after",
        error_metric_name="groupwise 95th-percentile relative error",
    )

    save_moment_summary_plot_log(
        outdir=moment_dir,
        group_nos=group_nos,
        stat_err=moment_max_relerr,
        filename="moment_max_relerr_by_group.png",
        ylabel_tex=r"Maximum relative moment error",
        title=f"Per-group maximum sampled mixed-moment relative error (mode={OUTPUT_MODE})",
    )
    save_moment_summary_plot_log(
        outdir=moment_dir,
        group_nos=group_nos,
        stat_err=moment_q95_relerr,
        filename="moment_q95_relerr_by_group.png",
        ylabel_tex=r"Q0.95 relative moment error",
        title=f"Per-group Q0.95 sampled mixed-moment relative error (mode={OUTPUT_MODE})",
    )

    if WARN_LOG:
        with complex_log_path.open("w", encoding="utf-8") as wf:
            wf.write(f"IMAG_TOL = {IMAG_TOL:.3e}\n")
            wf.write(f"NEG_TOL  = {NEG_TOL:.3e}\n")
            wf.write("All complex warnings (console messages duplicated here):\n\n")
            for m in WARN_LOG:
                wf.write(m + "\n")
        print(f"\nComplex warning log saved to: {complex_log_path}")
    else:
        with complex_log_path.open("w", encoding="utf-8") as wf:
            wf.write(f"IMAG_TOL = {IMAG_TOL:.3e}\n")
            wf.write(f"NEG_TOL  = {NEG_TOL:.3e}\n")
            wf.write("No complex warnings were triggered.\n")
        print(f"\nNo complex warnings. Log file still created: {complex_log_path}")

    print("\nDone. Outputs saved to:", outdir)
    print("Summary written to:", summary_path)
    print("Saved REF values:", ref_txt_path)
    print("Saved reorth mode results:", reorth_txt_path)
    print("Saved:", outdir / "max_err_by_group.png")
    if "full" in modes:
        print("Saved:", outdir / "max_err_by_group_full.png")
    print("Saved:", outdir / "orth_loss_by_group.png")
    print("Saved:", outdir / f"sigma0_error_by_group_{OUTPUT_MODE}.txt")

    print("\n[NEW] Extra plots saved to:", extra_dir)
    print("[NEW] Saved:", extra_dir / "q95_err_by_group_trunc.png")
    print("[NEW] Saved per-group absolute error plots:", extra_dir / f"group_XXX_abs_err_{OUTPUT_MODE}.png")

    print("\n[NEW][MOMENTS] Folder:", moment_dir)
    print("[NEW][MOMENTS] Saved detailed txt:", moment_txt_path)
    print("[NEW][MOMENTS] Saved per-group plots:", moment_dir / "group_XXX_moment_relerr.png")
    print("[NEW][MOMENTS] Saved summary plots:", moment_dir / "moment_max_relerr_by_group.png")
    print("[NEW][MOMENTS] Saved summary plots:", moment_dir / "moment_q95_relerr_by_group.png")

    print("\n[UNCOMPRESSED] Folder:", uncomp_dir)
    print("[UNCOMPRESSED] Saved per-group plots:", uncomp_dir / "group_XXX_err_uncompressed.png")
    print("[UNCOMPRESSED] Saved summary Q0.95:", uncomp_dir / "q95_err_by_group_uncompressed.png")
    print("[UNCOMPRESSED] Saved summary Q0.95 txt:", uncomp_dir / "q95_err_by_group_uncompressed.txt")

    print("\n[COMPRESSION] Folder:", comp_dir)
    print("[COMPRESSION] Saved per-group plots:", comp_dir / "group_XXX_err_compression.png")
    print("[COMPRESSION] Saved summary Q0.95:", comp_dir / "q95_err_by_group_compression.png")
    print("[COMPRESSION] Saved summary Q0.95 txt:", comp_dir / "q95_err_by_group_compression.txt")

    print("\n=== Nonphysical group counts by quantity (complex or negative) ===")
    print(f"Criteria: complex if max|Im| > {IMAG_TOL:.3e}; negative if min Re(.) < {NEG_TOL:.3e}")

    for mode in modes:
        n_eval = int(np.count_nonzero(mode_success[mode]))
        if n_eval == 0:
            print(f"\nmode={mode}: evaluated groups = 0")
            continue

        n_sigma_t_complex = int(np.count_nonzero(mode_success[mode] & flag_sigma_t_complex[mode]))
        n_sigma_t_negative = int(np.count_nonzero(mode_success[mode] & flag_sigma_t_negative[mode]))
        n_sigma_t_nonphys = int(np.count_nonzero(
            mode_success[mode] & (flag_sigma_t_complex[mode] | flag_sigma_t_negative[mode])
        ))

        n_p_complex = int(np.count_nonzero(mode_success[mode] & flag_p_complex[mode]))
        n_p_negative = int(np.count_nonzero(mode_success[mode] & flag_p_negative[mode]))
        n_p_nonphys = int(np.count_nonzero(
            mode_success[mode] & (flag_p_complex[mode] | flag_p_negative[mode])
        ))

        n_eff_complex = int(np.count_nonzero(mode_success[mode] & flag_eff_complex[mode]))
        n_eff_negative = int(np.count_nonzero(mode_success[mode] & flag_eff_negative[mode]))
        n_eff_nonphys = int(np.count_nonzero(
            mode_success[mode] & (flag_eff_complex[mode] | flag_eff_negative[mode])
        ))

        print(f"\nmode={mode}: evaluated groups = {n_eval}")
        print(
            f"  sigma_t_nodes : complex = {n_sigma_t_complex:4d}, "
            f"negative = {n_sigma_t_negative:4d}, "
            f"nonphysical = {n_sigma_t_nonphys:4d} / {n_eval}"
        )
        print(
            f"  p             : complex = {n_p_complex:4d}, "
            f"negative = {n_p_negative:4d}, "
            f"nonphysical = {n_p_nonphys:4d} / {n_eval}"
        )
        print(
            f"  PT_eff_x      : complex = {n_eff_complex:4d}, "
            f"negative = {n_eff_negative:4d}, "
            f"nonphysical = {n_eff_nonphys:4d} / {n_eval}"
        )

# ============================================================
# ProcessPool worker wrapper
# ============================================================
def run_single_job_worker(job: dict) -> tuple[str, int, bool, str]:
    try:
        run_single_job(
            file_total=Path(job["file_total"]),
            file_partial=Path(job["file_partial"]),
            case_name=str(job["case_name"]),
            input_tag=str(job["input_tag"]),
            N_MAX_current=int(job["N_MAX_current"]),
            config=job["config"],
        )
        return str(job["case_name"]), int(job["N_MAX_current"]), True, ""
    except Exception as e:
        return str(job["case_name"]), int(job["N_MAX_current"]), False, str(e)

# ============================================================
# Main
# ============================================================
def main() -> None:
    try:
        HERE = Path(__file__).resolve().parent
        SCRIPT_STEM = Path(__file__).resolve().stem
    except Exception:
        HERE = Path(".").resolve()
        SCRIPT_STEM = "lanczos_run"

    # ----------------------------
    # user edit area
    # ----------------------------
    ES = 2

    # 批量 N_MAX
    N_MAX_LIST = [5, 10, 20, 30, 50]

    # 并行进程数上限
    MAX_WORKERS = 15

    # None 表示 N_DENSE 跟随当前 N_MAX；也可以手动固定。
    N_DENSE_OVERRIDE = None

    # realization mode
    REALIZATION_MODE = "ana"
    # REALIZATION_MODE = "trap"

    # sigma_x reconstruction
    SIGMA_X_METHOD = "fullcorr"
    # SIGMA_X_METHOD = "positive_hat"
    # SIGMA_X_METHOD = "eigsolve"

    # Controls used by the full-first correction branch
    # QP_RETENTION_COUNT:
    #   1 : retain n = 0 only
    #   2 : retain n = -1 and n = 0
    # RHS is built from the original discrete realization.
    QP_RETENTION_COUNT = 1
    QP_OBJECTIVE_WEIGHT_MODE = "identity"
    QP_GAMMA_MODE = "weighted_total"
    QP_RELAX_GAMMA_TO_SEED = True
    QP_SOLVER = "active-set"
    QP_MAXITER = 500
    QP_GTOL = 1e-10
    QP_XTOL = 1e-12
    QP_BARRIER_TOL = 1e-12
    QP_REAL_IMAG_TOL = 1e-10

    GL_NQ = 4
    M_MIN = 2
    SIGMA0_LOG10_MIN = -1.0
    SIGMA0_LOG10_MAX = 6.0
    SIGMA0_N = 200

    SIGMA0_TRUNC_LOG10_MAX = 5.0

    EXTRA_PLOT_SUBFOLDER = "extra_tail_diagnostics"
    MOMENT_FOLDER_NAME = "moment_diagnostics"

    # REF densification control:
    #   "auto" -> ana: None, trap: DENSIFY_GAUSS_N
    #   None   -> no REF densification
    #   int    -> explicit densification target
    REF_DENSIFY_GAUSS_N = "auto"

    Z_AFFINE_NORMALIZE = True
    Z_NORM_METHOD = "minmax"

    RUN_MODES = ("none", "selective", "full")
    OUTPUT_MODE = "selective"

    DEBUG_GROUPS = {215}
    DEBUG_TAIL_N = 10

    SEL_REORTH_TOL = 1e-9
    SEL_CHECK_EVERY = 1
    FULL_REORTH_PASSES = 2
    
    #  selective 锁定检查门控
    SELECTIVE_LOCK_MIN_K = 6
    SELECTIVE_LOCK_BETA_REL_TRIGGER = 5e-3
    SELECTIVE_LOCK_FORCE_EVERY = 8
    
    SKIP_FULL_IF_MAXERR_LEQ = None

    INPUT_CASES = [
        {
            "total": "U-238_TOT.txt",
            "partial": "U-238_capture.txt",
        },
        #{
        #    "total": "U-235_TOT.txt",
        #    "partial": "U-235_fission.txt",
        #},
        #{
        #    "total": "U-235_TOT.txt",
        #    "partial": "U-235_capture.txt",
        #},
        #{
        #    "total": "Pu-239_TOT.txt",
        #    "partial": "Pu-239_capture.txt",
        #},
        #{
        #    "total": "Am-241_TOT.txt",
        #    "partial": "Am-241_capture.txt",
        #},
    ]
    # ----------------------------

    if REALIZATION_MODE not in ("ana", "trap"):
        raise ValueError("REALIZATION_MODE must be 'ana' or 'trap'.")
    if SIGMA_X_METHOD not in ("eigsolve", "positive_hat", "fullcorr", "active_set", "admissible_correction", "full_first", "qp", "constrained_qp"):
        raise ValueError("SIGMA_X_METHOD must be 'eigsolve', 'positive_hat', or 'fullcorr'.")
    if QP_RETENTION_COUNT not in (1, 2):
        raise ValueError("QP_RETENTION_COUNT must be 1 or 2.")
    if QP_OBJECTIVE_WEIGHT_MODE not in ("identity", "probability", "p"):
        raise ValueError("QP_OBJECTIVE_WEIGHT_MODE must be 'identity' or 'probability'.")
    if QP_GAMMA_MODE not in ("weighted_total", "total_average", "default", "seed_average", "seed"):
        raise ValueError("QP_GAMMA_MODE must be 'weighted_total' or 'seed_average'.")
    if QP_SOLVER not in ("active-set", "active_set", "trust-constr", "trust_constr", "trust", "SLSQP", "slsqp"):
        raise ValueError("QP_SOLVER must be 'active-set' or one of the legacy QP solver names.")
    if GL_NQ < 1:
        raise ValueError("GL_NQ must be >= 1.")
    if M_MIN < 2:
        raise ValueError("Set M_MIN >= 2.")
    if OUTPUT_MODE not in RUN_MODES:
        raise ValueError("OUTPUT_MODE must be contained in RUN_MODES.")
    if not N_MAX_LIST:
        raise ValueError("N_MAX_LIST cannot be empty.")
    if any(int(n) < 1 for n in N_MAX_LIST):
        raise ValueError("All entries in N_MAX_LIST must be >= 1.")
    if MAX_WORKERS < 1:
        raise ValueError("MAX_WORKERS must be >= 1.")

    file_edges_es1 = HERE / "Energy_structure1.txt"
    file_edges_es2 = HERE / "Energy_structure2.txt"

    INPUT_CASES_NORM = normalize_input_cases(INPUT_CASES, HERE)

    print("\n" + "#" * 90)
    print("# BATCH SETTINGS")
    print("#" * 90)
    print(f"Script = {SCRIPT_STEM}")
    print(f"ES = {ES}")
    print(f"N_MAX_LIST = {N_MAX_LIST}")
    print(f"MAX_WORKERS = {MAX_WORKERS}")
    print(f"N_DENSE_OVERRIDE = {N_DENSE_OVERRIDE}")
    print(f"REALIZATION_MODE = {REALIZATION_MODE}")
    print(f"SIGMA_X_METHOD = {SIGMA_X_METHOD}")
    print(f"QP_RETENTION_COUNT = {QP_RETENTION_COUNT}")
    print(f"QP_OBJECTIVE_WEIGHT_MODE = {QP_OBJECTIVE_WEIGHT_MODE}")
    print(f"QP_GAMMA_MODE = {QP_GAMMA_MODE}")
    print(f"QP_RELAX_GAMMA_TO_SEED = {QP_RELAX_GAMMA_TO_SEED}")
    print(f"QP_SOLVER (legacy name; ignored by fullcorr) = {QP_SOLVER}")
    print(f"QP_MAXITER = {QP_MAXITER}")
    print(f"GL_NQ = {GL_NQ}")
    print(f"M_MIN = {M_MIN}")
    print(f"RUN_MODES = {RUN_MODES}, OUTPUT_MODE = {OUTPUT_MODE}")
    print(f"INPUT_CASES count = {len(INPUT_CASES_NORM)}")
    print(f"SELECTIVE_LOCK_MIN_K = {SELECTIVE_LOCK_MIN_K}")
    print(f"SELECTIVE_LOCK_BETA_REL_TRIGGER = {SELECTIVE_LOCK_BETA_REL_TRIGGER}")
    print(f"SELECTIVE_LOCK_FORCE_EVERY = {SELECTIVE_LOCK_FORCE_EVERY}")
    for k, case in enumerate(INPUT_CASES_NORM, start=1):
        print(
            f"  [{k}] name={case['name']} | "
            f"total={case['total'].name} | partial={case['partial'].name}"
        )
    print("#" * 90)

    config = {
        "HERE": str(HERE),
        "SCRIPT_STEM": SCRIPT_STEM,
        "ES": ES,
        "N_DENSE_OVERRIDE": N_DENSE_OVERRIDE,
        "REALIZATION_MODE": REALIZATION_MODE,
        "SIGMA_X_METHOD": SIGMA_X_METHOD,
        "QP_RETENTION_COUNT": QP_RETENTION_COUNT,
        "QP_OBJECTIVE_WEIGHT_MODE": QP_OBJECTIVE_WEIGHT_MODE,
        "QP_GAMMA_MODE": QP_GAMMA_MODE,
        "QP_RELAX_GAMMA_TO_SEED": QP_RELAX_GAMMA_TO_SEED,
        "QP_SOLVER": QP_SOLVER,
        "QP_MAXITER": QP_MAXITER,
        "QP_GTOL": QP_GTOL,
        "QP_XTOL": QP_XTOL,
        "QP_BARRIER_TOL": QP_BARRIER_TOL,
        "QP_REAL_IMAG_TOL": QP_REAL_IMAG_TOL,
        "GL_NQ": GL_NQ,
        "M_MIN": M_MIN,
        "SIGMA0_LOG10_MIN": SIGMA0_LOG10_MIN,
        "SIGMA0_LOG10_MAX": SIGMA0_LOG10_MAX,
        "SIGMA0_N": SIGMA0_N,
        "SIGMA0_TRUNC_LOG10_MAX": SIGMA0_TRUNC_LOG10_MAX,
        "EXTRA_PLOT_SUBFOLDER": EXTRA_PLOT_SUBFOLDER,
        "MOMENT_FOLDER_NAME": MOMENT_FOLDER_NAME,
        "REF_DENSIFY_GAUSS_N": REF_DENSIFY_GAUSS_N,
        "Z_AFFINE_NORMALIZE": Z_AFFINE_NORMALIZE,
        "Z_NORM_METHOD": Z_NORM_METHOD,
        "RUN_MODES": list(RUN_MODES),
        "OUTPUT_MODE": OUTPUT_MODE,
        "DEBUG_GROUPS": sorted(DEBUG_GROUPS) if DEBUG_GROUPS is not None else None,
        "DEBUG_TAIL_N": DEBUG_TAIL_N,
        "SEL_REORTH_TOL": SEL_REORTH_TOL,
        "SEL_CHECK_EVERY": SEL_CHECK_EVERY,
        "FULL_REORTH_PASSES": FULL_REORTH_PASSES,
        "SKIP_FULL_IF_MAXERR_LEQ": SKIP_FULL_IF_MAXERR_LEQ,
        "file_edges_es1": str(file_edges_es1),
        "file_edges_es2": str(file_edges_es2),
        "SELECTIVE_LOCK_MIN_K": SELECTIVE_LOCK_MIN_K,
        "SELECTIVE_LOCK_BETA_REL_TRIGGER": SELECTIVE_LOCK_BETA_REL_TRIGGER,
        "SELECTIVE_LOCK_FORCE_EVERY": SELECTIVE_LOCK_FORCE_EVERY,
    }

    jobs = []
    for case in INPUT_CASES_NORM:
        for nmax in N_MAX_LIST:
            jobs.append({
                "file_total": str(case["total"]),
                "file_partial": str(case["partial"]),
                "case_name": case["name"],
                "input_tag": case["input_tag"],
                "N_MAX_current": int(nmax),
                "config": config,
            })

    total_jobs = len(jobs)
    cpu_count = os.cpu_count() or 1
    max_workers = min(int(MAX_WORKERS), int(total_jobs), int(cpu_count))

    print("\n" + "#" * 90)
    print(f"# START PARALLEL BATCH: total_jobs={total_jobs}, cpu_count={cpu_count}, max_workers={max_workers}")
    print("#" * 90)

    # Windows-safe spawn context
    ctx = get_context("spawn")

    done_count = 0
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        future_map = {ex.submit(run_single_job_worker, job): job for job in jobs}

        for fut in as_completed(future_map):
            done_count += 1
            job = future_map[fut]

            try:
                case_name, nmax, ok, msg = fut.result()
            except Exception as e:
                case_name = str(job["case_name"])
                nmax = int(job["N_MAX_current"])
                ok = False
                msg = str(e)

            print("\n" + "=" * 100)
            print(
                f"[BATCH] Finished {done_count}/{total_jobs} | "
                f"case={case_name} | N_MAX={nmax}"
            )
            if ok:
                print("[BATCH][OK]")
            else:
                print(f"[BATCH][FAIL] error={msg}")
            print("=" * 100)

    print("\n" + "#" * 90)
    print("# ALL PARALLEL BATCH JOBS FINISHED")
    print("#" * 90)
    

if __name__ == "__main__":
    freeze_support()
    main()
