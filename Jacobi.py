# -*- coding: utf-8 -*-
"""
Author: Beichen Zheng

Lanczos-based probability table construction with NRA error diagnostics (Chiba Case C).

Per group:
  - Build a discrete positive measure on a unionized energy grid (trapezoid weights).
  - Apply Lanczos--Golub--Welsch Gauss compression with optional reorthogonalization.
  - Compare the PT effective cross section against the NRA reference curve.
Outputs:
  - Per-group error scatter plots (pcm)
  - Summary plots (max / q95 errors by group; orthogonality loss by group)
  - summary.txt and ref_values_by_group.txt
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
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


IMAG_TOL = 1e-12
NEG_TOL = -1e-14

POINT_SIZE_GROUP = 6
POINT_SIZE_SUMMARY = 10

WARN_LOG: list[str] = []


def log_warning(msg: str) -> None:
    print(msg)
    WARN_LOG.append(msg)


def _safe_stem(s: str) -> str:
    """Return a filesystem-safe stem."""
    s = str(s).strip()
    if not s:
        return "run"
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_") or "run"


def build_outdir_name(
    *,
    script_stem: str,
    ES: int,
    N_MAX: int,
    N_DENSE: int,
    M_MIN: int,
    Z_AFFINE_NORMALIZE: bool,
    Z_NORM_METHOD: str,
) -> str:
    """Build the output folder name."""
    stem = _safe_stem(script_stem)
    if Z_AFFINE_NORMALIZE:
        z_tag = "Z" + _safe_stem(Z_NORM_METHOD)
    else:
        z_tag = "Znone"
    return f"{stem}_ES{int(ES)}_Nmax{int(N_MAX)}_Ndense{int(N_DENSE)}_{z_tag}_Mmin{int(M_MIN)}_reorth"


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


def warn_if_complex(x: np.ndarray, *, name: str, group_no: int, tol: float = IMAG_TOL) -> bool:
    mi = _max_abs_imag(x)
    if np.isfinite(mi) and mi > tol:
        log_warning(f"[WARNING][Group {group_no}] {name}: max|Im| = {mi:.3e}")
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


def read_cross_sections(file_path: Path) -> np.ndarray:
    """Read (E, sigma) from a whitespace-separated text file."""
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


def crop_edges_to_overlap_with_chiba(edges_asc: np.ndarray) -> np.ndarray:
    """Crop ES=2 edges to the overlap with the Chiba range."""
    edges = np.asarray(edges_asc, dtype=float).ravel()
    edges = edges[np.isfinite(edges)]
    edges = np.unique(np.sort(edges))
    if edges.size < 2:
        raise ValueError("Energy structure must have >=2 valid boundaries.")

    E_hi = 9.1188e3
    E_lo = 5.0435

    if edges[0] > E_lo or edges[-1] < E_hi:
        raise ValueError(
            f"Energy structure does not cover Chiba range [{E_lo}, {E_hi}]. "
            f"edges=[{edges[0]}, {edges[-1]}]"
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


def densify_union_nodes_linear(nodes: np.ndarray, target_count: int) -> np.ndarray:
    """Densify a sorted node array by linear subdivision until target size is reached."""
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


def build_group_discrete_samples_trapz_nodes(
    XS_t: np.ndarray,
    XS_x: np.ndarray,
    edges_asc: np.ndarray,
    g_low: int,
    *,
    densify_gauss_N: int | None = None,
):
    """Build union-grid node samples and normalized trapezoid weights for one group."""
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


def tridiag_matrix(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    alpha = np.asarray(alpha, dtype=float).ravel()
    beta = np.asarray(beta, dtype=float).ravel()
    N = alpha.size
    T = np.diag(alpha)
    if N >= 2:
        T += np.diag(beta, 1) + np.diag(beta, -1)
    return T


def diagnostics_lanczos(z_work: np.ndarray, Q: np.ndarray, alpha: np.ndarray, beta: np.ndarray) -> tuple[float, float, float]:
    """Compute orthogonality and tridiagonalization diagnostics for the Lanczos basis."""
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


def affine_normalize_z(z: np.ndarray, w_norm: np.ndarray, method: str = "minmax") -> tuple[np.ndarray, float, float]:
    """Apply an affine normalization z = s * z_work + c."""
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


def lanczos_tridiag_from_diag(
    z: np.ndarray,
    w_norm: np.ndarray,
    N: int,
    *,
    reorth_mode: str = "none",
    sel_tol: float = 1e-10,
    sel_check_every: int = 1,
    full_passes: int = 2,
):
    """Run symmetric Lanczos on A = diag(z) with optional reorthogonalization."""
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

    M = z.size
    Q = np.zeros((M, N), dtype=float)

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
            if reorth_mode == "full":
                V = Q[:, : (k + 1)]
                for _ in range(full_passes):
                    coeff = V.T @ r
                    r = r - V @ coeff

            elif reorth_mode == "selective":
                if (k % sel_check_every) == 0:
                    V = Q[:, : (k + 1)]
                    coeff0 = V.T @ r
                    if float(np.max(np.abs(coeff0))) > float(sel_tol):
                        for _ in range(full_passes):
                            coeff = V.T @ r
                            r = r - V @ coeff

            b = float(np.linalg.norm(r))
            if not np.isfinite(b) or b == 0.0:
                raise ValueError(f"Lanczos breakdown at k={k}, beta={b}")
            beta[k] = b
            v_prev = v
            beta_prev = b
            Q[:, k + 1] = r / b

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


def nra_effective_x_reference(
    XS_t,
    XS_x,
    edges_asc,
    g_low,
    sigma0_grid,
    *,
    densify_gauss_N: int | None = None,
):
    """Compute the NRA reference effective cross section on the union grid."""
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


def nra_effective_x_probability_table_chiba_ubar(sigma_t_nodes, p, ubar, a, sigma0_grid):
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


def append_ref_curve_to_txt(
    txt_path: Path,
    group_no: int,
    g_low: int,
    e_high: float,
    e_low: float,
    sigma0_grid: np.ndarray,
    ref_curve: np.ndarray,
) -> None:
    """Append one group's reference curve to a long-table txt file."""
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


def compute_error_pcm(ref_curve: np.ndarray, pt_curve: np.ndarray) -> np.ndarray:
    """Compute the relative error in pcm."""
    pt_plot = np.real(pt_curve) if np.iscomplexobj(pt_curve) else pt_curve
    with np.errstate(divide="ignore", invalid="ignore"):
        err_pcm = (pt_plot - ref_curve) / ref_curve * 1e5
    return np.asarray(err_pcm, dtype=float)


def _finite_percentile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=float).ravel()
    m = np.isfinite(x)
    if not np.any(m):
        return float("nan")
    return float(np.percentile(x[m], q))


def compute_abs_error_stats_pcm(ref_curve: np.ndarray, pt_curve: np.ndarray) -> tuple[float, float]:
    """Return max and 95th-percentile absolute errors in pcm."""
    err_pcm = compute_error_pcm(ref_curve, pt_curve)
    abs_err = np.abs(err_pcm)
    max_abs = float(np.nanmax(abs_err)) if np.any(np.isfinite(abs_err)) else float("nan")
    q95_abs = _finite_percentile(abs_err, 95.0)
    return max_abs, q95_abs


def save_group_error_plot_pcm_scatter(
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
    tag: str = "",
) -> tuple[float, float]:
    """Save per-group error scatter plot (pcm) and return (max, q95) |error| stats."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pt_plot = as_real_for_plot(pt_curve, name="PT_eff_x", group_no=group_no, tol=IMAG_TOL)
    with np.errstate(divide="ignore", invalid="ignore"):
        err_pcm = (pt_plot - ref_curve) / ref_curve * 1e5

    max_abs = float(np.nanmax(np.abs(err_pcm))) if np.any(np.isfinite(err_pcm)) else float("nan")
    q95_abs = _finite_percentile(np.abs(err_pcm), 95.0)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(sigma0_grid, err_pcm, s=POINT_SIZE_GROUP)
    ax.set_xscale("log")
    ax.set_xlabel(r"Background (dilution) cross section $\sigma_0$ (barn)")
    ax.set_ylabel(r"Relative error $\varepsilon_g(\sigma_0)$ (pcm)")
    ax.set_title(
        r"Lanczos--Golub--Welsch (Case C): "
        f"g={group_no}, E=[{e_high:.6g},{e_low:.6g}], N_g={N_g}, b={b_g:.6g}"
        + (f" [{tag}]" if tag else "")
    )

    suffix = f"_{tag}" if tag else ""
    fig.savefig(outdir / f"group_{group_no:03d}_err{suffix}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    return max_abs, q95_abs


def _decimal_log_formatter(y, _pos):
    if not np.isfinite(y) or y <= 0:
        return ""
    if 1e-6 <= y < 1e3:
        return f"{y:g}"
    return f"{y:.0e}"


def save_error_stat_by_group_plot_log(
    outdir: Path,
    group_nos: np.ndarray,
    stat_err_pcm: np.ndarray,
    flag_complex: np.ndarray,
    flag_negative: np.ndarray,
    *,
    Nmax: int,
    ES: int,
    filename: str,
    ylabel_tex: str,
    title: str,
):
    """Save a log-scale summary plot of a per-group error statistic."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gno = np.asarray(group_nos, dtype=int).ravel()
    stat_pcm = np.asarray(stat_err_pcm, dtype=float).ravel()
    fc = np.asarray(flag_complex, dtype=bool).ravel()
    fn = np.asarray(flag_negative, dtype=bool).ravel()

    if not (gno.size == stat_pcm.size == fc.size == fn.size):
        raise ValueError("group_nos/stat_err_pcm/flag_complex/flag_negative must have same length.")

    ok = np.isfinite(stat_pcm) & (stat_pcm > 0)
    if not np.any(ok):
        print(f"[SummaryPlot] No positive finite values for {filename}.")
        return

    y = stat_pcm * 1e-5

    both = ok & fc & fn
    comp_only = ok & fc & (~fn)
    neg_only = ok & fn & (~fc)
    normal = ok & (~fc) & (~fn)

    y_min = float(np.min(y[ok]))
    y_max = float(np.max(y[ok]))
    ymin = 10.0 ** np.floor(np.log10(y_min))
    ymax = 10.0 ** np.ceil(np.log10(y_max))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    if np.any(normal):
        ax.scatter(gno[normal], y[normal], s=POINT_SIZE_SUMMARY, c="tab:blue", label="real & nonnegative")
    if np.any(neg_only):
        ax.scatter(gno[neg_only], y[neg_only], s=POINT_SIZE_SUMMARY + 4, c="tab:orange", label="negative detected")
    if np.any(comp_only):
        ax.scatter(gno[comp_only], y[comp_only], s=POINT_SIZE_SUMMARY + 4, c="tab:red", label="complex detected")
    if np.any(both):
        ax.scatter(gno[both], y[both], s=POINT_SIZE_SUMMARY + 6, c="tab:purple", label="complex & negative")

    ax.set_xlabel(r"Energy-group index $g$ (high $\rightarrow$ low)")
    ax.set_ylabel(ylabel_tex)
    ax.set_title(f"{title}  (Case C, adaptive $N_g\\leq {Nmax}$; ES={ES})")

    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=12))
    ax.yaxis.set_major_formatter(FuncFormatter(_decimal_log_formatter))
    ax.legend(loc="best", fontsize=9)

    fig.savefig(outdir / filename, dpi=260, bbox_inches="tight")
    plt.close(fig)


def save_orth_loss_plot(
    outdir: Path,
    group_nos: np.ndarray,
    orth_off_dict: dict[str, np.ndarray],
    *,
    Nmax: int,
    ES: int,
    filename: str = "orth_loss_by_group.png",
):
    """Save the orthogonality-loss summary plot."""
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
            ax.plot(gno[ok], y[ok], linestyle="None", marker="o", markersize=3.5, label=mode)

    if not any_ok:
        print(f"[OrthPlot] No positive finite values for {filename}.")
        plt.close(fig)
        return

    ax.set_xlabel(r"Energy-group index $g$ (high $\rightarrow$ low)")
    ax.set_ylabel(r"$\max_{i\neq j}\,|q_i^{T} q_j|$")
    ax.set_title(f"Loss of orthogonality (Case C, $N_g\\leq {Nmax}$; ES={ES})")
    ax.set_yscale("log")
    ax.legend(loc="best", fontsize=9)

    fig.savefig(outdir / filename, dpi=260, bbox_inches="tight")
    plt.close(fig)


def effective_support_neff(w_norm: np.ndarray) -> float:
    """Compute Neff = 1 / sum_j w_j^2 (weights sum to 1)."""
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
    """Choose N_g from M and Neff constraints."""
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


if __name__ == "__main__":
    try:
        HERE = Path(__file__).resolve().parent
        SCRIPT_STEM = Path(__file__).resolve().stem
    except Exception:
        HERE = Path(".").resolve()
        SCRIPT_STEM = "lanczos_run"

    ES = 2
    N_MAX = 50
    N_DENSE = 9

    M_MIN = 2
    SIGMA0_LOG10_MIN = -1.0
    SIGMA0_LOG10_MAX = 6.0
    SIGMA0_N = 200

    Z_AFFINE_NORMALIZE = True
    Z_NORM_METHOD = "minmax"  # "minmax" or "wmean_wstd"

    RUN_MODES = ("none", "selective", "full")
    OUTPUT_MODE = "selective"

    SEL_REORTH_TOL = 1e-13
    SEL_CHECK_EVERY = 1
    FULL_REORTH_PASSES = 2

    SKIP_FULL_IF_MAXERR_LEQ_PCM = None

    if N_MAX < 1:
        raise ValueError("N_MAX must be >= 1.")
    if N_DENSE < 2:
        raise ValueError("N_DENSE must be >= 2.")
    if M_MIN < 2:
        raise ValueError("Set M_MIN >= 2.")
    if OUTPUT_MODE not in RUN_MODES:
        raise ValueError("OUTPUT_MODE must be contained in RUN_MODES.")

    DENSIFY_GAUSS_N = int(min(N_DENSE, N_MAX))

    file_total = HERE / "U238_TOT.txt"
    file_partial = HERE / "U238_(N,G).txt"
    file_edges_es1 = HERE / "Energy_structure1.txt"
    file_edges_es2 = HERE / "Energy_structure2.txt"
    file_edges = file_edges_es1 if ES == 1 else file_edges_es2

    XS_t = read_cross_sections(file_total)
    XS_x = read_cross_sections(file_partial)
    if XS_t.size == 0 or XS_x.size == 0:
        raise RuntimeError("Empty XS data. Check input files.")

    edges_asc_full = read_energy_structure_1col(file_edges)

    if ES == 2:
        edges_asc = crop_edges_to_overlap_with_chiba(edges_asc_full)
        print(f"[ES=2] edges: full={edges_asc_full.size}, cropped={edges_asc.size}, "
              f"range=[{edges_asc[0]:.6g},{edges_asc[-1]:.6g}]")
    else:
        edges_asc = edges_asc_full
        print(f"[ES=1] edges={edges_asc.size}, range=[{edges_asc[0]:.6g},{edges_asc[-1]:.6g}]")

    a = -1.0
    sigma0_grid = np.logspace(SIGMA0_LOG10_MIN, SIGMA0_LOG10_MAX, int(SIGMA0_N))

    outdir_name = build_outdir_name(
        script_stem=SCRIPT_STEM,
        ES=ES,
        N_MAX=N_MAX,
        N_DENSE=N_DENSE,
        M_MIN=M_MIN,
        Z_AFFINE_NORMALIZE=Z_AFFINE_NORMALIZE,
        Z_NORM_METHOD=Z_NORM_METHOD,
    )
    outdir = HERE / outdir_name
    outdir.mkdir(parents=True, exist_ok=True)

    G = edges_asc.size - 1

    modes = list(RUN_MODES)
    max_err_pcm = {m: np.full(G, np.nan, dtype=float) for m in modes}
    q95_err_pcm = {m: np.full(G, np.nan, dtype=float) for m in modes}
    time_sec = {m: np.full(G, np.nan, dtype=float) for m in modes}
    orth_inf = {m: np.full(G, np.nan, dtype=float) for m in modes}
    orth_off = {m: np.full(G, np.nan, dtype=float) for m in modes}
    tri_res = {m: np.full(G, np.nan, dtype=float) for m in modes}

    flag_complex = {m: np.zeros(G, dtype=bool) for m in modes}
    flag_negative = {m: np.zeros(G, dtype=bool) for m in modes}
    min_re_sigma_t = {m: np.full(G, np.nan, dtype=float) for m in modes}
    min_re_p = {m: np.full(G, np.nan, dtype=float) for m in modes}

    print("Settings:",
          f"ES={ES}, N_MAX={N_MAX}, N_DENSE={N_DENSE}, M_MIN={M_MIN}, "
          f"Znorm={'on' if Z_AFFINE_NORMALIZE else 'off'}({Z_NORM_METHOD}), "
          f"modes={RUN_MODES}, output={OUTPUT_MODE}")
    print(f"Groups: G={G}, sigma0: [{sigma0_grid[0]:.3e}..{sigma0_grid[-1]:.3e}], n={sigma0_grid.size}")
    print("Out:", outdir)

    summary_path = outdir / "summary.txt"
    complex_log_path = outdir / "complex_warnings.txt"
    ref_txt_path = outdir / "ref_values_by_group.txt"

    with summary_path.open("w", encoding="utf-8") as sf:
        sf.write(
            "GroupNo  g_low  E_high  E_low  width  M_nodes  Neff  N_g  b_g  cover_frac  skip_frac  "
            "m0  z_shift  z_scale  "
            "mode  elapsed_s  "
            "cond_partialM  rcond_partialM  sum(p)  minRe(p)  maxRe(p)  "
            "max_abs_err_pcm  q95_abs_err_pcm  "
            "orth_inf  orth_off  tri_res  "
            "FLAG_COMPLEX  FLAG_NEG  minRe_sigma_t  minRe_p  "
            "maxIm_sigma_t  maxIm_p  maxIm_sigma_x  maxIm_ubar  maxIm_PT\n"
        )

    with ref_txt_path.open("w", encoding="utf-8") as rf:
        rf.write("# Reference effective cross section values\n")
        rf.write("# GroupNo  g_low  E_high  E_low  sigma0  ref\n")

    for idx_hi2lo in range(G):
        group_no = idx_hi2lo + 1
        g_low = (G - 1) - idx_hi2lo
        e_low = float(edges_asc[g_low])
        e_high = float(edges_asc[g_low + 1])
        width = e_high - e_low

        ref = nra_effective_x_reference(
            XS_t, XS_x, edges_asc, g_low, sigma0_grid,
            densify_gauss_N=DENSIFY_GAUSS_N,
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

        rt_node, rx_node, w_base, width, cover_dx, skip_dx, skip_seg = build_group_discrete_samples_trapz_nodes(
            XS_t, XS_x, edges_asc, g_low,
            densify_gauss_N=DENSIFY_GAUSS_N,
        )

        M = int(rt_node.size)
        cover_frac = (cover_dx / width) if (np.isfinite(width) and width > 0) else float("nan")
        skip_frac = (skip_dx / width) if (np.isfinite(width) and width > 0) else float("nan")

        if M < M_MIN:
            print(f"[Group {group_no}] skip: M={M} < {M_MIN}")
            continue
        if np.any(rt_node <= 0) or not np.all(np.isfinite(rt_node)):
            print(f"[Group {group_no}] skip: invalid rt_node")
            continue

        pi = w_base * np.power(rt_node, a)
        m0 = float(np.sum(pi))
        if not np.isfinite(m0) or m0 <= 0.0:
            print(f"[Group {group_no}] skip: m0={m0}")
            continue
        w_norm = pi / m0

        N_g, Neff, N_cap_M, N_cap_eff = choose_adaptive_N_case_c_neff(
            M=M, Nmax=N_MAX, w_norm=w_norm, c_eff=0.9, N_min=2
        )
        if N_g < 2:
            print(f"[Group {group_no}] skip: N_g={N_g} (M={M}, Neff={Neff})")
            continue

        b_g = 1.0 / (N_g - 1)
        z = np.power(rt_node, b_g)

        z_shift = 0.0
        z_scale = 1.0
        z_work = z.copy()
        if Z_AFFINE_NORMALIZE:
            z_work, z_shift, z_scale = affine_normalize_z(z, w_norm, method=Z_NORM_METHOD)

        def run_once(mode: str):
            t0 = perf_counter()

            if mode == "selective":
                alpha, beta, Qk = lanczos_tridiag_from_diag(
                    z=z_work,
                    w_norm=w_norm,
                    N=N_g,
                    reorth_mode="selective",
                    sel_tol=SEL_REORTH_TOL,
                    sel_check_every=SEL_CHECK_EVERY,
                    full_passes=FULL_REORTH_PASSES,
                )
            elif mode == "full":
                alpha, beta, Qk = lanczos_tridiag_from_diag(
                    z=z_work,
                    w_norm=w_norm,
                    N=N_g,
                    reorth_mode="full",
                    sel_tol=SEL_REORTH_TOL,
                    sel_check_every=SEL_CHECK_EVERY,
                    full_passes=FULL_REORTH_PASSES,
                )
            else:
                alpha, beta, Qk = lanczos_tridiag_from_diag(
                    z=z_work,
                    w_norm=w_norm,
                    N=N_g,
                    reorth_mode="none",
                )

            oi, oo, tr = diagnostics_lanczos(z_work, Qk, alpha, beta)
            lam_work, Qeig, p_norm = golub_welsch(alpha, beta)

            lam_z = (z_scale * lam_work + z_shift) if Z_AFFINE_NORMALIZE else lam_work
            lam_z_c = np.asarray(lam_z, dtype=np.complex128)
            sigma_t_nodes = np.power(lam_z_c, 1.0 / b_g)

            pi_nodes = m0 * p_norm
            p = pi_nodes / np.power(sigma_t_nodes, a)

            t = np.sqrt(w_norm) * rx_node
            bvec = (Qk.T @ t).astype(float)

            sigma_x_nodes, condM, rcondM = solve_sigma_x_nodes_from_eigbasis(Q=Qeig, b=bvec)
            ubar = pi_nodes * sigma_x_nodes

            pt = nra_effective_x_probability_table_chiba_ubar(sigma_t_nodes, p, ubar, a, sigma0_grid)
            maxe, q95e = compute_abs_error_stats_pcm(ref, pt)

            elapsed = perf_counter() - t0
            return {
                "orth_inf": oi, "orth_off": oo, "tri_res": tr,
                "sigma_t_nodes": sigma_t_nodes, "p": p, "sigma_x_nodes": sigma_x_nodes,
                "ubar": ubar, "pt": pt,
                "condM": condM, "rcondM": rcondM,
                "maxerr_pcm": maxe, "q95_pcm": q95e,
                "elapsed": elapsed,
            }

        results = {}
        for mode in modes:
            if mode == "full" and (SKIP_FULL_IF_MAXERR_LEQ_PCM is not None) and (OUTPUT_MODE in results):
                max_out = results[OUTPUT_MODE]["maxerr_pcm"]
                if np.isfinite(max_out) and (max_out <= SKIP_FULL_IF_MAXERR_LEQ_PCM):
                    continue
            try:
                results[mode] = run_once(mode)
            except Exception as e:
                print(f"[Group {group_no}] mode={mode} failed: {e}")
                continue

        if OUTPUT_MODE not in results:
            print(f"[Group {group_no}] output mode '{OUTPUT_MODE}' not available.")
            continue

        for mode, res in results.items():
            sigma_t_nodes = res["sigma_t_nodes"]
            p = res["p"]
            sigma_x_nodes = res["sigma_x_nodes"]
            ubar = res["ubar"]
            pt = res["pt"]

            mi_sigma_t = _max_abs_imag(sigma_t_nodes)
            mi_p = _max_abs_imag(p)
            mi_sigma_x = _max_abs_imag(sigma_x_nodes)
            mi_ubar = _max_abs_imag(ubar)
            mi_pt = _max_abs_imag(pt)
            comp_flag = (mi_sigma_t > IMAG_TOL) or (mi_p > IMAG_TOL) or (mi_sigma_x > IMAG_TOL) or (mi_ubar > IMAG_TOL) or (mi_pt > IMAG_TOL)

            min_sigma = _min_real(sigma_t_nodes)
            min_pv = _min_real(p)
            neg_flag = (np.isfinite(min_sigma) and (min_sigma < NEG_TOL)) or (np.isfinite(min_pv) and (min_pv < NEG_TOL))

            flag_complex[mode][idx_hi2lo] = bool(comp_flag)
            flag_negative[mode][idx_hi2lo] = bool(neg_flag)
            min_re_sigma_t[mode][idx_hi2lo] = float(min_sigma)
            min_re_p[mode][idx_hi2lo] = float(min_pv)

            max_err_pcm[mode][idx_hi2lo] = float(res["maxerr_pcm"])
            q95_err_pcm[mode][idx_hi2lo] = float(res["q95_pcm"])
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
                    f"{m0:.12g}  {z_shift:.6g}  {z_scale:.6g}  "
                    f"{mode:9s}  {res['elapsed']:.6g}  "
                    f"{res['condM']:.6g}  {res['rcondM']:.6g}  {sum_p:.12g}  {min_p_print:.6g}  {max_p_print:.6g}  "
                    f"{res['maxerr_pcm']:.6g}  {res['q95_pcm']:.6g}  "
                    f"{res['orth_inf']:.6g}  {res['orth_off']:.6g}  {res['tri_res']:.6g}  "
                    f"{int(comp_flag):11d}  {int(neg_flag):8d}  {min_sigma:.6g}  {min_pv:.6g}  "
                    f"{mi_sigma_t:.6g}  {mi_p:.6g}  {mi_sigma_x:.6g}  {mi_ubar:.6g}  {mi_pt:.6g}\n"
                )

        res_used = results[OUTPUT_MODE]
        sigma_t_nodes = res_used["sigma_t_nodes"]
        p = res_used["p"]
        sigma_x_nodes = res_used["sigma_x_nodes"]
        ubar = res_used["ubar"]
        pt = res_used["pt"]

        warn_if_complex(sigma_t_nodes, name="sigma_t_nodes", group_no=group_no)
        warn_if_complex(p, name="p", group_no=group_no)
        warn_if_complex(sigma_x_nodes, name="sigma_x_nodes", group_no=group_no)
        warn_if_complex(ubar, name="ubar", group_no=group_no)
        warn_if_complex(pt, name="PT_eff_x", group_no=group_no)

        save_group_error_plot_pcm_scatter(
            outdir=outdir,
            group_no=group_no,
            e_high=e_high,
            e_low=e_low,
            sigma0_grid=sigma0_grid,
            ref_curve=ref,
            pt_curve=pt,
            N_g=N_g,
            b_g=b_g,
            tag=OUTPUT_MODE,
        )

        print(f"[Group {group_no}] E=[{e_high:g}->{e_low:g}] M={M} N_g={N_g} "
              f"max={res_used['maxerr_pcm']:.3f}pcm q95={res_used['q95_pcm']:.3f}pcm "
              f"orth_off={res_used['orth_off']:.3e} tri_res={res_used['tri_res']:.3e} "
              f"t={res_used['elapsed']:.3g}s")

    group_nos = np.arange(1, G + 1, dtype=int)

    save_error_stat_by_group_plot_log(
        outdir=outdir,
        group_nos=group_nos,
        stat_err_pcm=max_err_pcm[OUTPUT_MODE],
        flag_complex=flag_complex[OUTPUT_MODE],
        flag_negative=flag_negative[OUTPUT_MODE],
        Nmax=N_MAX,
        ES=ES,
        filename="max_err_by_group.png",
        ylabel_tex=r"$\max_{\sigma_0}\,|\varepsilon_g(\sigma_0)|$",
        title=f"Per-group worst-case relative error (mode={OUTPUT_MODE})",
    )
    save_error_stat_by_group_plot_log(
        outdir=outdir,
        group_nos=group_nos,
        stat_err_pcm=q95_err_pcm[OUTPUT_MODE],
        flag_complex=flag_complex[OUTPUT_MODE],
        flag_negative=flag_negative[OUTPUT_MODE],
        Nmax=N_MAX,
        ES=ES,
        filename="q95_err_by_group.png",
        ylabel_tex=r"$Q_{0.95}\!\left(|\varepsilon_g(\sigma_0)|\right)$",
        title=f"Per-group 95th-percentile relative error (mode={OUTPUT_MODE})",
    )

    if "full" in modes:
        save_error_stat_by_group_plot_log(
            outdir=outdir,
            group_nos=group_nos,
            stat_err_pcm=max_err_pcm["full"],
            flag_complex=flag_complex["full"],
            flag_negative=flag_negative["full"],
            Nmax=N_MAX,
            ES=ES,
            filename="max_err_by_group_full.png",
            ylabel_tex=r"$\max_{\sigma_0}\,|\varepsilon_g(\sigma_0)|$",
            title="Per-group worst-case relative error (mode=full)",
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
        filename="orth_loss_by_group.png",
    )

    if WARN_LOG:
        with complex_log_path.open("w", encoding="utf-8") as wf:
            wf.write(f"IMAG_TOL = {IMAG_TOL:.3e}\n")
            wf.write(f"NEG_TOL  = {NEG_TOL:.3e}\n")
            wf.write("Warnings:\n\n")
            for m in WARN_LOG:
                wf.write(m + "\n")
    else:
        with complex_log_path.open("w", encoding="utf-8") as wf:
            wf.write(f"IMAG_TOL = {IMAG_TOL:.3e}\n")
            wf.write(f"NEG_TOL  = {NEG_TOL:.3e}\n")
            wf.write("No complex warnings.\n")

    print("Done.")
    print("Out:", outdir)
    print("Summary:", summary_path)
    print("REF:", ref_txt_path)