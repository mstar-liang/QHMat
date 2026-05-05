#!/usr/bin/env python3
"""
Aggregate NextHAM test metrics for paper tables: real-space H(R) MAE (diagonal / off-diagonal / all)
and reciprocal-space eigenenergy MAE, in eV.

**H(R)** uses ``MaskedMAELosswithGuage`` in ``train_val.py`` (gauge fix via overlap). Diagonal / off-diagonal
split the **edge-stacked** tensor: first ``num_atoms`` blocks = onsite; remaining = hopping.

**H0 baseline** — same gauge MAE between ``H0`` and converged ``H_gt`` (prediction = H0), reported as
``mae_h0_vs_gt_{diag,off,all}`` alongside ``mae_h_{diag,off,all}`` for the model.

**epsilon (mesh / gamma / random_k)** — eigenenergy MAE from the same Fourier GEVP as ``lmdb_to_nextham_pth``.
By default (``--eps-gauge``) the **reference** edge Hamiltonian is ``H_gt + μ·S`` with scalar ``μ`` chosen
like ``convert_label_with_overlap(H_pred, H_gt, S)`` (same single-parameter gauge as H MAE); prediction
uses ``H_pred`` unchanged. Linearity of Bloch sums gives ``H_gt(k)+μ S(k)``, so eigenvalues shift by ``μ``
relative to raw ``H_gt``. Pass ``--no-eps-gauge`` for raw ``ε(H_gt)`` vs ``ε(H_pred)``.

Table aggregate = mean over **structures** (macro average).

**CLI knobs (ε reporting)**

- ``--eps-mode mesh|gamma|random_k``: **mesh** averages |Δε| over many k-points (stabler headline than a
  single **random_k** draw). **gamma** is Γ only.
- ``--eps-gauge`` / ``--no-eps-gauge``: remove global μ·S ambiguity on ``H_gt`` before ε (default: on).
- ``--eps-report-both-gauges``: also emit ε metrics for **both** ``H_gt+μS`` and raw ``H_gt`` (``mae_eps_ref_*``
  keys); ``mae_eps`` / LaTeX still follow ``--eps-gauge``. H(R) MAE remains gauge-adjusted target only.
- ``--show-eps-fermi-window`` / ``--hide-eps-fermi-window``: when Fermi is known, also print and aggregate
  ``mae_eps_zoom`` / ``rmse_eps_zoom`` (**inside** ``|ε_ref - E_F| ≤`` half-width) and
  ``mae_eps_zoom_outside`` / ``rmse_eps_zoom_outside`` (**outside** that window, same ref bands).
  Use ``--eps-energy-units hartree|ev`` so the default half-width matches how ``ε`` and ``E_F`` are
  stored: **hartree** (default) uses ``±10 eV`` as ``10/27.211386…`` Ha; **ev** uses half-width ``10``
  in eV. Override half-width anytime with ``--eps-zoom-window-ha`` (same units as ``ε``, ``E_F``;
  flag name is historical).
- ``--latex-epsilon-metric full_spectrum|fermi_window``: which mean |Δε| is used for the **LaTeX** row last
  column (default keeps full spectrum; ``fermi_window`` uses aggregated zoom when finite).
- ``--checkpoint-glob-ood`` / ``--checkpoints-ood``: optional **second** ensemble for the OOD list only;
  if omitted, OOD reuses the ID ``--checkpoint-glob`` / ``--checkpoints``.
- ``--filter-max-mae-h0-vs-gt-all EV``: only aggregate structures whose baseline ``mae_h0_vs_gt_all`` (eV) is
  ≤ ``EV``; others are skipped **before** the model forward (cheap gate on H0 vs converged GT).

Requires ``train_val.py`` importable from repo root or ``NextHAM-main/``.
"""

from __future__ import annotations

import argparse
import atexit
import glob
import hashlib
import math
import json
import logging
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# ----- I/O caches (huge win for thousands of samples; see --log-level) -----
_shard_index_tensor_cache: Dict[str, torch.Tensor] = {}
_lmdb_envs: Dict[str, Any] = {}


def _get_shard_index_tensor(database_indices_file: str) -> torch.Tensor:
    """Load ``merged_mat_ham_database_indices_matched.pt`` once per path (was reloaded every sample)."""
    key = str(Path(database_indices_file).resolve())
    t = _shard_index_tensor_cache.get(key)
    if t is None:
        idx = torch.load(database_indices_file, map_location="cpu", weights_only=True)
        if not torch.is_tensor(idx):
            idx = torch.as_tensor(idx)
        _shard_index_tensor_cache[key] = idx
        return idx
    return t


def _close_lmdb_envs() -> None:
    for env in list(_lmdb_envs.values()):
        try:
            env.close()
        except Exception:
            pass
    _lmdb_envs.clear()


atexit.register(_close_lmdb_envs)


def _get_lmdb_env(mat_ham_lmdb_path: str) -> Any:
    """Reuse one LMDB Environment per DB path (was open/close every sample)."""
    try:
        import lmdb
    except ModuleNotFoundError as e:
        raise RuntimeError("pip install lmdb required for --mat-ham-lmdb") from e
    key = str(Path(mat_ham_lmdb_path).resolve())
    env = _lmdb_envs.get(key)
    if env is None:
        env = lmdb.open(
            key,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        _lmdb_envs[key] = env
    return env


# Fermi-window ε zoom uses the same energy unit as ε and E_F (see --eps-energy-units).
EV_PER_HARTREE = 27.21138602451721
DEFAULT_EPS_ZOOM_HALF_WIDTH_EV = 10.0
DEFAULT_EPS_ZOOM_WINDOW_HA = DEFAULT_EPS_ZOOM_HALF_WIDTH_EV / EV_PER_HARTREE

ROOT = Path(__file__).resolve().parent


def _ensure_import_paths() -> None:
    for p in (ROOT, ROOT / "NextHAM-main"):
        s = str(p)
        if p.is_dir() and s not in sys.path:
            sys.path.insert(0, s)


_ensure_import_paths()

import train_val as tv  # noqa: E402
from nets import model_entrypoint  # noqa: E402


# ----- Monkhorst / GEVP (inlined; no lmdb import) -----


def _eigvals_generalized(hk_r: torch.Tensor, sk_r: torch.Tensor, jitter: float, device: torch.device) -> torch.Tensor:
    nb = hk_r.shape[-1]
    eye = torch.eye(nb, dtype=sk_r.dtype, device=device)
    sk_j = sk_r + jitter * eye
    try:
        g = torch.linalg.cholesky(sk_j)
        g_inv = torch.linalg.inv(g)
        m_star = g_inv @ hk_r @ g_inv.conj().transpose(-1, -2)
        m_star = 0.5 * (m_star + m_star.conj().transpose(-1, -2))
        w = torch.linalg.eigvalsh(m_star.real)
    except RuntimeError:
        w = torch.linalg.eigvalsh(hk_r.real)
    return w.real.float()


def monkhorst_pack_kpoints(
    nk_points: int = 10, kpts_grids: Tuple[int, int, int] = (6, 6, 6), seed: int = 0, dtype=torch.float32
):
    gx, gy, gz = kpts_grids
    grid = []
    for i in range(gx):
        for j in range(gy):
            for k in range(gz):
                grid.append([i / gx, j / gy, k / gz])
    kpoints = torch.tensor(grid, dtype=dtype)
    if kpoints.shape[0] <= nk_points:
        return kpoints
    g = torch.Generator()
    g.manual_seed(seed)
    pick = torch.randperm(kpoints.shape[0] - 1, generator=g)[: nk_points - 1]
    pick = torch.cat([torch.tensor([0]), pick + 1], dim=0)
    return kpoints[pick]


def reconstruct_hk_from_edges(h_edge, edge_src, edge_dst, edge_r, kpoints, num_nodes, n_orb):
    total_orb = num_nodes * n_orb
    nk = kpoints.shape[0]
    dev = kpoints.device
    hk = torch.zeros((nk, total_orb, total_orb), dtype=torch.complex64, device=dev)
    for e in range(h_edge.shape[0]):
        i = int(edge_src[e].item())
        j = int(edge_dst[e].item())
        r = edge_r[e].to(kpoints.dtype)
        phase = torch.exp(-1j * 2 * torch.pi * (kpoints * r.unsqueeze(0)).sum(dim=-1))
        si, ei = i * n_orb, (i + 1) * n_orb
        sj, ej = j * n_orb, (j + 1) * n_orb
        block = h_edge[e].to(torch.complex64)
        hk[:, si:ei, sj:ej] += block.unsqueeze(0) * phase.view(-1, 1, 1)
    return hk


def eigenvalues_monkhorst(
    h_edge: torch.Tensor,
    overlap_edge: torch.Tensor,
    mask_edge: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    edge_r: torch.Tensor,
    num_nodes: int,
    n_orb: int,
    nk_points: int,
    kpts_grids: Tuple[int, int, int],
    jitter: float,
    device: torch.device,
) -> torch.Tensor:
    edge_src = edge_src.to(device)
    edge_dst = edge_dst.to(device)
    edge_r = edge_r.to(device)
    h_eff = (h_edge * mask_edge).to(device)
    s_eff = (overlap_edge * mask_edge).to(device)
    kpoints = monkhorst_pack_kpoints(
        nk_points=nk_points, kpts_grids=kpts_grids, seed=0, dtype=torch.float32
    ).to(device)
    hk = reconstruct_hk_from_edges(h_eff, edge_src, edge_dst, edge_r, kpoints, num_nodes, n_orb)
    sk = reconstruct_hk_from_edges(s_eff, edge_src, edge_dst, edge_r, kpoints, num_nodes, n_orb)
    row_norms = torch.sum(torch.abs(sk), dim=[0, -1])
    col_norms = torch.sum(torch.abs(sk), dim=[0, -2])
    active = (row_norms > 1e-12) & (col_norms > 1e-12)
    hk = hk[:, active][:, :, active]
    sk = sk[:, active][:, :, active]
    nk = hk.shape[0]
    eps_list = []
    for ik in range(nk):
        eps_list.append(_eigvals_generalized(hk[ik], sk[ik], jitter, device))
    return torch.stack(eps_list, dim=0)


def eigenvalues_single_fractional_k(
    h_edge: torch.Tensor,
    overlap_edge: torch.Tensor,
    mask_edge: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    edge_r: torch.Tensor,
    num_nodes: int,
    n_orb: int,
    k_frac: torch.Tensor,
    jitter: float,
    device: torch.device,
) -> torch.Tensor:
    kpoints = k_frac.to(device).float().view(1, 3)
    edge_src = edge_src.to(device)
    edge_dst = edge_dst.to(device)
    edge_r = edge_r.to(device)
    h_eff = (h_edge * mask_edge).to(device)
    s_eff = (overlap_edge * mask_edge).to(device)
    hk = reconstruct_hk_from_edges(h_eff, edge_src, edge_dst, edge_r, kpoints, num_nodes, n_orb)
    sk = reconstruct_hk_from_edges(s_eff, edge_src, edge_dst, edge_r, kpoints, num_nodes, n_orb)
    sk0 = sk[0]
    row_norms = torch.sum(torch.abs(sk0), dim=-1)
    col_norms = torch.sum(torch.abs(sk0), dim=-2)
    active = (row_norms > 1e-12) & (col_norms > 1e-12)
    hk_r = hk[0][active][:, active]
    sk_r = sk0[active][:, active]
    return _eigvals_generalized(hk_r, sk_r, jitter, device)


def stable_rng_k_index(base_seed: int, sample_path: str, n_k: int) -> int:
    h = int(hashlib.md5(sample_path.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng((base_seed + h) % (2**32))
    return int(rng.integers(0, n_k))


def _mean_abs_rmse_on_mask(d: torch.Tensor, mask: torch.Tensor) -> Tuple[float, float]:
    if not mask.any():
        return float("nan"), float("nan")
    t = d[mask]
    return torch.mean(torch.abs(t)).item(), torch.sqrt(torch.mean(t**2)).item()


def fermi_energy_window_in_out(
    d: torch.Tensor,
    ref_eps: torch.Tensor,
    fermi_ha: float,
    zoom_window: float,
) -> Tuple[float, float, float, float]:
    """
    Mean |Δε| and RMSE for reference eigenvalues inside vs outside |ref - E_F| ≤ zoom_window
    (same units as ref_eps, fermi_ha, zoom_window).
    """
    if math.isnan(fermi_ha):
        return float("nan"), float("nan"), float("nan"), float("nan")
    lo = fermi_ha - zoom_window
    hi = fermi_ha + zoom_window
    zm = (ref_eps >= lo) & (ref_eps <= hi)
    mae_in, rmse_in = _mean_abs_rmse_on_mask(d, zm)
    mae_out, rmse_out = _mean_abs_rmse_on_mask(d, ~zm)
    return mae_in, rmse_in, mae_out, rmse_out


def compute_epsilon_spectrum_block(
    *,
    h_gt_ref: torch.Tensor,
    h_pd: torch.Tensor,
    ov: torch.Tensor,
    mk: torch.Tensor,
    es: torch.Tensor,
    ed: torch.Tensor,
    er: torch.Tensor,
    num_nodes: int,
    matrix_orbitals: int,
    fermi_ha: float,
    zoom_window: float,
    eps_mode: str,
    nk_points: int,
    kpts_grids: Tuple[int, int, int],
    sjitter: float,
    eps_random_k_seed: int,
    sample_pth: str,
) -> Tuple[float, float, float, float, float, float, int, int]:
    """|ε(H_gt_ref)−ε(H_pred)| stats; zoom = inside Fermi window, zoom_out = outside."""
    cpu = torch.device("cpu")
    eps_mae = float("nan")
    eps_rmse = float("nan")
    eps_mae_zoom = float("nan")
    eps_rmse_zoom = float("nan")
    eps_mae_zoom_out = float("nan")
    eps_rmse_zoom_out = float("nan")
    nk_eps = 0
    nb_eps = 0

    if eps_mode == "mesh":
        eps_gt = eigenvalues_monkhorst(
            h_edge=h_gt_ref,
            overlap_edge=ov,
            mask_edge=mk,
            edge_src=es,
            edge_dst=ed,
            edge_r=er,
            num_nodes=num_nodes,
            n_orb=matrix_orbitals,
            nk_points=nk_points,
            kpts_grids=kpts_grids,
            jitter=sjitter,
            device=cpu,
        )
        eps_pd = eigenvalues_monkhorst(
            h_edge=h_pd,
            overlap_edge=ov,
            mask_edge=mk,
            edge_src=es,
            edge_dst=ed,
            edge_r=er,
            num_nodes=num_nodes,
            n_orb=matrix_orbitals,
            nk_points=nk_points,
            kpts_grids=kpts_grids,
            jitter=sjitter,
            device=cpu,
        )
        nb = min(eps_gt.shape[1], eps_pd.shape[1])
        nk_eps = int(eps_gt.shape[0])
        nb_eps = int(nb)
        d = eps_gt[:, :nb] - eps_pd[:, :nb]
        eps_mae = torch.mean(torch.abs(d)).item()
        eps_rmse = torch.sqrt(torch.mean(d**2)).item()
        ref = eps_gt[:, :nb]
        eps_mae_zoom, eps_rmse_zoom, eps_mae_zoom_out, eps_rmse_zoom_out = fermi_energy_window_in_out(
            d, ref, fermi_ha, zoom_window
        )
    elif eps_mode == "gamma":
        k_g = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        eps_gt = eigenvalues_single_fractional_k(
            h_gt_ref, ov, mk, es, ed, er, num_nodes, matrix_orbitals, k_g, sjitter, cpu
        )
        eps_pd = eigenvalues_single_fractional_k(
            h_pd, ov, mk, es, ed, er, num_nodes, matrix_orbitals, k_g, sjitter, cpu
        )
        nb = min(eps_gt.shape[0], eps_pd.shape[0])
        nb_eps = int(nb)
        nk_eps = 1
        d1 = eps_gt[:nb] - eps_pd[:nb]
        eps_mae = torch.mean(torch.abs(d1)).item()
        eps_rmse = torch.sqrt(torch.mean(d1**2)).item()
        ref = eps_gt[:nb]
        eps_mae_zoom, eps_rmse_zoom, eps_mae_zoom_out, eps_rmse_zoom_out = fermi_energy_window_in_out(
            d1, ref, fermi_ha, zoom_window
        )
    elif eps_mode == "random_k":
        k_pool = monkhorst_pack_kpoints(
            nk_points=nk_points,
            kpts_grids=kpts_grids,
            seed=0,
            dtype=torch.float32,
        )
        ik = stable_rng_k_index(eps_random_k_seed, sample_pth, k_pool.shape[0])
        k1 = k_pool[ik].clone()
        eps_gt = eigenvalues_single_fractional_k(
            h_gt_ref, ov, mk, es, ed, er, num_nodes, matrix_orbitals, k1, sjitter, cpu
        )
        eps_pd = eigenvalues_single_fractional_k(
            h_pd, ov, mk, es, ed, er, num_nodes, matrix_orbitals, k1, sjitter, cpu
        )
        nb = min(eps_gt.shape[0], eps_pd.shape[0])
        nb_eps = int(nb)
        nk_eps = 1
        d2 = eps_gt[:nb] - eps_pd[:nb]
        eps_mae = torch.mean(torch.abs(d2)).item()
        eps_rmse = torch.sqrt(torch.mean(d2**2)).item()
        ref = eps_gt[:nb]
        eps_mae_zoom, eps_rmse_zoom, eps_mae_zoom_out, eps_rmse_zoom_out = fermi_energy_window_in_out(
            d2, ref, fermi_ha, zoom_window
        )
    else:
        raise ValueError(eps_mode)

    return (
        eps_mae,
        eps_rmse,
        eps_mae_zoom,
        eps_rmse_zoom,
        eps_mae_zoom_out,
        eps_rmse_zoom_out,
        nk_eps,
        nb_eps,
    )


# ----- .pth / LMDB -----


def load_nextham_pth(path: str) -> Tuple[Any, ...]:
    return torch.load(path, map_location="cpu", weights_only=True)


def unpack_sample(packed: Sequence[Any]) -> Dict[str, Any]:
    (
        file_path,
        lattice_vector_torch,
        position_torch,
        tot_basis_num_torch,
        band_cut_index_torch,
        tot_kunm_torch,
        kpt_torch,
        eigenvectors_enlager_torch,
        H0_ds,
        H0,
        overlap_tensor,
        mask_tensor,
        edge_vec,
        edge_src,
        edge_dst,
        ele_list,
        mp_stru_name,
        delta_H_dp,
        H0_raw,
        overlap_tensor_raw,
        mask_tensor_raw,
        delta_H_raw,
        *extra,
    ) = packed
    out = {
        "file_path": file_path,
        "lattice_vector_torch": lattice_vector_torch,
        "position_torch": position_torch,
        "edge_vec": edge_vec,
        "edge_src": edge_src.long(),
        "edge_dst": edge_dst.long(),
        "ele_list": ele_list,
        "mp_stru_name": mp_stru_name,
        "H0_ds": H0_ds,
        "H0_raw": H0_raw.float(),
        "overlap_tensor_raw": overlap_tensor_raw.float(),
        "mask_tensor_raw": mask_tensor_raw.float(),
        "delta_H_raw": delta_H_raw.float(),
        "extra": extra,
    }
    if len(extra) >= 7:
        # LMDB convert with --precompute-band-ref: k-grid, ref eigs, coeffs, masks, edge_r, fermi.
        out["edge_r"] = extra[5].float()
        out["fermi_level"] = extra[6]
    elif len(extra) == 2:
        # LMDB convert without full band ref: [edge_r, fermi_level] only (see lmdb_to_nextham_pth.py).
        out["edge_r"] = extra[0].float()
        out["fermi_level"] = extra[1]
    return out


def edge_r_from_lmdb_data(data: Any) -> torch.Tensor:
    num_nodes = int(data.pos.shape[0])
    zero_r = torch.zeros((num_nodes, 3), dtype=torch.float32)
    return torch.cat([zero_r, data.lattice_translation_vector_full.float()], dim=0)


def load_merged_lmdb_sample(mat_ham_lmdb_path: str, lmdb_key: int) -> Any:
    env = _get_lmdb_env(mat_ham_lmdb_path)
    with env.begin(write=False) as txn:
        raw = txn.get(int(lmdb_key).to_bytes(length=4, byteorder="big"))
    if raw is None:
        raise RuntimeError(f"LMDB key {lmdb_key} not found")
    return pickle.loads(raw)


def infer_lmdb_key_from_shard_name(sample_pth: str, database_indices_file: str) -> int:
    """Map ``.../samples/00000042.pth`` → LMDB key ``indices[42]`` (same order as ``lmdb_to_nextham_pth``)."""
    idx = _get_shard_index_tensor(database_indices_file)
    n = int(idx.shape[0])
    stem = Path(sample_pth).stem
    if not stem.isdigit():
        raise ValueError(f"Cannot infer LMDB key from basename {stem}")
    out_i = int(stem)
    if out_i < 0 or out_i >= n:
        raise ValueError(f"Shard index {out_i} out of range (len={n})")
    k = idx[out_i]
    return int(k.item()) if torch.is_tensor(k) else int(k)


def build_node_atom(edge_src: torch.Tensor, edge_dst: torch.Tensor, ele_list, device: torch.device) -> torch.Tensor:

    def _unwrap_symbol(x):
        while isinstance(x, (list, tuple)) and len(x) > 0:
            x = x[0]
        return x

    node_num = max(int(edge_src.max().item()) + 1, int(edge_dst.max().item()) + 1)
    node_atom = [-1] * node_num
    for ele_idx in range(len(ele_list)):
        src_symbol = _unwrap_symbol(ele_list[ele_idx][0])
        dst_symbol = _unwrap_symbol(ele_list[ele_idx][1]) if len(ele_list[ele_idx]) > 1 else src_symbol
        si = int(edge_src[ele_idx].item())
        di = int(edge_dst[ele_idx].item())
        if src_symbol in tv.ele_dict:
            node_atom[si] = tv.ele_dict[src_symbol]
        if dst_symbol in tv.ele_dict:
            node_atom[di] = tv.ele_dict[dst_symbol]
    node_atom = [0 if v < 0 else v for v in node_atom]
    return torch.tensor(node_atom, dtype=torch.long, device=device)


@torch.no_grad()
def forward_ensemble(
    models: List[torch.nn.Module],
    construct_kernel,
    ctx: Dict[str, Any],
    orbital_block_sizes: List[int],
    ns: argparse.Namespace,
    device: torch.device,
    deadline_check: Optional[Callable[[], None]] = None,
) -> torch.Tensor:
    edge_src = ctx["edge_src"].to(device)
    edge_dst = ctx["edge_dst"].to(device)
    edge_vec = ctx["edge_vec"].to(device)
    H0_ds = ctx["H0_ds"].to(device)
    H0_raw = ctx["H0_raw"].to(device)
    ls = orbital_block_sizes
    node_num = max(int(edge_src.max().item()) + 1, int(edge_dst.max().item()) + 1)
    batch = torch.zeros((node_num,), dtype=torch.int32, device=device)
    node_atom = build_node_atom(edge_src, edge_dst, ctx["ele_list"], device)
    range_dis = tv.build_distance_ranges(len(models), max_dist=6.0)
    pred_parts = []
    for mi, model in enumerate(models):
        if deadline_check is not None:
            deadline_check()
        weak_in = H0_ds if len(models) == 1 else H0_ds.clone()
        kw = {
            "weak_ham_in": weak_in,
            "node_num": node_num,
            "edge_src": edge_src,
            "edge_dst": edge_dst,
            "edge_vec": edge_vec,
            "batch": batch,
            "node_atom": node_atom,
            "use_sep": True,
            "range_dis": range_dis[mi],
        }
        pred_h_direct_sum, _, _ = model(**kw)
        pred_parts.append(pred_h_direct_sum)
    pred_h = torch.sum(torch.stack(pred_parts), dim=0)
    pred_h = construct_kernel.get_H(pred_h)
    if ns.spinful:
        delta_H_pred_real = tv.reverse_transform_matrix(pred_h[:, 0, :].real, ls)
        H_pred = H0_raw.clone()
        H_pred = H_pred.reshape(-1, 2, ns.num_orbitals, 2, ns.num_orbitals)
        H_pred[:, 0, :, 0, :].real = H_pred[:, 0, :, 0, :].real + delta_H_pred_real
        H_pred[:, 1, :, 1, :].real = H_pred[:, 1, :, 1, :].real + delta_H_pred_real
        H_pred = H_pred.reshape(-1, ns.matrix_orbitals, ns.matrix_orbitals)
    else:
        delta_H_pred_real = tv.reverse_transform_matrix(pred_h.real, ls)
        H_pred = H0_raw + delta_H_pred_real.to(H0_raw.dtype)
    return H_pred


def _sorted_checkpoint_paths(pattern: str) -> List[str]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No checkpoints matched: {pattern}")

    def sort_key(p: str) -> Tuple[int, str]:
        m = re.search(r"model_range(\d+)", Path(p).name)
        idx = int(m.group(1)) if m else 0
        return idx, p

    return sorted(paths, key=sort_key)


def hamiltonian_mae_decomposed(
    H_pred: torch.Tensor,
    H_gt: torch.Tensor,
    overlap: torch.Tensor,
    mask: torch.Tensor,
    num_on_site_edges: int,
    threshold_max: float = 100000000.0,
    threshold_min: float = -100000000.0,
) -> Tuple[float, float, float]:
    mae_fn = tv.MaskedMAELosswithGuage(threshold_max=threshold_max, threshold_min=threshold_min)
    _, mae_all_t = mae_fn(H_pred, H_gt, overlap, mask)
    mae_all = float(mae_all_t.real if hasattr(mae_all_t, "real") else mae_all_t)

    target_adj = tv.convert_label_with_overlap(H_pred, H_gt, overlap)
    diff = (H_pred - target_adj).abs().real
    thr = ((H_gt.abs() > threshold_min) & (H_gt.abs() < threshold_max)).float()
    m_all = mask * thr

    m_diag = m_all.clone()
    m_diag[num_on_site_edges:] = 0
    m_off = m_all.clone()
    m_off[:num_on_site_edges] = 0

    def _mean(mm: torch.Tensor) -> float:
        return (diff * mm).sum() / (mm.sum() + 1e-7)

    return _mean(m_diag).item(), _mean(m_off).item(), mae_all


def hamiltonian_rmse_all(
    H_pred: torch.Tensor,
    H_gt: torch.Tensor,
    overlap: torch.Tensor,
    mask: torch.Tensor,
    num_on_site_edges: int,
    threshold_max: float = 100000000.0,
    threshold_min: float = -100000000.0,
) -> float:
    target_adj = tv.convert_label_with_overlap(H_pred, H_gt, overlap)
    diff = (H_pred - target_adj).abs().real
    thr = ((H_gt.abs() > threshold_min) & (H_gt.abs() < threshold_max)).float()
    m_all = mask * thr
    return torch.sqrt((diff**2 * m_all).sum() / (m_all.sum() + 1e-7)).item()


def load_models(
    args: argparse.Namespace,
    device: torch.device,
    *,
    checkpoint_glob: Optional[str] = None,
    checkpoints: Optional[Sequence[str]] = None,
):
    ns = argparse.Namespace(
        orbital_layout=args.orbital_layout,
        spinful=args.spinful,
        no_parity=args.no_parity,
        convert_net_out=args.convert_net_out,
        target=args.target,
        target_blocks_type=args.target_blocks_type,
        with_trace=args.with_trace,
        trace_out_len=args.trace_out_len,
        start_layer=args.start_layer,
        drop_path=args.drop_path,
    )
    irreps_edge, construct_kernel, orbital_block_sizes, num_orbitals, matrix_orbitals = tv.get_hamiltonian_size(
        ns, spinful=args.spinful
    )
    ns.num_orbitals = num_orbitals
    ns.matrix_orbitals = matrix_orbitals
    ns.orbital_block_sizes = orbital_block_sizes

    if checkpoints is not None and len(checkpoints) > 0:
        ckpts = list(checkpoints)
    elif checkpoint_glob is not None:
        ckpts = _sorted_checkpoint_paths(checkpoint_glob)
    elif args.checkpoints:
        ckpts = list(args.checkpoints)
    else:
        ckpts = _sorted_checkpoint_paths(args.checkpoint_glob)

    create_model = model_entrypoint(args.model_name)
    models = []
    for i, ck in enumerate(ckpts):
        m = create_model(
            irreps_in=args.input_irreps,
            irreps_edge=irreps_edge,
            radius=args.radius,
            num_basis=args.num_basis,
            task_mean=0.0,
            task_std=1.0,
            atomref=None,
            start_layer=args.start_layer,
            drop_path_rate=args.drop_path,
            with_trace=args.with_trace,
            trace_out_len=args.trace_out_len,
            use_w2v=False,
        ).to(device)
        state = torch.load(ck, map_location="cpu")
        inner = state.get("state_dict", state)
        cur = m.state_dict()
        compat = {k: v for k, v in inner.items() if k in cur and v.size() == cur[k].size()}
        cur.update(compat)
        m.load_state_dict(cur)
        m.eval()
        models.append(m)
        logger.info("Loaded checkpoint[%d] %s (%d tensors)", i, ck, len(compat))

    return models, construct_kernel, ns, orbital_block_sizes, matrix_orbitals


def read_paths(list_file: str) -> List[str]:
    out = []
    with open(list_file, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip()
            if p:
                out.append(p)
    return out


def subsample_paths(
    paths: List[str], max_n: int, random_draw: bool, seed: int
) -> Tuple[List[str], Dict[str, Any]]:
    n_all = len(paths)
    meta: Dict[str, Any] = {"n_list_file": n_all, "max_n": max_n, "random": random_draw, "seed": seed}
    if max_n <= 0 or max_n >= n_all:
        meta["n_selected"] = n_all
        meta["used_all"] = True
        return paths, meta
    if not random_draw:
        sel = paths[:max_n]
        meta["n_selected"] = len(sel)
        meta["used_all"] = False
        meta["mode"] = "first_n"
        return sel, meta
    rng = np.random.default_rng(seed)
    pick = rng.choice(n_all, size=max_n, replace=False)
    pick.sort()
    sel = [paths[i] for i in pick]
    meta["n_selected"] = len(sel)
    meta["used_all"] = False
    meta["mode"] = "random_without_replacement"
    meta["indices"] = pick.tolist()
    return sel, meta


def evaluate_sample(
    sample_pth: str,
    models: List[torch.nn.Module],
    construct_kernel,
    ns: argparse.Namespace,
    orbital_block_sizes: List[int],
    matrix_orbitals: int,
    args: argparse.Namespace,
    device: torch.device,
    run_counters: Optional[Dict[str, int]] = None,
) -> Optional[Dict[str, float]]:
    packed = load_nextham_pth(sample_pth)
    ctx = unpack_sample(packed)
    num_nodes = int(ctx["position_torch"].shape[0])
    H_gt = ctx["delta_H_raw"] + ctx["H0_raw"]

    resolved_key: Optional[int] = None
    if args.mat_ham_lmdb:
        if args.database_indices_file:
            resolved_key = infer_lmdb_key_from_shard_name(sample_pth, args.database_indices_file)
            logger.debug(
                "Inferred LMDB key %s from shard %s via %s",
                resolved_key,
                Path(sample_pth).name,
                args.database_indices_file,
            )
        elif args.lmdb_key is not None:
            resolved_key = int(args.lmdb_key)

    edge_r = ctx.get("edge_r")
    lmdb_data: Any = None
    if edge_r is None:
        if args.mat_ham_lmdb is None or resolved_key is None:
            logger.warning("Skip %s: no edge_r; set --mat-ham-lmdb and --database-indices-file", sample_pth)
            return None
        lmdb_data = load_merged_lmdb_sample(args.mat_ham_lmdb, resolved_key)
        edge_r = edge_r_from_lmdb_data(lmdb_data)
        fermi_log = float("nan")
        if hasattr(lmdb_data, "fermi_level"):
            fl = lmdb_data.fermi_level
            fermi_log = float(fl.item()) if torch.is_tensor(fl) else float(fl)
        logger.debug(
            "Using edge_r and fermi from LMDB key=%s (fermi=%s)",
            resolved_key,
            fermi_log,
        )
    elif (
        args.mat_ham_lmdb is not None
        and resolved_key is not None
        and ctx.get("fermi_level") is None
    ):
        lmdb_data = load_merged_lmdb_sample(args.mat_ham_lmdb, resolved_key)
        logger.debug(
            "Loaded LMDB key=%s for fermi_level only (.pth has edge_r but no fermi in unpack tail)",
            resolved_key,
        )

    num_on_site = num_nodes
    H0_raw = ctx["H0_raw"].float()
    h0_diag, h0_off, h0_all = hamiltonian_mae_decomposed(
        H0_raw.detach().cpu(),
        H_gt.detach().cpu(),
        ctx["overlap_tensor_raw"].detach().cpu(),
        ctx["mask_tensor_raw"].detach().cpu(),
        num_on_site_edges=num_on_site,
    )
    if args.filter_max_mae_h0_vs_gt_all is not None:
        thr = float(args.filter_max_mae_h0_vs_gt_all)
        if not math.isfinite(h0_all) or h0_all > thr:
            if run_counters is not None:
                run_counters["skipped_h0_vs_gt_filter"] = run_counters.get("skipped_h0_vs_gt_filter", 0) + 1
            logger.info(
                "Skip %s: mae_h0_vs_gt_all=%s exceeds filter_max_mae_h0_vs_gt_all=%.6f",
                sample_pth,
                h0_all,
                thr,
            )
            return None

    H_pred = forward_ensemble(models, construct_kernel, ctx, orbital_block_sizes, ns, device)

    m_diag, m_off, m_all = hamiltonian_mae_decomposed(
        H_pred.detach().cpu(),
        H_gt.detach().cpu(),
        ctx["overlap_tensor_raw"].detach().cpu(),
        ctx["mask_tensor_raw"].detach().cpu(),
        num_on_site_edges=num_on_site,
    )
    rmse_h_all = hamiltonian_rmse_all(
        H_pred.detach().cpu(),
        H_gt.detach().cpu(),
        ctx["overlap_tensor_raw"].detach().cpu(),
        ctx["mask_tensor_raw"].detach().cpu(),
        num_on_site_edges=num_on_site,
    )

    fermi_ha = float("nan")
    if ctx.get("fermi_level") is not None:
        fl = ctx["fermi_level"]
        fermi_ha = float(fl.item()) if torch.is_tensor(fl) else float(fl)
    if lmdb_data is not None and hasattr(lmdb_data, "fermi_level"):
        fl = lmdb_data.fermi_level
        fermi_ha = float(fl.item()) if torch.is_tensor(fl) else float(fl)

    eps_mae = float("nan")
    eps_rmse = float("nan")
    eps_mae_zoom = float("nan")
    eps_rmse_zoom = float("nan")
    eps_mae_zoom_outside = float("nan")
    eps_rmse_zoom_outside = float("nan")
    nk_eps = 0
    nb_eps = 0
    zoom_window = args.eps_zoom_window_ha
    eps_both: Dict[str, float] = {}
    if not args.skip_eps:
        h_gt = H_gt.cpu().float()
        h_pd = H_pred.cpu().float()
        ov = ctx["overlap_tensor_raw"].cpu().float()
        mk = ctx["mask_tensor_raw"].cpu().float()
        es, ed = ctx["edge_src"].cpu(), ctx["edge_dst"].cpu()
        er = edge_r.cpu().float()

        h_gt_gauged = tv.convert_label_with_overlap(h_pd, h_gt, ov)
        if args.eps_report_both_gauges:
            g_mae, g_rmse, g_mae_z, g_rmse_z, g_mae_zo, g_rmse_zo, nk_eps, nb_eps = compute_epsilon_spectrum_block(
                h_gt_ref=h_gt_gauged,
                h_pd=h_pd,
                ov=ov,
                mk=mk,
                es=es,
                ed=ed,
                er=er,
                num_nodes=num_nodes,
                matrix_orbitals=matrix_orbitals,
                fermi_ha=fermi_ha,
                zoom_window=zoom_window,
                eps_mode=args.eps_mode,
                nk_points=args.nk_points,
                kpts_grids=tuple(args.kpts_grids),
                sjitter=args.sjitter,
                eps_random_k_seed=args.eps_random_k_seed,
                sample_pth=sample_pth,
            )
            r_mae, r_rmse, r_mae_z, r_rmse_z, r_mae_zo, r_rmse_zo, _, _ = compute_epsilon_spectrum_block(
                h_gt_ref=h_gt,
                h_pd=h_pd,
                ov=ov,
                mk=mk,
                es=es,
                ed=ed,
                er=er,
                num_nodes=num_nodes,
                matrix_orbitals=matrix_orbitals,
                fermi_ha=fermi_ha,
                zoom_window=zoom_window,
                eps_mode=args.eps_mode,
                nk_points=args.nk_points,
                kpts_grids=tuple(args.kpts_grids),
                sjitter=args.sjitter,
                eps_random_k_seed=args.eps_random_k_seed,
                sample_pth=sample_pth,
            )
            eps_both = {
                "mae_eps_ref_gauged_gt": g_mae,
                "rmse_eps_ref_gauged_gt": g_rmse,
                "mae_eps_zoom_ref_gauged_gt": g_mae_z,
                "rmse_eps_zoom_ref_gauged_gt": g_rmse_z,
                "mae_eps_zoom_outside_ref_gauged_gt": g_mae_zo,
                "rmse_eps_zoom_outside_ref_gauged_gt": g_rmse_zo,
                "mae_eps_ref_raw_gt": r_mae,
                "rmse_eps_ref_raw_gt": r_rmse,
                "mae_eps_zoom_ref_raw_gt": r_mae_z,
                "rmse_eps_zoom_ref_raw_gt": r_rmse_z,
                "mae_eps_zoom_outside_ref_raw_gt": r_mae_zo,
                "rmse_eps_zoom_outside_ref_raw_gt": r_rmse_zo,
            }
            if args.eps_gauge:
                eps_mae, eps_rmse, eps_mae_zoom, eps_rmse_zoom = g_mae, g_rmse, g_mae_z, g_rmse_z
                eps_mae_zoom_outside, eps_rmse_zoom_outside = g_mae_zo, g_rmse_zo
            else:
                eps_mae, eps_rmse, eps_mae_zoom, eps_rmse_zoom = r_mae, r_rmse, r_mae_z, r_rmse_z
                eps_mae_zoom_outside, eps_rmse_zoom_outside = r_mae_zo, r_rmse_zo
        else:
            h_gt_primary = h_gt_gauged if args.eps_gauge else h_gt
            (
                eps_mae,
                eps_rmse,
                eps_mae_zoom,
                eps_rmse_zoom,
                eps_mae_zoom_outside,
                eps_rmse_zoom_outside,
                nk_eps,
                nb_eps,
            ) = compute_epsilon_spectrum_block(
                h_gt_ref=h_gt_primary,
                h_pd=h_pd,
                ov=ov,
                mk=mk,
                es=es,
                ed=ed,
                er=er,
                num_nodes=num_nodes,
                matrix_orbitals=matrix_orbitals,
                fermi_ha=fermi_ha,
                zoom_window=zoom_window,
                eps_mode=args.eps_mode,
                nk_points=args.nk_points,
                kpts_grids=tuple(args.kpts_grids),
                sjitter=args.sjitter,
                eps_random_k_seed=args.eps_random_k_seed,
                sample_pth=sample_pth,
            )

    out: Dict[str, float] = {
        "mae_h_diag": m_diag,
        "mae_h_off": m_off,
        "mae_h_all": m_all,
        "mae_h0_vs_gt_diag": h0_diag,
        "mae_h0_vs_gt_off": h0_off,
        "mae_h0_vs_gt_all": h0_all,
        "rmse_h_all": rmse_h_all,
        "mae_eps": eps_mae,
        "rmse_eps": eps_rmse,
        "mae_eps_zoom": eps_mae_zoom,
        "rmse_eps_zoom": eps_rmse_zoom,
        "mae_eps_zoom_outside": eps_mae_zoom_outside,
        "rmse_eps_zoom_outside": eps_rmse_zoom_outside,
        "fermi_ha": fermi_ha,
        "nk_eps": float(nk_eps),
        "nb_eps": float(nb_eps),
    }
    out.update(eps_both)
    return out


def aggregate(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: float(np.nanmean([r[k] for r in rows])) for k in keys}


def _epsilon_value_for_latex(agg: Dict[str, float], latex_epsilon_metric: str) -> float:
    if latex_epsilon_metric == "fermi_window":
        z = float(agg.get("mae_eps_zoom", float("nan")))
        if math.isfinite(z):
            return z
    return float(agg["mae_eps"])


def format_latex_row(label: str, agg: Dict[str, float], latex_epsilon_metric: str) -> str:
    eps_latex = _epsilon_value_for_latex(agg, latex_epsilon_metric)
    h0_all = float(agg.get("mae_h0_vs_gt_all", float("nan")))
    h0_s = f"{h0_all:.6f}" if math.isfinite(h0_all) else "nan"
    return (
        rf"{label} & NextHAM & {agg['mae_h_diag']:.5f} & {agg['mae_h_off']:.6f} & "
        rf"{agg['mae_h_all']:.6f} & {h0_s} & {eps_latex:.4f} \\"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paper-style NextHAM test metrics (H MAE + epsilon MAE)")
    p.add_argument("--checkpoint-glob", type=str, default=None)
    p.add_argument("--checkpoints", type=str, nargs="*", default=None)
    p.add_argument(
        "--checkpoint-glob-ood",
        type=str,
        default=None,
        help=(
            "Checkpoint glob for the OOD split only (with --test-list-ood). "
            "If unset, OOD reuses --checkpoint-glob / --checkpoints (ID)."
        ),
    )
    p.add_argument(
        "--checkpoints-ood",
        type=str,
        nargs="*",
        default=None,
        help="Explicit OOD checkpoint paths (mutually exclusive with --checkpoint-glob-ood when non-empty).",
    )
    p.add_argument("--test-list-id", type=str, default=None)
    p.add_argument("--test-list-ood", type=str, default=None)
    p.add_argument(
        "--max-samples-per-split",
        type=int,
        default=-1,
        help="Cap structures per split (-1 = all). Use with --subsample-random for random N.",
    )
    p.add_argument(
        "--subsample-random",
        action="store_true",
        help="With --max-samples-per-split N, draw N paths uniformly without replacement.",
    )
    p.add_argument("--subsample-seed", type=int, default=42)
    p.add_argument(
        "--filter-max-mae-h0-vs-gt-all",
        type=float,
        default=None,
        metavar="EV",
        help=(
            "Include only structures with mae_h0_vs_gt_all ≤ this value (eV). "
            "Skips expensive forward + ε before the model runs when the filter fails. "
            "Default: no filter."
        ),
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--model-name", type=str, default="graph_attention_transformer_nonlinear_materials_ham_soc")
    p.add_argument("--input-irreps", type=str, required=True)
    p.add_argument("--radius", type=float, default=8.0)
    p.add_argument("--num-basis", type=int, default=64)
    p.add_argument("--orbital-layout", type=str, default="sssppddf")
    p.add_argument("--start-layer", type=int, default=0)
    p.add_argument("--drop-path", type=float, default=0.0)
    p.add_argument("--with-trace", action="store_true")
    p.add_argument("--trace-out-len", type=int, default=81)
    p.add_argument("--target", type=str, default="hamiltonian")
    p.add_argument("--target-blocks-type", type=str, default="all")
    p.add_argument("--no-parity", action="store_true")
    p.add_argument("--convert-net-out", action="store_true")
    p.add_argument("--spinful", action="store_true")
    p.add_argument("--no-spinful", action="store_false", dest="spinful")
    p.set_defaults(spinful=False)
    p.add_argument("--mat-ham-lmdb", type=str, default=None)
    p.add_argument("--database-indices-file", type=str, default=None)
    p.add_argument("--lmdb-key", type=int, default=None)
    p.add_argument(
        "--nk-points",
        type=int,
        default=10,
        help="Max k-points from the Monkhorst grid used for ε (mesh / random_k pool); subsample if grid is larger.",
    )
    p.add_argument("--kpts-grids", type=int, nargs=3, default=[6, 6, 6])
    p.add_argument("--sjitter", type=float, default=1e-6)
    p.add_argument(
        "--eps-mode",
        type=str,
        choices=("mesh", "gamma", "random_k"),
        default="mesh",
        help="mesh = mean |Δε| over many k (stabler); gamma = Γ only; random_k = one reproducible k from pool.",
    )
    p.add_argument("--eps-random-k-seed", type=int, default=0)
    p.add_argument(
        "--eps-gauge",
        dest="eps_gauge",
        action="store_true",
        default=True,
        help="Apply scalar μ·S gauge to H_gt before ε reference (same μ as convert_label_with_overlap).",
    )
    p.add_argument(
        "--no-eps-gauge",
        dest="eps_gauge",
        action="store_false",
        help="Compare ε from raw H_gt vs H_pred (legacy epsilon MAE).",
    )
    p.add_argument(
        "--eps-report-both-gauges",
        action="store_true",
        help=(
            "Compute ε MAE/RMSE (full spectrum + Fermi zoom) for both H_gt+μS and raw H_gt references; "
            "emits mae_eps_ref_gauged_gt / mae_eps_ref_raw_gt (and RMSE + zoom variants). "
            "~2× ε eigen-solve cost. Primary mae_eps / LaTeX still follow --eps-gauge."
        ),
    )
    p.add_argument(
        "--eps-energy-units",
        type=str,
        choices=("hartree", "ev"),
        default="hartree",
        help=(
            "Unit of ε, E_F, and --eps-zoom-window-ha for the Fermi window. "
            "Default zoom half-width is ±10 eV expressed in this unit (Ha or eV) unless you pass "
            "--eps-zoom-window-ha explicitly."
        ),
    )
    p.add_argument(
        "--eps-zoom-window-ha",
        type=float,
        default=None,
        metavar="WIDTH",
        help=(
            "Half-width on the ε axis around E_F (same units as ε, E_F; name is historical). "
            "If omitted: with --eps-energy-units hartree, default is "
            f"{DEFAULT_EPS_ZOOM_WINDOW_HA:.6f} Ha (±10 eV); with ev, default is {DEFAULT_EPS_ZOOM_HALF_WIDTH_EV} eV."
        ),
    )
    p.add_argument(
        "--show-eps-fermi-window",
        dest="show_eps_fermi_window",
        action="store_true",
        default=True,
        help="Print Fermi-window ε MAE/RMSE (mae_eps_zoom) when Fermi is known (often << full-spectrum).",
    )
    p.add_argument(
        "--hide-eps-fermi-window",
        dest="show_eps_fermi_window",
        action="store_false",
        help="Do not print the Fermi-window ε lines.",
    )
    p.add_argument(
        "--latex-epsilon-metric",
        type=str,
        choices=("full_spectrum", "fermi_window"),
        default="full_spectrum",
        help="LaTeX row last column: full_spectrum=mae_eps; fermi_window=mae_eps_zoom when aggregate is finite.",
    )
    p.add_argument("--skip-eps", action="store_true")
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Root log level. Use WARNING for long runs (fewer messages; per-sample LMDB lines are DEBUG).",
    )
    p.add_argument("--output-json", type=str, default=None)
    p.add_argument("--latex-row-id", type=str, default="ID")
    p.add_argument("--latex-row-ood", type=str, default="OOD")
    return p.parse_args()


def _resolve_eps_zoom_default(args: argparse.Namespace) -> None:
    """If --eps-zoom-window-ha omitted, set ±10 eV half-width in the active energy unit."""
    if args.eps_zoom_window_ha is not None:
        return
    args.eps_zoom_window_ha = (
        DEFAULT_EPS_ZOOM_HALF_WIDTH_EV
        if args.eps_energy_units == "ev"
        else DEFAULT_EPS_ZOOM_WINDOW_HA
    )


def main() -> None:
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, str(args.log_level).upper(), logging.INFO))
    _resolve_eps_zoom_default(args)
    if not args.checkpoints and not args.checkpoint_glob:
        raise SystemExit("Provide --checkpoint-glob or --checkpoints")
    if not args.test_list_id and not args.test_list_ood:
        raise SystemExit("Provide at least one of --test-list-id / --test-list-ood")
    ood_ckpts_list = args.checkpoints_ood if args.checkpoints_ood else None
    if ood_ckpts_list is not None and len(ood_ckpts_list) == 0:
        ood_ckpts_list = None
    if args.checkpoint_glob_ood and ood_ckpts_list:
        raise SystemExit("Use only one of --checkpoint-glob-ood or non-empty --checkpoints-ood")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    models_id, construct_kernel, ns, orbital_block_sizes, matrix_orbitals = load_models(args, device)

    use_separate_ood_ensemble = bool(args.test_list_ood) and (
        args.checkpoint_glob_ood is not None or (ood_ckpts_list is not None and len(ood_ckpts_list) > 0)
    )
    models_ood = models_id
    if use_separate_ood_ensemble:
        models_ood, _, _, _, _ = load_models(
            args,
            device,
            checkpoint_glob=args.checkpoint_glob_ood,
            checkpoints=ood_ckpts_list,
        )
        logger.info(
            "Loaded separate OOD ensemble (%d models) from %s",
            len(models_ood),
            args.checkpoint_glob_ood if args.checkpoint_glob_ood else "checkpoints-ood",
        )

    results: Dict[str, Any] = {}

    def run_split(name: str, list_file: Optional[str], latex_label: str, models: List[torch.nn.Module]) -> None:
        if not list_file:
            return
        paths_full = read_paths(list_file)
        paths, sub_meta = subsample_paths(
            paths_full,
            args.max_samples_per_split,
            random_draw=args.subsample_random,
            seed=args.subsample_seed,
        )
        logger.info(
            "[%s] subsample: %s (list=%d, evaluating %d)",
            name,
            sub_meta.get("mode", "all"),
            sub_meta["n_list_file"],
            len(paths),
        )
        rows: List[Dict[str, float]] = []
        run_counters: Dict[str, int] = {}
        for i, pth in enumerate(paths):
            if (i + 1) % 50 == 0 or i == 0:
                logger.info("[%s] %d / %d  %s", name, i + 1, len(paths), pth)
            try:
                m = evaluate_sample(
                    pth,
                    models,
                    construct_kernel,
                    ns,
                    orbital_block_sizes,
                    matrix_orbitals,
                    args,
                    device,
                    run_counters=run_counters,
                )
                if m is not None:
                    rows.append(m)
            except Exception:
                logger.exception("Failed %s", pth)
        agg = aggregate(rows)
        ck_meta: Dict[str, Any] = {}
        if name == "in_distribution":
            if args.checkpoint_glob:
                ck_meta["checkpoint_glob"] = args.checkpoint_glob
            if args.checkpoints:
                ck_meta["checkpoints"] = list(args.checkpoints)
        elif name == "out_of_distribution":
            if use_separate_ood_ensemble:
                if args.checkpoint_glob_ood:
                    ck_meta["checkpoint_glob"] = args.checkpoint_glob_ood
                if ood_ckpts_list:
                    ck_meta["checkpoints"] = list(ood_ckpts_list)
            else:
                ck_meta["same_as_in_distribution"] = True
                if args.checkpoint_glob:
                    ck_meta["checkpoint_glob"] = args.checkpoint_glob
                if args.checkpoints:
                    ck_meta["checkpoints"] = list(args.checkpoints)

        results[name] = {
            "n_ok": len(rows),
            "n_total": len(paths),
            "checkpoint_source": ck_meta or None,
            "subsample": sub_meta,
            "eps_mode": args.eps_mode,
            "eps_gauge": bool(args.eps_gauge),
            "eps_report_both_gauges": bool(args.eps_report_both_gauges),
            "show_eps_fermi_window": bool(args.show_eps_fermi_window),
            "latex_epsilon_metric": args.latex_epsilon_metric,
            "eps_energy_units": args.eps_energy_units,
            "eps_zoom_window_ha": float(args.eps_zoom_window_ha),
            "metrics": agg,
            "latex_label": latex_label,
            "h0_vs_gt_filter": (
                None
                if args.filter_max_mae_h0_vs_gt_all is None
                else {
                    "max_mae_h0_vs_gt_all": float(args.filter_max_mae_h0_vs_gt_all),
                    "n_skipped_above_threshold": int(run_counters.get("skipped_h0_vs_gt_filter", 0)),
                }
            ),
        }
        filt_note = ""
        if args.filter_max_mae_h0_vs_gt_all is not None:
            filt_note = (
                f"; h0_filter skips={run_counters.get('skipped_h0_vs_gt_filter', 0)} "
                f"(keep mae_h0_vs_gt_all≤{args.filter_max_mae_h0_vs_gt_all})"
            )
        print(
            f"\n=== {name} (ok={len(rows)}/{len(paths)}{filt_note}; "
            f"list_file_n={sub_meta['n_list_file']}) ==="
        )
        print(
            f"  epsilon MAE mode: {args.eps_mode}  (eps_gauge={bool(args.eps_gauge)}; "
            f"eps_report_both_gauges={bool(args.eps_report_both_gauges)}; "
            f"latex_epsilon_metric={args.latex_epsilon_metric})"
        )
        if agg:
            print(f"  H diagonal MAE [eV] (pred vs GT):     {agg['mae_h_diag']:.6f}")
            print(f"  H off-diagonal MAE [eV] (pred vs GT): {agg['mae_h_off']:.6f}")
            print(f"  H all MAE [eV] (pred vs GT):          {agg['mae_h_all']:.6f}")
            print(f"  H0 baseline diagonal MAE [eV] (H0 vs GT): {agg['mae_h0_vs_gt_diag']:.6f}")
            print(f"  H0 baseline off-diagonal MAE [eV]:       {agg['mae_h0_vs_gt_off']:.6f}")
            print(f"  H0 baseline all MAE [eV]:                  {agg['mae_h0_vs_gt_all']:.6f}")
            ref_lbl = "H_gt+μS ref" if args.eps_gauge else "raw H_gt"
            print(f"  epsilon MAE [eV] (full spectrum, {ref_lbl}): {agg['mae_eps']:.6f}")
            if args.eps_report_both_gauges and "mae_eps_ref_gauged_gt" in agg:
                print(
                    f"  epsilon MAE [eV] (full spectrum, H_gt+μS ref, side channel): "
                    f"{agg['mae_eps_ref_gauged_gt']:.6f}"
                )
                print(
                    f"  epsilon MAE [eV] (full spectrum, raw H_gt ref, side channel): "
                    f"{agg['mae_eps_ref_raw_gt']:.6f}"
                )
            if args.show_eps_fermi_window:
                z_mae = float(agg.get("mae_eps_zoom", float("nan")))
                z_rmse = float(agg.get("rmse_eps_zoom", float("nan")))
                z_out_mae = float(agg.get("mae_eps_zoom_outside", float("nan")))
                z_out_rmse = float(agg.get("rmse_eps_zoom_outside", float("nan")))
                w = float(args.eps_zoom_window_ha)
                if args.eps_energy_units == "ev":
                    win_desc = f"|E_ref-E_F|≤{w:.2f} eV hw"
                    win_desc_na = f"Fermi window ±{w:.2f} eV hw"
                    win_out_desc = f"|E_ref-E_F|>{w:.2f} eV (half-width)"
                else:
                    w_ev = w * EV_PER_HARTREE
                    win_desc = f"|E_ref-E_F|≤{w:.4f} Ha ≈ ±{w_ev:.2f} eV hw"
                    win_desc_na = f"Fermi window ±{w:.4f} Ha ≈ ±{w_ev:.2f} eV hw"
                    win_out_desc = f"|E_ref-E_F|>{w:.4f} Ha (half-width)"
                if math.isfinite(z_mae) or math.isfinite(z_out_mae):
                    if math.isfinite(z_mae):
                        print(
                            f"  epsilon MAE [eV] (inside {win_desc}, {ref_lbl}): {z_mae:.6f}"
                            + (f"  ; RMSE: {z_rmse:.6f}" if math.isfinite(z_rmse) else "")
                        )
                    else:
                        print(
                            f"  epsilon MAE [eV] (inside {win_desc}, {ref_lbl}): n/a "
                            "(no reference bands in window)"
                        )
                    if math.isfinite(z_out_mae):
                        print(
                            f"  epsilon MAE [eV] (outside {win_out_desc}, {ref_lbl}): {z_out_mae:.6f}"
                            + (f"  ; RMSE: {z_out_rmse:.6f}" if math.isfinite(z_out_rmse) else "")
                        )
                    else:
                        print(
                            f"  epsilon MAE [eV] (outside {win_out_desc}, {ref_lbl}): n/a "
                            "(no reference bands outside window)"
                        )
                else:
                    print(
                        f"  epsilon MAE [eV] ({win_desc_na}): n/a "
                        "(missing/invalid Fermi or no usable bands for in/out split)"
                    )
            if (
                args.eps_report_both_gauges
                and args.show_eps_fermi_window
                and "mae_eps_zoom_ref_gauged_gt" in agg
            ):
                gz = float(agg.get("mae_eps_zoom_ref_gauged_gt", float("nan")))
                rz = float(agg.get("mae_eps_zoom_ref_raw_gt", float("nan")))
                gzo = float(agg.get("mae_eps_zoom_outside_ref_gauged_gt", float("nan")))
                rzo = float(agg.get("mae_eps_zoom_outside_ref_raw_gt", float("nan")))
                if math.isfinite(gz) or math.isfinite(rz) or math.isfinite(gzo) or math.isfinite(rzo):
                    gs = f"{gz:.6f}" if math.isfinite(gz) else "n/a"
                    rs = f"{rz:.6f}" if math.isfinite(rz) else "n/a"
                    print(
                        f"  epsilon MAE [eV] (inside window side channels — H_gt+μS: {gs}; raw H_gt: {rs})"
                    )
                    gos = f"{gzo:.6f}" if math.isfinite(gzo) else "n/a"
                    ros = f"{rzo:.6f}" if math.isfinite(rzo) else "n/a"
                    print(
                        f"  epsilon MAE [eV] (outside window side channels — H_gt+μS: {gos}; raw H_gt: {ros})"
                    )
            latex_eps = _epsilon_value_for_latex(agg, args.latex_epsilon_metric)
            le_desc = (
                "fermi_window (mae_eps_zoom)" if args.latex_epsilon_metric == "fermi_window" else "full_spectrum (mae_eps)"
            )
            print(f"  LaTeX ε column uses: {le_desc} → value {latex_eps:.6f}")
            print("LaTeX row (diag, off, pred_all, H0_vs_GT_all, ε last col per --latex-epsilon-metric):")
            print(format_latex_row(latex_label, agg, args.latex_epsilon_metric))

    run_split("in_distribution", args.test_list_id, args.latex_row_id, models_id)
    run_split("out_of_distribution", args.test_list_ood, args.latex_row_ood, models_ood)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
