import argparse
import os
import pickle
import random
import sys

import lmdb
import torch

# When this file lives at the v2 repo root, imports resolve via NextHAM-main on sys.path.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_NEXTHAM_MAIN = os.path.join(_SCRIPT_DIR, "NextHAM-main")
if os.path.isfile(os.path.join(_NEXTHAM_MAIN, "dataset_nano.py")) and _NEXTHAM_MAIN not in sys.path:
    sys.path.insert(0, _NEXTHAM_MAIN)

CHEMICAL_SYMBOLS = [
    "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]


from dataset_nano import config_set_target
from tg_src.e3modules import e3TensorDecomp


class AttributeDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(f"No such attribute: {name}") from exc

    def __setattr__(self, name, value):
        self[name] = value


def parse_orbital_layout(layout: str):
    orbital_map = {"s": 0, "p": 1, "d": 2, "f": 3}
    layout = layout.strip().lower()
    if not layout:
        raise ValueError("orbital layout cannot be empty")
    if any(ch not in orbital_map for ch in layout):
        raise ValueError(f"Unsupported orbital layout '{layout}'. Only s/p/d/f are supported.")
    orbital_types = [orbital_map[ch] for ch in layout]
    orbital_block_sizes = [2 * l + 1 for l in orbital_types]
    num_orbitals = sum(orbital_block_sizes)
    return orbital_types, orbital_block_sizes, num_orbitals


def get_hamiltonian_converter(args):
    orbital_types, orbital_block_sizes, num_orbitals = parse_orbital_layout(args.orbital_layout)
    dataset_info = AttributeDict(
        spinful=args.spinful,
        index_to_Z=torch.arange(118).long(),
        Z_to_index=torch.arange(118).long(),
        orbital_types=[orbital_types],
    )
    _, _, net_out_irreps, net_out_info = config_set_target(dataset_info, args, verbose="target.txt")
    construct_kernel = e3TensorDecomp(
        net_out_irreps,
        net_out_info.js,
        default_dtype_torch=torch.float32,
        spinful=args.spinful,
        no_parity=args.no_parity,
        if_sort=args.convert_net_out,
        device_torch=torch.device("cpu"),
    )
    return construct_kernel, orbital_block_sizes, num_orbitals


def flatten_hamiltonian_by_blocks(h_raw, overlap_raw, mask_raw, orbital_block_sizes, num_orbitals, spinful):
    if spinful:
        h_5d = h_raw.reshape((h_raw.shape[0], 2, num_orbitals, 2, num_orbitals))
        s_5d = overlap_raw.reshape((overlap_raw.shape[0], 2, num_orbitals, 2, num_orbitals))
        m_5d = mask_raw.reshape((mask_raw.shape[0], 2, num_orbitals, 2, num_orbitals))
        spin_pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    else:
        h_5d = h_raw.reshape((h_raw.shape[0], 1, num_orbitals, 1, num_orbitals))
        s_5d = overlap_raw.reshape((overlap_raw.shape[0], 1, num_orbitals, 1, num_orbitals))
        m_5d = mask_raw.reshape((mask_raw.shape[0], 1, num_orbitals, 1, num_orbitals))
        spin_pairs = [(0, 0)]

    h_list, s_list, m_list = [], [], []
    for d1, d2 in spin_pairs:
        h_blocks, s_blocks, m_blocks = [], [], []
        a = 0
        for i in orbital_block_sizes:
            b = 0
            for j in orbital_block_sizes:
                h_blocks.append(h_5d[:, d1, a : a + i, d2, b : b + j].reshape(h_5d.shape[0], -1))
                s_blocks.append(s_5d[:, d1, a : a + i, d2, b : b + j].reshape(s_5d.shape[0], -1))
                m_blocks.append(m_5d[:, d1, a : a + i, d2, b : b + j].reshape(m_5d.shape[0], -1))
                b += j
            a += i
        channel_size = num_orbitals * num_orbitals
        h_list.append(torch.cat(h_blocks, dim=-1).reshape((-1, 1, channel_size)))
        s_list.append(torch.cat(s_blocks, dim=-1).reshape((-1, 1, channel_size)))
        m_list.append(torch.cat(m_blocks, dim=-1).reshape((-1, 1, channel_size)))

    h_out = torch.cat(h_list, dim=1)
    s_out = torch.cat(s_list, dim=1)
    m_out = torch.cat(m_list, dim=1)
    # e3TensorDecomp.get_net_out/get_H_trace expect 2D tensors in spinless mode.
    if not spinful:
        h_out = h_out[:, 0, :]
        s_out = s_out[:, 0, :]
        m_out = m_out[:, 0, :]
    return h_out, s_out, m_out


def build_edge_tensors(data):
    pos = data.pos.float()
    lattice = data.lattice.float()
    off_edge_index = data.multi_edge_index_full.long()
    off_r = data.lattice_translation_vector_full.float()

    num_nodes = pos.shape[0]
    diag_idx = torch.arange(num_nodes, dtype=torch.long)
    diag_edge_index = torch.stack([diag_idx, diag_idx], dim=0)
    diag_vec = torch.zeros((num_nodes, 3), dtype=torch.float32)

    off_src = off_edge_index[0]
    off_dst = off_edge_index[1]
    off_vec = pos[off_dst] - pos[off_src] + off_r @ lattice.T

    edge_index = torch.cat([diag_edge_index, off_edge_index], dim=1)
    edge_vec = torch.cat([diag_vec, off_vec], dim=0)
    return edge_index[0], edge_index[1], edge_vec


def make_wfc_placeholders(num_nodes, matrix_orbitals):
    tot_basis_num = int(num_nodes * matrix_orbitals)
    band_cut_index = max(1, tot_basis_num // 2)
    tot_knum = 1
    kpt = torch.zeros((1, 3), dtype=torch.float64)
    # Placeholder wavefunctions: identity basis at a single k-point.
    eigenvectors = torch.eye(tot_basis_num, dtype=torch.complex64)
    return (
        torch.tensor(tot_basis_num, dtype=torch.int64),
        torch.tensor(band_cut_index, dtype=torch.int64),
        torch.tensor(tot_knum, dtype=torch.int64),
        kpt,
        eigenvectors,
    )


def make_wfc_precomputed(kpoints, eps_ref, c_ref, fermi_level, energy_window):
    tk = int(kpoints.shape[0])
    nbasis = int(c_ref.shape[-1])
    # Match the historical layout expected by training:
    # [tot_knum * nband, nbasis], where rows index bands for each k-point.
    eigenvectors_enlager = c_ref.transpose(-1, -2).reshape(tk * nbasis, nbasis).to(torch.complex64)
    # Keep the same energy-window convention as BandStuctureAlignmentLoss.
    # kpoints[0] is gamma due to monkhorst_pack_kpoints selection.
    gamma_eps = eps_ref[0]
    band_cut_index = int((gamma_eps < (float(fermi_level) + float(energy_window))).sum().item())
    band_cut_index = max(1, min(nbasis - 1, band_cut_index))
    return (
        torch.tensor(nbasis, dtype=torch.int64),
        torch.tensor(band_cut_index, dtype=torch.int64),
        torch.tensor(tk, dtype=torch.int64),
        kpoints.to(torch.float64),
        eigenvectors_enlager,
    )


def monkhorst_pack_kpoints(nk_points=50, kpts_grids=(6, 6, 6), seed=0, dtype=torch.float32):
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
    hk = torch.zeros((nk, total_orb, total_orb), dtype=torch.complex64)
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


@torch.no_grad()
def precompute_band_reference(
    h_target_edge,
    overlap_edge,
    mask_edge,
    edge_src,
    edge_dst,
    edge_r,
    num_nodes,
    n_orb,
    fermi_level,
    nk_points=50,
    kpts_grids=(6, 6, 6),
    energy_window=20.0,
    seed=0,
):
    h_eff = h_target_edge * mask_edge
    s_eff = overlap_edge * mask_edge
    kpoints = monkhorst_pack_kpoints(
        nk_points=nk_points, kpts_grids=kpts_grids, seed=seed, dtype=edge_r.dtype
    )
    hk = reconstruct_hk_from_edges(h_eff, edge_src, edge_dst, edge_r, kpoints, num_nodes, n_orb)
    sk = reconstruct_hk_from_edges(s_eff, edge_src, edge_dst, edge_r, kpoints, num_nodes, n_orb)

    row_norms = torch.sum(torch.abs(sk), dim=[0, -1])
    col_norms = torch.sum(torch.abs(sk), dim=[0, -2])
    active = (row_norms > 1e-12) & (col_norms > 1e-12)
    hk = hk[:, active][:, :, active]
    sk = sk[:, active][:, :, active]

    eye = torch.eye(sk.shape[-1], dtype=sk.dtype).unsqueeze(0)
    sk = sk + 1e-6 * eye
    gk = torch.linalg.cholesky(sk)
    gk_inv = torch.linalg.inv(gk)
    mstar = torch.bmm(gk_inv, torch.bmm(hk, gk_inv.transpose(-1, -2).conj()))
    eps_ref, qstar = torch.linalg.eigh(mstar)
    c_ref = gk_inv.transpose(-1, -2).conj() @ qstar

    mask_band = eps_ref < (float(fermi_level) + float(energy_window))
    return kpoints, eps_ref.real.float(), c_ref.to(torch.complex64), active, mask_band


def convert_one_sample(data, construct_kernel, args, orbital_block_sizes, num_orbitals, matrix_orbitals):
    h0_diag = data.diagonal_hamiltonian_h0.float()
    h0_off = data.off_diagonal_hamiltonian_h0.float()
    hc_diag = data.diagonal_hamiltonian_converged.float()
    hc_off = data.off_diagonal_hamiltonian_converged.float()

    m_diag = data.diagonal_hamiltonian_h0_mask.float()
    m_off = data.off_diagonal_hamiltonian_h0_mask.float()

    if hasattr(data, "diagonal_overlap") and hasattr(data, "off_diagonal_overlap"):
        s_diag = data.diagonal_overlap.float()
        s_off = data.off_diagonal_overlap.float()
    else:
        s_diag = torch.zeros_like(h0_diag)
        s_off = torch.zeros_like(h0_off)

    h0_raw = torch.cat([h0_diag, h0_off], dim=0)
    hconv_raw = torch.cat([hc_diag, hc_off], dim=0)
    delta_h_raw = hconv_raw - h0_raw
    overlap_raw = torch.cat([s_diag, s_off], dim=0)
    mask_raw = torch.cat([m_diag, m_off], dim=0)

    h0, overlap, mask = flatten_hamiltonian_by_blocks(
        h0_raw, overlap_raw, mask_raw, orbital_block_sizes, num_orbitals, args.spinful
    )
    delta_h_dp, _, _ = flatten_hamiltonian_by_blocks(
        delta_h_raw, overlap_raw, mask_raw, orbital_block_sizes, num_orbitals, args.spinful
    )
    h0_ds = construct_kernel.get_net_out(h0)

    edge_src, edge_dst, edge_vec = build_edge_tensors(data)
    num_nodes = int(data.atoms.shape[0])
    zero_r = torch.zeros((num_nodes, 3), dtype=torch.float32)
    edge_r = torch.cat([zero_r, data.lattice_translation_vector_full.float()], dim=0)

    atomic_numbers = data.atoms.to(torch.int64)
    symbols = [CHEMICAL_SYMBOLS[int(z.item())] for z in atomic_numbers]
    ele_list = [([symbols[int(src)]], [symbols[int(dst)]]) for src, dst in zip(edge_src.tolist(), edge_dst.tolist())]

    sample_id = getattr(data, "raw_basename", f"sample_{int(data.database_idx)}")
    mp_name = getattr(data, "mp_id", str(sample_id))
    fermi_level = float(data.fermi_level.item()) if torch.is_tensor(data.fermi_level) else float(data.fermi_level)
    # Precompute k-point/eigenvector data for P/Q space split.
    # Fallback to placeholders if decomposition fails for a sample.
    band_ref = None
    try:
        band_kpoints, band_eps_ref, band_c_ref, band_active, band_mask = precompute_band_reference(
            h_target_edge=hconv_raw,
            overlap_edge=overlap_raw,
            mask_edge=mask_raw,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_r=edge_r,
            num_nodes=num_nodes,
            n_orb=matrix_orbitals,
            fermi_level=fermi_level,
            nk_points=args.band_nk_points,
            kpts_grids=tuple(args.band_kpts_grids),
            energy_window=args.band_energy_window,
            seed=args.seed,
        )
        tot_basis_num, band_cut_index, tot_knum, kpt, eigenvectors = make_wfc_precomputed(
            kpoints=band_kpoints,
            eps_ref=band_eps_ref,
            c_ref=band_c_ref,
            fermi_level=fermi_level,
            energy_window=args.band_energy_window,
        )
        band_ref = (
            band_kpoints,
            band_eps_ref,
            band_c_ref,
            band_active,
            band_mask,
        )
    except Exception as exc:
        print(
            f"[WARN] Falling back to placeholder WFC for sample '{sample_id}' "
            f"(mp_id='{mp_name}'): {exc}"
        )
        tot_basis_num, band_cut_index, tot_knum, kpt, eigenvectors = make_wfc_placeholders(
            num_nodes=atomic_numbers.shape[0], matrix_orbitals=matrix_orbitals
        )

    packed = [
        sample_id,
        data.lattice.float(),
        data.pos.float(),
        tot_basis_num,
        band_cut_index,
        tot_knum,
        kpt,
        eigenvectors,
        h0_ds.float(),
        h0.float(),
        overlap.float(),
        mask.float(),
        edge_vec.float(),
        edge_src.long(),
        edge_dst.long(),
        ele_list,
        mp_name,
        delta_h_dp.float(),
        h0_raw.float(),
        overlap_raw.float(),
        mask_raw.float(),
        delta_h_raw.float(),
    ]
    # Tail A (7 tensors): full band-reference precompute — same order as train_val / unpack_sample.
    # Tail B (2 tensors): always store edge lattice shifts + Fermi level from LMDB when Tail A is absent,
    # so noprecompute .pth files still carry fermi/edge_r for band plots and metrics without re-opening LMDB.
    if args.precompute_band_ref and band_ref is not None:
        band_kpoints, band_eps_ref, band_c_ref, band_active, band_mask = band_ref
        packed.extend([
            band_kpoints.float(),
            band_eps_ref.float(),
            band_c_ref.to(torch.complex64),
            band_active.bool(),
            band_mask.bool(),
            edge_r.float(),
            torch.tensor(fermi_level, dtype=torch.float32),
        ])
    else:
        packed.extend([
            edge_r.float(),
            torch.tensor(fermi_level, dtype=torch.float32),
        ])
    return packed


def split_paths(paths, train_ratio, val_ratio, seed):
    rng = random.Random(seed)
    shuffled = paths[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    if n_train + n_val > n:
        n_val = max(0, n - n_train)
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


def write_split_file(path, entries):
    with open(path, "w", encoding="utf-8") as f:
        for p in entries:
            f.write(p + "\n")


def main():
    parser = argparse.ArgumentParser("Convert merged LMDB to NextHAM .pth dataset")
    parser.add_argument("--mat-ham-lmdb-path", type=str, required=True)
    parser.add_argument("--database-indices-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--orbital-layout", type=str, default="sssppddf")
    parser.add_argument("--spinful", action="store_true", default=False)
    parser.add_argument("--no-parity", action="store_true")
    parser.add_argument("--convert-net-out", action="store_true")
    parser.add_argument("--target", type=str, default="hamiltonian")
    parser.add_argument("--target-blocks-type", type=str, default="all")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--precompute-band-ref", action="store_true")
    parser.add_argument("--band-nk-points", type=int, default=50)
    parser.add_argument("--band-kpts-grids", type=int, nargs=3, default=[6, 6, 6])
    parser.add_argument("--band-energy-window", type=float, default=20.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    samples_dir = os.path.join(args.output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    construct_kernel, orbital_block_sizes, num_orbitals = get_hamiltonian_converter(args)
    matrix_orbitals = num_orbitals * (2 if args.spinful else 1)

    indices = torch.load(args.database_indices_file, weights_only=True)
    if args.max_samples > 0:
        indices = indices[: args.max_samples]

    env = lmdb.open(args.mat_ham_lmdb_path, readonly=True, lock=False)
    saved_paths = []

    with env.begin(write=False) as txn:
        for out_i, idx in enumerate(indices):
            key = int(idx).to_bytes(4, "big")
            raw = txn.get(key)
            if raw is None:
                continue
            data = pickle.loads(raw)
            packed = convert_one_sample(
                data=data,
                construct_kernel=construct_kernel,
                args=args,
                orbital_block_sizes=orbital_block_sizes,
                num_orbitals=num_orbitals,
                matrix_orbitals=matrix_orbitals,
            )
            out_path = os.path.join(samples_dir, f"{out_i:08d}.pth")
            torch.save(packed, out_path)
            saved_paths.append(os.path.abspath(out_path))
            if (out_i + 1) % 100 == 0:
                print(f"Converted {out_i + 1}/{len(indices)} samples")

    env.close()

    train_paths, val_paths, test_paths = split_paths(
        saved_paths, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed
    )
    write_split_file(os.path.join(args.output_dir, "train.txt"), train_paths)
    write_split_file(os.path.join(args.output_dir, "val.txt"), val_paths)
    write_split_file(os.path.join(args.output_dir, "test.txt"), test_paths)

    print("Conversion done.")
    print(f"Total samples: {len(saved_paths)}")
    print(f"Train/Val/Test: {len(train_paths)}/{len(val_paths)}/{len(test_paths)}")
    print(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
