import argparse
import datetime
import glob
import itertools
import pickle
import subprocess
import time
import torch
import numpy as np
import random
#torch.autograd.set_detect_anomaly(True)
import sys
#from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset, DataLoader
from tg_src.e3modules import e3TensorDecomp, get_random_R
from output_data_convert import get_hamiltion_data
import gc
import os
from logger import FileLogger
from pathlib import Path
from typing import Iterable, Optional
import copy
import torch.multiprocessing as mp

import nets
from nets import model_entrypoint

from timm.utils import ModelEmaV2, get_state_dict
from timm.scheduler import create_scheduler

from engine import AverageMeter, compute_stats
from dataset_nano import nanotube_weak, config_set_target, DatasetInfo
from operator import itemgetter
from scipy.linalg import block_diag
from torch.nn.parallel import parallel_apply


ModelEma = ModelEmaV2

elements_index_info = [
    (1, "H", 1, 1), (2, "He", 18, 1),
    (3, "Li", 1, 2), (4, "Be", 2, 2), (5, "B", 13, 2), (6, "C", 14, 2), 
    (7, "N", 15, 2), (8, "O", 16, 2), (9, "F", 17, 2), (10, "Ne", 18, 2),
    (11, "Na", 1, 3), (12, "Mg", 2, 3), (13, "Al", 13, 3), (14, "Si", 14, 3), 
    (15, "P", 15, 3), (16, "S", 16, 3), (17, "Cl", 17, 3), (18, "Ar", 18, 3),
    (19, "K", 1, 4), (20, "Ca", 2, 4), (21, "Sc", 3, 4), (22, "Ti", 4, 4), 
    (23, "V", 5, 4), (24, "Cr", 6, 4), (25, "Mn", 7, 4), (26, "Fe", 8, 4), 
    (27, "Co", 9, 4), (28, "Ni", 10, 4), (29, "Cu", 11, 4), (30, "Zn", 12, 4), 
    (31, "Ga", 13, 4), (32, "Ge", 14, 4), (33, "As", 15, 4), (34, "Se", 16, 4), 
    (35, "Br", 17, 4), (36, "Kr", 18, 4),
    (37, "Rb", 1, 5), (38, "Sr", 2, 5), (39, "Y", 3, 5), (40, "Zr", 4, 5), 
    (41, "Nb", 5, 5), (42, "Mo", 6, 5), (43, "Tc", 7, 5), (44, "Ru", 8, 5), 
    (45, "Rh", 9, 5), (46, "Pd", 10, 5), (47, "Ag", 11, 5), (48, "Cd", 12, 5), 
    (49, "In", 13, 5), (50, "Sn", 14, 5), (51, "Sb", 15, 5), (52, "Te", 16, 5), 
    (53, "I", 17, 5), (54, "Xe", 18, 5),
    (55, "Cs", 1, 6), (56, "Ba", 2, 6), 
    (72, "Hf", 4, 6), (73, "Ta", 5, 6), (74, "W", 6, 6), (75, "Re", 7, 6), 
    (76, "Os", 8, 6), (77, "Ir", 9, 6), (78, "Pt", 10, 6), (79, "Au", 11, 6), 
    (80, "Hg", 12, 6), (81, "Tl", 13, 6), (82, "Pb", 14, 6), (83, "Bi", 15, 6), 
    (84, "Po", 16, 6), (85, "At", 17, 6), (86, "Rn", 18, 6)
]

ele_dict = {}

for tuple_ele in elements_index_info:
    if not tuple_ele[1] in ele_dict:
        ele_dict[tuple_ele[1]] = int(tuple_ele[0])-1


def parse_orbital_layout(layout: str):
    orbital_map = {'s': 0, 'p': 1, 'd': 2, 'f': 3}
    layout = layout.strip().lower()
    if not layout:
        raise ValueError("orbital layout cannot be empty")
    if any(ch not in orbital_map for ch in layout):
        raise ValueError(f"Unsupported orbital layout '{layout}'. Only s/p/d/f are supported.")
    orbital_types = [orbital_map[ch] for ch in layout]
    orbital_block_sizes = [2 * l + 1 for l in orbital_types]
    num_orbitals = sum(orbital_block_sizes)
    return orbital_types, orbital_block_sizes, num_orbitals

def get_args_parser():
    parser = argparse.ArgumentParser('Training general equivariant networks for electronic-structure prediction', add_help=False)
    parser.add_argument('--output-dir', type=str, default=None)
    # network architecture
    parser.add_argument('--model-name', type=str, default='graph_attention_transformer_nonlinear_l2_md17')
    parser.add_argument('--input-irreps', type=str, default=None)
    parser.add_argument('--radius', type=float, default=8.0)
    parser.add_argument('--num-basis', type=int, default=128)
    # training hyper-parameters
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=24)
    # regularization
    parser.add_argument('--drop-path', type=float, default=0.0)
    # optimizer (timm)
    parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adam"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='weight decay (default: 5e-3)')

    # learning rate schedule parameters (timm)
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine")')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.0, metavar='PERCENT',
                        help='learning rate noise limit percent (set to 0.0 for off)')
    parser.add_argument('--lr-noise-std', type=float, default=0.0, metavar='STDDEV',
                        help='learning rate noise std-dev (set to 0.0 for off)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--decay-epochs', type=float, default=0, metavar='N',
                        help='not used for cosine scheduler')
    parser.add_argument('--decay-rate', '--dr', type=float, default=1.0, metavar='RATE',
                        help='not used for cosine scheduler')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--cooldown-epochs', type=int, default=0, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=0, metavar='N',
                        help='not used for cosine scheduler')

    # logging
    parser.add_argument("--print-freq", type=int, default=20)
    # task and dataset
    parser.add_argument("--target", type=str, default='hamiltonian')
    parser.add_argument("--target-blocks-type", type=str, default='all')
    parser.add_argument("--no-parity", action='store_true')
    parser.add_argument("--convert-net-out", action='store_true')
    parser.add_argument("--data-path", type=str, default='datasets/md17')
    parser.add_argument("--weakdata-path", type=str, default='datasets/md17')
    parser.add_argument("--data-ratio", type=float, default=0.1)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument(
        "--train-subsample-ratio",
        type=float,
        default=1.0,
        help="Each training epoch, randomly keep this fraction of batches (1.0 = full epoch).",
    )
    parser.add_argument(
        "--val-subsample-ratio",
        type=float,
        default=1.0,
        help="Each epoch, randomly keep this fraction of validation batches (1.0 = full val pass).",
    )
    parser.add_argument(
        "--test-subsample-ratio",
        type=float,
        default=1.0,
        help="When the test loader is evaluated, fraction of test batches (1.0 = full pass).",
    )
    parser.add_argument("--is-accurate-label", action='store_true')
    parser.add_argument("--with-trace", action='store_true')
    parser.add_argument("--trace-out-len", type=int, default=25)
    parser.add_argument("--select-stru-id", type=int, default=-1)
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--num-models", type=int, default=4)
    parser.add_argument("--orbital-layout", type=str, default='ssssppddf',
                        help="Orbital basis layout string, e.g. ssssppddf (27) or sssppddf (26)")
    parser.add_argument('--spinful', dest='spinful', action='store_true',
                        help='Use spinful/SOC setting for Hamiltonian tensors')
    parser.add_argument('--no-spinful', dest='spinful', action='store_false',
                        help='Use spinless/non-SOC setting for Hamiltonian tensors')
    parser.set_defaults(spinful=True)
    parser.add_argument("--band-loss-weight", type=float, default=0.0)
    parser.add_argument("--use-precomputed-band-loss", action="store_true")
    parser.add_argument("--use-wa-loss-lmdb", action="store_true")
    parser.add_argument("--wa-k-grid", type=int, nargs=3, default=[4, 4, 4])
    parser.add_argument("--wa-spd-eps", type=float, default=1e-4)
    parser.add_argument("--wa-filter-tol", type=float, default=1e-12)

    parser.add_argument('--compute-stats', action='store_true', dest='compute_stats')
    parser.set_defaults(compute_stats=False)
    parser.add_argument('--test-interval', type=int, default=10, 
                        help='epoch interval to evaluate on the testing set')
    parser.add_argument('--test-max-iter', type=int, default=1000, 
                        help='max iteration to evaluate on the testing set')

    # random
    parser.add_argument("--seed", type=int, default=1)
    # data loader config
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    # evaluation
    parser.add_argument('--checkpoint-path1', type=str, default=None)
    parser.add_argument('--checkpoint-path2', type=str, default=None)
    parser.add_argument('--checkpoint-path3', type=str, default=None)
    parser.add_argument('--checkpoint-path4', type=str, default=None)
    parser.add_argument('--checkpoint-paths', type=str, nargs='*', default=None,
                        help='Optional list of checkpoint paths overriding checkpoint-path1..4')
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='If set, load training state from output_dir/train_state.pth.tar when that file exists.',
    )
    parser.add_argument(
        '--resume-state-path',
        type=str,
        default='',
        help='Explicit path to train_state.pth.tar (overrides --auto-resume default location).',
    )
    parser.add_argument(
        '--auto-split',
        action='store_true',
        help='Build train/val/test list files under data-path from samples/*.pth before training.',
    )
    parser.add_argument(
        '--split-tag',
        type=str,
        default='',
        help='If set, read/write train_<tag>.txt / val_<tag>.txt / test_<tag>.txt instead of train.txt etc.',
    )
    parser.add_argument(
        '--split-mode',
        type=str,
        default='random',
        choices=['random', 'ood_atom_count'],
        help='How to assign samples when --auto-split is used.',
    )
    parser.add_argument('--split-train-ratio', type=float, default=0.8)
    parser.add_argument('--split-val-ratio', type=float, default=0.1)
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument(
        '--split-force-overwrite',
        action='store_true',
        help='Regenerate split txts even if they already exist.',
    )
    parser.add_argument(
        '--split-ood-atom-cutoff',
        type=int,
        default=33,
        help='OOD mode: samples with atom count < cutoff go to train; larger structures go to val/test.',
    )
    parser.add_argument(
        '--split-ood-val-ratio',
        type=float,
        default=0.5,
        help='OOD mode: among OOD pool, fraction assigned to val (rest to test).',
    )

    parser.add_argument('--evaluate', action='store_true', dest='evaluate')
    parser.set_defaults(evaluate=False)
    return parser 

def reverse_transform_matrix(tensor, ls):
    # 获取原始通道数
    C = tensor.shape[0]
    # 计算原始张量的高度和宽度（sum(ls)）
    total_HW = sum(ls)
    # 初始化原始形状的张量
    original = torch.zeros((C, total_HW, total_HW), dtype=tensor.dtype, device=tensor.device)
    total_idx = 0 
    a = 0
    for i in ls:
        b = 0
        for j in ls:
            original[:, a:a+i, b:b+j] = tensor[:, total_idx:total_idx+i*j].reshape((C, i, j))
            b += j
            total_idx += i*j
        a += i
    return original


def build_distance_ranges(num_models: int, max_dist: float = 6.0):
    if num_models <= 0:
        return []
    step = max_dist / num_models
    return [[i * step, (i + 1) * step] for i in range(num_models)]

def convert_label_with_overlap(pred_h, label, overlap):
    Denominator = torch.sum(overlap * torch.conj(overlap))
    Numerator =  torch.real(torch.sum((pred_h-label) * torch.conj(overlap)))
    delta_mu = Numerator/(Denominator+1e-6)
    new_label = label + delta_mu*overlap
    return new_label


class MaskedMAELoss(torch.nn.Module):
    def __init__(self, threshold_max=10000, threshold_min=-10000, factor=1.0):
        super(MaskedMAELoss, self).__init__()
        self.mae_loss = torch.nn.L1Loss(reduction='none')
        self.threshold_max = threshold_max
        self.threshold_min = threshold_min
        self.factor = factor

    def forward(self, input, target, mask):
        loss = self.mae_loss(input, target)
        threshold_mask = ((self.threshold_min < target.abs()) & (target.abs() < self.threshold_max)).float()
        combined_mask = mask * threshold_mask
        loss = loss * combined_mask * self.factor
        masked_loss = loss.sum() / (combined_mask.sum() + 1e-6) 
        return masked_loss
    

class MaskedMAELosswithGuage(torch.nn.Module):
    def __init__(self, threshold_max=100000000, threshold_min=-100000000, factor=1.0):
        super(MaskedMAELosswithGuage, self).__init__()
        self.mae_loss = torch.nn.L1Loss(reduction='none')
        self.threshold_max = threshold_max
        self.threshold_min = threshold_min
        self.factor = factor

    def forward(self, input, target, overlap, mask):
        target = convert_label_with_overlap(input, target, overlap)
        loss = self.mae_loss(input, target)
        threshold_mask = ((self.threshold_min < target.abs()) & (target.abs() < self.threshold_max)).float()
        combined_mask = mask * threshold_mask
        loss = loss * combined_mask * self.factor
        combined_mask_sum = combined_mask.sum()
        masked_loss = loss.sum() / (combined_mask_sum+1e-7)
        return combined_mask_sum, masked_loss.real

class MaskedWALoss_Guage(torch.nn.Module):
    def __init__(self, threshold_max=10000, threshold_min=-10000, factor_pspace = 0.0002, factor_qspace = 0.0001, factor_overlap = 0.00015, unify_orb_num=None):
        super(MaskedWALoss_Guage, self).__init__()
        self.wa_loss =  torch.nn.L1Loss(reduction='none')
        self.threshold_max = threshold_max
        self.threshold_min = threshold_min
        self.factor_pspace = factor_pspace
        self.factor_qspace = factor_qspace
        self.factor_overlap = factor_overlap
        self.unify_orb_num = 27 * 2 if unify_orb_num is None else unify_orb_num
    
    def _switch_wa_loss_mse(self):
        self.wa_loss = torch.nn.MSELoss(reduction='none')

    def _switch_wa_loss_mae(self):
        self.wa_loss = torch.nn.L1Loss(reduction='none')

    def get_R_list(self, edge_vec, edge_src, edge_dst, lattice_vector, atoms_positions):
        self.pair_num = edge_dst.shape[0]
        cell_inv = torch.inverse(torch.transpose(lattice_vector,0,1))
        threshold = 0.001
        self.R_tot_list = []
        self.tot_num = atoms_positions.shape[0]
        for ii in range(self.pair_num):
            # 注意：edge_src[ii] 与 edge_dst[ii]若为单个整数，直接作为索引使用即可
            posit_ii = atoms_positions[edge_src[ii]]
            posit_jj = atoms_positions[edge_dst[ii]]
            # 计算两原子之间的位移差，再与 edge_vec 修正后获得差值向量
            R_dis = edge_vec[ii] - (posit_jj - posit_ii)
            # 用 cell_inv 与 R_dis 得到晶胞内的坐标差
            R_temp = cell_inv @ R_dis
            # 四舍五入得到最近整数晶胞平移向量
            R_tot = torch.round(R_temp)
            diff = torch.abs(R_tot - R_temp)
            # 当任一分量的差值超过阈值，则报错退出
            if torch.any(diff > threshold):
                print("转换数据出错，请检查结构或者输入数据")
                import sys
                sys.exit(2)
            self.R_tot_list.append(R_tot)
        return self.R_tot_list
    
    def divide_space(self, tot_kunm_torch, tot_basis_num_torch, kpt_data, band_cut_index, eigenvectors_enlager_torch):
        # 随机选取一个 k 点索引
        rand_kpt_index = torch.randint(low=0, high=tot_kunm_torch, size=(1,)).item()
        # 获取该 k 点的坐标
        kpt_coord = kpt_data[rand_kpt_index]  # shape: [3]
        # 获取波函数数据
        nbasis = eigenvectors_enlager_torch.shape[1]
        eigenvectors_recovered = eigenvectors_enlager_torch.view(tot_kunm_torch, tot_basis_num_torch, nbasis)
        # 随机获取波函数信息并分为 P 和 Q 两部分空间
        eigenvectors_P_space = eigenvectors_recovered[rand_kpt_index, :band_cut_index, :]
        eigenvectors_Q_space = eigenvectors_recovered[rand_kpt_index, band_cut_index:, :]

        return kpt_coord, eigenvectors_P_space, eigenvectors_Q_space 

    def cal_wfc_hk_vectorized(self, edge_src, edge_dst, hr_matrix, kpt_coord, eigenvectors_P_space, eigenvectors_Q_space):
        device = hr_matrix.device

        # 1) 堆成 (pair_count, 3) 的 R_tot 张量
        R_tot_tensor = torch.stack(self.R_tot_list, dim=0).to(device)      # (pair_count, 3)
        pair_count   = R_tot_tensor.shape[0]

        # 2) 基本尺寸
        orb_per_atom     = self.unify_orb_num                             # 每原子轨道数
        total_atom_count = self.tot_num                                   # 原子总数
        total_kpoints    = 1
        matrix_dim       = orb_per_atom * total_atom_count               # hk 矩阵维度

        # 3) 准备 kpt_data 和 hr_matrix
        kpt_tensor = kpt_coord.to(device).float()                         # (total_kpoints, 3)
        hr_tensor  = hr_matrix.to(device).view(pair_count, orb_per_atom, orb_per_atom)  # (pair_count, orb, orb)

        # 4) 计算 phase_factors：exp(2πi k·R)
        dot_products  = kpt_tensor @ R_tot_tensor.t()                     # (total_kpoints, pair_count)
        phase_factors = torch.exp(2j * torch.pi * dot_products)          # (total_kpoints, pair_count)

        # 5) 生成每个 k, pair 的贡献 contrib
        hr_expand    = hr_tensor.unsqueeze(0)                            # (1, pair_count, orb, orb)
        phase_expand = phase_factors.view(total_kpoints, pair_count, 1, 1) 
        contrib      = phase_expand * hr_expand                           # (total_kpoints, pair_count, orb, orb)
        contrib_flat = contrib.reshape(total_kpoints, -1)                 # (total_kpoints, pair_count*orb*orb)

        # 6) 计算扁平化索引，用于 scatter_add
        idx_local = torch.arange(orb_per_atom, device=device)
        row_local = idx_local.view(orb_per_atom,1).expand(orb_per_atom,orb_per_atom)
        col_local = idx_local.view(1,orb_per_atom).expand(orb_per_atom,orb_per_atom)

        start_row = (edge_src.to(device) * orb_per_atom).view(pair_count,1,1)
        start_col = (edge_dst.to(device) * orb_per_atom).view(pair_count,1,1)

        global_row_idx = (start_row + row_local).reshape(-1)  # (pair_count*orb*orb,)
        global_col_idx = (start_col + col_local).reshape(-1)

        flat_index   = global_row_idx * matrix_dim + global_col_idx
        index_tensor = flat_index.unsqueeze(0).expand(total_kpoints, -1)  # (total_kpoints, pair_count*orb*orb)

        # 7) scatter_add 到 flat hk 矩阵
        hk_flat = torch.zeros((total_kpoints, matrix_dim * matrix_dim),
                            dtype=torch.complex64,
                            device=device)
        hk_flat.scatter_add_(1, index_tensor, contrib_flat)

        # 8) reshape 回 (total_kpoints, matrix_dim, matrix_dim)
        hk_matrix = hk_flat.view(total_kpoints, matrix_dim, matrix_dim)

        # 9) 计算 reduce_space
        eigenvectors_P_space = eigenvectors_P_space.unsqueeze(0)
        eigenvectors_Q_space = eigenvectors_Q_space.unsqueeze(0)
        reduce_P_space  = eigenvectors_P_space.conj() @ hk_matrix @ eigenvectors_P_space.transpose(1, 2)
        reduce_Q_space  = eigenvectors_Q_space.conj() @ hk_matrix @ eigenvectors_Q_space.transpose(1, 2)
        reduce_PQ_space = eigenvectors_P_space.conj() @ hk_matrix @ eigenvectors_Q_space.transpose(1, 2)
        return reduce_P_space, reduce_Q_space, reduce_PQ_space 

    def grep_min_mu(self,  reduce_P_space1, reduce_Q_space1,  
                           reduce_P_space2, reduce_Q_space2, 
                           H_gt, pred_H, overlap, mask_tensor):
        
        # 由于实空间 H(R) 矩阵与k空间 H(k) 矩阵mae定义的角度不一致，两者进行混合时需调节有效的factor参数
        self.factor_R = 1 - self.factor_pspace - self.factor_qspace - self.factor_overlap

        # 计算总的矩阵数目
        self.N_number1 = torch.sum(mask_tensor)
        self.N_number2 = reduce_P_space1.shape[0] * reduce_P_space1.shape[1] * reduce_P_space1.shape[2]
        self.N_number3 = reduce_Q_space1.shape[0] * reduce_Q_space1.shape[1] * reduce_Q_space1.shape[2]

        # H(R)的贡献
        n1 = self.factor_R * torch.real(torch.sum((pred_H - H_gt) * torch.conj(overlap)))/self.N_number1
        d1 = self.factor_R * torch.real(torch.sum(overlap * torch.conj(overlap)))/self.N_number1

        # P空间的贡献
        reduce_P_space =  reduce_P_space2 - reduce_P_space1
        n2 = self.factor_pspace * torch.real(reduce_P_space.diagonal(dim1=1, dim2=2).sum())/self.N_number2 
        d2 = self.factor_pspace / reduce_P_space1.shape[1]

        # Q空间的贡献
        reduce_Q_space =  reduce_Q_space2 - reduce_Q_space1
        n3 = self.factor_qspace * torch.real(reduce_Q_space.diagonal(dim1=1, dim2=2).sum())/self.N_number3
        d3 = self.factor_qspace / reduce_Q_space1.shape[1]

        # PQ空间耦合与mu值无关

        # 计算mu值
        Numerator = n1 + n2 + n3
        Denominator = d1 + d2 + d3
        self.mu = Numerator/Denominator  
        # print(d1,d2,d3)
        return self.mu
    
    def cal_loss(self, 
                reduce_P_space1, reduce_Q_space1, reduce_PQ_space1,
                reduce_P_space2, reduce_Q_space2, reduce_PQ_space2,
                H_gt, pred_H, overlap):

        try:
            self.mu = float(self.mu)
        except:
            print('mu except!')
            self.mu = 0.0
          
        # 获取单位矩阵
        eye_matrix_p = torch.eye(reduce_P_space1.shape[1]).to(H_gt.device)          
        eye_batch_p = eye_matrix_p.unsqueeze(0).repeat(reduce_P_space1.shape[0], 1, 1).float().to(H_gt.device)
        eye_matrix_q = torch.eye(reduce_Q_space1.shape[1]).to(H_gt.device)
        eye_batch_q = eye_matrix_q.unsqueeze(0).repeat(reduce_Q_space1.shape[0], 1, 1).float().to(H_gt.device)

        # 计算 mae
        loss_hr = self.wa_loss(torch.real(H_gt + self.mu * overlap).float(), torch.real(pred_H).float())

        loss_p_space_real = self.wa_loss(torch.real(reduce_P_space1).float() + self.mu*eye_batch_p, torch.real(reduce_P_space2).float())
        loss_p_space_imag = self.wa_loss(torch.imag(reduce_P_space1).float(), torch.imag(reduce_P_space2).float())

        loss_q_space_real = self.wa_loss(torch.real(reduce_Q_space1).float() + self.mu*eye_batch_q, torch.real(reduce_Q_space2).float())
        loss_q_space_imag = self.wa_loss(torch.imag(reduce_Q_space1).float(), torch.imag(reduce_Q_space2).float())

        loss_pq_space_real = self.wa_loss(torch.real(reduce_PQ_space1).float(), torch.real(reduce_PQ_space2).float())
        loss_pq_space_imag = self.wa_loss(torch.imag(reduce_PQ_space1).float(), torch.imag(reduce_PQ_space2).float())

        self.N_number4 = reduce_PQ_space1.shape[0] * reduce_PQ_space1.shape[1] * reduce_PQ_space1.shape[2]
        tot_loss = self.factor_R * loss_hr.sum() /self.N_number1 + \
                   self.factor_pspace * (loss_p_space_real.sum() + loss_p_space_imag.sum()) / self.N_number2 + \
                   self.factor_qspace * (loss_q_space_real.sum() + loss_q_space_imag.sum()) / self.N_number3 + \
                   self.factor_overlap * (loss_pq_space_real.sum() + loss_pq_space_imag.sum()) / self.N_number4


        return tot_loss


class WALoss_LMDB(torch.nn.Module):
    """WA loss that reconstructs k-space terms from LMDB-native tensors."""

    def __init__(self, factor_pspace=0.0002, factor_qspace=0.0001, factor_overlap=0.00015, k_grid=(4, 4, 4), spd_eps=1e-4, filter_tol=1e-12):
        super(WALoss_LMDB, self).__init__()
        self.wa_loss = torch.nn.L1Loss(reduction='none')
        self.factor_pspace = factor_pspace
        self.factor_qspace = factor_qspace
        self.factor_overlap = factor_overlap
        self.k_grid = k_grid
        self.spd_eps = spd_eps
        self.filter_tol = filter_tol
        self.mu = 0.0

    def _switch_wa_loss_mse(self):
        self.wa_loss = torch.nn.MSELoss(reduction='none')

    def _switch_wa_loss_mae(self):
        self.wa_loss = torch.nn.L1Loss(reduction='none')

    def _sample_kpoints(self, device, dtype):
        gx, gy, gz = self.k_grid
        pts = torch.tensor(
            [[i / gx, j / gy, k / gz] for i in range(gx) for j in range(gy) for k in range(gz)],
            device=device, dtype=dtype
        )
        if pts.shape[0] <= 1:
            return pts
        ridx = torch.randint(1, pts.shape[0], (1,), device=device)
        return torch.cat([pts[:1], pts[ridx]], dim=0)

    def _get_R(self, edge_vec, edge_src, edge_dst, lattice_vector, atoms_positions):
        cell_inv = torch.inverse(torch.transpose(lattice_vector, 0, 1))
        pos_i = atoms_positions[edge_src]
        pos_j = atoms_positions[edge_dst]
        r_dis = edge_vec - (pos_j - pos_i)
        return torch.round((cell_inv @ r_dis.T).T)

    def _build_hk(self, hr, edge_src, edge_dst, r_tot, kpts, matrix_dim):
        pair_count = hr.shape[0]
        orb = hr.shape[1]
        nk = kpts.shape[0]
        phase = torch.exp(2j * torch.pi * (kpts @ r_tot.T))
        hk = torch.zeros((nk, matrix_dim, matrix_dim), dtype=torch.complex64, device=hr.device)
        for e in range(pair_count):
            i = int(edge_src[e].item())
            j = int(edge_dst[e].item())
            si, ei = i * orb, (i + 1) * orb
            sj, ej = j * orb, (j + 1) * orb
            h_block = hr[e].to(torch.complex64)
            if i == j and torch.all(r_tot[e] == 0):
                # On-site: enforce Hermitian contribution split as in evaluator.
                h_sym = 0.5 * (h_block + h_block.mH)
                hk[:, si:ei, sj:ej] += h_sym.unsqueeze(0)
            else:
                ph = phase[:, e].view(-1, 1, 1)
                hk[:, si:ei, sj:ej] += h_block.unsqueeze(0) * ph
        return hk

    def _filter_zero_rows_cols(self, mat):
        # mat shape [nk, n, n]
        row_norms = torch.sum(torch.abs(mat), dim=(0, -1))
        col_norms = torch.sum(torch.abs(mat), dim=(0, -2))
        nonzero = (row_norms > self.filter_tol) & (col_norms > self.filter_tol)
        if torch.sum(nonzero) == mat.shape[-1]:
            return mat, nonzero
        return mat[:, nonzero][:, :, nonzero], nonzero

    def _has_valid_precomputed_split(self, tot_kunm, tot_basis_num, band_cut_index, kpt_data, eigenvectors_enlager):
        if any(v is None for v in [tot_kunm, tot_basis_num, band_cut_index, kpt_data, eigenvectors_enlager]):
            return False
        if kpt_data.numel() == 0 or eigenvectors_enlager.numel() == 0:
            return False
        tk = int(tot_kunm.item()) if torch.is_tensor(tot_kunm) else int(tot_kunm)
        tb = int(tot_basis_num.item()) if torch.is_tensor(tot_basis_num) else int(tot_basis_num)
        bc = int(band_cut_index.item()) if torch.is_tensor(band_cut_index) else int(band_cut_index)
        if tk <= 0 or tb <= 1 or bc <= 0 or bc >= tb:
            return False
        if kpt_data.dim() != 2 or kpt_data.shape[0] < tk or kpt_data.shape[1] != 3:
            return False
        if eigenvectors_enlager.dim() != 2 or eigenvectors_enlager.shape[0] != tk * tb:
            return False
        # Placeholder heuristics from lmdb_to_nextham_pth.py:
        # kpt == [[0,0,0]], tot_knum == 1, eigenvectors == identity.
        if tk == 1:
            if torch.allclose(kpt_data[0].to(torch.float32), torch.zeros(3, device=kpt_data.device), atol=1e-7):
                nb = eigenvectors_enlager.shape[1]
                if nb == tb:
                    eye = torch.eye(tb, device=eigenvectors_enlager.device, dtype=eigenvectors_enlager.dtype)
                    if torch.allclose(eigenvectors_enlager, eye, atol=1e-6, rtol=1e-5):
                        return False
        return True

    def compute_loss(
        self,
        H_gt,
        H_pred,
        overlap,
        mask,
        edge_vec,
        edge_src,
        edge_dst,
        lattice_vector,
        atoms_positions,
        tot_kunm=None,
        tot_basis_num=None,
        band_cut_index=None,
        kpt_data=None,
        eigenvectors_enlager=None,
    ):
        total_atoms = atoms_positions.shape[0]
        orb = H_gt.shape[-1]
        matrix_dim = total_atoms * orb
        r_tot = self._get_R(edge_vec, edge_src, edge_dst, lattice_vector, atoms_positions).to(dtype=torch.float32)
        use_precomputed_split = self._has_valid_precomputed_split(
            tot_kunm, tot_basis_num, band_cut_index, kpt_data, eigenvectors_enlager
        )
        if use_precomputed_split:
            tk = int(tot_kunm.item()) if torch.is_tensor(tot_kunm) else int(tot_kunm)
            tb = int(tot_basis_num.item()) if torch.is_tensor(tot_basis_num) else int(tot_basis_num)
            bc = int(band_cut_index.item()) if torch.is_tensor(band_cut_index) else int(band_cut_index)
            ridx = torch.randint(low=0, high=tk, size=(1,), device=H_gt.device).item()
            kpts = kpt_data[ridx:ridx + 1].to(H_gt.device, dtype=torch.float32)
            nb = eigenvectors_enlager.shape[1]
            ev = eigenvectors_enlager.to(H_gt.device).reshape(tk, tb, nb)
            cp = ev[ridx, :bc, :].unsqueeze(0).to(torch.complex64)
            cq = ev[ridx, bc:, :].unsqueeze(0).to(torch.complex64)
        else:
            kpts = self._sample_kpoints(H_gt.device, torch.float32)
            cp = cq = None
        hk_gt = self._build_hk(H_gt, edge_src, edge_dst, r_tot, kpts, matrix_dim)
        hk_pd = self._build_hk(H_pred, edge_src, edge_dst, r_tot, kpts, matrix_dim)
        sk = self._build_hk(overlap, edge_src, edge_dst, r_tot, kpts, matrix_dim)
        # Enforce Hermitian matrices before decomposition.
        hk_gt = 0.5 * (hk_gt + hk_gt.transpose(-1, -2).conj())
        hk_pd = 0.5 * (hk_pd + hk_pd.transpose(-1, -2).conj())
        sk = 0.5 * (sk + sk.transpose(-1, -2).conj())
        # Filter singular/empty subspace.
        sk, nonzero = self._filter_zero_rows_cols(sk)
        hk_gt = hk_gt[:, nonzero][:, :, nonzero]
        hk_pd = hk_pd[:, nonzero][:, :, nonzero]
        matrix_dim_eff = sk.shape[-1]
        if matrix_dim_eff <= 1:
            self.mu = 0.0
            loss_simple = self.wa_loss(torch.real(H_gt).float(), torch.real(H_pred).float()).mean()
            return loss_simple.real, self.mu

        if cp is not None and cq is not None:
            # Original-like: use precomputed eigenvectors/band cut split if available.
            p_gt = cp.conj() @ hk_gt @ cp.transpose(1, 2)
            q_gt = cq.conj() @ hk_gt @ cq.transpose(1, 2)
            pq_gt = cp.conj() @ hk_gt @ cq.transpose(1, 2)
            p_pd = cp.conj() @ hk_pd @ cp.transpose(1, 2)
            q_pd = cq.conj() @ hk_pd @ cq.transpose(1, 2)
            pq_pd = cp.conj() @ hk_pd @ cq.transpose(1, 2)
        else:
            # Fallback approximation when precomputed split is missing/placeholder.
            cut = max(1, matrix_dim_eff // 2)
            p_gt = hk_gt[:, :cut, :cut]
            q_gt = hk_gt[:, cut:, cut:]
            pq_gt = hk_gt[:, :cut, cut:]
            p_pd = hk_pd[:, :cut, :cut]
            q_pd = hk_pd[:, cut:, cut:]
            pq_pd = hk_pd[:, :cut, cut:]

        factor_R = 1 - self.factor_pspace - self.factor_qspace - self.factor_overlap
        msum = torch.sum(mask) + 1e-8
        n1 = factor_R * torch.real(torch.sum((H_pred - H_gt) * torch.conj(overlap))) / msum
        d1 = factor_R * torch.real(torch.sum(overlap * torch.conj(overlap))) / msum
        n2 = self.factor_pspace * torch.real((p_pd - p_gt).diagonal(dim1=1, dim2=2).sum()) / (p_gt.numel() + 1e-8)
        n3 = self.factor_qspace * torch.real((q_pd - q_gt).diagonal(dim1=1, dim2=2).sum()) / (q_gt.numel() + 1e-8)
        d2 = self.factor_pspace / max(1, p_gt.shape[1])
        d3 = self.factor_qspace / max(1, q_gt.shape[1])
        self.mu = (n1 + n2 + n3) / (d1 + d2 + d3 + 1e-8)

        ep = torch.eye(p_gt.shape[1], device=H_gt.device)[None, :, :].repeat(p_gt.shape[0], 1, 1)
        eq = torch.eye(q_gt.shape[1], device=H_gt.device)[None, :, :].repeat(q_gt.shape[0], 1, 1)
        loss_hr = self.wa_loss(torch.real(H_gt + self.mu * overlap).float(), torch.real(H_pred).float())
        loss_p = self.wa_loss(torch.real(p_gt).float() + self.mu * ep, torch.real(p_pd).float()) + self.wa_loss(torch.imag(p_gt).float(), torch.imag(p_pd).float())
        if q_gt.shape[1] > 0:
            loss_q = self.wa_loss(torch.real(q_gt).float() + self.mu * eq, torch.real(q_pd).float()) + self.wa_loss(torch.imag(q_gt).float(), torch.imag(q_pd).float())
        else:
            loss_q = torch.zeros_like(loss_p[:, :1, :1])
        loss_pq = self.wa_loss(torch.real(pq_gt).float(), torch.real(pq_pd).float()) + self.wa_loss(torch.imag(pq_gt).float(), torch.imag(pq_pd).float())
        total = factor_R * loss_hr.mean() + self.factor_pspace * loss_p.mean() + self.factor_qspace * loss_q.mean() + self.factor_overlap * loss_pq.mean()
        return total.real, self.mu


def compute_precomputed_band_loss(
    H_pred,
    edge_src,
    edge_dst,
    edge_r,
    band_kpoints,
    band_eps_ref,
    band_c_ref,
    band_active,
    band_mask,
    matrix_orbitals,
    node_num,
):
    total_orb = node_num * matrix_orbitals
    nk = band_kpoints.shape[0]
    hk = torch.zeros((nk, total_orb, total_orb), dtype=torch.complex64, device=H_pred.device)
    for e in range(H_pred.shape[0]):
        i = int(edge_src[e].item())
        j = int(edge_dst[e].item())
        r = edge_r[e]
        phase = torch.exp(-1j * 2 * torch.pi * (band_kpoints * r.unsqueeze(0)).sum(dim=-1))
        si, ei = i * matrix_orbitals, (i + 1) * matrix_orbitals
        sj, ej = j * matrix_orbitals, (j + 1) * matrix_orbitals
        hk[:, si:ei, sj:ej] += H_pred[e].to(torch.complex64).unsqueeze(0) * phase.view(-1, 1, 1)
    hk = hk[:, band_active][:, :, band_active]
    eps_approx = torch.einsum('bij,bjk,bki->bi', band_c_ref.transpose(-1, -2).conj(), hk, band_c_ref).real
    diff = (eps_approx - band_eps_ref) * band_mask.to(band_eps_ref.dtype)
    return torch.sqrt((diff.square()).mean() + 1e-12)
    

class AttributeDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"No such attribute: {name}")

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"No such attribute: {name}")


# from https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/modules/loss.py#L7
class L2MAELoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        assert reduction in ["mean", "sum"]

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        dists = torch.norm(input - target, p=2, dim=-1)
        if self.reduction == "mean":
            return torch.mean(dists)
        elif self.reduction == "sum":
            return torch.sum(dists)

class Material_Project_Dataset(torch.utils.data.Dataset):
    def __init__(self, mode, construct_kernel, device, dataset_root='/your_path/NextHAM/datasets/', split_tag=''):
        super().__init__()
        self.mode = mode
        self.construct_kernel = construct_kernel
        self.samples = []
        self.label_norm_tensor = None
        self.descriptor_norm_tensor = None
        self.norm_mask_tensor = None
        time1 = time.time()
        tag_part = ('_' + split_tag.strip()) if (split_tag or '').strip() else ''
        dataset_file = open(dataset_root + mode + tag_part + '.txt', 'r')
        self.file_list = []
        for line in dataset_file.readlines():                          
            self.file_list.append(line.strip())
        print('total load time: ', time.time()-time1)
        print('len of self.samples: ', len(self.file_list))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        return torch.load(file_path, weights_only=True)
    
def save_sample(sample, save_dir, sample_name):
    """Save individual sample as a .pth file."""
    os.makedirs(save_dir, exist_ok=True)
    sample_path = os.path.join(save_dir, f"{sample_name}.pth")
    torch.save(sample, sample_path)  # Save using torch.save


def get_material_project_dataset(construct_kernel, device, dataset_root, split_tag=''):
    """Process and save datasets individually for train, val, test."""
    datasets = {}

    datasets["train"], datasets["val"], datasets["test"] = (
        Material_Project_Dataset('train', construct_kernel, device, dataset_root=dataset_root, split_tag=split_tag),
        Material_Project_Dataset('val', construct_kernel, device, dataset_root=dataset_root, split_tag=split_tag),
        Material_Project_Dataset('test', construct_kernel, device, dataset_root=dataset_root, split_tag=split_tag),
    )

    return datasets["train"], datasets["val"], datasets["test"]

def get_hamiltonian_size(args, spinful):
    orbital_types, orbital_block_sizes, num_orbitals = parse_orbital_layout(args.orbital_layout)
    matrix_orbitals = num_orbitals * (2 if spinful else 1)
    dataset_info = AttributeDict(
        spinful=spinful,
        index_to_Z=torch.Tensor([idx for idx in range(118)]).long(),
        Z_to_index=torch.Tensor([idx for idx in range(118)]).long(),
        orbital_types=[orbital_types]
    )
    _, _, net_out_irreps, net_out_info = config_set_target(dataset_info, args, verbose='target.txt')
    irreps_edge = net_out_irreps
    js = net_out_info.js
    spinful = dataset_info.spinful
    no_parity = args.no_parity
    if_sort = args.convert_net_out
    construct_kernel = e3TensorDecomp(irreps_edge, 
                                    js, 
                                    default_dtype_torch=torch.get_default_dtype(), 
                                    spinful=spinful,
                                    no_parity=no_parity, 
                                    if_sort=if_sort, 
                                    device_torch=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    return irreps_edge, construct_kernel, orbital_block_sizes, num_orbitals, matrix_orbitals

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)    
    torch.manual_seed(seed)    
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    
    os.environ["PYTHONHASHSEED"] = str(seed)


def _default_train_state_path(output_dir: str) -> str:
    return os.path.join(output_dir, 'train_state.pth.tar')


def _serialize_mu(mu):
    if mu is None:
        return None
    if torch.is_tensor(mu):
        return float(mu.detach().cpu().item())
    try:
        return float(mu)
    except (TypeError, ValueError):
        return None


def _apply_mu(module, val):
    if val is None or not hasattr(module, 'mu'):
        return
    module.mu = val


def save_training_state(
    path,
    epoch,
    best_val_err,
    best_metrics,
    models,
    optimizers,
    lr_schedulers,
    criterion,
    criterion_lmdb,
):
    sched_states = []
    for s in lr_schedulers:
        fn = getattr(s, 'state_dict', None)
        sched_states.append(fn() if callable(fn) else {})

    payload = {
        'epoch': int(epoch),
        'best_val_err': float(best_val_err),
        'best_metrics': dict(best_metrics),
        'model_states': [{k: v.cpu() for k, v in m.state_dict().items()} for m in models],
        'optimizer_states': [o.state_dict() for o in optimizers],
        'scheduler_states': sched_states,
        'criterion_mu': _serialize_mu(getattr(criterion, 'mu', None)),
        'criterion_lmdb_mu': _serialize_mu(getattr(criterion_lmdb, 'mu', None)),
        'torch_rng': torch.get_rng_state(),
        'numpy_rng': np.random.get_state(),
        'python_rng': random.getstate(),
    }
    tmp = path + '.tmp'
    torch.save(payload, tmp)
    os.replace(tmp, path)


def try_load_training_state(
    path,
    models,
    optimizers,
    lr_schedulers,
    criterion,
    criterion_lmdb,
):
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location='cpu')
    m_states = ckpt.get('model_states') or ckpt.get('models') or []
    for i, m in enumerate(models):
        if i < len(m_states):
            m.load_state_dict(m_states[i], strict=False)
    o_states = ckpt.get('optimizer_states') or ckpt.get('optimizers') or []
    for i, o in enumerate(optimizers):
        if i < len(o_states):
            o.load_state_dict(o_states[i])
    sched_saved = ckpt.get('scheduler_states') or []
    for i, s in enumerate(lr_schedulers):
        if i < len(sched_saved) and sched_saved[i]:
            fn = getattr(s, 'load_state_dict', None)
            if callable(fn):
                try:
                    fn(sched_saved[i])
                except Exception:
                    pass
    _apply_mu(criterion, ckpt.get('criterion_mu'))
    _apply_mu(criterion_lmdb, ckpt.get('criterion_lmdb_mu'))
    try:
        torch.set_rng_state(ckpt['torch_rng'])
    except Exception:
        pass
    try:
        np.random.set_state(ckpt['numpy_rng'])
    except Exception:
        pass
    try:
        random.setstate(ckpt['python_rng'])
    except Exception:
        pass
    return {
        'epoch': int(ckpt['epoch']),
        'best_val_err': float(ckpt['best_val_err']),
        'best_metrics': ckpt.get('best_metrics')
        or {
            'val_epoch': 0,
            'test_epoch': 0,
            'val_ham_err': float('inf'),
            'val_trace_err': float('inf'),
            'test_ham_err': float('inf'),
            'test_trace_err': float('inf'),
        },
    }


def _collect_sample_pths(data_path: str):
    root = os.path.normpath(data_path)
    paths = sorted(glob.glob(os.path.join(root, 'samples', '*.pth')))
    if paths:
        return paths
    return sorted(glob.glob(os.path.join(root, '*.pth')))


def _atom_count_from_pth(path: str) -> int:
    obj = torch.load(path, weights_only=True)
    if isinstance(obj, (list, tuple)) and len(obj) > 2:
        t = obj[2]
        if torch.is_tensor(t) and t.dim() >= 1:
            return int(t.shape[0])
    raise ValueError(f'Cannot infer atom count (pos) from {path}')


def _split_paths_random(paths, train_ratio, val_ratio, seed):
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


def _split_paths_ood_atom(paths, seed, cutoff, ood_val_ratio):
    """Train = all samples with atom count < cutoff; val/test split the remainder by ood_val_ratio."""
    small, large = [], []
    for p in paths:
        try:
            n = _atom_count_from_pth(p)
        except Exception:
            large.append(p)
            continue
        (small if n < cutoff else large).append(p)
    rng = random.Random(seed)
    rng.shuffle(large)
    n_val = int(len(large) * float(ood_val_ratio))
    val = large[:n_val]
    test = large[n_val:]
    return small, val, test


def _write_split_lines(path, entries):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for p in entries:
            f.write(p + '\n')


def maybe_auto_split(args):
    if not getattr(args, 'auto_split', False):
        return
    dataset_root = args.data_path
    if not dataset_root.endswith('/'):
        dataset_root = dataset_root + '/'
    tag = (args.split_tag or '').strip()
    suffix = f'_{tag}' if tag else ''
    train_p = os.path.join(dataset_root, f'train{suffix}.txt')
    val_p = os.path.join(dataset_root, f'val{suffix}.txt')
    test_p = os.path.join(dataset_root, f'test{suffix}.txt')
    if (
        os.path.isfile(train_p)
        and os.path.isfile(val_p)
        and os.path.isfile(test_p)
        and not getattr(args, 'split_force_overwrite', False)
    ):
        print(f'[auto-split] Using existing {train_p} / {val_p} / {test_p}')
        return
    paths = _collect_sample_pths(dataset_root)
    if not paths:
        raise FileNotFoundError(
            f'[auto-split] No sample .pth files under {dataset_root}samples/ or {dataset_root}'
        )
    if args.split_mode == 'random':
        train, val, test = _split_paths_random(
            paths, args.split_train_ratio, args.split_val_ratio, args.split_seed
        )
    else:
        train, val, test = _split_paths_ood_atom(
            paths, args.split_seed, args.split_ood_atom_cutoff, args.split_ood_val_ratio
        )
    _write_split_lines(train_p, train)
    _write_split_lines(val_p, val)
    _write_split_lines(test_p, test)
    print(
        f'[auto-split] Wrote {len(train)} train / {len(val)} val / {len(test)} test -> '
        f'{train_p}, {val_p}, {test_p}'
    )


def main(args):
    
    _log = FileLogger(is_master=True, is_rank0=True, output_dir=args.output_dir)
    _log.info(args)
    

    ''' Config '''
    irreps_edge, construct_kernel, orbital_block_sizes, num_orbitals, matrix_orbitals = get_hamiltonian_size(args, spinful=args.spinful)
    args.orbital_block_sizes = orbital_block_sizes
    args.num_orbitals = num_orbitals
    args.matrix_orbitals = matrix_orbitals

    mean = 0.
    std = 1. 

    # since dataset needs random 
    set_seed(args.seed)
    maybe_auto_split(args)

    ''' Network '''
    create_model = model_entrypoint(args.model_name)

    if args.num_models < 1:
        raise ValueError("--num-models must be >= 1")
    devices = [[i] for i in range(args.num_models)]

    models = []

    for model_idx in range(args.num_models):
        models.append(create_model(irreps_in=args.input_irreps, irreps_edge=irreps_edge,
            radius=args.radius, 
            num_basis=args.num_basis, 
            task_mean=mean, 
            task_std=std, 
            atomref=None,
            start_layer=args.start_layer,
            drop_path_rate=args.drop_path,
            with_trace=args.with_trace,
            trace_out_len=args.trace_out_len,
            use_w2v=False,
            ).to(f'cuda:{devices[model_idx][0]}'))

    if args.checkpoint_paths:
        checkpoint_paths = list(args.checkpoint_paths)
    else:
        checkpoint_paths = [args.checkpoint_path1, args.checkpoint_path2, args.checkpoint_path3, args.checkpoint_path4]
    if len(checkpoint_paths) < args.num_models:
        checkpoint_paths.extend([None] * (args.num_models - len(checkpoint_paths)))
    checkpoint_paths = checkpoint_paths[:args.num_models]

    n_parameters = sum(p.numel() for p in models[0].parameters()) * len(models)
    _log.info('Number of params: {}'.format(n_parameters))
  
    ''' Dataset '''
    dataset_root = args.data_path
    if not dataset_root.endswith('/'):
        dataset_root = dataset_root + '/'
    train_dataset, val_dataset, test_dataset = get_material_project_dataset(
        construct_kernel=construct_kernel,
        device=devices[0][0],
        dataset_root=dataset_root,
        split_tag=(args.split_tag or '').strip(),
    )

    _log.info('')
    _log.info('Training set size:   {}'.format(len(train_dataset)))
    _log.info('Validation set size: {}'.format(len(val_dataset)))
    _log.info('Testing set size:    {}\n'.format(len(test_dataset)))

    ''' Data Loader '''
    from tg_src.graph import Collater
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=args.workers, pin_memory = True)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=args.workers, pin_memory = True)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=args.workers, pin_memory = True)

    ''' Optimizer and LR Scheduler '''
    optimizers = []
    lr_schedulers = []
    for model_idx in range(len(models)):
        params = list(filter(lambda p: p.requires_grad, models[model_idx].parameters()))
        optimizer_h = torch.optim.Adam(params, lr=args.lr, betas=(0.9, 0.999))
        optimizers.append(optimizer_h)
        lr_scheduler_h, _ = create_scheduler(args, optimizer_h)
        lr_schedulers.append(lr_scheduler_h)

    criterion = MaskedWALoss_Guage(unify_orb_num=args.matrix_orbitals) 
    criterion_lmdb = WALoss_LMDB(
        k_grid=tuple(args.wa_k_grid),
        spd_eps=args.wa_spd_eps,
        filter_tol=args.wa_filter_tol,
    )

    resume_path = None
    rp = getattr(args, 'resume_state_path', '') or ''
    if str(rp).strip():
        resume_path = str(rp).strip()
    elif getattr(args, 'auto_resume', False) and args.output_dir:
        cand = _default_train_state_path(args.output_dir)
        if os.path.isfile(cand):
            resume_path = cand

    if resume_path and os.path.isfile(resume_path):
        info = try_load_training_state(
            resume_path, models, optimizers, lr_schedulers, criterion, criterion_lmdb
        )
        start_epoch = info['epoch']
        best_val_err = info['best_val_err']
        best_metrics = info['best_metrics']
        _log.info('Loaded training state from {} (next epoch index {})'.format(resume_path, start_epoch))
    else:
        start_epoch = 0
        best_val_err = 1000.0
        best_metrics = {
            'val_epoch': 0,
            'test_epoch': 0,
            'val_ham_err': float('inf'),
            'val_trace_err': float('inf'),
            'test_ham_err': float('inf'),
            'test_trace_err': float('inf'),
        }
        for model_idx in range(args.num_models):
            checkpoint_path = checkpoint_paths[model_idx]
            if checkpoint_path is not None:
                state_dict = torch.load(checkpoint_path, map_location='cpu')
                model_range_state_dict = models[model_idx].state_dict()
                compatible_state_dict = {
                    k: v
                    for k, v in state_dict['state_dict'].items()
                    if k in model_range_state_dict and v.size() == model_range_state_dict[k].size()
                }
                model_range_state_dict.update(compatible_state_dict)
                models[model_idx].load_state_dict(model_range_state_dict)
                print(
                    'model_idx, len(compatible_state_dict), len(model_range_state_dict): ',
                    model_idx,
                    len(compatible_state_dict),
                    len(model_range_state_dict),
                )
            else:
                print('no pre-trained model')

    ''' Compute stats '''
    if args.compute_stats:
        compute_stats(train_loader, max_radius=args.radius, logger=_log, print_freq=args.print_freq)
        return

    epoch = start_epoch

    while epoch < args.epochs:
        
        for model_idx in range(len(models)):
            lr_schedulers[model_idx].step(epoch)

        train_err = train_eval_one_epoch(args=args,  models=models, devices=devices, criterion=criterion, criterion_lmdb=criterion_lmdb, data_loader=train_loader, optimizers=optimizers, epoch=epoch, print_freq=args.print_freq, logger=_log, construct_kernel=construct_kernel)

        val_err = train_eval_one_epoch(args=args, models=models, devices=devices, criterion=criterion, criterion_lmdb=criterion_lmdb, data_loader=val_loader, optimizers=optimizers, epoch=epoch, print_freq=args.print_freq, logger=_log, construct_kernel=construct_kernel, train = False, print_progress=True)

        if val_err < best_val_err:
            best_val_err = val_err
            for model_idx in range(len(models)):
                torch.save(
                    {'state_dict': models[model_idx].state_dict()}, 
                    os.path.join(args.output_dir, 'model_range'+str(model_idx)+'_best.pth.tar')
                )

        epoch += 1
        if args.output_dir:
            save_training_state(
                _default_train_state_path(args.output_dir),
                epoch,
                best_val_err,
                best_metrics,
                models,
                optimizers,
                lr_schedulers,
                criterion,
                criterion_lmdb,
            )


def train_eval_one_epoch(args, 
                    models: list, 
                    devices: list, 
                    criterion: torch.nn.Module,
                    criterion_lmdb: torch.nn.Module,
                    data_loader: Iterable,
                    optimizers: list,
                    epoch: int, 
                    print_freq: int = 100, 
                    logger=None, construct_kernel = None, train = True,
                    print_progress=False):

    global ele_dict

    if train:
        for model_idx in range(len(models)):
            models[model_idx].train()
        if epoch > 130:
            criterion._switch_wa_loss_mse()
            criterion_lmdb._switch_wa_loss_mse()
        criterion.train()
        criterion_lmdb.train()
    else:
        for model_idx in range(len(models)):
            models[model_idx].eval()  
        criterion._switch_wa_loss_mae()
        criterion_lmdb._switch_wa_loss_mae()
        criterion.eval()
        criterion_lmdb.eval()

    loss_metrics = {'ham': AverageMeter(), 'trace': AverageMeter(), 'baseline_ham': AverageMeter()}
    mae_metrics  = {'ham': AverageMeter(), 'ham_lt_10': AverageMeter(), 'ham_10_100': AverageMeter(), 'ham_100_1000': AverageMeter(), 'ham_gt_1000': AverageMeter(), 'trace': AverageMeter(), 'baseline_ham': AverageMeter(), 'baseline_ham_lt_10': AverageMeter(), 'baseline_ham_10_100': AverageMeter(), 'baseline_ham_l00_1000': AverageMeter(), 'baseline_ham_gt_1000': AverageMeter(), 'ham_ratio': AverageMeter(), 'ham_lt_10_ratio': AverageMeter(), 'ham_10_100_ratio': AverageMeter(), 'ham_100_1000_ratio': AverageMeter(), 'ham_gt_1000_ratio': AverageMeter(), 'ham_on_site': AverageMeter(),  'ham_1_2': AverageMeter(), 'ham_2_4': AverageMeter(), 'ham_4_6': AverageMeter(),}
    loss_h_all = []
    mae_h_all = []
    
    sample_num = 0 
    start_time = time.perf_counter()

    MAE_metric = MaskedMAELosswithGuage()
    MAE_metric_lt_10 = MaskedMAELosswithGuage(threshold_max=0.01, threshold_min=-100000000)
    MAE_metric_10_100 = MaskedMAELosswithGuage(threshold_max=0.1, threshold_min=0.01)
    MAE_metric_100_1000 = MaskedMAELosswithGuage(threshold_max=1, threshold_min=0.1)
    MAE_metric_gt_1000 = MaskedMAELosswithGuage(threshold_max=100000000, threshold_min=1)

    criterion_trace = MaskedMAELoss()

    def _unwrap_symbol(x):
        while isinstance(x, (list, tuple)) and len(x) > 0:
            x = x[0]
        return x

    ls = args.orbital_block_sizes
    range_dis = build_distance_ranges(len(models), max_dist=6.0)

    subset_indices = None
    loader_len = None
    try:
        loader_len = len(data_loader)
    except TypeError:
        loader_len = None

    if train:
        tr = float(getattr(args, "train_subsample_ratio", 1.0))
        if loader_len is not None and tr < 1.0:
            if tr <= 0 or tr > 1.0:
                raise ValueError("--train-subsample-ratio must be in (0, 1].")
            k = max(1, int(loader_len * tr))
            rng = random.Random(int(args.seed) + int(epoch))
            subset_indices = set(rng.sample(range(loader_len), k))
            if logger is not None:
                logger.info(
                    "Train subsample active (epoch %d): keep %d/%d batches (%.3f).",
                    epoch,
                    k,
                    loader_len,
                    tr,
                )
    elif loader_len is not None:
        if print_progress:
            sr = float(getattr(args, "val_subsample_ratio", 1.0))
            label = "Val"
        else:
            sr = float(getattr(args, "test_subsample_ratio", 1.0))
            label = "Test"
        if sr < 1.0:
            if sr <= 0 or sr > 1.0:
                raise ValueError("--val-subsample-ratio and --test-subsample-ratio must be in (0, 1].")
            k = max(1, int(loader_len * sr))
            rng = random.Random(int(args.seed) + int(epoch) + (7 if print_progress else 13))
            subset_indices = set(rng.sample(range(loader_len), k))
            if logger is not None:
                logger.info(
                    "%s subsample active (epoch %d): keep %d/%d batches (%.3f).",
                    label,
                    epoch,
                    k,
                    loader_len,
                    sr,
                )

    for step, data in enumerate(data_loader):
        if subset_indices is not None and step not in subset_indices:
            continue
        (
            file_path, lattice_vector_torch, position_torch, tot_basis_num_torch, band_cut_index_torch,
            tot_kunm_torch, kpt_torch, eigenvectors_enlager_torch, H0_ds, H0, overlap_tensor, mask_tensor,
            edge_vec, edge_src, edge_dst, ele_list, mp_stru_name, delta_H_dp, H0_raw, overlap_tensor_raw,
            mask_tensor_raw, delta_H_raw, *extra_data
        ) = data
        file_path, lattice_vector_torch, position_torch, tot_basis_num_torch, band_cut_index_torch, tot_kunm_torch, kpt_torch, eigenvectors_enlager_torch, H0_ds, H0, overlap_tensor, mask_tensor, edge_vec, edge_src, edge_dst, delta_H_dp, H0_raw, overlap_tensor_raw, mask_tensor_raw, delta_H_raw = file_path[0], lattice_vector_torch[0].to(devices[0][0], non_blocking=True), position_torch[0].to(devices[0][0], non_blocking=True), tot_basis_num_torch[0].to(devices[0][0], non_blocking=True), band_cut_index_torch[0].to(devices[0][0], non_blocking=True), tot_kunm_torch[0].to(devices[0][0], non_blocking=True), kpt_torch[0].to(devices[0][0], non_blocking=True), eigenvectors_enlager_torch[0].to(devices[0][0], non_blocking=True), H0_ds[0].to(devices[0][0], non_blocking=True), H0[0].to(devices[0][0], non_blocking=True), overlap_tensor[0].to(devices[0][0], non_blocking=True), mask_tensor[0].to(devices[0][0], non_blocking=True), edge_vec[0].to(devices[0][0], non_blocking=True), edge_src.to(torch.int64)[0].to(devices[0][0], non_blocking=True), edge_dst.to(torch.int64)[0].to(devices[0][0], non_blocking=True), delta_H_dp[0].to(devices[0][0], non_blocking=True), H0_raw[0].to(devices[0][0], non_blocking=True), overlap_tensor_raw[0].to(devices[0][0], non_blocking=True), mask_tensor_raw[0].to(devices[0][0], non_blocking=True), delta_H_raw[0].to(devices[0][0], non_blocking=True)        
        band_inputs = None
        if len(extra_data) >= 7:
            band_kpoints = extra_data[0][0].to(devices[0][0], non_blocking=True)
            band_eps_ref = extra_data[1][0].to(devices[0][0], non_blocking=True)
            band_c_ref = extra_data[2][0].to(devices[0][0], non_blocking=True)
            band_active = extra_data[3][0].to(devices[0][0], non_blocking=True)
            band_mask = extra_data[4][0].to(devices[0][0], non_blocking=True)
            band_edge_r = extra_data[5][0].to(devices[0][0], non_blocking=True)
            band_inputs = (band_kpoints, band_eps_ref, band_c_ref, band_active, band_mask, band_edge_r)
        node_num = max(int(max(edge_src)+1), int(max(edge_dst)+1))
        batch = torch.ones((node_num,), dtype=torch.int32).to(devices[0][0], non_blocking=True)
        node_atom = [-1 for _ in range(node_num)]
        for ele_idx in range(len(ele_list)):
            src_symbol = _unwrap_symbol(ele_list[ele_idx][0])
            dst_symbol = _unwrap_symbol(ele_list[ele_idx][1]) if len(ele_list[ele_idx]) > 1 else src_symbol
            src_idx = int(edge_src[ele_idx].item())
            dst_idx = int(edge_dst[ele_idx].item())
            if src_symbol in ele_dict:
                node_atom[src_idx] = ele_dict[src_symbol]
            if dst_symbol in ele_dict:
                node_atom[dst_idx] = ele_dict[dst_symbol]
        if any(v < 0 for v in node_atom):
            # Fallback to a valid index to avoid invalid embedding gather on GPU.
            node_atom = [0 if v < 0 else v for v in node_atom]

        node_atom = torch.tensor(node_atom, dtype=torch.long, device=devices[0][0])

        data_list_device = [[] for _ in range(len(models))]

        for model_idx in range(len(models)):
            if model_idx > 0:
                data_list_device[model_idx] = [H0_ds.to(devices[model_idx][0], non_blocking=True), edge_src.to(devices[model_idx][0], non_blocking=True), edge_dst.to(devices[model_idx][0], non_blocking=True),  edge_vec.to(devices[model_idx][0], non_blocking=True),  batch.to(devices[model_idx][0], non_blocking=True), node_atom.to(devices[model_idx][0], non_blocking=True)]
            else:
                data_list_device[model_idx] = [H0_ds, edge_src, edge_dst, edge_vec, batch, node_atom]    
        pred_h_all = []
        pred_h_trace_all = []  
        mask_dis_list = []
        inputs_list = []      
        kwargs_list = []      
        for model_idx in range(len(models)):
            kw_params = {
                'weak_ham_in':    data_list_device[model_idx][0],
                'node_num':       node_num,
                'edge_src':       data_list_device[model_idx][1],
                'edge_dst':       data_list_device[model_idx][2],
                'edge_vec':       data_list_device[model_idx][3],
                'batch':          data_list_device[model_idx][4],
                'node_atom':      data_list_device[model_idx][5],
                'use_sep':        True,
                'range_dis':      range_dis[model_idx]
            }
            inputs_list.append(())
            kwargs_list.append(kw_params)

        outputs = parallel_apply(models, inputs_list, kwargs_tup=kwargs_list)

        pred_h_all = []
        pred_h_trace_all = []
        mask_dis_list = []

        for i, output_tuple in enumerate(outputs):
            pred_h_direct_sum, pred_h_trace, mask_dis = output_tuple
            pred_h_all.append(pred_h_direct_sum.to(devices[0][0]))
            pred_h_trace_all.append(pred_h_trace.to(devices[0][0]))
            mask_dis_list.append(mask_dis.to(devices[0][0]))

        pred_h = torch.sum(torch.stack(pred_h_all), dim=0)
        pred_h_trace = torch.sum(torch.stack(pred_h_trace_all), dim=0)
        pred_h = construct_kernel.get_H(pred_h)

        if args.spinful:
            delta_H_pred_real = reverse_transform_matrix(pred_h[:, 0, :].real, ls)
        else:
            delta_H_pred_real = reverse_transform_matrix(pred_h.real, ls)

        H_gt = delta_H_raw + H0_raw
        H_pred = H0_raw.clone()
        if args.spinful:
            if H_pred.shape[-1] != args.matrix_orbitals:
                raise ValueError(
                    f"Expected spinful Hamiltonian size {args.matrix_orbitals}, got {H_pred.shape[-1]}. "
                    f"Check --orbital-layout/--spinful against your dataset."
                )
            H_pred = H_pred.reshape(-1, 2, args.num_orbitals, 2, args.num_orbitals)
            H_pred[:, 0, :, 0, :].real = H_pred[:, 0, :, 0, :].real + delta_H_pred_real
            H_pred[:, 1, :, 1, :].real = H_pred[:, 1, :, 1, :].real + delta_H_pred_real
            H_pred = H_pred.reshape(-1, args.matrix_orbitals, args.matrix_orbitals)
        else:
            if H_pred.shape[-1] != args.matrix_orbitals:
                raise ValueError(
                    f"Expected spinless Hamiltonian size {args.matrix_orbitals}, got {H_pred.shape[-1]}. "
                    f"Check --orbital-layout/--spinful against your dataset."
                )
            H_pred = H_pred + delta_H_pred_real.to(H_pred.dtype)

        edge_vec, edge_src, edge_dst = edge_vec.to(devices[0][0], non_blocking=True), edge_src.to(devices[0][0], non_blocking=True), edge_dst.to(devices[0][0], non_blocking=True)

        if args.use_wa_loss_lmdb:
            loss_h, mu = criterion_lmdb.compute_loss(
                H_gt=H_gt,
                H_pred=H_pred,
                overlap=overlap_tensor_raw,
                mask=mask_tensor_raw,
                edge_vec=edge_vec,
                edge_src=edge_src,
                edge_dst=edge_dst,
                lattice_vector=lattice_vector_torch,
                atoms_positions=position_torch,
                tot_kunm=tot_kunm_torch,
                tot_basis_num=tot_basis_num_torch,
                band_cut_index=band_cut_index_torch,
                kpt_data=kpt_torch,
                eigenvectors_enlager=eigenvectors_enlager_torch,
            )
        elif args.use_precomputed_band_loss:
            mu = torch.zeros([], dtype=H_gt.real.dtype, device=H_gt.device)
            diff = torch.abs(H_pred - H_gt) * mask_tensor_raw.to(H_pred.real.dtype)
            loss_h = diff.mean().real
        else:
            R_list = criterion.get_R_list(edge_vec, edge_src, edge_dst, lattice_vector_torch, position_torch)

            # 随机获得一个k点波函数，并且通过 band_cut_index_torch 指标把空间分为 P & Q 两部分
            kpt_coord, eigenvectors_P_space, eigenvectors_Q_space = criterion.divide_space(tot_kunm_torch, tot_basis_num_torch, kpt_torch, band_cut_index_torch, eigenvectors_enlager_torch)

            # 获取子哈密顿量波函数投影矩阵
            reduce_P_space1, reduce_Q_space1, reduce_PQ_space1  = criterion.cal_wfc_hk_vectorized(edge_src, edge_dst, H_gt, kpt_coord, eigenvectors_P_space, eigenvectors_Q_space)
            reduce_P_space2, reduce_Q_space2, reduce_PQ_space2  = criterion.cal_wfc_hk_vectorized(edge_src, edge_dst, H_pred, kpt_coord, eigenvectors_P_space, eigenvectors_Q_space)

            # 计算 mu 与 mae
            mu = criterion.grep_min_mu(reduce_P_space1, reduce_Q_space1, reduce_P_space2, reduce_Q_space2, 
                                    H_gt, H_pred, overlap_tensor_raw, mask_tensor_raw)

            loss_h = criterion.cal_loss(reduce_P_space1, reduce_Q_space1, reduce_PQ_space1,
                                    reduce_P_space2, reduce_Q_space2, reduce_PQ_space2,
                                    H_gt, H_pred, overlap_tensor_raw).real      

        if band_inputs is not None and args.band_loss_weight > 0:
            band_kpoints, band_eps_ref, band_c_ref, band_active, band_mask, band_edge_r = band_inputs
            band_loss = compute_precomputed_band_loss(
                H_pred=H_pred,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_r=band_edge_r,
                band_kpoints=band_kpoints,
                band_eps_ref=band_eps_ref,
                band_c_ref=band_c_ref,
                band_active=band_active,
                band_mask=band_mask,
                matrix_orbitals=args.matrix_orbitals,
                node_num=node_num,
            )
        else:
            band_loss = torch.zeros_like(loss_h)


        sample_num += 1

        if torch.isnan(loss_h).any() or torch.isinf(loss_h).any():
            print('nan or inf loss')
            continue

        trace_label = construct_kernel.get_H_trace(delta_H_dp + mu*overlap_tensor)

        trace_mask = construct_kernel.get_H_trace(mask_tensor).to(torch.bool).to(mask_tensor.real.dtype)
        if pred_h_trace.shape[-1] != trace_label.shape[-1]:
            trace_dim = min(pred_h_trace.shape[-1], trace_label.shape[-1], trace_mask.shape[-1])
            pred_h_trace = pred_h_trace[..., :trace_dim]
            trace_label = trace_label[..., :trace_dim]
            trace_mask = trace_mask[..., :trace_dim]

        if args.with_trace:
            loss_t = criterion_trace(pred_h_trace, trace_label, trace_mask) 
            loss_all = 0.8 * loss_h + 0.2 * loss_h.item() / loss_t.item() * loss_t
        else:
            loss_t = torch.zeros_like(loss_h)
            loss_all = loss_h
        loss_all = loss_all + args.band_loss_weight * band_loss
            
        loss_h_all.append(loss_h.item())

        if train:
            for model_idx in range(len(models)):
                optimizers[model_idx].zero_grad()
            loss_all.backward()
            for model_idx in range(len(models)):
                optimizers[model_idx].step()

        loss_metrics['ham'].update(loss_h.item(), n=1)
        loss_metrics['trace'].update(loss_t.item(), n=1)
        if args.with_trace:
            mae_trace = torch.mean(torch.abs(pred_h_trace.detach()-trace_label)).item()
        else:
            mae_trace = 0
        mae_metrics['trace'].update(mae_trace, n=1)

        combined_mask_sum, mae_ham = MAE_metric(H_pred.detach(), H_gt.detach(), overlap_tensor_raw.detach(), mask_tensor_raw.detach())
        range_mae = []
        for ridx in range(len(mask_dis_list)):
            _, r_mae = MAE_metric(
                H0_raw.detach(),
                H_gt.detach(),
                overlap_tensor_raw.detach(),
                mask_tensor_raw.detach() * mask_dis_list[ridx][:, :, None],
            )
            range_mae.append(r_mae.item())
        mae_ham_on_site = range_mae[0] if len(range_mae) > 0 else 0.0
        mae_ham_1_2 = range_mae[1] if len(range_mae) > 1 else 0.0
        mae_ham_2_4 = range_mae[2] if len(range_mae) > 2 else 0.0
        mae_ham_4_6 = range_mae[3] if len(range_mae) > 3 else 0.0

        _, mae_baseline_ham = MAE_metric(H0_raw.detach(), H_gt.detach(), overlap_tensor_raw.detach(), mask_tensor_raw.detach())

        mae_metrics['ham'].update(mae_ham.item(), n=1)
        mae_metrics['ham_on_site'].update(mae_ham_on_site, n=1)
        mae_metrics['ham_1_2'].update(mae_ham_1_2, n=1)
        mae_metrics['ham_2_4'].update(mae_ham_2_4, n=1)
        mae_metrics['ham_4_6'].update(mae_ham_4_6, n=1)
        mae_metrics['baseline_ham'].update(mae_baseline_ham.item(), n=1)

        mae_h_all.append(mae_ham)
    
        # logging
        if train:


            if step % print_freq == 0 or step == len(data_loader) - 1: 
                e = (step + 1) / len(data_loader)
                info_str = 'Epoch: [{epoch}][{step}/{length}] \t'.format(epoch=epoch, step=step, length=len(data_loader))
                info_str +=  'loss_ham: {loss_ham:.9f}, loss_trace: {loss_trace:.9f}, ham_MAE: {ham_mae:.9f},  baseline_ham_MAE: {baseline_ham_mae:.9f}, trace_MAE: {trace_mae:.9f}'.format(
                    loss_ham=loss_metrics['ham'].avg, loss_trace=loss_metrics['trace'].avg, ham_mae=mae_metrics['ham'].avg,  baseline_ham_mae=mae_metrics['baseline_ham'].avg, trace_mae=mae_metrics['trace'].avg, 
                )                
                logger.info(info_str)

            if sample_num % 100 == 0:
                for model_idx in range(len(models)):
                    torch.save(
                        {'state_dict': models[model_idx].state_dict()}, 
                        os.path.join(args.output_dir, 'model_range'+str(model_idx)+'_curr.pth.tar')
                    )
        else:
            if (step % print_freq == 0 or step == len(data_loader) - 1) and print_progress: 
                e = (step + 1) / len(data_loader)
                info_str = '[{step}/{length}] \t'.format(step=step, length=len(data_loader))

                info_str +=  'loss_ham: {loss_ham:.9f}, loss_trace: {loss_trace:.9f}, ham_MAE: {ham_mae:.9f}, ham_on_site: {ham_on_site:.9f}, ham_1_2: {ham_1_2:.9f}, ham_2_4: {ham_2_4:.9f}, ham_4_6: {ham_4_6:.9f}, baseline_ham_MAE: {baseline_ham_mae:.9f}, trace_MAE: {trace_mae:.9f}'.format(
                    loss_ham = loss_metrics['ham'].avg, loss_trace = loss_metrics['trace'].avg, ham_mae = mae_metrics['ham'].avg,  ham_on_site = mae_metrics['ham_on_site'].avg,  ham_1_2 = mae_metrics['ham_1_2'].avg,  ham_2_4 = mae_metrics['ham_2_4'].avg,  ham_4_6 = mae_metrics['ham_4_6'].avg, baseline_ham_mae = mae_metrics['baseline_ham'].avg, trace_mae = mae_metrics['trace'].avg, 
                )

                logger.info(info_str)

    del loss_all, loss_h, pred_h
    torch.cuda.empty_cache()
    return mae_metrics['ham'].avg

        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser('Training equivariant networks on Material Project', parents=[get_args_parser()])
    args = parser.parse_args()  
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)