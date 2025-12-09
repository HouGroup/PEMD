"""
PEMD code library.

Developed by: Tan Shendong
Date: 2025.05.23
"""

import random
import logging
import numpy as np
import pandas as pd
import PEMD.io as io
import PEMD.constants as const

from rdkit import Chem
from pathlib import Path
from copy import deepcopy
from rdkit import RDLogger
from rdkit.Chem import AllChem
from scipy.spatial import cKDTree
from rdkit.Chem import Descriptors
from rdkit.Geometry import Point3D
from collections import defaultdict
from openbabel import openbabel as ob
from scipy.spatial.transform import Rotation as R


lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)

# Set up logging
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.ERROR)
    logger.propagate = False


def gen_sequence_copolymer_3D(name,
                              smiles_A,
                              smiles_B,
                              sequence,
                              bond_length=1.5,
                              left_cap_smiles=None,
                              right_cap_smiles=None,
                              retry_step=100,
                              growth_axis = 'auto',
                              cone_half_deg = 15.0,
                              keep_axis_weight = 0.7,):
    """
    通用序列构建：sequence 是一个列表，如 ['A','B','B','A',…]
    """

    # 1. 预先初始化 A、B 单体的信息
    dumA1, dumA2, atomA1, atomA2 = Init_info(name, smiles_A)
    dumB1, dumB2, atomB1, atomB2 = Init_info(name, smiles_B)

    first_unit = sequence[0]
    if first_unit == 'A':
        dum1, dum2, atom1, atom2, smiles_mid = dumA1, dumA2, atomA1, atomA2, smiles_A
    else:
        dum1, dum2, atom1, atom2, smiles_mid = dumB1, dumB2, atomB1, atomB2, smiles_B

    mol_1, h_1, t_1 = prepare_monomer_nocap(smiles_mid, dum1, dum2, atom1, atom2)
    print(">>> mol_1 num conf:", mol_1.GetNumConformers())

    chi = gen_chi_array(len(sequence))
    if chi[0]:
        mol_1 = mirror_inversion_mol(mol_1, confId=0)
    else:
        mol_1 = deepcopy_mol(mol_1)

    # 全局生长轴：从用户指定或自动由首单体 head→tail 推断
    global_dir = _parse_growth_axis(growth_axis, mol_1, h_1, t_1)

    connecting_mol = Chem.RWMol(mol_1)

    tail_idx = t_1
    num_atom = connecting_mol.GetNumAtoms()

    k=1
    for unit in sequence[1:]:
        if unit == 'A':
            dum1, dum2, atom1, atom2, smiles_mid = dumA1, dumA2, atomA1, atomA2, smiles_A
        else:
            dum1, dum2, atom1, atom2, smiles_mid = dumB1, dumB2, atomB1, atomB2, smiles_B

        mon, h, t = prepare_monomer_nocap(smiles_mid, dum1, dum2, atom1, atom2)

        if chi[k]:
            mon = mirror_inversion_mol(mon, confId=0)
        else:
            mon = deepcopy_mol(mon)
        k+=1

        R_unit = _unit_bounding_radius(mon, h, skip_h=True)
        conf_poly = connecting_mol.GetConformer()
        tail_pos = np.array(conf_poly.GetAtomPosition(tail_idx))

        _, local_dir = get_vector(connecting_mol, tail_idx)
        ideal_direction = _norm((1.0 - keep_axis_weight) * local_dir + keep_axis_weight * global_dir)

        best_dir, best_offset, best_margin = _choose_extension_direction_and_offset(
            connecting_mol=connecting_mol,
            tail_idx=tail_idx,
            base_dir=ideal_direction,  # ✅ 用全局轴偏置后的方向
            R_unit=R_unit,
            bond_length=bond_length,
            cone_half_deg=cone_half_deg,  # ✅ 缩小圆锥半角，沿轴前进
        )

        z_tail = int(connecting_mol.GetAtomWithIdx(tail_idx).GetAtomicNum())
        z_head = int(mon.GetAtomWithIdx(h).GetAtomicNum())
        bl_est = estimate_bond_length(z_tail, z_head, fallback=bond_length)
        target_pos = tail_pos + (bl_est + best_offset + 0.12) * best_dir

        new_unit = Chem.Mol(mon)
        new_unit = align_monomer_unit(new_unit, h, target_pos, best_dir)

        for i in range(retry_step):

            extra_angle = 0.10
            atom_indices_to_rotate = [j for j in range(new_unit.GetNumAtoms()) if j != h_1]
            rotate_substructure_around_axis(new_unit, atom_indices_to_rotate,
                                            ideal_direction, target_pos, extra_angle)

            combined = Chem.CombineMols(connecting_mol, new_unit)
            editable = Chem.EditableMol(combined)
            head_idx = num_atom + h
            editable.AddBond(tail_idx, head_idx, order=Chem.rdchem.BondType.SINGLE)

            combined_mol = editable.GetMol()
            combined_mol = Chem.RWMol(combined_mol)

            # combined_mol.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(combined_mol)
            AllChem.MMFFOptimizeMolecule(combined_mol, maxIters=50, confId=0)

            if check_3d_structure(combined_mol, dist_min=0.7):
                print(k)
                break

        connecting_mol = Chem.RWMol(combined_mol)
        tail_idx = num_atom + t
        num_atom = num_atom + new_unit.GetNumAtoms()

    length = len(sequence)
    final_poly = gen_3D_withcap(
        connecting_mol,
        h_1,
        tail_idx,
        length,
        left_cap_smiles=left_cap_smiles,
        right_cap_smiles=right_cap_smiles,
    )

    return final_poly


def gen_chi_array(n, atac_ratio=0.5):
    chi = np.full(n, False)
    chi[int(n*atac_ratio):] = True
    random.shuffle(chi)
    return chi

def mirror_inversion_mol(mol, confId=0):
    mol_c = deepcopy_mol(mol)
    coord = np.array(mol_c.GetConformer(confId).GetPositions())
    coord[:, 2] = coord[:, 2] * -1.0
    for i in range(mol_c.GetNumAtoms()):
        mol_c.GetConformer(confId).SetAtomPosition(i, Point3D(coord[i, 0], coord[i, 1], coord[i, 2]))

    return mol_c

def deepcopy_mol(mol):
    mol = picklable(mol)
    copy_mol = deepcopy(mol)

    return copy_mol

def picklable(mol):
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
    return mol

def _vdw_radius(Z: int) -> float:
    table = {
        1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47,
        15: 1.80, 16: 1.80, 17: 1.75, 35: 1.85, 53: 1.98
    }
    return table.get(Z, 1.8)

def _polymer_kdtree(mol: Chem.Mol, exclude_idx: set[int] | None = None, skip_h: bool = True):
    conf = mol.GetConformer()
    pts, zs = [], []
    for i in range(mol.GetNumAtoms()):
        if exclude_idx and i in exclude_idx:
            continue
        Z = mol.GetAtomWithIdx(i).GetAtomicNum()
        if skip_h and Z == 1:
            continue
        pts.append(np.array(conf.GetAtomPosition(i), dtype=float))
        zs.append(Z)
    if not pts:
        pts = [np.array([1e9,1e9,1e9])]
    return cKDTree(np.vstack(pts)), np.array(zs, dtype=int)

def _unit_bounding_radius(unit: Chem.Mol, head_idx: int, skip_h: bool = True, include_vdw: bool = True, scale: float = 1.0) -> float:
    conf = unit.GetConformer()
    c = np.array(conf.GetAtomPosition(head_idx), dtype=float)
    r = 0.0
    for i in range(unit.GetNumAtoms()):
        if i == head_idx:
            continue
        Zi = unit.GetAtomWithIdx(i).GetAtomicNum()
        if skip_h and Zi == 1:
            continue
        d = np.linalg.norm(np.array(conf.GetAtomPosition(i), dtype=float) - c)
        if include_vdw:
            d += _vdw_radius(Zi)
        r = max(r, d)
    return r * scale

def _orthonormal_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = n / (np.linalg.norm(n) + 1e-12)
    h = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(h, n)) > 0.9:
        h = np.array([0.0, 1.0, 0.0])
    u = h - np.dot(h, n) * n
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(n, u)
    v /= np.linalg.norm(v) + 1e-12
    return u, v

def _clearance_margin_at_point(poly_tree: cKDTree, poly_Z: np.ndarray, pt: np.ndarray, R_unit: float, scale: float = 0.85) -> float:
    idxs = poly_tree.query_ball_point(pt, r=R_unit + 2.6)
    if not idxs:
        return 1e3  # 非常安全
    margins = []
    for j in idxs:
        d = np.linalg.norm(poly_tree.data[j] - pt)
        margins.append(d - (R_unit + scale * _vdw_radius(int(poly_Z[j]))))
    return min(margins) if margins else 1e3

def _direction_clearance_score(poly_tree: cKDTree, poly_Z: np.ndarray,
                               tail_pos: np.ndarray, direction: np.ndarray,
                               R_unit: float, s_start: float, s_window: float,
                               n_samples: int = 8, scale: float = 0.85) -> float:
    """
    沿 direction 从 s_start 开始、长度 s_window 的线段上均匀采样，取最小裕度。
    """
    mins = []
    for s in np.linspace(s_start, s_start + s_window, n_samples):
        pt = tail_pos + s*direction
        mins.append(_clearance_margin_at_point(poly_tree, poly_Z, pt, R_unit, scale))
    return min(mins) if mins else 1e3

def _choose_extension_direction_and_offset(connecting_mol: Chem.Mol,
                                           tail_idx: int,
                                           base_dir: np.ndarray,
                                           R_unit: float,
                                           bond_length: float,
                                           lookahead: float = 1.2,
                                           allow_offsets: tuple[float,...] = (0.0, 0.2, 0.4, 0.6),
                                           cone_half_deg: float = 30.0) -> tuple[np.ndarray, float, float]:
    conf = connecting_mol.GetConformer()
    tail_pos = np.array(conf.GetAtomPosition(tail_idx), dtype=float)
    poly_tree, poly_Z = _polymer_kdtree(connecting_mol, exclude_idx={tail_idx}, skip_h=True)

    dirs = _directions_in_cone_fibonacci(base_dir, half_deg=cone_half_deg, n=96)
    dirs = _early_clearance_prune(connecting_mol, tail_idx, R_unit, dirs, bond_length,
                                  lookahead=0.8, samples=4, min_margin=0.0)

    best = (dirs[0], 0.0, -1e9)
    for d in dirs:
        for off in allow_offsets:
            margin = _direction_clearance_score(poly_tree, poly_Z, tail_pos, d,
                                                R_unit=R_unit, s_start=bond_length+off,
                                                s_window=max(lookahead, 0.6), n_samples=8, scale=0.85)
            if margin > best[2] or (np.isclose(margin, best[2]) and off < best[1]):
                best = (d, off, margin)
    return best


# Processes a polymer’s SMILES string with dummy atoms to set up connectivity and identify the connecting atoms.
def Init_info(name, smiles_mid):
    # Get index of dummy atoms and atoms associated with them
    dum_index, bond_type = FetchDum(smiles_mid)
    dum1 = dum_index[0]
    dum2 = dum_index[1]

    # Assign dummy atom according to bond type
    dum = None
    if bond_type == 'SINGLE':
        dum = 'Cl'

    # Replace '*' with dummy atom
    smiles_each = smiles_mid.replace(r'*', dum)

    # Convert SMILES to XYZ coordinates
    xyz_filename = io.smile_toxyz(
        name,
        smiles_each,       # Replace '*' with dummy atom
    )

    # Collect valency and connecting information for each atom according to XYZ coordinates
    neigh_atoms_info = connec_info(xyz_filename)

    # Find connecting atoms associated with dummy atoms.
    # Dum1 and dum2 are connected to atom1 and atom2, respectively.
    atom1 = neigh_atoms_info['NeiAtom'][dum1].copy()[0]
    atom2 = neigh_atoms_info['NeiAtom'][dum2].copy()[0]

    Path(xyz_filename).unlink(missing_ok=True)  # Clean up the temporary XYZ file

    return dum1, dum2, atom1, atom2,

# Get index of dummy atoms and bond type associated with it
def FetchDum(smiles):
    m = Chem.MolFromSmiles(smiles)
    dummy_index = []
    bond_type = None
    if m is not None:
        for atom in m.GetAtoms():
            if atom.GetSymbol() == '*':
                dummy_index.append(atom.GetIdx())
        for bond in m.GetBonds():
            if (
                bond.GetBeginAtom().GetSymbol() == '*'
                or bond.GetEndAtom().GetSymbol() == '*'
            ):
                bond_type = bond.GetBondType()
                break
    return dummy_index, str(bond_type)

def connec_info(name):
    # Collect valency and connecting information for each atom according to XYZ coordinates
    obConversion = ob.OBConversion()
    obConversion.SetInFormat("xyz")
    mol = ob.OBMol()
    obConversion.ReadFile(mol, name)
    neigh_atoms_info = []

    for atom in ob.OBMolAtomIter(mol):
        neigh_atoms = []
        bond_orders = []
        for allatom in ob.OBAtomAtomIter(atom):
            neigh_atoms.append(allatom.GetIndex())
            bond_orders.append(atom.GetBond(allatom).GetBondOrder())
        neigh_atoms_info.append([neigh_atoms, bond_orders])
    neigh_atoms_info = pd.DataFrame(neigh_atoms_info, columns=['NeiAtom', 'BO'])
    return neigh_atoms_info

def prepare_monomer_nocap(smiles_mid: str,
                          dum1: int,
                          dum2: int,
                          atom1: int,
                          atom2: int) -> tuple[Chem.Mol, int, int]:
    from rdkit.Chem import rdMolTransforms as MT
    # 可调偏好：与原 mol_from_smiles 一致
    version = 2         # ETKDG 版本；需要 v3 可改成 3
    ez_pref = 'E'       # 未指定双键偏好：'E' 或 'Z'
    chiral_pref = 'S'   # 未指定手性偏好：'R' 或 'S'

    # --- 1) 规范连接位到 [3H] ---
    n_conn = smiles_mid.count('[*]') + smiles_mid.count('*') + smiles_mid.count('[3H]')
    smi = smiles_mid.replace('[*]', '[3H]').replace('*', '[3H]')

    # --- 2) 生成分子，加氢 ---
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles_mid}")
    mol = Chem.AddHs(mol)
    Chem.AssignStereochemistry(mol)

    # --- 3) 找主链（两连接位最短路径）、搜集“未指定”的主链双键，后续拉直 ---
    backbone_atoms: list[int] = []
    backbone_bonds: list[int] = []
    backbone_dih: list[tuple[int,int,int,int]] = []
    if n_conn == 2:
        link_idx = [a.GetIdx() for a in mol.GetAtoms()
                    if (a.GetSymbol() == "H" and a.GetIsotope() == 3)]
        if len(link_idx) == 2:
            backbone_atoms = list(Chem.GetShortestPath(mol, link_idx[0], link_idx[1]))
            for i in range(len(backbone_atoms) - 1):
                b = mol.GetBondBetweenAtoms(backbone_atoms[i], backbone_atoms[i+1])
                if b is None:
                    continue
                backbone_bonds.append(b.GetIdx())
                # 未指定且非环的双键，构造一个（粗略）二面角四元组
                if b.GetBondTypeAsDouble() == 2 and str(b.GetStereo()) == 'STEREONONE' and (not b.IsInRing()):
                    # 注：沿用你给的写法（i-1,i,i+1,i+2）；若越界 RDKit 会报错，这里加保护
                    if i-1 >= 0 and i+2 < len(backbone_atoms):
                        backbone_dih.append((backbone_atoms[i-1], backbone_atoms[i],
                                             backbone_atoms[i+1], backbone_atoms[i+2]))

    # --- 4) 收集“非主链/非环”的未指派双键，稍后做 E/Z 偏好筛选 ---
    db_list: list[int] = []
    for b in mol.GetBonds():
        if b.GetBondTypeAsDouble() == 2 and str(b.GetStereo()) == 'STEREONONE' and (not b.IsInRing()):
            if n_conn == 2 and b.GetIdx() in backbone_bonds:
                continue
            db_list.append(b.GetIdx())

    # --- 5) 立体枚举：按 E/Z 偏好与手性偏好挑一个 ---
    try:
        from rdkit.Chem import EnumerateStereoisomers as ESI
        opts = ESI.StereoEnumerationOptions(unique=True, tryEmbedding=True)
        isomers = tuple(ESI.EnumerateStereoisomers(mol, options=opts))
    except Exception:
        isomers = (mol,)

    if len(isomers) > 1:
        chiral_num_max = -1
        picked = None
        for iso in isomers:
            Chem.AssignStereochemistry(iso)

            # E/Z 偏好仅作用于 db_list 中的双键
            ez_ok = True
            if db_list:
                marks = []
                for idx in db_list:
                    b = iso.GetBondWithIdx(idx)
                    st = str(b.GetStereo())
                    if st in ('STEREOANY', 'STEREONONE'):
                        continue
                    if ez_pref == 'E' and st in ('STEREOE', 'STEREOTRANS'):
                        marks.append(True)
                    elif ez_pref == 'Z' and st in ('STEREOZ', 'STEREOCIS'):
                        marks.append(True)
                    else:
                        marks.append(False)
                ez_ok = (len(marks) == 0) or bool(np.all(np.array(marks)))

            # 手性偏好：尽量挑选最多满足 chiral_pref 的那个
            chiral_list = Chem.FindMolChiralCenters(iso, includeUnassigned=False)
            if chiral_list:
                tags = [c[1] for c in chiral_list]  # e.g. 'R'/'S'
                cnum = sum(1 for t in tags if t == chiral_pref)
                # 全满足则直接选
                if cnum == len(chiral_list) and ez_ok:
                    picked = iso
                    break
                # 否则记录下“最多满足”的那个
                if ez_ok and cnum > chiral_num_max:
                    chiral_num_max = cnum
                    picked = iso
            else:
                # 没有手性中心则只看 E/Z
                if ez_ok:
                    picked = iso
                    break

        mol = Chem.Mol(picked if picked is not None else isomers[0])

    # --- 6) 3D 坐标（ETKDG） ---
    if version == 3:
        etkdg = AllChem.ETKDGv3()
    elif version == 2:
        etkdg = AllChem.ETKDGv2()
    else:
        etkdg = AllChem.ETKDG()
    etkdg.enforceChirality = True
    etkdg.useRandomCoords = False
    etkdg.maxAttempts = 100

    res = AllChem.EmbedMolecule(mol, etkdg)
    if res != 0:
        raise RuntimeError(f"ETKDG embedding failed for {smiles_mid}")

    # --- 7) 把主链未指派双键“拉直”为 180°，并给侧向一个 0° 参考 ---
    if backbone_dih:
        for (i, j, k, l) in backbone_dih:
            try:
                MT.SetDihedralDeg(mol.GetConformer(0), i, j, k, l, 180.0)
                # 再把 k 的某个非 j/l 邻居拉到 0°，提供一致参考
                for na in mol.GetAtomWithIdx(k).GetNeighbors():
                    na_idx = na.GetIdx()
                    if na_idx != j and na_idx != l:
                        MT.SetDihedralDeg(mol.GetConformer(0), i, j, k, na_idx, 0.0)
                        break
            except Exception:
                # 任何越界/几何异常都忽略，尽量继续
                pass

    # --- 8) 找到两个 [3H] 的“重原子邻居”作为 head/tail ---
    linkers = [a.GetIdx() for a in mol.GetAtoms()
               if (a.GetSymbol() == "H" and a.GetIsotope() == 3)]
    if len(linkers) != 2:
        raise ValueError(f"Monomer must contain exactly 2 linkers ([3H]/[*]). Found {len(linkers)}.")

    def _first_heavy_neighbor(idx: int) -> int:
        for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
            if nb.GetAtomicNum() != 1:  # 非氢（普通氢/三氢都排除）
                return nb.GetIdx()
        # 如果只连到了氢，也退而求其次拿第一个邻居
        nbs = list(mol.GetAtomWithIdx(idx).GetNeighbors())
        return nbs[0].GetIdx() if nbs else idx

    head_heavy = _first_heavy_neighbor(linkers[0])
    tail_heavy = _first_heavy_neighbor(linkers[1])

    # --- 10) 移除两个 [3H]，并修正 head/tail 索引回退 ---
    to_remove = sorted(linkers, reverse=True)
    rw = Chem.RWMol(mol)
    for idx in to_remove:
        try:
            rw.RemoveAtom(idx)
        except Exception:
            pass
    monomer = rw.GetMol()
    monomer.UpdatePropertyCache(False)
    try:
        Chem.SanitizeMol(monomer)
    except Exception:
        pass

    def adjust(i: int) -> int:
        """按删除的 [3H] 回退索引"""
        out = i
        for d in to_remove:
            if out > d:
                out -= 1
        return out

    new_head = adjust(head_heavy)
    new_tail = adjust(tail_heavy)
    # 统一顺序（可选）
    if new_head > new_tail:
        new_head, new_tail = new_tail, new_head

    return monomer, new_head, new_tail

def prepare_cap_monomer(smiles_cap: str) -> tuple[Chem.Mol, int, np.ndarray]:
    """Prepare a capping fragment defined by a SMILES string containing a single dummy atom."""
    mol = Chem.MolFromSmiles(smiles_cap)
    if mol is None:
        raise ValueError(f"Invalid cap SMILES: {smiles_cap}")

    dummy_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummy_indices) != 1:
        raise ValueError("Cap SMILES must contain exactly one dummy atom '*' or '[*]'.")

    dummy_idx = dummy_indices[0]
    dummy_atom = mol.GetAtomWithIdx(dummy_idx)
    neighbors = list(dummy_atom.GetNeighbors())
    if len(neighbors) != 1:
        raise ValueError("Cap dummy atom must be connected to exactly one atom.")

    connection_idx = neighbors[0].GetIdx()

    rw = Chem.RWMol(mol)
    rw.GetAtomWithIdx(dummy_idx).SetAtomicNum(53)  # Use iodine as a placeholder heavy atom

    rw = Chem.RWMol(Chem.AddHs(rw))
    params = AllChem.ETKDGv3()
    params.randomSeed = -1
    if AllChem.EmbedMolecule(rw, params) != 0:
        logger.warning("3D embedding failed for cap %s.", smiles_cap)
    try:
        AllChem.MMFFOptimizeMolecule(rw)
    except Exception as exc:  # pragma: no cover - RDKit errors are data dependent
        logger.warning("MMFF optimization failed for cap %s: %s", smiles_cap, exc)

    conf = rw.GetConformer()
    attachment_vec = np.array(conf.GetAtomPosition(dummy_idx)) - np.array(conf.GetAtomPosition(connection_idx))
    if np.linalg.norm(attachment_vec) < const.MIN_DIRECTION_NORM:
        logger.warning("Attachment direction too small for cap %s; using default.", smiles_cap)
        attachment_vec = const.DEFAULT_DIRECTION
    else:
        attachment_vec = attachment_vec / np.linalg.norm(attachment_vec)

    rw.RemoveAtom(dummy_idx)
    if connection_idx > dummy_idx:
        connection_idx -= 1

    cap_mol = rw.GetMol()
    try:
        Chem.SanitizeMol(cap_mol)
    except Exception as exc:  # pragma: no cover - depends on specific SMILES
        logger.warning("Sanitization failed for cap %s: %s", smiles_cap, exc)

    return cap_mol, connection_idx, attachment_vec


def get_vector(mol, index):
    """
    对于指定原子，返回其位置及其与所有邻接原子连线方向的平均单位向量。
    若无邻居或平均向量过小，则返回默认方向。
    """
    conf = mol.GetConformer()
    pos = np.array(conf.GetAtomPosition(index))
    atom = mol.GetAtomWithIdx(index)
    neighbors = atom.GetNeighbors()
    if not neighbors:
        return pos, const.DEFAULT_DIRECTION
    vecs = []
    for nbr in neighbors:
        nbr_pos = np.array(conf.GetAtomPosition(nbr.GetIdx()))
        v = pos - nbr_pos
        if np.linalg.norm(v) > 1e-6:
            vecs.append(v / np.linalg.norm(v))
    if not vecs:
        return pos, const.DEFAULT_DIRECTION
    avg = np.mean(vecs, axis=0)
    norm_avg = np.linalg.norm(avg)
    if norm_avg < const.MIN_DIRECTION_NORM:
        # logger.warning("Atom %s: Computed local direction norm too small (%.3f); using default.", index, norm_avg)
        return pos, const.DEFAULT_DIRECTION
    return pos, avg / norm_avg


def align_monomer_unit(monomer,
                       connection_atom_idx,
                       target_position,
                       target_direction,
                       local_reference_direction=None):

    conf = monomer.GetConformer()
    B = np.array(conf.GetAtomPosition(connection_atom_idx))
    if np.linalg.norm(target_direction) < const.MIN_DIRECTION_NORM:
        logger.warning("Target direction is too small; using default direction.")
        target_direction = const.DEFAULT_DIRECTION
    if local_reference_direction is None:
        _, local_dir = get_vector(monomer, connection_atom_idx)
    else:
        local_dir = np.array(local_reference_direction, dtype=float)
    if np.linalg.norm(local_dir) < const.MIN_DIRECTION_NORM:
        logger.warning("Local direction of atom %s is too small; using default.", connection_atom_idx)
        local_dir = const.DEFAULT_DIRECTION
    rot_obj = rotate_vector_to_align(local_dir, -target_direction)
    for i in range(monomer.GetNumAtoms()):
        pos_i = np.array(conf.GetAtomPosition(i))
        new_pos = B + rot_obj.apply(pos_i - B)
        conf.SetAtomPosition(i, new_pos)
    B_rot = np.array(conf.GetAtomPosition(connection_atom_idx))
    translation = target_position - B_rot
    for i in range(monomer.GetNumAtoms()):
        pos_i = np.array(conf.GetAtomPosition(i))
        conf.SetAtomPosition(i, pos_i + translation)
    return monomer

def rotate_substructure_around_axis(mol, atom_indices, axis, anchor, angle_rad):
    """
    对分子中给定 atom_indices 列表中的原子，
    以 anchor 为中心绕单位向量 axis 旋转 angle_rad 弧度。
    """
    conf = mol.GetConformer()
    rot = R.from_rotvec(axis * angle_rad)
    for idx in atom_indices:
        pos = np.array(conf.GetAtomPosition(idx))
        pos_shifted = pos - anchor
        pos_rot = rot.apply(pos_shifted)
        conf.SetAtomPosition(idx, pos_rot + anchor)

def place_h_in_tetrahedral(mol, atom_idx, h_indices):
    """
    重新定位中心原子 atom_idx 上的氢原子，使局部几何尽量符合预期构型。
    针对 NH2（氮原子、1 个重邻居、2 个氢）单独处理，
    对于其他情况仍采用正四面体方法。
    """
    conf = mol.GetConformer()
    center_pos = np.array(conf.GetAtomPosition(atom_idx))
    center_atom = mol.GetAtomWithIdx(atom_idx)
    heavy_neighbors = [nbr.GetIdx() for nbr in center_atom.GetNeighbors() if nbr.GetAtomicNum() != 1]

    # 检测是否为 NH2 型：氮原子、1 个重邻居、传入2个氢
    if center_atom.GetAtomicNum() == 7 and len(heavy_neighbors) == 1 and len(h_indices) == 2:
        hv_idx = heavy_neighbors[0]
        hv_pos = np.array(conf.GetAtomPosition(hv_idx))
        v = hv_pos - center_pos
        if np.linalg.norm(v) < 1e-6:
            logger.warning("Atom %s: heavy neighbor vector too small; using default.", atom_idx)
            v = np.array([0, 0, 1])
        else:
            v = v / np.linalg.norm(v)

        # 获取理想正四面体方向
        tet_dirs = _get_ideal_tetrahedral_vectors()  # 返回4个单位向量

        # 1. 找出与 v 最一致的方向（应对应于重邻居方向）
        dots = [np.dot(d, v) for d in tet_dirs]
        idx_heavy = np.argmax(dots)

        # 2. 在剩下的3个方向中，找出与 -v 最一致的方向（对应孤对，暂不放氢）
        remaining = [(i, d) for i, d in enumerate(tet_dirs) if i != idx_heavy]
        dots_neg = [np.dot(d, -v) for i, d in remaining]
        idx_lonepair = remaining[np.argmax(dots_neg)][0]

        # 3. 剩下的两个方向用来放置氢原子
        h_dirs = [d for i, d in enumerate(tet_dirs) if i not in (idx_heavy, idx_lonepair)]
        if len(h_dirs) != 2:
            logger.error("Internal error: expected 2 hydrogen directions, got %s", len(h_dirs))
            return

        CH_BOND = 1.09  # 典型 C–H 键长
        # 首先为两个氢原子设定新的位置
        new_pos_1 = center_pos + CH_BOND * h_dirs[0]
        new_pos_2 = center_pos + CH_BOND * h_dirs[1]

        # 检查氢原子之间的距离，避免重叠
        for i, h_idx in enumerate(h_indices):
            if i == 0:
                new_pos = new_pos_1
            else:
                new_pos = new_pos_2
            for other_h_idx in h_indices:
                if other_h_idx != h_idx:
                    other_h_pos = np.array(conf.GetAtomPosition(other_h_idx))
                    if np.linalg.norm(new_pos - other_h_pos) < 0.8:  # 检查阈值，防止重叠
                        logger.warning(f"Hydrogen atoms {h_idx} and {other_h_idx} overlap! Adjusting.")
                        new_pos += np.random.uniform(0.1, 0.2, size=3)  # 轻微调整位置

        # 更新氢原子位置
        conf.SetAtomPosition(h_indices[0], new_pos_1)
        conf.SetAtomPosition(h_indices[1], new_pos_2)
        return


def rotate_vector_to_align(a, b):
    """
    返回一个旋转对象，使得向量 a 旋转后与向量 b 对齐。
    """
    a_norm = a / np.linalg.norm(a) if np.linalg.norm(a) > 1e-6 else const.DEFAULT_DIRECTION
    b_norm = b / np.linalg.norm(b) if np.linalg.norm(b) > 1e-6 else const.DEFAULT_DIRECTION
    cross_prod = np.cross(a_norm, b_norm)
    norm_cross = np.linalg.norm(cross_prod)
    if norm_cross < 1e-6:
        arbitrary = np.array([1, 0, 0])
        if np.allclose(a_norm, arbitrary) or np.allclose(a_norm, -arbitrary):
            arbitrary = np.array([0, 1, 0])
        rotation_axis = np.cross(a_norm, arbitrary)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        dot_prod = np.dot(a_norm, b_norm)
        angle_rad = np.pi if dot_prod < 0 else 0
    else:
        rotation_axis = cross_prod / norm_cross
        dot_prod = np.dot(a_norm, b_norm)
        dot_prod = np.clip(dot_prod, -1.0, 1.0)
        angle_rad = np.arccos(dot_prod)
    return R.from_rotvec(rotation_axis * angle_rad)

def _get_ideal_tetrahedral_vectors():
    """
    返回理想正四面体状态下4个顶点的归一化参考向量。
    """
    vs = [
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
    ]
    return [np.array(v) / np.linalg.norm(v) for v in vs]


def estimate_bond_length(atom_num1: int, atom_num2: int, fallback: float = 1.5) -> float:
    """Estimate a bond length based on covalent radii with a safe fallback."""
    pt = Chem.GetPeriodicTable()
    try:
        length = pt.GetRcovalent(atom_num1) + pt.GetRcovalent(atom_num2)
    except Exception:
        return fallback
    if not np.isfinite(length) or length <= 0:
        return fallback
    return float(length)


def attach_fragment(base_mol, fragment, terminal_idx, fragment_connection_idx):
    n_base = base_mol.GetNumAtoms()
    combo = Chem.CombineMols(base_mol, fragment)
    ed = Chem.EditableMol(combo)
    new_idx = fragment_connection_idx + n_base
    ed.AddBond(terminal_idx, new_idx, order=Chem.rdchem.BondType.SINGLE)
    combined = ed.GetMol()

    rw = Chem.RWMol(combined)
    h_inds = [nbr.GetIdx() for nbr in rw.GetAtomWithIdx(new_idx).GetNeighbors()
              if rw.GetAtomWithIdx(nbr.GetIdx()).GetAtomicNum() == 1]
    if h_inds:
        place_h_in_tetrahedral(rw, new_idx, h_inds)

    mol_out = rw.GetMol()
    # 🔧 新增：更新缓存并消毒
    mol_out.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(mol_out)

    return mol_out


def attach_hydrogen_cap(base_mol: Chem.Mol, terminal_idx: int) -> Chem.Mol:
    terminal_pos, v_norm = get_vector(base_mol, terminal_idx)
    atom_num = base_mol.GetAtomWithIdx(terminal_idx).GetAtomicNum()
    bond_length = estimate_bond_length(atom_num, 1, fallback=1.1)
    H_pos = terminal_pos + v_norm * bond_length

    editable_mol = Chem.EditableMol(base_mol)
    new_H_idx = editable_mol.AddAtom(Chem.Atom(1))
    editable_mol.AddBond(terminal_idx, new_H_idx, Chem.BondType.SINGLE)
    capped = editable_mol.GetMol()

    conformer = capped.GetConformer()
    conformer.SetAtomPosition(new_H_idx, Point3D(*H_pos))

    # 🔧 关键补充：更新缓存并消毒
    capped.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(capped)

    return capped

def attach_methyl_cap(base_mol: Chem.Mol, terminal_idx: int) -> Chem.Mol:
    fragment = Chem.AddHs(Chem.MolFromSmiles('C'))
    params = AllChem.ETKDG()
    params.randomSeed = -1
    if AllChem.EmbedMolecule(fragment, params) != 0:
        logger.warning("3D embedding failed for methyl cap; proceeding without optimization.")
    h_atoms = [a.GetIdx() for a in fragment.GetAtoms() if a.GetSymbol() == 'H']
    if not h_atoms:
        raise ValueError("Failed to construct methyl fragment with hydrogens.")
    em = Chem.EditableMol(fragment)
    em.RemoveAtom(h_atoms[0])  # 删除一个 H 以连接主链
    fragment = em.GetMol()

    connection_idx = [a.GetIdx() for a in fragment.GetAtoms() if a.GetSymbol() == 'C'][0]
    tail_pos, vec = get_vector(base_mol, terminal_idx)
    atom_poly = base_mol.GetAtomWithIdx(terminal_idx).GetAtomicNum()
    atom_cap = fragment.GetAtomWithIdx(connection_idx).GetAtomicNum()
    bond_length = estimate_bond_length(atom_poly, atom_cap)
    target_pos = tail_pos + (bond_length + 0.1) * vec

    aligned_fragment = align_monomer_unit(
        Chem.Mol(fragment),
        connection_idx,
        target_pos,
        vec,
    )
    return attach_fragment(base_mol, aligned_fragment, terminal_idx, connection_idx)


def attach_custom_cap(base_mol: Chem.Mol, terminal_idx: int, cap_smiles: str) -> Chem.Mol:
    cap_mol, connection_idx, attachment_vec = prepare_cap_monomer(cap_smiles)

    tail_pos, vec = get_vector(base_mol, terminal_idx)
    atom_poly = base_mol.GetAtomWithIdx(terminal_idx).GetAtomicNum()
    atom_cap = cap_mol.GetAtomWithIdx(connection_idx).GetAtomicNum()
    bond_length = estimate_bond_length(atom_poly, atom_cap)
    target_pos = tail_pos + (bond_length + 0.1) * vec

    aligned_fragment = align_monomer_unit(
        Chem.Mol(cap_mol),
        connection_idx,
        target_pos,
        vec,
        local_reference_direction=attachment_vec,
    )
    return attach_fragment(base_mol, aligned_fragment, terminal_idx, connection_idx)


def attach_default_cap(base_mol: Chem.Mol, terminal_idx: int) -> Chem.Mol:
    atom = base_mol.GetAtomWithIdx(terminal_idx)
    h_count = sum(1 for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() == 1)
    if atom.GetAtomicNum() == 6 and h_count == 2:
        return attach_hydrogen_cap(base_mol, terminal_idx)
    return attach_methyl_cap(base_mol, terminal_idx)


def gen_3D_withcap(mol, start_atom, end_atom, length, left_cap_smiles=None, right_cap_smiles=None):
    capped_mol = Chem.Mol(mol)
    terminal_data = [(start_atom, left_cap_smiles), (end_atom, right_cap_smiles)]

    for terminal_idx, cap_smiles in terminal_data:
        if cap_smiles:
            try:
                capped_mol = attach_custom_cap(capped_mol, terminal_idx, cap_smiles)
            except ValueError as exc:
                logger.error("Failed to apply custom cap %s at atom %s: %s. Using default capping.",
                             cap_smiles, terminal_idx, exc)
                capped_mol = attach_default_cap(capped_mol, terminal_idx)
        else:
            capped_mol = attach_default_cap(capped_mol, terminal_idx)

        # ✅ 每次加完一个帽，都立刻更新+消毒，避免后续步骤踩坑
        try:
            capped_mol.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(capped_mol)
        except Exception as exc:
            logger.warning("Sanitization after capping terminal %s failed: %s", terminal_idx, exc)

    # ✅ 在 MMFF 前再做一道保险
    capped_mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(capped_mol)

    AllChem.MMFFOptimizeMolecule(capped_mol, maxIters=50, confId=0)
    valid_structure = check_3d_structure(capped_mol)
    if length <= 3 or valid_structure:
        return capped_mol

    logger.warning("Failed to generate the final PDB file.")
    return None

def _benzene_rings(mol: Chem.Mol):
    rings = []
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        if len(ring) != 6:
            continue
        if all(
            mol.GetAtomWithIdx(i).GetIsAromatic() and mol.GetAtomWithIdx(i).GetAtomicNum() == 6
            for i in ring
        ):
            rings.append(list(ring))
    return rings

def _ring_center_normal(mol: Chem.Mol, ring_idx_list, confId=0):
    conf = mol.GetConformer(confId)
    pts = np.array([conf.GetAtomPosition(i) for i in ring_idx_list], dtype=float)
    center = pts.mean(axis=0)
    # 用 SVD 求环面法向量（最小奇异向量）
    P = pts - center
    _, _, vh = np.linalg.svd(P, full_matrices=False)
    normal = vh[-1]
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    return center, normal

def distance_matrix(coord1, coord2=None):
    coord1 = np.array(coord1)
    coord2 = np.array(coord2) if coord2 is not None else coord1
    return np.sqrt(np.sum((coord1[:, np.newaxis, :] - coord2[np.newaxis, :, :])**2, axis=-1))

def check_3d_structure(
    mol,
    confId=0,
    dist_min=0.7,
    bond_s=2.7, bond_a=1.9, bond_d=1.8, bond_t=1.4,
    # —— 改成苯环面积检查 ——
    check_ring_area=True,
    ring_area_max=6.0,      # ⭐ 苯环面积阈值：>6 Å² 视为不合理
    enforce_fh=False,
    fh_min=1.8,
    auto_fix_fh=True,      # 如果你后面还要用自动修 F–H，就保留
):
    coord = np.array(mol.GetConformer(confId).GetPositions())

    dist_m = distance_matrix(coord)
    dist_m = np.where(dist_m == 0, dist_min, dist_m)

    # ========== 1) 键长检查 ==========
    bond_l_c = True
    for b in mol.GetBonds():
        i = b.GetBeginAtom().GetIdx()
        j = b.GetEndAtom().GetIdx()
        bond_l = dist_m[i, j]
        bt = b.GetBondTypeAsDouble()
        if (bt == 1.0 and bond_l > bond_s) or \
           (bt == 1.5 and bond_l > bond_a) or \
           (bt == 2.0 and bond_l > bond_d) or \
           (bt == 3.0 and bond_l > bond_t):
            bond_l_c = False
            break

    # ========== 2) 苯环面积检查 ==========
    ring_area_ok = True
    ring_areas = []

    if check_ring_area:
        rings = _benzene_rings(mol)
        if rings:
            for r in rings:
                area = _ring_area(mol, r, confId=confId)
                ring_areas.append(area)
                # 如果某个苯环面积 > ring_area_max，直接不通过
                if area > ring_area_max:
                    ring_area_ok = False
                    # 你可以打印调试信息：
                    # print(f"[WARN] Benzene ring area too large: {area:.3f} Å^2, ring={r}")
                    break

    # ========== 3) 非键合 F–H 距离检查（可选自动修复） ==========
    fh_ok = True
    if enforce_fh:
        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]

        bonded_pairs = set()
        for b in mol.GetBonds():
            i = b.GetBeginAtomIdx()
            j = b.GetEndAtomIdx()
            bonded_pairs.add((i, j))
            bonded_pairs.add((j, i))

        F_indices = [i for i, s in enumerate(symbols) if s == "F"]
        H_indices = [i for i, s in enumerate(symbols) if s == "H"]

        for i in F_indices:
            for j in H_indices:
                if (i, j) in bonded_pairs:
                    continue  # 跳过成键 F–H

                # 每次用最新坐标算距离
                coord = np.array(mol.GetConformer(confId).GetPositions())
                dist_m = distance_matrix(coord)
                dist_m = np.where(dist_m == 0, dist_min, dist_m)

                d_ij = dist_m[i, j]
                if d_ij < fh_min:
                    if auto_fix_fh:
                        ok = _separate_FH_pair(
                            mol,
                            idx_F=i,
                            idx_H=j,
                            target_dist=fh_min,
                            margin=0.05,
                            confId=confId,
                        )
                        if not ok:
                            fh_ok = False
                            break
                    else:
                        fh_ok = False
                        break
            if not fh_ok:
                break

    # ========== 4) 总体检查 ==========
    # ⭐ 这里把 fh_ok 和 ring_area_ok 都加进去
    coord = np.array(mol.GetConformer(confId).GetPositions())
    dist_m = distance_matrix(coord)
    dist_m = np.where(dist_m == 0, dist_min, dist_m)

    check = (dist_m.min() >= dist_min) and bond_l_c and ring_area_ok and fh_ok
    return check

def _separate_FH_pair(mol, idx_F, idx_H,
                      target_dist: float,
                      margin: float = 0.00,
                      confId: int = 0) -> bool:
    conf = mol.GetConformer(confId)
    pos_F = np.array(conf.GetAtomPosition(idx_F), dtype=float)
    pos_H = np.array(conf.GetAtomPosition(idx_H), dtype=float)

    v_FH = pos_H - pos_F
    d = np.linalg.norm(v_FH)
    if d < 1e-6:
        # F 和 H 几乎重合，方向不好定义
        return False

    v_dir = v_FH / d
    desired = target_dist + margin
    delta = desired - d
    if delta <= 0:
        # 已经够远，不用动
        return True

    # 两个原子各位移一半
    shift_F = -0.5 * delta * v_dir
    shift_H = +0.5 * delta * v_dir

    new_F = pos_F + shift_F
    new_H = pos_H + shift_H

    conf.SetAtomPosition(idx_F, Point3D(*new_F))
    conf.SetAtomPosition(idx_H, Point3D(*new_H))
    return True

def _ring_area(mol: Chem.Mol, ring_idx_list, confId=0) -> float:
    """
    计算一个苯环（6 元芳香碳）的面积，单位 Å^2。

    简要做法：
    1. 取环上 6 个原子的 3D 坐标；
    2. 用 SVD 求出环面的法向量；
    3. 在该平面上构造 2D 坐标 (u, v)；
    4. 用多边形面积公式（shoelace formula）算面积。
    """
    conf = mol.GetConformer(confId)
    pts = np.array([conf.GetAtomPosition(i) for i in ring_idx_list], dtype=float)  # (6, 3)
    center = pts.mean(axis=0)
    P = pts - center  # 去中心化

    # 用 SVD 求平面法向量
    _, _, vh = np.linalg.svd(P, full_matrices=False)
    normal = vh[-1]
    normal = normal / (np.linalg.norm(normal) + 1e-12)

    # 在环面上构造 2D 基底（利用你已有的 _orthonormal_basis）
    u, v = _orthonormal_basis(normal)   # 两个互相正交并与 normal 正交的单位向量

    # 投影到 (u, v) 平面，得到 2D 坐标
    xs = P @ u   # (6,)
    ys = P @ v   # (6,)

    # Shoelace formula
    area = 0.5 * np.abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))
    return float(area)

def calculate_box_size(numbers, pdb_files, density):
    total_mass = 0
    for num, file in zip(numbers, pdb_files):

        molecular_weight = calc_mol_weight(file)  # in g/mol
        total_mass += molecular_weight * num / 6.022e23  # accumulate mass of each molecule in grams

    total_volume = total_mass / density  # volume in cm^3
    length = (total_volume * 1e24) ** (1 / 3)  # convert to Angstroms
    return length


def calc_mol_weight(pdb_file):
    try:
        mol = Chem.MolFromPDBFile(pdb_file, removeHs=False, sanitize=False)
        if mol:
            Chem.SanitizeMol(mol)
            return Descriptors.MolWt(mol)
        else:
            raise ValueError(f"RDKit 无法解析 PDB 文件: {pdb_file}")
    except (Chem.rdchem.AtomValenceException, Chem.rdchem.KekulizeException, ValueError):
        # 如果 RDKit 解析失败，尝试手动计算分子量
        try:
            atom_counts = defaultdict(int)
            with open(pdb_file, 'r') as f:
                for line in f:
                    if line.startswith(("ATOM", "HETATM")):
                        element = line[76:78].strip()
                        if not element:
                            # 从原子名称推断元素符号
                            atom_name = line[12:16].strip()
                            element = ''.join([char for char in atom_name if char.isalpha()]).upper()[:2]
                        atom_counts[element] += 1

            # 常见元素的原子质量（g/mol）
            atomic_weights = {
                'H': 1.008,
                'C': 12.011,
                'N': 14.007,
                'O': 15.999,
                'F': 18.998,
                'P': 30.974,
                'S': 32.06,
                'CL': 35.45,
                'BR': 79.904,
                'I': 126.904,
                'FE': 55.845,
                'ZN': 65.38,
                # 根据需要添加更多元素
            }

            mol_weight = 0.0
            for atom, count in atom_counts.items():
                weight = atomic_weights.get(atom.upper())
                if weight is None:
                    raise ValueError(f"未知的原子类型 '{atom}' 在 PDB 文件: {pdb_file}")
                mol_weight += weight * count
            return mol_weight
        except Exception as e:
            raise ValueError(f"无法计算分子量，PDB 文件: {pdb_file}，错误: {e}")

def _fibonacci_sphere(n=64):
    import numpy as np
    phi = (1 + 5**0.5) / 2
    i = np.arange(n)
    z = 1 - 2*(i + 0.5)/n
    r = np.sqrt(np.maximum(0.0, 1 - z*z))
    theta = 2*np.pi*i/phi
    x, y = r*np.cos(theta), r*np.sin(theta)
    return np.vstack([x, y, z]).T

def _directions_in_cone_fibonacci(base_dir: np.ndarray, half_deg: float = 30.0, n: int = 96):
    base = base_dir / (np.linalg.norm(base_dir) + 1e-12)
    cands = _fibonacci_sphere(n)
    cos_half = np.cos(np.deg2rad(half_deg))
    dots = cands @ base
    mask = dots >= cos_half
    # 保留圆锥内方向，并且把 base_dir 本身放在第一位
    dirs = [base.copy()]
    if mask.any():
        sel = cands[mask]
        # 简单按与 base_dir 的点积降序（更贴近基向量）
        order = np.argsort(-(sel @ base))
        dirs += [d/np.linalg.norm(d) for d in sel[order]]
    return dirs

def _early_clearance_prune(connecting_mol: Chem.Mol, tail_idx: int, R_unit: float,
                           dirs: list[np.ndarray], bond_length: float,
                           lookahead: float = 1.0, samples: int = 4, scale: float = 0.85,
                           min_margin: float = 0.0):
    tree, Z = _polymer_kdtree(connecting_mol, exclude_idx={tail_idx}, skip_h=True)
    conf = connecting_mol.GetConformer()
    tail_pos = np.array(conf.GetAtomPosition(tail_idx), dtype=float)
    kept = []
    for d in dirs:
        s_vals = np.linspace(bond_length, bond_length + lookahead, samples)
        ok = True
        for s in s_vals:
            pt = tail_pos + s*d
            margin = _clearance_margin_at_point(tree, Z, pt, R_unit, scale=scale)
            if margin < min_margin:
                ok = False
                break
        if ok:
            kept.append(d)
    return kept if kept else dirs[:16]  # 全部不达标时保留少量兜底

def _norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

def _parse_growth_axis(growth_axis, mol_first: Chem.Mol, h_idx: int, t_idx: int) -> np.ndarray:
    """
    growth_axis 支持：
      - 字符串：'+z','-z','+x','-x','+y','-y','auto'
      - 长度为3的数组或 np.ndarray
    'auto'：用首个单体的 (tail - head) 方向。
    """
    if isinstance(growth_axis, (list, tuple, np.ndarray)):
        return _norm(np.array(growth_axis, dtype=float))

    if isinstance(growth_axis, str):
        s = growth_axis.lower().strip()
        if s in ['+z', 'z', 'up']:        return np.array([0., 0., 1.])
        if s in ['-z', 'down']:           return np.array([0., 0., -1.])
        if s in ['+x', 'x', 'right']:     return np.array([1., 0., 0.])
        if s in ['-x', 'left']:           return np.array([-1., 0., 0.])
        if s in ['+y', 'y', 'front']:     return np.array([0., 1., 0.])
        if s in ['-y', 'back']:           return np.array([0., -1., 0.])
        if s == 'auto':
            conf = mol_first.GetConformer()
            v = np.array(conf.GetAtomPosition(t_idx)) - np.array(conf.GetAtomPosition(h_idx))
            if np.linalg.norm(v) < 1e-6:
                return np.array([0., 0., 1.])
            return _norm(v)

    # fallback
    return np.array([0., 0., 1.])









