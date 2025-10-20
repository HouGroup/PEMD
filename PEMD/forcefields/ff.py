

import os
import shutil
import pandas as pd
import parmed as pmd
import importlib.resources as pkg_resources

from pathlib import Path
from foyer import Forcefield
from collections import defaultdict

from PEMD import io
from PEMD.forcefields.xml import XMLGenerator
from PEMD.forcefields.ligpargen import PEMDLigpargen


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module=r"foyer\.forcefield")


def get_xml_ligpargen(
        work_dir: Path | str,
        name: str,
        resname: str,
        *,
        pdb_file: str | None = None,
        charge: float = 0,
        charge_model: str = 'CM1A-LBCC',
):

    work_path = Path(work_dir)
    ligpargen_dir = work_path / f'ligpargen_{name}'
    ligpargen_dir.mkdir(parents=True, exist_ok=True)

    pdb_path = work_path / pdb_file

    PEMDLigpargen(
        ligpargen_dir,
        name,
        resname,
        charge,
        charge_model,
        filename = pdb_path,
    ).run_local()

    mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=False)

    itp_path = ligpargen_dir / f"{name}.gmx.itp"
    xml_path = work_path / f"{name}.xml"
    generator = XMLGenerator(
        itp_path,
        mol,
        xml_path
    )
    generator.run()

    temp_sdf = work_path / "temp.sdf"
    temp_sdf.unlink(missing_ok=True)
    pdb_path.unlink(missing_ok=True)

    csv_path = ligpargen_dir / f"{name}.csv"
    chg_df = pd.read_csv(csv_path)

    return chg_df


def get_oplsaa_xml(
        work_dir: Path | str,
        name: str,
        pdb_file: Path | str,
) -> str:

    work_path = Path(work_dir)
    md_dir = work_path / "MD_dir"
    md_dir.mkdir(parents=True, exist_ok=True)

    untyped_str = pmd.load_file(pdb_file, structure=True)

    xml_path = work_path / f"{name}.xml"
    oplsaa = Forcefield(forcefield_files = xml_path)
    typed_str = oplsaa.apply(untyped_str, verbose=True, use_residue_map=True)

    top_path = md_dir / f"{name}.top"
    gro_path = md_dir / f"{name}.gro"
    typed_str.save(str(top_path), overwrite=True)
    typed_str.save(str(gro_path), overwrite=True)

    nonbonded_itp = md_dir / f"{name}_nonbonded.itp"
    bonded_itp    = md_dir / f"{name}_bonded.itp"

    io.extract_from_top(top_path, nonbonded_itp, nonbonded=True, bonded=False)
    io.extract_from_top(top_path, bonded_itp, nonbonded=False, bonded=True)

    pdb_path = os.path.join(md_dir, f'{name}.pdb')
    io.convert_gro_to_pdb(gro_path, pdb_path, )

    top_path.unlink(missing_ok=True)
    xml_path.unlink(missing_ok=True)

    print(f'Generate {name}_bonded.itp and {name}_nonbonded.itp in ./{md_dir} path.')

    return f'{name}_bonded.itp'


def get_oplsaa_ligpargen(work_dir, name, resname, chg, smiles, charge_model, ):

    ligpargen_dir = os.path.join(work_dir, f'ligpargen_{name}')
    os.makedirs(ligpargen_dir, exist_ok=True)

    MD_dir = os.path.join(work_dir, 'MD_dir')
    os.makedirs(MD_dir, exist_ok=True)

    xyz_filename = f'{name}.xyz'
    io.smiles_to_xyz(smiles, os.path.join(work_dir,  xyz_filename))

    PEMDLigpargen(
        ligpargen_dir,
        name,
        resname,
        chg,
        charge_model,
        filename = xyz_filename,
    ).run_local()

    nonbonditp_filename = os.path.join(MD_dir, f'{name}_nonbonded.itp')
    bonditp_filename = os.path.join(MD_dir, f'{name}_bonded.itp')
    pdb_filename = os.path.join(MD_dir, f'{name}.pdb')

    top_filename = os.path.join(ligpargen_dir, f"{name}.gmx.itp")
    gro_filename = os.path.join(ligpargen_dir, f'{name}.gmx.gro')
    io.extract_from_top(top_filename, nonbonditp_filename, nonbonded=True, bonded=False)
    io.extract_from_top(top_filename, bonditp_filename, nonbonded=False, bonded=True)
    io.convert_gro_to_pdb(gro_filename, pdb_filename,)

    return f'{name}_bonded.itp'


def gen_ff_from_data(work_dir, compound_name, corr_factor, target_sum_chg):

    MD_dir = os.path.join(work_dir, 'MD_dir')
    os.makedirs(MD_dir, exist_ok=True)

    files_to_copy = [
        f"pdb/{compound_name}.pdb",
        f"itp/{compound_name}_bonded.itp",
        f"itp/{compound_name}_nonbonded.itp"
    ]
    for file_path in files_to_copy:
        try:
            resource_dir = pkg_resources.files('PEMD.forcefields')
            resource_path = resource_dir.joinpath(file_path)
            shutil.copy(str(resource_path), MD_dir)
            print(f"Copied {file_path} to {MD_dir} successfully.")
        except Exception as e:
            print(f"Failed to copy {file_path}: {e}")

    filename = f"{compound_name}_bonded.itp"
    scale_chg_itp(MD_dir, filename, corr_factor, target_sum_chg)
    # print(f"scale charge successfully.")

# def find_substruct_matches(target_mol, query_mol, *, uniquify=True):
#     """Return substructure matches with fallbacks for aromatic systems."""
#
#     if target_mol is None or query_mol is None:
#         return []
#
#     matches = list(target_mol.GetSubstructMatches(
#         query_mol,
#         uniquify=uniquify,
#         useChirality=False,
#     ))
#     if matches:
#         return matches
#
#     target_copy = Chem.Mol(target_mol)
#     query_copy = Chem.Mol(query_mol)
#
#     try:
#         Chem.Kekulize(target_copy, clearAromaticFlags=True)
#         Chem.Kekulize(query_copy, clearAromaticFlags=True)
#     except KekulizeException:
#         return matches
#
#     return list(target_copy.GetSubstructMatches(
#         query_copy,
#         uniquify=uniquify,
#         useChirality=False,
#     ))
from rdkit import Chem
from rdkit.Chem.rdchem import KekulizeException

def find_substruct_matches(
    target_mol,
    query_mol,
    *,
    uniquify=False,
    ignore_stereo=True,          # 忽略手性/顺反（默认更宽松）
    allow_aromatic_conj=True,    # 芳香↔共轭兼容（新版本更稳）
    match_hs=True,               # 显式氢也参与匹配（和你“保持氢”的需求一致）
    try_remove_stereo=True       # 第一层回退：移除立体信息
):
    """Return substructure matches with robust fallbacks and cross-version guards."""
    if target_mol is None or query_mol is None:
        return []

    # --- 小工具：跨版本获取匹配，保证确定性排序 ---
    def _get_matches(t, q, params):
        try:
            ms = list(t.GetSubstructMatches(q, params=params))
        except TypeError:
            # 老版本 RDKit 没有 params=，退化为手动传常用参数
            ms = list(t.GetSubstructMatches(
                q,
                useChirality=getattr(params, 'useChirality', False),
                uniquify=uniquify
            ))
        # 确定性排序：先按长度，再按元组字典序
        return sorted(ms, key=lambda m: (len(m), m))

    # --- 优先路径：使用 SubstructMatchParameters 控制开关 ---
    p = Chem.SubstructMatchParameters()
    # 顶层开关
    p.uniquify = uniquify
    if hasattr(p, 'useChirality'):
        p.useChirality = not ignore_stereo
    if hasattr(p, 'useEnhancedStereo'):
        p.useEnhancedStereo = not ignore_stereo
    if hasattr(p, 'aromaticMatchesConjugated'):
        p.aromaticMatchesConjugated = allow_aromatic_conj
    if hasattr(p, 'useHs'):
        p.useHs = match_hs
    if hasattr(p, 'maxMatches'):
        p.maxMatches = 0  # 0 = 不限数量

    # 直匹配
    matches = _get_matches(target_mol, query_mol, p)
    if matches:
        return matches

    # --- 回退 1：完全去除立体信息再匹配（更宽容的等价拓扑） ---
    if try_remove_stereo:
        t2 = Chem.Mol(target_mol); Chem.RemoveStereochemistry(t2)
        q2 = Chem.Mol(query_mol);  Chem.RemoveStereochemistry(q2)
        matches = _get_matches(t2, q2, p)
        if matches:
            return matches

    # --- 回退 2：Kekulize，清除芳香标记再匹配（应对芳香/共轭体系差异） ---
    t3 = Chem.Mol(target_mol)
    q3 = Chem.Mol(query_mol)
    try:
        Chem.Kekulize(t3, clearAromaticFlags=True)
        Chem.Kekulize(q3, clearAromaticFlags=True)
    except KekulizeException:
        return []

    # 清了芳香标记后，必要时把 aromaticMatchesConjugated 关掉
    if hasattr(p, 'aromaticMatchesConjugated'):
        p.aromaticMatchesConjugated = False

    return _get_matches(t3, q3, p)

def select_non_overlapping_matches(matches, used_atoms=None):
    """Select non-overlapping matches in a deterministic order."""

    if not matches:
        return []

    if used_atoms is None:
        used_atoms = set()
    else:
        used_atoms = set(used_atoms)

    selected = []
    for match in sorted(matches, key=lambda m: (min(m), m)):
        if any(atom_idx in used_atoms for atom_idx in match):
            continue
        selected.append(match)
        used_atoms.update(match)
    return selected


def assign_partial_charges(mol_poly, sub_mol, matches):
    """Assign partial charges from ``sub_mol`` to the corresponding atoms in ``mol_poly``."""
    if sub_mol is None:
        return
    for match in matches:
        for sub_atom_idx, poly_atom_idx in enumerate(match):
            sub_atom = sub_mol.GetAtomWithIdx(sub_atom_idx)
            if not sub_atom.HasProp('partial_charge'):
                continue
            charge = float(sub_atom.GetProp('partial_charge'))
            mol_poly.GetAtomWithIdx(poly_atom_idx).SetDoubleProp('partial_charge', charge)

def mol_to_charge_df(mol):

    data = []
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        atom_symbol = atom.GetSymbol()
        # Check whether the atom has a ``partial_charge`` property
        if atom.HasProp('partial_charge'):
            charge = float(atom.GetProp('partial_charge'))
        else:
            charge = None  # Or fall back to 0.0 or another default
        data.append({'atom_index': atom_idx, 'atom': atom_symbol, 'charge': charge})

    # Build a DataFrame sorted by atom index
    df = pd.DataFrame(data)
    df = df.sort_values('atom_index').reset_index(drop=True)

    return df

def ensure_all_atoms_have_charges(mol):
    """Ensure that every atom in ``mol`` has an assigned partial charge."""

    def missing_atoms(current_mol):
        return [atom.GetIdx() for atom in current_mol.GetAtoms()
                if not atom.HasProp('partial_charge')]

    missing = missing_atoms(mol)
    if not missing:
        return

    # Propagate charges from neighbouring atoms first
    for _ in range(3):
        updated = False
        still_missing = []
        for atom_idx in missing:
            atom = mol.GetAtomWithIdx(atom_idx)
            neighbour_charges = [
                neigh.GetDoubleProp('partial_charge')
                for neigh in atom.GetNeighbors()
                if neigh.HasProp('partial_charge')
            ]
            if neighbour_charges:
                average_charge = sum(neighbour_charges) / len(neighbour_charges)
                atom.SetDoubleProp('partial_charge', average_charge)
                updated = True
            else:
                still_missing.append(atom_idx)
        missing = still_missing
        if not missing or not updated:
            break

    # Final fallback for any isolated atoms that still miss charges
    for atom_idx in missing:
        mol.GetAtomWithIdx(atom_idx).SetDoubleProp('partial_charge', 0.0)


def apply_chg_to_poly(
        work_dir,
        mol_short,
        mol_long,
        itp_file,
        resp_chg_df,
        repeating_unit,
        end_repeating,
        scale,
        charge,
):

    MD_dir = os.path.join(work_dir, 'MD_dir')
    os.makedirs(MD_dir, exist_ok=True)

    left_mol, right_mol, mid_mol = apply_chg2mol(
        resp_chg_df,
        mol_short,
        repeating_unit,
        end_repeating
    )

    Chem.SanitizeMol(mol_long)
    mol_poly = Chem.AddHs(mol_long)

    # Match ``left_mol`` within ``mol_poly``
    left_matches = []
    rw_mol = Chem.RWMol(mol_poly)
    used_atoms = set()
    # all_left = list(rw_mol.GetSubstructMatches(left_mol, uniquify=True, useChirality=False))
    all_left = find_substruct_matches(rw_mol, left_mol)

    if all_left:
        left_match = min(all_left, key=lambda m: sum(m)/len(m))
        left_matches.append(left_match)
        used_atoms.update(left_match)
    print(f"Matches for left_mol: {left_matches}")

    # Match ``right_mol`` within ``mol_poly``
    right_matches = []
    rw_mol = Chem.RWMol(mol_poly)
    # all_right = list(rw_mol.GetSubstructMatches(right_mol, uniquify=True, useChirality=False))
    all_right = find_substruct_matches(rw_mol, right_mol)
    if all_right:
        right_match = max(all_right, key=lambda m: sum(m)/len(m))
        if not any(atom_idx in used_atoms for atom_idx in right_match):
            right_matches.append(right_match)
            used_atoms.update(right_match)
    print(f"Matches for right_mol: {right_matches}")

    # Assign partial charges for the matching atoms in ``mol_poly``
    assign_partial_charges(mol_poly, left_mol, left_matches)
    assign_partial_charges(mol_poly, right_mol, right_matches)

    # Match ``mid_mol`` to the repeating units in ``mol_poly`` and assign charges
    mid_matches = []
    for match in select_non_overlapping_matches(
            find_substruct_matches(rw_mol, mid_mol),
            used_atoms,
    ):
    # for match in rw_mol.GetSubstructMatches(mid_mol, uniquify=True, useChirality=False):
    #     if any(atom_idx in used_atoms for atom_idx in match):
    #         continue  # Skip overlapping matches
        mid_matches.append(match)
        used_atoms.update(match)  # Mark atoms as used
    print(f"Matches for mid_mol: {mid_matches}")

    # Assign partial charges to the ``mid_mol`` matches
    assign_partial_charges(mol_poly, mid_mol, mid_matches)

    ensure_all_atoms_have_charges(mol_poly)

    # Extract the updated charges into a DataFrame
    charge_update_df = mol_to_charge_df(mol_poly)
    # print(charge_update_df)

    # Charge neutralize and scale
    charge_update_df_cor = charge_neutralize_scale(charge_update_df, scale, charge, )

    # Update the itp file
    update_itp_file(MD_dir, itp_file, charge_update_df_cor)


def apply_chg2mol(resp_chg_df, mol_poly, repeating_unit, end_repeating):
    # 1. Generate the polymer molecule from SMILES and add hydrogens
    # mol_poly = Chem.Mol(mol_poly)
    # Chem.SanitizeMol(mol_poly)
    # mol_poly = Chem.AddHs(mol_poly)
    # print("mol_poly SMILES:", Chem.MolToSmiles(mol_poly))
    # Chem.MolToPDBFile(mol_poly, "test.pdb")

    # 2. Write RESP charges into ``mol_poly``
    resp_chg_df = resp_chg_df.copy()
    max_idx = resp_chg_df['position'].max()
    if max_idx == mol_poly.GetNumAtoms():
        resp_chg_df['position'] = resp_chg_df['position'] - 1

    for _, row in resp_chg_df.iterrows():
        pos = int(row['position'])
        charge = float(row['charge'])
        if pos < 0 or pos >= mol_poly.GetNumAtoms():
            # Skip invalid index
            continue
        atom = mol_poly.GetAtomWithIdx(pos)
        atom.SetDoubleProp('partial_charge', charge)  # Store the charge via SetDoubleProp
        # if not atom.HasProp('partial_charge'):
        #     print(f"Atom {pos} has no attribute!")
        # else:
        #     print(f"Atom {pos} partial_charge = {atom.GetDoubleProp('partial_charge')}")

    partial_charges = [
        float(row['charge'])
        for _, row in resp_chg_df.sort_values('position').iterrows()
    ]
    mol_poly.SetProp("partial_charges", ','.join(map(str, partial_charges)))

    # ==========  Generate forward mol_unit_fwd  ==========
    mol_unit_fwd = Chem.MolFromSmiles(repeating_unit)
    mol_unit_fwd = Chem.AddHs(mol_unit_fwd)
    # Remove star atoms
    edit_fwd = Chem.EditableMol(mol_unit_fwd)
    for atom in reversed(list(mol_unit_fwd.GetAtoms())):
        if atom.GetSymbol() == '*':
            edit_fwd.RemoveAtom(atom.GetIdx())
    mol_unit_fwd = edit_fwd.GetMol()

    # ==========  Generate reverse mol_unit_rev  ==========
    # Rearrange atoms based on mol_unit_fwd
    num_atoms_fwd = mol_unit_fwd.GetNumAtoms()
    new_order = list(range(num_atoms_fwd - 1, -1, -1))
    mol_unit_rev = Chem.RenumberAtoms(mol_unit_fwd, new_order)

    # ==========  Perform substructure matching on mol_poly  ==========
    rw_mol = Chem.RWMol(mol_poly)

    # Forward matches
    fwd_used_atoms = set()
    fwd_matches = []
    for match in rw_mol.GetSubstructMatches(mol_unit_fwd, uniquify=True, useChirality=False):
        if any(atom_idx in fwd_used_atoms for atom_idx in match):
            continue
        fwd_matches.append(match)
        fwd_used_atoms.update(match)

    # Reverse matches
    rev_used_atoms = set()
    rev_matches = []
    for match in rw_mol.GetSubstructMatches(mol_unit_rev, uniquify=True, useChirality=False):
        if any(atom_idx in rev_used_atoms for atom_idx in match):
            continue
        rev_matches.append(match)
        rev_used_atoms.update(match)

    # print("fwd_matches:", fwd_matches)
    # print("rev_matches:", rev_matches)

    # ==========  Select the best match  ==========
    # Use the match with the highest atom count as the metric; adjust as needed
    if len(fwd_matches) >= len(rev_matches):
        best_matches = fwd_matches
        best_mol_unit = mol_unit_fwd
        best_direction = "forward"
    else:
        best_matches = rev_matches
        best_mol_unit = mol_unit_rev
        best_direction = "reverse"

    # print(f"Best direction: {best_direction}, matches found: {len(best_matches)}")

    # Return immediately if no matches are found
    if not best_matches:
        print("No matches found in either direction!")
        return None, None, None

    # ==========  Subsequent operations rely on best_matches and best_mol_unit  ==========
    # The following logic mirrors the original but uses the selected best match variables
    matched_atoms = set()
    for match in best_matches:
        matched_atoms.update(match)

    no_matched_atoms = [
        atom.GetIdx() for atom in mol_poly.GetAtoms()
        if atom.GetIdx() not in matched_atoms
    ]

    # Initialize the terminal atom list
    left_end_atoms = []
    right_end_atoms = []

    # Ensure best_matches is not empty
    if best_matches:
        left_neighbor = set(best_matches[0])
        right_neighbor = set(best_matches[-1])

        # Iterate over a copy to avoid mutating the original list
        for idx in no_matched_atoms[:]:
            atom = mol_poly.GetAtomWithIdx(idx)
            neighbors = atom.GetNeighbors()
            for neighbor in neighbors:
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx in left_neighbor:
                    left_end_atoms.append(idx)
                    left_neighbor.add(idx)
                    no_matched_atoms.remove(idx)
                    break
                elif neighbor_idx in right_neighbor:
                    right_end_atoms.append(idx)
                    right_neighbor.add(idx)
                    no_matched_atoms.remove(idx)
                    break
    # print(f"Left end atoms: {left_end_atoms}")
    # print(f"Right end atoms: {right_end_atoms}")

    # Assemble the left and right terminal atoms
    left_atoms = left_end_atoms + [
        atom_idx for match in best_matches[:end_repeating] for atom_idx in match
    ]
    right_atoms = right_end_atoms + [
        atom_idx for match in best_matches[-end_repeating:] for atom_idx in match
    ]
    # print("left_end_atoms:", left_end_atoms, "right_end_atoms:", right_end_atoms)

    # Assume a helper ``gen_molfromindex(mol, index_list)`` extracts submolecules by atom indices
    left_mol = gen_molfromindex(mol_poly, left_atoms)
    right_mol = gen_molfromindex(mol_poly, right_atoms)

    # Print and fetch partial charges, assigning them to left_mol / right_mol
    for i, atom_idx in enumerate(left_atoms):
        charge = mol_poly.GetAtomWithIdx(atom_idx).GetDoubleProp('partial_charge')
        if i < left_mol.GetNumAtoms():
            left_mol.GetAtomWithIdx(i).SetDoubleProp('partial_charge', charge)
        # print(f"Average charge at left position {atom_idx}: {charge}")

    for i, atom_idx in enumerate(right_atoms):
        charge = mol_poly.GetAtomWithIdx(atom_idx).GetDoubleProp('partial_charge')
        if i < right_mol.GetNumAtoms():
            right_mol.GetAtomWithIdx(i).SetDoubleProp('partial_charge', charge)
        # print(f"Average charge at right position {atom_idx}: {charge}")

    # Assume each match in the middle of best_matches represents a repeating unit
    mid_atoms = best_matches[1:-1]
    num_repeats = len(mid_atoms)
    # print(f"Number of repeating units in the middle: {num_repeats}")

    # Determine the atom count per repeating unit
    if num_repeats > 0:
        atoms_per_unit = len(mid_atoms[0])
        # print(f"Atoms per repeating unit: {atoms_per_unit}")
    else:
        atoms_per_unit = 0

    # Compute the average charge for each position within the middle units
    charge_dict = defaultdict(list)
    for repeat_idx, match in enumerate(mid_atoms):
        for pos, atom_idx in enumerate(match):
            charge = mol_poly.GetAtomWithIdx(atom_idx).GetDoubleProp('partial_charge')
            charge_dict[pos].append(charge)

    avg_charges = {}
    for pos, charges in charge_dict.items():
        avg_charge = sum(charges) / len(charges)
        avg_charges[pos] = avg_charge
        # print(f"Average charge at position {pos}: {avg_charge}")

    # Use the first repeating unit as the template
    if num_repeats > 0:
        template_match = mid_atoms[0]
        template_atoms = list(template_match)
        mid_mol = gen_molfromindex(mol_poly, template_atoms)

        # Reassign partial charges to mid_mol
        for pos, atom in enumerate(mid_mol.GetAtoms()):
            atom.SetDoubleProp('partial_charge', avg_charges[pos])
    else:
        mid_mol = None

    return left_mol, right_mol, mid_mol


def ave_chg_to_df(resp_chg_df, N_mid, N_mid_h, N_left, N_right, N_left_h, N_right_h, ):
    # deal with non-H atoms
    nonH_df = resp_chg_df[resp_chg_df['atom'] != 'H']

    top_N_noH_df = nonH_df.head(N_left)
    tail_N_noH_df = nonH_df.tail(N_right)
    mid_df_noH_df = nonH_df.drop(nonH_df.head(N_left).index.union(nonH_df.tail(N_right).index)).reset_index(drop=True)
    mid_ave_chg_noH_df = ave_mid_chg(mid_df_noH_df, N_mid)

    # deal with H atoms
    H_df = resp_chg_df[resp_chg_df['atom'] == 'H']

    top_N_H_df = H_df.head(N_left_h)
    tail_N_H_df = H_df.tail(N_right_h)
    mid_df_H_df = H_df.drop(H_df.head(N_left_h).index.union(H_df.tail(N_right_h).index)).reset_index(drop=True)
    mid_ave_chg_H_df = ave_mid_chg(mid_df_H_df, N_mid_h)

    return top_N_noH_df, tail_N_noH_df, mid_ave_chg_noH_df, top_N_H_df, tail_N_H_df, mid_ave_chg_H_df


def update_itp_file(MD_dir, itp_file, charge_update_df_cor):
    itp_filepath = os.path.join(MD_dir, itp_file)
    with open(itp_filepath, 'r') as file:
        lines = file.readlines()

    # Locate the start and end of the [ atoms ] section
    in_section = False  # Flag whether we are inside the section
    start_index = end_index = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("[") and 'atoms' in line.split():
            in_section = True
            continue
        if in_section:
            if line.strip().startswith(";"):
                start_index = i + 1  # Skip the section header and column names
                continue
            if line.strip() == "":
                end_index = i
                break

    # update the charge value in the [ atoms ] section
    charge_index = 0  # Track the current index within the charge DataFrame
    for i in range(start_index, end_index):
        parts = lines[i].split()
        if charge_index < len(charge_update_df_cor):
            new_charge = charge_update_df_cor.iloc[charge_index]['charge']
            parts[6] = f'{new_charge:.8f}'  # Update the charge value (assumed in column 7)
            lines[i] = ' '.join(parts) + '\n'
            charge_index += 1

    # save the updated itp file
    new_itp_filepath = os.path.join(MD_dir, itp_file)
    with open(new_itp_filepath, 'w') as file:
        file.writelines(lines)


def ave_end_chg(df, N):
    # Process average charges for terminal atoms
    top_N_df = df.head(N)
    tail_N_df = df.tail(N).iloc[::-1].reset_index(drop=True)
    average_charge = (top_N_df['charge'].reset_index(drop=True) + tail_N_df['charge']) / 2
    average_df = pd.DataFrame({
        'atom': top_N_df['atom'].reset_index(drop=True),  # Preserve atom names
        'charge': average_charge
    })
    return average_df


def ave_mid_chg(df, atom_count):
    # Process average charges for interior atoms
    average_charges = []
    for i in range(atom_count):
        same_atoms = df[df.index % atom_count == i]
        avg_charge = same_atoms['charge'].mean()
        average_charges.append({'atom': same_atoms['atom'].iloc[0], 'charge': avg_charge})
    return pd.DataFrame(average_charges)


def smiles_to_df(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    mol = Chem.AddHs(mol)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]

    df = pd.DataFrame(atoms, columns=['atom'])
    df['charge'] = None
    return df

def gen_molfromindex(mol, idx):
    editable_mol = Chem.EditableMol(Chem.Mol())

    atom_map = {}  # Map from original to new indices
    for old_idx in idx:
        atom = mol.GetAtomWithIdx(old_idx)
        new_idx = editable_mol.AddAtom(atom)
        atom_map[old_idx] = new_idx

    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        if begin_idx in idx  and end_idx in idx:
            new_begin = atom_map[begin_idx]
            new_end = atom_map[end_idx]
            bond_type = bond.GetBondType()
            editable_mol.AddBond(new_begin, new_end, bond_type)

    return editable_mol.GetMol()

def charge_neutralize_scale(df, correction_factor=1, target_total_charge=0, ):
    current_total_charge = df['charge'].sum()
    charge_difference = target_total_charge - current_total_charge
    charge_adjustment_per_atom = charge_difference / len(df)
    df['charge'] = (df['charge'] + charge_adjustment_per_atom) * correction_factor

    return df

def apply_chg_to_molecule(work_dir, itp_file, chg_df, scale, charge, ):
    MD_dir = os.path.join(work_dir, 'MD_dir')
    os.makedirs(MD_dir, exist_ok=True)

    # charge neutralize and scale
    charge_update_df_cor = charge_neutralize_scale(chg_df, scale, charge, )

    # update the itp file
    update_itp_file(MD_dir, itp_file, charge_update_df_cor)


# work_dir, filename, corr_factor, target_sum_chg
def scale_chg_itp(work_dir, filename, corr_factor, target_sum_chg):
    filename = os.path.join(work_dir, filename)
    start_reading = False
    atoms = []

    with open(filename, 'r') as file:
        for line in file:
            if line.strip().startswith("[") and 'atoms' in line.split():  # Locate the start of the atom data
                start_reading = True
                continue
            if start_reading:
                if line.strip() == "":  # Stop when encountering a blank line
                    break
                if line.strip().startswith(";"):  # Skip comment lines
                    continue
                parts = line.split()
                if len(parts) >= 7:  # Ensure the row has sufficient data
                    atom_id = parts[4]  # Assume atom type is in column 5
                    charge = float(parts[6])  # Assume charge is in column 7
                    atoms.append([atom_id, charge])

    # create DataFrame
    df = pd.DataFrame(atoms, columns=['atom', 'charge'])
    # charge neutralize and scale
    charge_update_df_cor = charge_neutralize_scale(df, corr_factor, target_sum_chg, )

    # reas itp file
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Locate the start and end of the [ atoms ] section
    in_section = False  # Flag whether we are inside the section
    start_index = end_index = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("[") and 'atoms' in line.split():
            in_section = True
            continue
        if in_section:
            if line.strip().startswith(";"):
                start_index = i + 1  # Skip the section header and column names
                continue
            if line.strip() == "":
                end_index = i
                break

    # update the charge value in the [ atoms ] section
    charge_index = 0  # Track the current index within the charge DataFrame
    for i in range(start_index, end_index):
        parts = lines[i].split()
        if charge_index < len(charge_update_df_cor):
            new_charge = charge_update_df_cor.iloc[charge_index]['charge']
            parts[6] = f'{new_charge:.8f}'  # Update the charge value (assumed in column 7)
            lines[i] = ' '.join(parts) + '\n'
            charge_index += 1

    # save the updated itp file
    with open(filename, 'w') as file:
        file.writelines(lines)











