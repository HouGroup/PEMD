
# ****************************************************************************** #
#        The module implements functions to calculate the coordination           #
# ****************************************************************************** #


import os
import re
import glob
import logging
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import MDAnalysis as mda
import matplotlib.pyplot as plt

from rdkit import Chem
from tqdm.auto import tqdm
from collections import deque
from rdkit.Geometry import Point3D
from MDAnalysis.analysis import rdf
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from PEMD.model import polymer, model_lib
from matplotlib.colors import LinearSegmentedColormap
from concurrent.futures import ProcessPoolExecutor, as_completed
from PEMD.analysis.utils import minimum_image_displacement


warnings.filterwarnings("ignore", category=UserWarning, module='MDAnalysis.coordinates.PDB')
logging.getLogger('MDAnalysis').setLevel(logging.WARNING)


def calc_rdf_coord(group1, group2, v, nbins=200, range_rdf=(0.0, 10.0)):
    # Initialize RDF analysis
    rdf_analysis = rdf.InterRDF(group1, group2, nbins=nbins, range=range_rdf)
    rdf_analysis.run()

    # Calculate coordination numbers
    rho = group2.n_atoms / v  # Density of the second group
    bins = rdf_analysis.results.bins
    rdf_values = rdf_analysis.results.rdf
    coord_numbers = np.cumsum(4 * np.pi * bins**2 * rdf_values * np.diff(np.append(0, bins)) * rho)

    return bins, rdf_values, coord_numbers

def obtain_rdf_coord(bins, rdf, coord_numbers):

    deriv_sign_changes = np.diff(np.sign(np.diff(rdf)))
    peak_index = np.where(deriv_sign_changes < 0)[0] + 1
    if len(peak_index) == 0:
        raise ValueError("No peak found in RDF data.")
    first_peak_index = peak_index[0]

    min_after_peak_index = np.where(deriv_sign_changes[first_peak_index:] > 0)[0] + first_peak_index + 1
    if len(min_after_peak_index) == 0:
        raise ValueError("No minimum found after the first peak in RDF data.")
    first_min_index = min_after_peak_index[0]

    x_val = round(float(bins[first_min_index]), 3)
    y_coord = round(float(np.interp(x_val, bins, coord_numbers)), 3)

    return x_val, y_coord

def load_md_trajectory(work_dir, tpr_filename='nvt_prod.tpr', xtc_filename='nvt_prod.xtc'):
    data_tpr_file = os.path.join(work_dir, tpr_filename)
    data_xtc_file = os.path.join(work_dir, xtc_filename)
    u = mda.Universe(data_tpr_file, data_xtc_file)
    return u

def analyze_coordination(universe, li_atoms, molecule_groups, cutoff_radii, run_start, run_end):
    num_timesteps = run_end - run_start
    num_li_atoms = len(li_atoms)
    coordination = np.zeros((num_timesteps, num_li_atoms), dtype=int)

    for ts_index, ts in enumerate(tqdm(universe.trajectory[run_start:run_end], desc='Processing')):
        box_size = ts.dimensions[0:3]
        for li_index, li in enumerate(li_atoms):
            encoded_coordination = 0
            factor = 10**(len(molecule_groups) - 1)  # Factor for encoding counts at different decimal places
            for group_name, group_atoms in molecule_groups.items():
                d_vec = minimum_image_displacement(group_atoms.positions, li.position, box_size)
                d = np.linalg.norm(d_vec, axis=1)
                close_atoms_index = np.where(d < cutoff_radii[group_name])[0]
                unique_resids = len(np.unique(group_atoms[close_atoms_index].resids))
                encoded_coordination += unique_resids * factor
                factor //= 10  # Increment factor for the next group encoding
            coordination[ts_index, li_index] = encoded_coordination

    return coordination

def plot_rdf_coordination(
    bins,
    rdf,
    coord_numbers,
):
    # Font sizes and color palette
    font_list = {"label": 18, "ticket": 18, "legend": 16}
    color_list = ["#DF543F", "#2286A9", "#FBBF7C", "#3C3846"]

    # Create the plotting canvas
    fig, ax1 = plt.subplots()
    fig.set_size_inches(5.5, 4)

    # Plot the RDF curve
    ax1.plot(
        bins,
        rdf,
        '-',
        linewidth=1.5,
        color=color_list[0],
        label='g(r)'
    )
    ax1.set_xlabel('Distance (Å)', fontsize=font_list["label"])
    ax1.set_ylabel('g(r)', fontsize=font_list["label"])
    ax1.tick_params(
        axis='both',
        which='both',
        direction='in',
        labelsize=font_list["ticket"]
    )

    # Add a second y-axis for the coordination number
    ax2 = ax1.twinx()
    ax2.plot(
        bins,
        coord_numbers,
        '--',
        linewidth=2,
        color="grey",
        label='Coord. Number'
    )
    ax2.set_ylabel('Coordination Number', fontsize=font_list["label"])
    ax2.tick_params(
        axis='y',
        which='both',
        direction='in',
        labelsize=font_list["ticket"]
    )

    # Axis range and grid styling
    ax1.set_xlim(0, 10)
    ax1.grid(True, linestyle='--')

    plt.tight_layout()
    plt.show()


def num_of_neighbor(
        work_dir,
        nvt_run,
        center_atom_name,
        distance_dict,
        select_dict,
        run_start,
        run_end,
        write,
        structure_code,
        write_freq,
        max_number
):
    # build the dir to store the cluster file
    write_path = os.path.join(work_dir, 'cluster_dir')
    os.makedirs(write_path, exist_ok=True)

    center_atoms = nvt_run.select_atoms(center_atom_name)
    trj_analysis = nvt_run.trajectory[run_start:run_end:10]
    cn_values = {}
    species = list(distance_dict.keys())

    for kw in species:
        cn_values[kw] = np.zeros(int(len(trj_analysis)))
    cn_values["total"] = np.zeros(int(len(trj_analysis)))

    written_structures = 0
    max_reached = False  # Initialize the flag variable

    for time_count, ts in enumerate(trj_analysis):
        if max_reached:
            break  # Exit the loop if max_number is reached
        for center_atom in center_atoms:
            digit_of_species = len(species) - 1
            for kw in species:
                selection = select_shell(select_dict, distance_dict, center_atom, kw)
                shell = nvt_run.select_atoms(selection, periodic=True)
                for _ in shell.atoms:
                    cn_values[kw][time_count] += 1
                    cn_values["total"][time_count] += 10 ** digit_of_species
                digit_of_species -= 1  # Simplify decrement

            if write and cn_values["total"][time_count] == structure_code:
                a = np.random.random()
                if a > 1 - write_freq:
                    selection_write = " or ".join(
                        "(same resid as (" + select_shell(select_dict, distance_dict, center_atom, kw) + "))"
                        for kw in species
                    )
                    center_resid_selection = "same resid as index " + str(center_atom.index)
                    selection_write = "((" + selection_write + ") or (" + center_resid_selection + "))"
                    structure = nvt_run.select_atoms(selection_write, periodic=True)
                    center_pos = ts[center_atom.index]
                    # path = write_path + 'num' + "_" + str(int(written_structures)) + ".xyz"
                    pdb_filename = 'num' + "_" + str(int(written_structures)) + ".pdb"
                    path = os.path.join(write_path, pdb_filename)
                    write_out(center_pos, structure, path)

                    written_structures += 1
                    if written_structures >= max_number:
                        # print(f"{max_number} structures have been written out in {write_path}!!!.")
                        max_reached = True  # Set the flag to True
                        break  # Break out of the innermost loop

                if max_reached:
                    break  # Check the flag and break if needed
        if max_reached:
            break  # Check the flag and break if needed
    else:
        print(f"Target of {max_number} structures not reached; only {written_structures} were written to {write_path}!!!.")

    return cn_values


def write_out(center_pos, neighbors, path):
    # Compute relative coordinates while accounting for periodic boundary conditions
    box = neighbors.dimensions
    half_box = box[:3] / 2.0
    new_positions = neighbors.positions - center_pos
    # Apply the minimum image convention
    new_positions = np.where(new_positions > half_box, new_positions - box[:3], new_positions)
    new_positions = np.where(new_positions < -half_box, new_positions + box[:3], new_positions)
    neighbors.positions = new_positions

    # Write the PDB file
    neighbors.write(path)


def select_shell(
        select,
        distance,
        center_atom,
        kw
):
    if isinstance(select, dict):
        species_selection = select[kw]
        if species_selection is None:
            raise ValueError("Species specified does not match entries in the select dict.")
    else:
        species_selection = select
    if isinstance(distance, dict):
        distance_value = distance[kw]
        if distance_value is None:
            raise ValueError("Species specified does not match entries in the distance dict.")
        distance_str = str(distance_value)
    else:
        distance_str = distance
    return "(" + species_selection + ") and (around " + distance_str + " index " + str(center_atom.index) + ")"

def pdb2mol(work_dir, pdb_filename):
    label_to_element = {
        'N': 'N',
        'S': 'S',
        'O': 'O',
        'C': 'C',
        'F': 'F',
        'CL': 'Cl',
        'BR': 'Br',
        'LI': 'Li',
        'SI':'Si'
    }

    pdb_filepath = os.path.join(work_dir, pdb_filename)

    # Collect atom labels, coordinates, and residue names
    atoms_data = []

    # Read the PDB file
    with open(pdb_filepath, 'r') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                # Relevant PDB columns:
                #   columns 13-16: atom name
                #   column 17: alternate location indicator
                #   columns 18-20: residue name
                #   column 22: chain identifier
                #   columns 23-26: residue sequence number
                #   columns 31-38: X coordinate
                #   columns 39-46: Y coordinate
                #   columns 47-54: Z coordinate
                atom_name = line[12:17].strip()
                res_name = line[17:20].strip()
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())

                # Infer the element symbol from the atom name if the element field is blank
                element = line[76:78].strip()
                if not element:
                    # Use the first alphabetical character(s) of the atom name as the element symbol
                    element = ''.join([char for char in atom_name if char.isalpha()])[0]

                element = label_to_element.get(element, element)

                atoms_data.append((element, res_name, atom_name, (x, y, z)))

    num_atoms = len(atoms_data)
    if num_atoms == 0:
        print("No atom information found in the PDB file.")
        return None

    # Create a new RDKit molecule
    mol = Chem.RWMol()

    # Add atoms to the molecule and store residue names as atom properties
    for atom_info in atoms_data:
        label, res_name, atom_name, coords = atom_info
        atomic_num = Chem.GetPeriodicTable().GetAtomicNumber(label)
        if atomic_num == 0:
            print(f"Unrecognized element symbol '{label}', skipping this atom.")
            continue
        atom = Chem.Atom(atomic_num)
        # Set the 'resname' atom property
        atom.SetProp("resname", res_name)
        atom.SetProp("name", atom_name)
        mol.AddAtom(atom)

    # Generate a 3D conformer
    conf = Chem.Conformer(num_atoms)
    for i, atom_info in enumerate(atoms_data):
        _, _, _, (x, y, z) = atom_info
        conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(x, y, z))
    mol.AddConformer(conf)

    # Add bonds based on interatomic distances and covalent radii
    tolerance = 0.4  # Å
    pt = Chem.GetPeriodicTable()
    for i in range(num_atoms):
        atom_i = mol.GetAtomWithIdx(i)
        for j in range(i + 1, num_atoms):
            atom_j = mol.GetAtomWithIdx(j)
            # Calculate the distance between atoms
            pos_i = np.array([conf.GetAtomPosition(i).x,
                              conf.GetAtomPosition(i).y,
                              conf.GetAtomPosition(i).z])
            pos_j = np.array([conf.GetAtomPosition(j).x,
                              conf.GetAtomPosition(j).y,
                              conf.GetAtomPosition(j).z])
            distance = np.linalg.norm(pos_i - pos_j)
            # Retrieve the covalent radii
            radius_i = pt.GetRcovalent(atom_i.GetSymbol())
            radius_j = pt.GetRcovalent(atom_j.GetSymbol())
            if radius_i == 0 or radius_j == 0:
                continue  # Skip if a radius is unavailable
            # Check whether the distance falls within the sum of radii plus tolerance
            if distance <= (radius_i + radius_j + tolerance):
                try:
                    mol.AddBond(i, j, order=Chem.rdchem.BondType.SINGLE)
                except Exception as e:
                    print(f"Failed to add bond: {e}")

    # Convert to an RDKit ``Mol`` and sanitize the result
    mol = mol.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Chem.rdchem.KekulizeException as e:
        print(f"Molecule sanitization failed: {e}")
        return None

    return mol


def parse_selection_string(selection_str):

    # Split on "and" with a regular expression and extract key-value pairs
    conditions = re.split(r'\s+and\s+', selection_str.strip(), flags=re.IGNORECASE)
    criteria = {}
    for condition in conditions:
        match = re.match(r'(\w+)\s+(\S+)', condition)
        if match:
            key, value = match.groups()
            criteria[key.lower()] = value
        else:
            raise ValueError(f"Unable to parse selection condition: '{condition}'")
    return criteria


def bfs_traverse(mol, start_idx, max_steps=None, ignore_H=False):
    """
    Perform a breadth-first traversal starting from ``start_idx`` and return the
    set of visited atom indices.

    - ``max_steps``: Maximum depth (``None`` means unlimited)
    - ``ignore_H``: Whether to skip hydrogen atoms
    """
    visited = {start_idx}
    queue = deque([(start_idx, 0)])
    while queue:
        idx, depth = queue.popleft()
        if max_steps is not None and depth >= max_steps:
            continue
        atom = mol.GetAtomWithIdx(idx)
        for nbr in atom.GetNeighbors():
            nid = nbr.GetIdx()
            if ignore_H and nbr.GetSymbol() == 'H':
                continue
            if nid not in visited:
                visited.add(nid)
                queue.append((nid, depth + 1))
    return visited


def select_atoms_by_criteria(mol, criteria):
    """
    Return all atoms in ``mol`` that satisfy the provided ``criteria`` dictionary
    (e.g., ``{'atom_name': 'C1', 'resname': 'PEO'}``).
    """
    return [
        atom for atom in mol.GetAtoms()
        if all(atom.HasProp(k) and atom.GetProp(k) == v for k, v in criteria.items())
    ]


def get_cluster_index(
    mol,
    center_atom_name,
    select_dict,
    distance_dict,
    poly_name,
    repeating_unit,
    length
):

    # Replace ``*`` with ``[H]`` in the repeating unit and build the RDKit molecule
    unit_smi_with_h1 = repeating_unit.replace('*', '[H]')
    unit_mol_with_h1 = Chem.MolFromSmiles(unit_smi_with_h1)
    unit_mol = Chem.RemoveHs(unit_mol_with_h1)
    num_unit = unit_mol.GetNumAtoms()
    max_steps = num_unit * (length - 1)

    # Retrieve the molecular conformer
    conf = mol.GetConformer()
    center_criteria = parse_selection_string(center_atom_name)
    select_criteria = parse_selection_string(select_dict[poly_name])

    # Step 1: Select the center atom based on the criteria (assuming a single match)
    center_atoms_list = select_atoms_by_criteria(mol, center_criteria)
    center_atom = center_atoms_list[0]
    n_idx = center_atom.GetIdx()

    # Gather the selected atoms and their coordinates using the criteria
    select_atoms_list = select_atoms_by_criteria(mol, select_criteria)
    select_positions = np.array([conf.GetAtomPosition(atom.GetIdx()) for atom in select_atoms_list])

    # Step 3: Find the selected atom closest to the center and confirm bonding
    n_pos = np.array([conf.GetAtomPosition(n_idx).x,
                      conf.GetAtomPosition(n_idx).y,
                      conf.GetAtomPosition(n_idx).z])
    distances = np.linalg.norm(select_positions - n_pos, axis=1)

    if center_atom.GetSymbol() != 'Li':
        center_atom_indices = bfs_traverse(mol, n_idx, max_steps=None, ignore_H=False)
        min_dist_idx = np.argmin(distances)
        closest_select_atom = select_atoms_list[min_dist_idx]

        # Identify the non-hydrogen neighbor bonded to the closest selected atom
        try:
            Chem.FastFindRings(mol)
        except Exception:
            pass
        bonded_heavy = [nbr for nbr in closest_select_atom.GetNeighbors()
                        if nbr.GetAtomicNum() > 1]
        if closest_select_atom.IsInRing() and len(bonded_heavy) >= 2:
            c_idx_list = [bonded_heavy[0].GetIdx(), bonded_heavy[1].GetIdx()]
        elif bonded_heavy:
            c_idx_list = [bonded_heavy[0].GetIdx()]
        else:
            c_idx_list = [closest_select_atom.GetIdx()]
    else:
        center_atom_indices = {n_idx}
        threshold = distance_dict[poly_name]
        mask = distances < threshold
        idxs = np.where(mask)[0]
        c_idx_list = [select_atoms_list[i].GetIdx() for i in idxs]

    # 6. Use BFS with a depth limit to locate the polymer fragment (ignoring hydrogens)
    selected_atom_indices = set()
    for start_idx in c_idx_list:
        selected_atom_indices |= bfs_traverse(
            mol,
            start_idx,
            max_steps=max_steps,
            ignore_H=True
        )
    selected_atom_indices = sorted(selected_atom_indices)

    # 7. Gather other atom indices
    other_atom_indices = []
    for name, sel_str in select_dict.items():
        if name != poly_name:
            other_criteria = parse_selection_string(select_dict[name])
            comp_idxs = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetProp('resname') == other_criteria['resname']]
            other_atom_indices.extend(comp_idxs)
    print(f"Center atom indices: {center_atom_indices}")
    print(f"Selected atom indices: {selected_atom_indices}")
    print(f"Other atom indices: {other_atom_indices}")
    print(f"c_idx_list: {c_idx_list}")

    return sorted(c_idx_list), sorted(center_atom_indices), sorted(selected_atom_indices), sorted(other_atom_indices)


# def find_poly_match_subindex(poly_name, repeating_unit, length, mol, selected_atom_idxs, c_idx_list, ):
#
#     (
#         dum1,
#         dum2,
#         atom1,
#         atom2,
#     ) = polymer.Init_info(
#         poly_name,
#         repeating_unit,
#     )
#
#     (
#         inti_mol3,
#         monomer_mol,
#         start_atom,
#         end_atom,
#     ) = model_lib.gen_smiles_nocap(
#         dum1,
#         dum2,
#         atom1,
#         atom2,
#         repeating_unit,
#         length,
#     )
#     main_smi = Chem.MolToSmiles(inti_mol3, canonical=False)
#     mol_test = Chem.MolFromSmiles(main_smi)
#
#     # Collect the indices of all neighbors of ``*`` atoms in a single comprehension
#     connected_idxs = [
#         nbr.GetIdx()
#         for atom in mol_test.GetAtoms() if atom.GetSymbol() == '*'
#         for nbr in atom.GetNeighbors()
#     ]
#     main_smi_with_h1 = main_smi.replace('*', '[H]')
#     main_mol_with_h1 = Chem.MolFromSmiles(main_smi_with_h1)
#
#     main_mol = Chem.RemoveHs(main_mol_with_h1)
#     main_smi = Chem.MolToSmiles(main_mol, canonical=False)
#
#     mol1 = Chem.MolFromSmiles(main_smi)
#     rw_mol = Chem.RWMol(mol1)
#     for bond in rw_mol.GetBonds():
#         bond.SetBondType(Chem.BondType.SINGLE)
#
#     mol2 = rw_mol.GetMol()
#     smi2 = Chem.MolToSmiles(mol2, canonical=False)
#     pattern = Chem.MolFromSmiles(smi2)
#     matches = _plainize_mol(mol).GetSubstructMatches(_plainize_mol(pattern), useChirality=False)
#     print(pattern)
#
#     best = pick_most_central_match(matches, c_idx_list, selected_atom_idxs)
#     if best is not None:
#         # return list(best), connected_idxs[0], connected_idxs[1]-1
#         return list(best), start_atom, connected_idxs[1]-1

def _plainize_mol(m: Chem.Mol) -> Chem.Mol:
    m2 = Chem.Mol(m)
    try:
        Chem.Kekulize(m2, clearAromaticFlags=True)
    except Exception:
        pass
    for b in m2.GetBonds():
        b.SetBondType(Chem.BondType.SINGLE)
        b.SetIsAromatic(False)
    for a in m2.GetAtoms():
        a.SetIsAromatic(False)
    return m2

def _promote_bonds_on_parent(parent: Chem.Mol, match_list, templ: Chem.Mol) -> Chem.Mol:
    """
    将模板 templ（严格键级/芳香）的键型映射到 parent 的匹配子图（按 match_list 对齐），
    仅改子图内的键；返回一个新的 mol（不原地改）。
    """
    ed = Chem.RWMol(parent)
    # 给模板做一次 Kekulize，保证芳香能被合理转写
    t2 = Chem.Mol(templ)
    try:
        Chem.Kekulize(t2, clearAromaticFlags=False)
    except Exception:
        pass
    # 逐键映射
    for b in t2.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        mi = int(match_list[i]); mj = int(match_list[j])
        bond = ed.GetBondBetweenAtoms(mi, mj)
        if bond is None:
            continue
        bond.SetBondType(b.GetBondType())
        bond.SetIsAromatic(b.GetIsAromatic())
    out = ed.GetMol()
    # 让 RDKit 重新做价态/芳香性判断
    try:
        Chem.SanitizeMol(out)
    except Exception:
        # 局部不一致也尽量忽略，让严格匹配再检验
        pass
    return out

def _has_aromatic_6ring(m: Chem.Mol) -> bool:
    """模板中是否含芳香 6 元环（决定分支策略）"""
    try:
        Chem.FastFindRings(m)
    except Exception:
        pass
    ri = m.GetRingInfo()
    # 只要存在 6 元环且该环中有 >=4 个芳香原子/或任一键芳香即认为“苯环模式”
    for ids in ri.AtomRings():
        if len(ids) == 6:
            arom_atoms = sum(1 for i in ids if m.GetAtomWithIdx(i).GetIsAromatic())
            if arom_atoms >= 4:
                return True
            # 或者看环内是否有芳香键
            bonds = []
            for i in range(6):
                a = ids[i]; b = ids[(i+1) % 6]
                bd = m.GetBondBetweenAtoms(a, b)
                if bd is not None and bd.GetIsAromatic():
                    return True
    return False

def find_poly_match_subindex(
    poly_name,
    repeating_unit,
    length,
    mol,
    selected_atom_idxs,
    c_idx_list,
):
    # === 1) 生成含 '*' 的主链（严格模板起点） ===
    dum1, dum2, atom1, atom2 = polymer.Init_info(poly_name, repeating_unit)
    inti_mol3, monomer_mol, _sa, _ea = model_lib.gen_smiles_nocap(
        dum1, dum2, atom1, atom2, repeating_unit, length,
    )
    main_smi_star = Chem.MolToSmiles(inti_mol3, canonical=False)
    if not main_smi_star:
        raise ValueError("模板 SMILES 为空")

    # === 2) 把 '*' 显式为 '[*]'，给端点邻居打标（101/102），删除哑原子 ===
    pat_star = Chem.MolFromSmiles(main_smi_star.replace("*","[*]"))
    if pat_star is None:
        raise ValueError(f"无法解析模板：{main_smi_star}")

    dummy_idxs = [a.GetIdx() for a in pat_star.GetAtoms() if a.GetAtomicNum()==0]
    end_neighbors = []
    for di in dummy_idxs:
        for nb in pat_star.GetAtomWithIdx(di).GetNeighbors():
            if nb.GetAtomicNum() > 1:
                end_neighbors.append(nb.GetIdx())
    end_neighbors = list(dict.fromkeys(end_neighbors))[:2]
    if len(end_neighbors) < 2:
        heavy = [a.GetIdx() for a in pat_star.GetAtoms() if a.GetAtomicNum()>1]
        if len(heavy) >= 2:
            end_neighbors = [heavy[0], heavy[-1]]
        else:
            raise ValueError("模板端点邻居识别失败")

    pat_star.GetAtomWithIdx(end_neighbors[0]).SetAtomMapNum(101)
    pat_star.GetAtomWithIdx(end_neighbors[1]).SetAtomMapNum(102)

    em = Chem.RWMol(pat_star)
    for di in sorted(dummy_idxs, reverse=True):
        em.RemoveAtom(di)
    pattern_strict = em.GetMol()               # 带正确键级/芳香（严格用）
    pattern_plain  = _plainize_mol(pattern_strict)  # 纯拓扑（先定位）

    # 记录 pattern 中端点的位置
    pat_idx_start = pat_idx_end = None
    for ai, a in enumerate(pattern_strict.GetAtoms()):
        amap = a.GetAtomMapNum()
        if   amap == 101: pat_idx_start = ai
        elif amap == 102: pat_idx_end   = ai
    if pat_idx_start is None or pat_idx_end is None:
        raise ValueError("未找到模板端点标记 101/102")

    # === 3) 纯拓扑匹配先定位 ===
    target_plain = _plainize_mol(mol)
    matches_plain = target_plain.GetSubstructMatches(pattern_plain, useChirality=False)
    if not matches_plain:
        raise ValueError("纯拓扑匹配为 0 —— 模板与目标拓扑不一致")

    anchors = set(c_idx_list)
    allowed0 = set(selected_atom_idxs) if selected_atom_idxs else set(range(mol.GetNumAtoms()))
    is_aromatic = _has_aromatic_6ring(pattern_strict)

    # 打分函数：锚点“中心性”（越靠 pattern 中心越好）
    def center_score(m):
        L = len(m); center = (L-1)/2
        pos = sorted([m.index(a) for a in anchors if a in m])
        if not pos: return 1e9
        return abs(pos[len(pos)//2] - center)

    # 几何就近打分：匹配原子到锚点集合的最小距离之和
    conf = mol.GetConformer()
    def _p(idx):
        p = conf.GetAtomPosition(int(idx))
        return np.array([p.x, p.y, p.z], dtype=float)
    def geo_score(m):
        if not anchors: return 0.0
        s = 0.0
        for i in m:
            p = _p(i)
            dmin = min(np.linalg.norm(p - _p(a)) for a in anchors)
            s += dmin
        return s

    M = [list(m) for m in matches_plain]

    # === 3a) 分支策略：苯环 vs 非苯环 ===
    cand = []
    if is_aromatic:
        # 苯环：优先强约束（全部锚点 + 允许集）
        filters = [
            (lambda m,allowed: anchors.issubset(set(m)) and all(i in allowed for i in m), allowed0),
            (lambda m,allowed: anchors.issubset(set(m)),                                      None),
            (lambda m,allowed: any(a in m for a in anchors) and all(i in allowed for i in m), allowed0),
            (lambda m,allowed: any(a in m for a in anchors),                                  None),
        ]
    else:
        # 非苯环：更宽松（至少一个锚点起步）
        filters = [
            (lambda m,allowed: any(a in m for a in anchors) and all(i in allowed for i in m), allowed0),
            (lambda m,allowed: any(a in m for a in anchors),                                  None),
            (lambda m,allowed: all(i in allowed for i in m),                                   allowed0),
        ]
    for pred, allowed in filters:
        if allowed is None:
            tmp = [m for m in M if pred(m, set())]
        else:
            tmp = [m for m in M if pred(m, allowed)]
        if tmp:
            cand = tmp
            break
    if not cand:
        cand = M  # 实在不行：全量候选

    # 选最好：优先中心性，若无锚点则用几何就近
    if any(a in cand[0] for a in anchors):
        best = min(cand, key=center_score)
    else:
        best = min(cand, key=geo_score)

    # === 4) 回填严格键级/芳香，做严格匹配验证 ===
    mol_strict = _promote_bonds_on_parent(mol, best, pattern_strict)
    strict_hits = mol_strict.GetSubstructMatches(pattern_strict, useChirality=False)
    if not strict_hits:
        # 再试一次 Kekulize 模板
        pat_k = Chem.Mol(pattern_strict)
        try:
            Chem.Kekulize(pat_k, clearAromaticFlags=False)
        except Exception:
            pass
        strict_hits = mol_strict.GetSubstructMatches(pat_k, useChirality=False)

    if not strict_hits:
        return best, pat_idx_start, pat_idx_end, mol

    # 选择与 best 重叠最多的严格匹配，避免落到其它重复单元
    def overlap_score(hit):  # 越大越好 → 用负号
        return -len(set(hit) & set(best))
    strict_best = min(strict_hits, key=overlap_score)
    best = list(strict_best)

    return best, pat_idx_start, pat_idx_end, mol_strict


def pick_most_central_match(matches, c_idx_list, selected_atom_idxs):
    best_match = None
    best_score = float("inf")
    # Assume all matches have the same length
    L = len(matches[0]) if matches else 0
    center = (L - 1) / 2

    for match in matches:
        # Basic filtering: every element of ``c_idx_list`` must appear in the match
        # and each index in the match must be present in ``selected_atom_idxs``
        if not all(c in match for c in c_idx_list):
            continue
        if not all(idx in selected_atom_idxs for idx in match):
            continue

        # ``c_idx_list`` currently contains a single element ``c``
        p_c = len(c_idx_list)//2
        c = c_idx_list[p_c]
        pos = match.index(c)

        # Deviation from the center of the match
        score = abs(pos - center)
        if score < best_score:
            best_score = score
            best_match = match

    return best_match


def get_cluster_withcap(work_dir, mol, match_list, center_atom_idx, other_atom_indices, start_atom, end_atom, out_xyz_filename):

    # 1. Normalize ``center_atom_idx`` to a list
    if isinstance(center_atom_idx, int):
        center_idxs = [center_atom_idx]
    else:
        center_idxs = list(center_atom_idx)

    # 2. Flatten ``other_atom_indices`` if it is a dict
    if isinstance(other_atom_indices, dict):
        other_idxs = []
        for lst in other_atom_indices.values():
            other_idxs.extend(lst)
    else:
        other_idxs = list(other_atom_indices)

    h_atom_idx = set()
    for idx in match_list:
        atom = mol.GetAtomWithIdx(idx)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() == 1:  # H atom
                h_atom_idx.add(neighbor.GetIdx())

    all_idxs = set(match_list) | h_atom_idx | set(center_idxs) | set(other_idxs)
    select_atom_idx_with_h = sorted(all_idxs)

    # select_atom_idx_with_h = sorted(set(match_list + list(h_atom_idx) + center_atom_idx + other_atom_indices))

    new_mol = Chem.RWMol()
    index_map = {}

    for old_idx in select_atom_idx_with_h:
        atom = mol.GetAtomWithIdx(old_idx)
        new_atom = Chem.Atom(atom.GetAtomicNum())
        new_idx = new_mol.AddAtom(new_atom)
        index_map[old_idx] = new_idx

    # Build a reverse index map (new molecule index -> original atom index)
    # reverse_index_map = {v: k for k, v in index_map.items()}
    reverse_index_map = {new_idx: old_idx for old_idx, new_idx in index_map.items()}

    # Add bonds between atoms when both are part of the extracted subset
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if begin in select_atom_idx_with_h and end in select_atom_idx_with_h:
            new_mol.AddBond(index_map[begin], index_map[end], bond.GetBondType())

    # terminal_atoms = [start_atom, end_atom]
    start_atom_old = match_list[start_atom]
    end_atom_old = match_list[end_atom]
    start_atom_new = index_map[start_atom_old]
    end_atom_new = index_map[end_atom_old]
    terminal_atoms = [start_atom_new, end_atom_new]

    # Define the capping fragments to add
    capping_info = []
    for terminal_idx in terminal_atoms:
        atom = new_mol.GetAtomWithIdx(terminal_idx)
        h_count = sum(1 for nbr in atom.GetNeighbors() if nbr.GetAtomicNum() == 1)
        if atom.GetAtomicNum() == 6 and h_count == 2:
            capping_info.append({'type': 'H', 'atom_idx': terminal_idx})
        else:
            capping_info.append({'type': 'CH3', 'atom_idx': terminal_idx})

    # Bond lengths in Ångström
    C_H_bond_length = 1.09
    C_C_bond_length = 1.50

    # Retrieve the conformer of the original molecule
    orig_conf = mol.GetConformer()
    new_atom_positions = {}

    for cap in capping_info:
        terminal_idx = cap['atom_idx']
        terminal_atom = new_mol.GetAtomWithIdx(terminal_idx)

        # Look up the corresponding atom index in the original molecule
        terminal_old_idx = reverse_index_map[terminal_idx]
        terminal_pos = np.array(orig_conf.GetAtomPosition(terminal_old_idx))

        # Identify non-hydrogen neighbors for orientation
        neighbor_indices = [nbr.GetIdx() for nbr in terminal_atom.GetNeighbors()
                            if nbr.GetAtomicNum() > 1 and nbr.GetIdx() != terminal_idx]

        if neighbor_indices:
            neighbor_idx = neighbor_indices[0]
            neighbor_old_idx = reverse_index_map[neighbor_idx]
            neighbor_pos = np.array(orig_conf.GetAtomPosition(neighbor_old_idx))
            bond_vec = terminal_pos - neighbor_pos
        else:
            # Fall back to a default direction if no heavy neighbor exists
            bond_vec = np.array([0.0, 0.0, 1.0])

        # Normalize the direction vector
        bond_vec = bond_vec / np.linalg.norm(bond_vec)

        # Compute two orthogonal vectors perpendicular to ``bond_vec``
        perp_vec1 = np.cross(bond_vec, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(perp_vec1) < 1e-3:
            perp_vec1 = np.cross(bond_vec, np.array([0.0, 1.0, 0.0]))
        perp_vec1 = perp_vec1 / np.linalg.norm(perp_vec1)
        perp_vec2 = np.cross(bond_vec, perp_vec1)
        perp_vec2 = perp_vec2 / np.linalg.norm(perp_vec2)

        if cap['type'] == 'H':
            connected_h_indices = [
                nbr.GetIdx() for nbr in terminal_atom.GetNeighbors()
                if nbr.GetAtomicNum() == 1
            ]

            if len(connected_h_indices) < 2:
                raise ValueError(f"Expected at least 2 hydrogen atoms connected to atom index {terminal_idx}, found {len(connected_h_indices)}.")

            h1_old_idx = reverse_index_map.get(connected_h_indices[0])
            h2_old_idx = reverse_index_map.get(connected_h_indices[1])
            if h1_old_idx is None or h2_old_idx is None:
                raise KeyError(f"One of the hydrogen atoms ({connected_h_indices[0]}, {connected_h_indices[1]}) is missing in reverse_index_map.")

            h1_pos = np.array(orig_conf.GetAtomPosition(h1_old_idx))
            h2_pos = np.array(orig_conf.GetAtomPosition(h2_old_idx))
            h1_vec = h1_pos - terminal_pos
            h2_vec = h2_pos - terminal_pos
            mid_h_vec = (h1_vec + h2_vec) / 2
            norm_mid_h_vec = np.linalg.norm(mid_h_vec)
            if norm_mid_h_vec < 1e-6:
                # Use ``perp_vec1`` if ``mid_h_vec`` is too small
                mid_h_vec = perp_vec1
            else:
                mid_h_vec /= norm_mid_h_vec

            # Project ``mid_h_vec`` onto the plane perpendicular to ``bond_vec``
            proj_mid_h_vec = mid_h_vec - np.dot(mid_h_vec, bond_vec) * bond_vec
            norm_proj_mid_h_vec = np.linalg.norm(proj_mid_h_vec)
            if norm_proj_mid_h_vec < 1e-6:
                # Use ``perp_vec1`` if the projection is too small
                proj_mid_h_vec = perp_vec1
            else:
                proj_mid_h_vec /= norm_proj_mid_h_vec

            # Determine the direction for the new hydrogen to maintain a tetrahedral angle
            theta = np.deg2rad(360-109.5)  # Tetrahedral bond angle
            direction = (
                np.cos(theta) * (-bond_vec) +
                np.sin(theta) * proj_mid_h_vec
            )
            direction /= np.linalg.norm(direction)

            # Compute the position of the new hydrogen atom
            H_pos = terminal_pos + direction * C_H_bond_length

            # Add the new hydrogen atom to the molecule
            new_H = Chem.Atom(1)
            new_H_idx = new_mol.AddAtom(new_H)
            new_atom_positions[new_H_idx] = H_pos
            new_mol.AddBond(terminal_idx, new_H_idx, Chem.BondType.SINGLE)

        elif cap['type'] == 'CH3':
            terminal_old_idx = reverse_index_map[terminal_idx]
            terminal_old_atom = mol.GetAtomWithIdx(terminal_old_idx)
            # neighbor_old_idx = [nbr.GetIdx() for nbr in terminal_old_atom.GetNeighbors()
            #                     if nbr.GetIdx() not in match_list]
            #
            # neighbor_old_pos = np.array(orig_conf.GetAtomPosition(neighbor_old_idx[0]))
            # if neighbor_old_idx[0] > terminal_old_idx:
            #     bond_vec = neighbor_old_pos - terminal_pos
            # else:
            #     bond_vec = - terminal_pos + neighbor_old_pos
            # direction = bond_vec / np.linalg.norm(bond_vec)
            neighbor_old_idx = [
                nbr.GetIdx() for nbr in terminal_old_atom.GetNeighbors()
                if nbr.GetIdx() not in match_list and nbr.GetAtomicNum() > 1
            ]

            neighbor_old_pos = np.array(orig_conf.GetAtomPosition(neighbor_old_idx[0]))
            bond_vec = neighbor_old_pos - terminal_pos
            direction = bond_vec / np.linalg.norm(bond_vec)
            C_pos = terminal_pos + direction * C_C_bond_length

            new_C = Chem.Atom(6)
            new_C_idx = new_mol.AddAtom(new_C)
            new_atom_positions[new_C_idx] = C_pos
            new_mol.AddBond(terminal_idx, new_C_idx, Chem.BondType.SINGLE)

            # Add three hydrogens arranged in a tetrahedral geometry
            # Recompute the bond vector for the new carbon
            bond_vec_new = C_pos - terminal_pos
            bond_vec_new = bond_vec_new / np.linalg.norm(bond_vec_new)

            # Recompute the two orthogonal vectors perpendicular to ``bond_vec_new``
            perp_vec1_new = np.cross(bond_vec_new, np.array([1.0, 0.0, 0.0]))
            if np.linalg.norm(perp_vec1_new) < 1e-3:
                perp_vec1_new = np.cross(bond_vec_new, np.array([0.0, 1.0, 0.0]))
            perp_vec1_new = perp_vec1_new / np.linalg.norm(perp_vec1_new)
            perp_vec2_new = np.cross(bond_vec_new, perp_vec1_new)
            perp_vec2_new = perp_vec2_new / np.linalg.norm(perp_vec2_new)

            # Define the azimuthal angles (degrees) for the three hydrogens
            angles = [0, 120, 240]
            for angle in angles:
                theta = np.deg2rad(109.5)  # Tetrahedral bond angle
                phi = np.deg2rad(angle)
                direction = (
                        np.cos(theta) * (-bond_vec_new) +
                        np.sin(theta) * (np.cos(phi) * perp_vec1_new + np.sin(phi) * perp_vec2_new)
                )
                H_pos = C_pos + direction * C_H_bond_length
                new_H = Chem.Atom(1)
                new_H_idx = new_mol.AddAtom(new_H)
                new_atom_positions[new_H_idx] = H_pos
                new_mol.AddBond(new_C_idx, new_H_idx, Chem.BondType.SINGLE)

    # Create a new conformer
    num_atoms_new = new_mol.GetNumAtoms()
    conf = Chem.Conformer(num_atoms_new)

    # Populate coordinates for the original atoms
    for old_idx, new_idx in index_map.items():
        pos = orig_conf.GetAtomPosition(old_idx)
        conf.SetAtomPosition(new_idx, pos)

    # Populate coordinates for the newly added atoms
    for idx, pos in new_atom_positions.items():
        conf.SetAtomPosition(idx, Point3D(*pos))

    # Attach the conformer to the molecule
    new_mol.AddConformer(conf)

    # Update cached properties and optionally sanitize
    # new_mol.UpdatePropertyCache(strict=False)
    # Chem.SanitizeMol(new_mol)

    # save to xyz file
    out_xyz_filepath = os.path.join(work_dir, out_xyz_filename)
    Chem.MolToXYZFile(new_mol, out_xyz_filepath)


def rotate_vector_around_axis(v, axis, theta):
    """
    Rotate vector ``v`` about the unit vector ``axis`` by ``theta`` radians using
    the Rodrigues rotation formula.
    """
    axis = axis / np.linalg.norm(axis)
    v_parallel = np.dot(v, axis) * axis
    v_perp = v - v_parallel
    w = np.cross(axis, v)
    return v_parallel + v_perp * np.cos(theta) + w * np.sin(theta)


def merge_xyz_files(xyz_dir: str,
                    pattern: str = "num_*_frag.xyz",
                    out_name: str = "merged.xyz") -> str:

    xyz_paths = sorted(glob.glob(os.path.join(xyz_dir, pattern)))
    if not xyz_paths:
        raise FileNotFoundError(f"No files matched pattern {pattern} in {xyz_dir}")

    traj_path = os.path.join(xyz_dir, out_name)
    with open(traj_path, 'w') as fw:
        for idx, path in enumerate(xyz_paths, 1):
            with open(path, 'r') as fr:
                lines = fr.readlines()
                # Optionally overwrite the comment line with the frame index
                lines[1] = f"Frame {idx}: {os.path.basename(path)}\n"
                fw.writelines(lines)

    # print(f"Merged {len(xyz_paths)} frames into trajectory: {traj_path}")
    return traj_path


def analyze_coordination_structure(
    run: mda.Universe,
    run_start: int,
    run_end: int,
    select_dict: dict[str, str],
    distance: float,
    center_atom: str = "cation",
    counter_atom: str = "anion",
    plot: bool = False,
) -> pd.DataFrame:

    def select_shell(select, distance, center_atom, kw):
        if isinstance(select, dict):
            species_selection = select[kw]
            if species_selection is None:
                raise ValueError("Species specified does not match entries in the select dict.")
        else:
            species_selection = select
        if isinstance(distance, dict):
            distance_value = distance[kw]
            if distance_value is None:
                raise ValueError("Species specified does not match entries in the distance dict.")
            distance_str = str(distance_value)
        else:
            distance_str = distance
        return "(" + species_selection + ") and (around " + distance_str + " index " + str(center_atom.index) + ")"

    def num_of_neighbor_simple(nvt_run, center_atom, distance_dict, select_dict, run_start, run_end):
        trj_analysis = nvt_run.trajectory[run_start:run_end:]
        species = next(iter(distance_dict.keys()))
        cn_values = np.zeros(int(len(trj_analysis)))
        for time_count, _ts in enumerate(trj_analysis):
            selection = select_shell(select_dict, distance_dict, center_atom, species)
            shell = nvt_run.select_atoms(selection, periodic=True)
            shell_molecules = shell.residues
            shell_len = len(shell_molecules)
            if shell_len == 0:
                cn_values[time_count] = 1
            elif shell_len == 1:
                coordination_atoms = shell_molecules.atoms.select_atoms("same type as index " + str(shell.atoms[0].index))
                unique_lithium_indices = set()
                for atom in coordination_atoms:
                    selection_species = select_shell("same type as index " + str(center_atom.index), distance_dict, atom, species)
                    shell_species = nvt_run.select_atoms(selection_species, periodic=True)
                    for li_atom in shell_species:
                        unique_lithium_indices.add(li_atom.index)
                shell_species_len = len(unique_lithium_indices) - 1
                if shell_species_len == 0:
                    cn_values[time_count] = 2
                else:
                    cn_values[time_count] = 3
            else:
                cn_values[time_count] = 3
        return {"total": cn_values}

    def concat_coord_array(nvt_run, func, center_atoms, distance_dict, select_dict, run_start, run_end):
        num_array = func(nvt_run, center_atoms[0], distance_dict, select_dict, run_start, run_end)
        for atom in tqdm(center_atoms[1::]):
            this_atom = func(nvt_run, atom, distance_dict, select_dict, run_start, run_end)
            for kw in num_array:
                num_array[kw] = np.concatenate((num_array.get(kw), this_atom.get(kw)), axis=0)
        return num_array

    distance_dict = {counter_atom: distance}
    center_atoms = run.select_atoms(select_dict.get(center_atom))
    num_array = concat_coord_array(
        run,
        num_of_neighbor_simple,
        center_atoms,
        distance_dict,
        select_dict,
        run_start,
        run_end,
    )["total"]

    shell_component, shell_count = np.unique(num_array.flatten(), return_counts=True)
    combined = np.vstack((shell_component, shell_count)).T
    item_dict = {"1": "ssip", "2": "cip", "3": "agg"}
    item_list = []
    percent_list = []
    for i in range(len(combined)):
        item = str(int(combined[i, 0]))
        item_list.append(item_dict.get(item))
        percent_list.append(f"{(combined[i, 1] / combined[:, 1].sum() * 100):.4f}%")
    df_dict = {"Solvation structure": item_list, "Percentage": percent_list}

    if plot:
        order = ["ssip", "cip", "agg"]
        perc_map = {k: float(v.strip("%")) for k, v in zip(item_list, percent_list)}
        plot_vals = [perc_map.get(k, 0.0) for k in order]

        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        bars = ax.bar(order, plot_vals,  width=0.35, edgecolor="black", linewidth=0.50)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Percentage (%)")
        ax.grid(axis="y", linestyle=":", alpha=0.35)

        # Annotate the top of each bar
        for rect, v in zip(bars, plot_vals):
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                rect.get_height() + 0.6,
                f"{v:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        plt.show()

    return pd.DataFrame(df_dict)


def calc_population_frame(ts, cations, anions, index):
    all_atoms = cations + anions
    merged_list = list(range(len(all_atoms)))
    box_size = ts.dimensions[0]
    all_clusters = []

    while merged_list:
        this_cluster = [merged_list[0]]
        merged_list.remove(merged_list[0])

        for i in this_cluster:
            for j in merged_list:
                d_vec = minimum_image_displacement(
                    all_atoms.positions[i],
                    all_atoms.positions[j],
                    box_size,
                )
                d = np.linalg.norm(d_vec)
                if d <= 3.4:
                    this_cluster.append(j)
                    merged_list.remove(j)

        all_clusters.append(this_cluster)

    type_id = 'Li'
    type_id3 = 'NBT'
    pop_matrix = np.zeros((50, 50, 1))

    for cluster in all_clusters:
        cations_count = 0
        anions_count = 0

        for atom_id in cluster:
            if all_atoms[atom_id].type == type_id:
                cations_count += 1
            if all_atoms[atom_id].type == type_id3:
                anions_count += 1

        if cations_count < 50 and anions_count < 50:
            pop_matrix[cations_count][anions_count] += 1

    return pop_matrix, index


def calc_population_parallel(run, run_start, run_end, select_cations, select_anions, core, plot):
    stacked_population = np.array([])

    with ProcessPoolExecutor(max_workers=core) as executor:
        futures = []
        for idx, ts in enumerate(run.trajectory[run_start:run_end]):
            cations = run.select_atoms(select_cations)
            anions = run.select_atoms(select_anions)
            futures.append(executor.submit(calc_population_frame, ts, cations, anions, idx))

        results = [future.result() for future in
                   tqdm(as_completed(futures), total=len(futures), desc='Processing trajectory')]

    # Sort results by index and combine
    sorted_results = sorted(results, key=lambda x: x[1])
    for current_population, _ in sorted_results:
        if stacked_population.size == 0:
            stacked_population = current_population
        else:
            stacked_population = np.dstack((stacked_population, current_population))

    avg_population = np.mean(stacked_population, axis=2)
    np.savetxt('avg_population.txt', avg_population, fmt='%.6f')

    if plot:
        plot_population_heatmap(avg_population)

    return 'avg_population.txt'

def plot_population_heatmap(avg_population):

    n = min(21, avg_population.shape[0], avg_population.shape[1])
    matrix = avg_population[:n, :n].T          # x: cation, y: anion -> x-axis represents cations
    matrix = np.flipud(matrix)                 # Flip vertically so the y-axis increases upward

    mat_plot = matrix.copy()
    mat_plot[mat_plot <= 0] = 1e-12

    colors = ["white", "yellow", "red"]
    cmap = LinearSegmentedColormap.from_list("custom_red_yellow_white", colors)
    norm = LogNorm(vmin=1e-5, vmax=1e2)

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        mat_plot,
        annot=False,
        fmt=".2f",
        cmap=cmap,
        square=True,
        linewidths=0.5,
        linecolor='white',
        norm=norm
    )

    cbar = ax.collections[0].colorbar
    cbar.ax.minorticks_off()
    cbar.set_ticks([10**i for i in range(-3, 3)])
    cbar.set_ticklabels([r'$\mathrm{10^{-3}}$', r'$\mathrm{10^{-2}}$', r'$\mathrm{10^{-1}}$',
                         r'$\mathrm{10^{0}}$', r'$\mathrm{10^{1}}$', r'$\mathrm{10^{2}}$'])
    cbar.ax.tick_params(labelsize=24)

    x_labels = list(range(0, n))
    y_labels = list(range(n-1, -1, -1))
    ax.set_xticks(np.arange(0.5, n, 4))
    ax.set_yticks(np.arange(0.5, n, 4))
    ax.set_xticklabels(x_labels[0::4])
    ax.set_yticklabels(y_labels[0::4])
    ax.tick_params(left=True, bottom=True, length=0)

    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_edgecolor("black")

    for i in range(n):
        for j in range(n):
            ax.add_patch(Rectangle((j + 0.05, i + 0.05), 0.9, 0.9, fill=False, edgecolor="#BFBFBF", lw=0.5))

    ax.plot([0, n], [n, 0], ls='-', lw=1, color="#172C51")

    plt.xlabel(r'$\mathrm{n_{+}}$')   # Number of cations
    plt.ylabel(r'$\mathrm{n_{-}}$')   # Number of anions

    plt.tight_layout()
    plt.show()



