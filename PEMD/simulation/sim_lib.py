# Copyright (c) 2024. PEMD developers. All rights reserved.
# Distributed under the terms of the MIT License.

# ******************************************************************************
# Module Docstring
# ******************************************************************************

import os
import re
import subprocess
import numpy as np
from rdkit import Chem


# Modified order_energy_xtb function
def order_energy_xtb(work_dir, xyz_file, numconf, output_file):

    sorted_xtb_file = os.path.join(work_dir, output_file)

    structures = []
    current_structure = []

    with open(xyz_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line.isdigit():
                if current_structure:
                    if len(current_structure) >= 2:
                        energy_line = current_structure[1]
                        try:
                            energy_match = re.search(r"[-+]?\d*\.\d+|\d+", energy_line)
                            if energy_match:
                                energy = float(energy_match.group())
                            else:
                                raise ValueError("No numeric value found")
                        except ValueError:
                            print(f"Could not parse energy value: {energy_line}")
                            energy = float('inf')
                        structures.append((energy, current_structure))
                    else:
                        print("Malformed structure encountered.")
                    current_structure = []
                current_structure.append(line)
            else:
                current_structure.append(line)

    if current_structure:
        if len(current_structure) >= 2:
            energy_line = current_structure[1]
            try:
                energy_match = re.search(r"[-+]?\d*\.\d+|\d+", energy_line)
                if energy_match:
                    energy = float(energy_match.group())
                else:
                    raise ValueError("No numeric value found")
            except ValueError:
                print(f"Could not parse energy value: {energy_line}")
                energy = float('inf')
            structures.append((energy, current_structure))
        else:
            print("Malformed structure encountered.")

    structures.sort(key=lambda x: x[0])
    selected_structures = structures[:numconf]

    with open(sorted_xtb_file, 'w') as outfile:
        for energy, structure in selected_structures:
            for line_num, line in enumerate(structure):
                if line_num == 1:
                    outfile.write(f"Energy = {energy}\n")
                else:
                    outfile.write(f"{line}\n")

    print(f"The lowest {numconf} energy structures have been written to {output_file}")
    # return sorted_xtb_file

# input: a xyz file
# output: a list store the xyz structure
# Description: read the xyz file and store the structure in a list
def _parse_comment_fields(s: str):
    """从XYZ第二行的注释里提取 ID / Energy / Success，未找到则返回 None。"""
    ID = None
    Energy = None
    Success = None

    if not s:
        return ID, Energy, Success

    # ID = 123
    m = re.search(r'\bid\s*=\s*([^\s;]+)', s, flags=re.IGNORECASE)
    if m:
        raw = m.group(1)
        try:
            ID = int(raw)
        except ValueError:
            ID = raw  # 若不是纯整数，保留原始字符串

    # Energy = -10684.9945835000
    m = re.search(r'\benergy\s*=\s*([+\-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)', s)
    if m:
        try:
            Energy = float(m.group(1))
        except ValueError:
            Energy = None

    # Success = True/False/Yes/No/1/0
    m = re.search(r'\bsuccess\s*=\s*([^\s;]+)', s, flags=re.IGNORECASE)
    if m:
        val = m.group(1).strip().lower()
        if val in {"true", "t", "yes", "y", "1"}:
            Success = True
        elif val in {"false", "f", "no", "n", "0"}:
            Success = False
        else:
            Success = None

    return ID, Energy, Success


def read_xyz_file(file_path):
    structures = []
    with open(file_path, 'r') as f:
        lines = [ln.rstrip('\n') for ln in f]

    i = 0
    n = len(lines)
    while i < n:
        num_atoms_line = (lines[i].strip() if i < n else "")
        if num_atoms_line.isdigit():
            num_atoms = int(num_atoms_line)

            # 需要至少一行注释 + num_atoms 行坐标
            if i + 1 >= n or i + 2 + num_atoms > n:
                # 文件不完整，终止解析
                break

            comment_line = lines[i + 1].strip()
            ID, Energy, Success = _parse_comment_fields(comment_line)

            atoms = []
            for j in range(i + 2, i + 2 + num_atoms):
                atom_line = lines[j].strip()
                atoms.append(atom_line)

            structure = {
                'num_atoms': num_atoms,
                'id': ID,           # 可能是 int 或 str，未解析到则为 None
                'energy': Energy,   # float 或 None
                'success': Success, # bool 或 None
                'atoms': atoms,
            }
            structures.append(structure)

            i = i + 2 + num_atoms
        else:
            i += 1

    return structures

def dedup_structures(structs: list[dict],
                      mode: str = "energy",         # 'energy' | 'rmsd' | 'energy_and_rmsd' | 'energy_or_rmsd'
                      energy_tol_in_input_unit: float = 0.000159,  # ≈ 0.10 kcal/mol (Hartree)
                      rmsd_tol: float = 0.30,
                      rmsd_heavy_only: bool = True) -> list[dict]:
    """
    对 final_structures 去重。每个结构字典含:
      {'num_atoms', 'atoms'(list[str]), 'id', 'energy'(float|None), 'success'(bool)}
    返回“保留”的子集（保持原顺序的贪心筛选）。
    """
    kept = []
    for s in structs:
        e_s = s.get('energy', None)
        # 预取坐标
        _, coords_s = _atoms_to_coords(s['atoms'], heavy_only=rmsd_heavy_only)
        is_dup = False
        for t in kept:
            cond_energy = False
            cond_rmsd = False

            # 能量判据
            e_t = t.get('energy', None)
            if (e_s is not None) and (e_t is not None):
                if abs(e_s - e_t) <= energy_tol_in_input_unit:
                    cond_energy = True

            # RMSD 判据
            _, coords_t = _atoms_to_coords(t['atoms'], heavy_only=rmsd_heavy_only)
            if coords_s.shape == coords_t.shape and coords_s.shape[0] >= 3:
                try:
                    val = _kabsch_rmsd(coords_s, coords_t)
                    if val <= rmsd_tol:
                        cond_rmsd = True
                except Exception:
                    pass

            if mode == "energy":
                is_dup = cond_energy
            elif mode == "rmsd":
                is_dup = cond_rmsd
            elif mode == "energy_and_rmsd":
                is_dup = cond_energy and cond_rmsd
            elif mode == "energy_or_rmsd":
                is_dup = cond_energy or cond_rmsd
            else:  # 未知模式，默认按 energy_or_rmsd
                is_dup = cond_energy or cond_rmsd

            if is_dup:
                break
        if not is_dup:
            kept.append(s)
    return kept

def _kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """最佳刚体叠合后的 RMSD，P/Q: (N,3)"""
    Pc = P - P.mean(axis=0, keepdims=True)
    Qc = Q - Q.mean(axis=0, keepdims=True)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    diff = Pc @ R - Qc
    return float(np.sqrt((diff * diff).sum() / P.shape[0]))

def _atoms_to_coords(atoms_lines, heavy_only=True):
    """
    将 ['C x y z', 'H x y z', ...] 解析为 (symbols, coords[N,3])。
    heavy_only=True 时丢弃氢（以 'H' 开头的元素/标签）。
    """
    symbols = []
    coords = []
    for line in atoms_lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        sym = parts[0]
        # 兼容 H*、H1 等：只要以 H/h 开头都视为氢
        is_h = sym.upper().startswith('H')
        if heavy_only and is_h:
            continue
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            continue
        symbols.append(sym)
        coords.append([x, y, z])
    if not coords:  # 若全被丢弃，则退回全原子
        for line in atoms_lines:
            parts = line.split()
            if len(parts) >= 4:
                symbols.append(parts[0])
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return symbols, np.asarray(coords, dtype=float)

def read_energy_from_gaussian(log_file_path: str):
    # Free energy (optimization + frequency) line
    pattern_free = re.compile(
        r"Sum of electronic and thermal Free Energies\s*=\s*(-?\d+\.\d+)"
    )
    # SCF single-point energy line
    pattern_scf = re.compile(
        r"SCF Done:\s+E\(\w+.*?\)\s+=\s+(-?\d+\.\d+)"
    )

    energy_free = None
    energy_scf = None

    with open(log_file_path, 'r') as f:
        for line in f:
            m_free = pattern_free.search(line)
            if m_free:
                energy_free = float(m_free.group(1))
            m_scf = pattern_scf.search(line)
            if m_scf:
                energy_scf = float(m_scf.group(1))

    # Prefer free energy; fall back to SCF energy otherwise
    if energy_free is not None:
        return energy_free
    return energy_scf


def read_final_structure_from_gaussian(log_file_path):

    if not os.path.exists(log_file_path):
        print(f"File not found: {log_file_path}")
        return None

    try:
        with open(log_file_path, 'r') as file:
            lines = file.readlines()
    except Exception as e:
        print(f"Error reading file {log_file_path}: {e}")
        return None

    # Define the sections to search for
    orientation_sections = ['Standard orientation:', 'Input orientation:']
    start_idx = None
    end_idx = None

    # Iterate through the file to find the last occurrence of the orientation sections
    for i, line in enumerate(lines):
        for section in orientation_sections:
            if section in line:
                # Assume that coordinate data starts 5 lines after the section header
                current_start = i + 5
                # Search for the line that indicates the end of the coordinate block
                for j in range(current_start, len(lines)):
                    if '-----' in lines[j]:
                        current_end = j
                        break
                else:
                    # If no separator line is found, skip to the next section
                    continue
                # Update start and end indices to the latest found section
                start_idx, end_idx = current_start, current_end

    if start_idx is None or end_idx is None or start_idx >= end_idx:
        print(f"No valid atomic coordinates found in {log_file_path}")
        return None

    atoms = []
    periodic_table = Chem.GetPeriodicTable()

    for line in lines[start_idx:end_idx]:
        tokens = line.strip().split()
        if len(tokens) < 6:
            continue  # Skip lines that do not have enough tokens
        try:
            atom_number = int(tokens[1])  # Atomic number is the second token
            x = float(tokens[3])
            y = float(tokens[4])
            z = float(tokens[5])
            atom_symbol = periodic_table.GetElementSymbol(atom_number)
            atoms.append(f"{atom_symbol}   {x:.6f}   {y:.6f}   {z:.6f}")
        except ValueError:
            # Handle cases where conversion to int or float fails
            continue
        except Exception as e:
            print(f"Unexpected error parsing line: {line}\nError: {e}")
            continue

    if not atoms:
        print(f"No valid atomic coordinates extracted from {log_file_path}")
        return None

    return atoms

def order_energy_gaussian(work_dir, filename, numconf, output_file):

    data = []
    escaped = re.escape(filename)
    file_pattern = re.compile(rf'^{escaped}_\d+\.log$')
    # Traverse all files in the specified folder
    for file in os.listdir(work_dir):
        if file_pattern.match(file):
            log_file_path = os.path.join(work_dir, file)
            energy = read_energy_from_gaussian(log_file_path)
            atoms = read_final_structure_from_gaussian(log_file_path)
            if energy is not None and atoms is not None:
                data.append({"Energy": energy, "Atoms": atoms})

    # Check if data is not empty
    if data:
        # Sort the structures by energy
        sorted_data = sorted(data, key=lambda x: x['Energy'])
        selected_data = sorted_data[:numconf]
        # Write the sorted structures to an .xyz file
        with open(output_file, 'w') as outfile:
            for item in selected_data:
                num_atoms = len(item['Atoms'])
                outfile.write(f"{num_atoms}\n")
                outfile.write(f"Energy = {item['Energy']}\n")
                for atom_line in item['Atoms']:
                    outfile.write(f"{atom_line}\n")

    else:
        print(f"No successful Gaussian output files found in {work_dir}")


def lmptoxyz(work_dir, pdb_file):

    file_prefix, file_extension = os.path.splitext(pdb_file)
    data_filepath = os.path.join(work_dir, f'{file_prefix}_gaff2.lmp')
    input_filepath = os.path.join(work_dir, f'{file_prefix}_lmp.xyz')
    output_filename = f'{file_prefix}_gmx.xyz'

    atom_map = parse_masses_from_lammps(data_filepath)

    with open(input_filepath, 'r') as fin, open(output_filename, 'w') as fout:
        for i, line in enumerate(fin):
            line = line.strip()
            if i < 2:
                fout.write(line + '\n')
            else:
                parts = line.split()
                if len(parts) >= 4 and parts[0].isdigit():
                    atom_id = int(parts[0])
                    if atom_id in atom_map:
                        parts[0] = atom_map[atom_id]
                    else:
                        print(f"Warning: Atom ID {atom_id} not found in atom_map.")
                fout.write(' '.join(parts) + '\n')

    print(f"the relaxed polymer chian has been written to {output_filename}\n")

    return output_filename

def parse_masses_from_lammps(data_filename):
    atom_map = {}
    with open(data_filename, 'r') as f:
        lines = f.readlines()

    masses_section = False
    for i, line in enumerate(lines):
        line_strip = line.strip()
        if line_strip == "Masses":
            masses_section = True
            start = i + 2  # Skip the next line (which is usually blank or comments)
            break

    if not masses_section:
        raise ValueError("Masses section not found in the LAMMPS data file.")

    # Now, parse the Masses section until an empty line or another section starts
    for line in lines[start:]:
        line_strip = line.strip()
        if line_strip == "" or any(line_strip.startswith(s) for s in ["Atoms", "Bonds", "Angles", "Dihedrals", "Impropers"]):
            break
        parts = line_strip.split()
        if len(parts) >= 2:
            atom_id = int(parts[0])
            mass = float(parts[1])
            atom_symbol = get_closest_element_by_mass(mass)
            atom_map[atom_id] = atom_symbol
    return atom_map

def get_closest_element_by_mass(target_mass, tolerance=0.5):

    element_masses = {
        'H': 1.008,  # hydrogen
        'B': 10.81,  # boron
        'C': 12.011,  # carbon
        'N': 14.007,  # nitrogen
        'O': 15.999,  # oxygen
        'F': 18.998,  # fluorine
        'Na': 22.990,  # sodium
        'Mg': 24.305,  # magnesium
        'Al': 26.982,  # aluminum
        'Si': 28.085,  # silicon
        'P': 30.974,  # phosphorus
        'S': 32.06,  # sulfur
        'Cl': 35.45,  # chlorine
        'K': 39.098,  # potassium
        'Ca': 40.078,  # calcium
        'Ti': 47.867,  # titanium
        'Cr': 51.996,  # chromium
        'Mn': 54.938,  # manganese
        'Fe': 55.845,  # iron
        'Ni': 58.693,  # nickel
        'Cu': 63.546,  # copper
        'Zn': 65.38,  # zinc
        'Br': 79.904,  # bromine
        'Ag': 107.87,  # silver
        'I': 126.90,  # iodine
        'Au': 196.97,  # gold
    }

    min_diff = np.inf
    closest_element = None

    for element, mass in element_masses.items():
        diff = abs(mass - target_mass)
        if diff < min_diff:
            min_diff = diff
            closest_element = element

    if min_diff > tolerance:
        print(f"Warning: No element found for mass {target_mass} within tolerance {tolerance}")
        closest_element = 'X'

    return closest_element

def smiles_to_atom_string(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    atom_list = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol != 'H':
            atom_list.append(symbol)
    atom_string = ''.join(atom_list)

    return atom_string