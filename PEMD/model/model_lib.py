"""
PEMD code library.

Developed by: Tan Shendong
Date: 2024.03.15
"""


import subprocess
import numpy as np
import networkx as nx

from PEMD import io
from rdkit import Chem
from PEMD import constants as const
from PEMD.model import polymer
from openbabel import openbabel as ob
from networkx.algorithms import isomorphism
from scipy.spatial.transform import Rotation as R


def gen_oligomer_smiles(poly_name, dum1, dum2, atom1, atom2, smiles_each, length, smiles_LCap_, smiles_RCap_, ):

    (
        inti_mol3,
        monomer_mol,
        start_atom,
        end_atom,
    ) = gen_smiles_nocap(
        dum1,
        dum2,
        atom1,
        atom2,
        smiles_each,
        length,
    )

    # Obtain the SMILES with cap
    main_mol_noDum = gen_smiles_with_cap(
        poly_name,
        inti_mol3,
        atom1,
        atom2 + (length -1) * monomer_mol.GetNumAtoms(),
        smiles_LCap_,
        smiles_RCap_,
    )

    return Chem.MolToSmiles(main_mol_noDum)

def gen_smiles_nocap(dum1, dum2, atom1, atom2, smiles_each, length, ):
    # Connect the units and caps to obtain SMILES structure
    input_mol = Chem.MolFromSmiles(smiles_each)
    edit_m1 = Chem.EditableMol(input_mol)
    edit_m2 = Chem.EditableMol(input_mol)
    edit_m3 = Chem.EditableMol(input_mol)

    # Remove the second dummy atom to build the left fragment
    edit_m1.RemoveAtom(dum2)
    mol_without_dum1 = Chem.Mol(edit_m1.GetMol())

    # Remove the first dummy atom to build the right fragment
    edit_m2.RemoveAtom(dum1)
    mol_without_dum2 = Chem.Mol(edit_m2.GetMol())

    # Build the monomer unit without either dummy atom
    edit_m3.RemoveAtom(dum1)
    if dum1 < dum2:  # If dum1 < dum2, then the index of dum2 is dum2 - 1
        edit_m3.RemoveAtom(dum2 - 1)
    else:
        edit_m3.RemoveAtom(dum2)
    monomer_mol = edit_m3.GetMol()
    inti_mol = monomer_mol

    # After removing dummy atoms,adjust the order and set first_atom and second_atom to represent connecting atoms
    if atom1 > atom2:
        atom1, atom2 = atom2, atom1

    if dum1 < atom1 and dum2 < atom1:
        first_atom = atom1 - 2
    elif (dum1 < atom1 < dum2) or (dum1 > atom1 > dum2):
        first_atom = atom1 - 1
    else:
        first_atom = atom1

    if dum1 < atom2 and dum2 < atom2:
        second_atom = atom2 - 2
    elif (dum1 < atom2 < dum2) or (dum1 > atom2 > dum2):
        second_atom = atom2 - 1
    else:
        second_atom = atom2

    inti_mol3 = None

    if length == 1:
        inti_mol3 = input_mol

    # Connect the units
    elif length > 1:

        if length > 2:
            for i in range(1, length - 2):      # First connect the middle n-2 units
                combo = Chem.CombineMols(inti_mol, monomer_mol)
                edcombo = Chem.EditableMol(combo)
                edcombo.AddBond(
                    second_atom + (i - 1) * monomer_mol.GetNumAtoms(),
                    first_atom + i * monomer_mol.GetNumAtoms(),
                    order=Chem.rdchem.BondType.SINGLE,
                )
                # Add bond according to the index of atoms to be connected

                inti_mol = edcombo.GetMol()

            inti_mol1 = inti_mol

            combo = Chem.CombineMols(mol_without_dum1, inti_mol1)   # Connect the leftmost unit
            edcombo = Chem.EditableMol(combo)
            edcombo.AddBond(
                atom2,
                first_atom + mol_without_dum1.GetNumAtoms(),
                order=Chem.rdchem.BondType.SINGLE
            )
            # print(atom2)
            # print(first_atom + mol_without_dum1.GetNumAtoms())
            inti_mol2 = edcombo.GetMol()

            combo = Chem.CombineMols(inti_mol2, mol_without_dum2)   # Connect the rightmost unit
            edcombo = Chem.EditableMol(combo)
            # print(atom2 + inti_mol1.GetNumAtoms())
            # print(first_atom + inti_mol2.GetNumAtoms())
            edcombo.AddBond(
                atom2 + inti_mol1.GetNumAtoms(),
                first_atom + inti_mol2.GetNumAtoms(),
                order=Chem.rdchem.BondType.SINGLE
            )
            inti_mol3 = edcombo.GetMol()

        else:
            combo = Chem.CombineMols(mol_without_dum1, mol_without_dum2)
            edcombo = Chem.EditableMol(combo)
            edcombo.AddBond(
                atom2,
                first_atom + mol_without_dum1.GetNumAtoms(),
                order=Chem.rdchem.BondType.SINGLE
            )
            inti_mol3 = edcombo.GetMol()

    return inti_mol3, monomer_mol, first_atom, second_atom + (length-1)*monomer_mol.GetNumAtoms()

def _kabsch_rotation(P, Q):
    """
    Compute the optimal rotation matrix with the Kabsch algorithm so that rotating
    ``P`` aligns it with ``Q`` as closely as possible. Both ``P`` and ``Q`` are
    expected to be ``(n, 3)`` arrays.
    """
    C = np.dot(P.T, Q)
    V, S, Wt = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(Wt)) < 0.0
    if d:
        V[:, -1] = -V[:, -1]
    return np.dot(V, Wt)

def rotate_mol_around_axis(mol, axis, anchor, angle_rad):
    """
    Rotate the entire molecule around ``axis`` (a unit vector) by ``angle_rad``
    radians, using ``anchor`` as the rotation center.
    """
    conf = mol.GetConformer()
    rot = R.from_rotvec(axis * angle_rad)
    for atom_idx in range(mol.GetNumAtoms()):
        pos = np.array(conf.GetAtomPosition(atom_idx))
        pos_shifted = pos - anchor
        pos_rot = rot.apply(pos_shifted)
        conf.SetAtomPosition(atom_idx, pos_rot + anchor)

# (symbol1 == 'C' and symbol2 == 'C') or
#            (symbol1 == 'O' and symbol2 == 'H') or (symbol1 == 'H' and symbol2 == 'O'):
def gen_smiles_with_cap(poly_name, inti_mol, first_atom, second_atom, smiles_LCap_, smiles_RCap_):

    # Add cap to main chain
    main_mol_Dum = inti_mol

    # Left Cap
    if not smiles_LCap_:
        # Decide whether to cap with H or CH3
        first_atom_obj = main_mol_Dum.GetAtomWithIdx(first_atom)
        atomic_num = first_atom_obj.GetAtomicNum() # Find its element
        degree = first_atom_obj.GetDegree() # Find the number of bonds associated with it
        num_implicit_hs = first_atom_obj.GetNumImplicitHs()
        num_explicit_hs = first_atom_obj.GetNumExplicitHs()
        total_hs = num_implicit_hs + num_explicit_hs # Find the number of hydrogen associated with it

        if atomic_num == 6 and degree == 2 and total_hs == 2:
            pass
        else:
            # Cap with CH3
            cap_mol = Chem.MolFromSmiles('C')
            cap_add = cap_mol.GetNumAtoms()
            combo = Chem.CombineMols(cap_mol, main_mol_Dum)
            edcombo = Chem.EditableMol(combo)
            edcombo.AddBond(0, cap_add + first_atom, order=Chem.rdchem.BondType.SINGLE)
            main_mol_Dum = edcombo.GetMol()
            second_atom += cap_add  # adjust the index of second_atom

    else:
        # Existing code for handling Left Cap
        (
            unit_name,
            dum_L,
            atom_L,
            neigh_atoms_info_L,
            flag_L
        ) = Init_info_Cap(
            poly_name,
            smiles_LCap_
        )

        # Reject if SMILES is not correct
        if flag_L == 'REJECT':
            return unit_name, 'REJECT', 0

        # Editable Mol for LeftCap
        LCap_m1 = Chem.MolFromSmiles(smiles_LCap_)
        LCap_edit_m1 = Chem.EditableMol(LCap_m1)

        # Remove dummy atoms
        LCap_edit_m1.RemoveAtom(dum_L)

        # Mol without dummy atom
        LCap_m1 = LCap_edit_m1.GetMol()
        LCap_add = LCap_m1.GetNumAtoms()
        # Linking atom
        if dum_L < atom_L:
            LCap_atom = atom_L - 1
        else:
            LCap_atom = atom_L

        # Join main chain with Left Cap
        combo = Chem.CombineMols(LCap_m1, inti_mol)
        edcombo = Chem.EditableMol(combo)
        edcombo.AddBond(
            LCap_atom, first_atom + LCap_add, order=Chem.rdchem.BondType.SINGLE
        )
        main_mol_Dum = edcombo.GetMol()
        second_atom += LCap_add     # adjust the index of second_atom

    # Right Cap
    if not smiles_RCap_:
        # Decide whether to cap with H or CH3
        second_atom_obj = main_mol_Dum.GetAtomWithIdx(second_atom)
        atomic_num = second_atom_obj.GetAtomicNum()
        degree = second_atom_obj.GetDegree()
        num_implicit_hs = second_atom_obj.GetNumImplicitHs()
        num_explicit_hs = second_atom_obj.GetNumExplicitHs()
        total_hs = num_implicit_hs + num_explicit_hs

        if atomic_num == 6 and degree == 2 and total_hs == 2:
            pass
        else:
            # Cap with CH3
            cap_mol = Chem.MolFromSmiles('C')
            cap_add = cap_mol.GetNumAtoms()
            combo = Chem.CombineMols(main_mol_Dum, cap_mol)
            edcombo = Chem.EditableMol(combo)
            edcombo.AddBond(
                second_atom,
                main_mol_Dum.GetNumAtoms(),  # Index of the cap atom
                order=Chem.rdchem.BondType.SINGLE,
            )
            main_mol_Dum = edcombo.GetMol()
    else:
        # Existing code for handling Right Cap
        (
            unit_name,
            dum_R,
            atom_R,
            neigh_atoms_info_R,
            flag_R
        ) = Init_info_Cap(
            poly_name,
            smiles_RCap_
        )

        # Reject if SMILES is not correct
        if flag_R == 'REJECT':
            return unit_name, 'REJECT', 0

        # Editable Mol for RightCap
        RCap_m1 = Chem.MolFromSmiles(smiles_RCap_)
        RCap_edit_m1 = Chem.EditableMol(RCap_m1)

        # Remove dummy atoms
        RCap_edit_m1.RemoveAtom(dum_R)

        # Mol without dummy atom
        RCap_m1 = RCap_edit_m1.GetMol()

        # Linking atom
        if dum_R < atom_R:
            RCap_atom = atom_R - 1
        else:
            RCap_atom = atom_R

        # Join main chain with Right Cap
        combo = Chem.CombineMols(main_mol_Dum, RCap_m1)
        edcombo = Chem.EditableMol(combo)
        edcombo.AddBond(
            second_atom,
            RCap_atom + main_mol_Dum.GetNumAtoms(),
            order=Chem.rdchem.BondType.SINGLE,
        )
        main_mol_Dum = edcombo.GetMol()

    # Remove remain dummy atoms (virtual atoms marked as [*])
    atoms_to_remove = []

    for atom in main_mol_Dum.GetAtoms():
        if atom.GetSymbol() == '*' or atom.GetAtomicNum() == 0:  # Checking for virtual atom
            atoms_to_remove.append(atom.GetIdx())

    # Create a new editable molecule to remove the atoms
    edmol_final = Chem.RWMol(main_mol_Dum)
    # Reverse traversal to prevent index changes
    for atom_idx in reversed(atoms_to_remove):
        edmol_final.RemoveAtom(atom_idx)

    main_mol_noDum = edmol_final.GetMol()
    main_mol_noDum = Chem.RemoveHs(main_mol_noDum)

    return main_mol_noDum

def Init_info_Cap(unit_name, smiles_each_ori):
    # Get index of dummy atoms and bond type associated with it in cap
    try:
        dum_index, bond_type = polymer.FetchDum(smiles_each_ori)
        if len(dum_index) == 1:
            dum1 = dum_index[0]
        else:
            print(
                unit_name,
                ": There are more or less than one dummy atoms in the SMILES string; ",
            )
            return unit_name, 0, 0, 0, 'REJECT'
    except Exception:
        print(
            unit_name,
            ": Couldn't fetch the position of dummy atoms. Hints: (1) In SMILES string, use '*' for a dummy atom,"
            "(2) Check RDKit installation.",
        )
        return unit_name, 0, 0, 0, 'REJECT'

    # Replace '*' with dummy atom to find atoms associated with it
    smiles_each = smiles_each_ori.replace(r'*', 'Cl')

    # Convert SMILES to XYZ coordinates
    convert_smiles2xyz = io.smile_toxyz(unit_name, smiles_each,)

    # if fails to get XYZ coordinates; STOP
    if convert_smiles2xyz == 'NOT_DONE':
        print(
            unit_name,
            ": Couldn't get XYZ coordinates from SMILES string. Hints: (1) Check SMILES string,"
            "(2) Check RDKit installation.",
        )
        return unit_name, 0, 0, 0, 'REJECT'

    # Collect valency and connecting information for each atom
    neigh_atoms_info = polymer.connec_info('./' + unit_name + '.xyz')

    try:
        # Find connecting atoms associated with dummy atoms.
        # dum1 and dum2 are connected to atom1 and atom2, respectively.
        atom1 = neigh_atoms_info['NeiAtom'][dum1].copy()[0]

    except Exception:
        print(
            unit_name,
            ": Couldn't get the position of connecting atoms. Hints: (1) XYZ coordinates are not acceptable,"
            "(2) Check Open Babel installation.",
        )
        return unit_name, 0, 0, 0, 'REJECT'
    return (
        unit_name,
        dum1,
        atom1,
        neigh_atoms_info,
        '',
    )


def read_energy_from_xtb(filename):
    """Read the energy value from an xTB output file."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    # Assume the energy is stored on the second line of the output file
    energy_line = lines[1]
    energy = float(energy_line.split()[1])
    return energy

def std_xyzfile(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    modified_lines = []
    structure_count = int(lines[0].strip())  # Assuming the first line contains the number of atoms

    # Process each structure in the file
    for i in range(0, len(lines), structure_count + 2):  # +2 for the atom count and energy lines
        # Copy the atom count line
        modified_lines.append(lines[i])

        # Extract and process the energy value line
        energy_line = lines[i + 1].strip()
        energy_value = energy_line.split()[1]  # Extract the energy value
        modified_lines.append(f" {energy_value}\n")  # Reconstruct the energy line with only the energy value

        # Copy the atom coordinate lines
        for j in range(i + 2, i + structure_count + 2):
            modified_lines.append(lines[j])

    # Overwrite the original file with modified lines
    with open(file_path, 'w') as file:
        file.writelines(modified_lines)

def convert_chk_to_fchk(chk_file_path):
    fchk_file_path = chk_file_path.replace('.chk', '.fchk')

    try:
        subprocess.run(
            ['formchk', chk_file_path, fchk_file_path],
            check=True,
            stdout=subprocess.DEVNULL,  # Redirect standard output
            stderr=subprocess.DEVNULL   # Redirect standard error output
        )
    except subprocess.CalledProcessError as e:
        print(f"Error converting chk to fchk: {e}")



def print_compounds(info_dict, key_name):
    """
    This function recursively searches for and prints 'compound' entries in a nested dictionary.
    """
    compounds = []
    for key, value in info_dict.items():
        # If the value is a dictionary, we make a recursive call
        if isinstance(value, dict):
            compounds.extend(print_compounds(value,key_name))
        # If the key is 'compound', we print its value
        elif key == key_name:
            compounds.append(value)
    return compounds


def extract_volume(partition, module_soft, edr_file, output_file='volume.xvg', option_id='21'):

    if partition == 'gpu':
        command = f"module load {module_soft} && echo {option_id} | gmx energy -f {edr_file} -o {output_file}"
    else:
        command = f"module load {module_soft} && echo {option_id} | gmx_mpi energy -f {edr_file} -o {output_file}"

    # Execute the command with subprocess.run; using ``bash -c`` requires adjusting
    # how stdin is passed.
    try:
        # Use ``capture_output=True`` to collect stdout instead of PIPE
        process = subprocess.run(['bash', '-c', command], capture_output=True, text=True, check=True)
        # ``check=True`` raises an exception when the command fails, so no explicit
        # return-code check is required.
        print(f"Output: {process.stdout}")
        return process.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e.stderr}")
        return None


def read_volume_data(volume_file):

    volumes = []
    with open(volume_file, 'r') as file:
        for line in file:
            if line.startswith(('@', '#')):
                continue
            parts = line.split()
            volumes.append(float(parts[1]))

    return np.array(volumes)


def analyze_volume(volumes, start, dt_collection):

    start_time = int(start) / dt_collection
    average_volume = np.mean(volumes[int(start_time):])
    closest_index = np.argmin(np.abs(volumes[int(start_time):] - average_volume))
    frame_time = start + closest_index * dt_collection
    return average_volume, frame_time


def extract_structure(partition, module_soft, tpr_file, xtc_file, save_gro_file, frame_time):

    if partition == 'gpu':
        command = (f"module load {module_soft} && echo 0 | gmx trjconv -s {tpr_file} -f {xtc_file} -o {save_gro_file} "
                   f"-dump {frame_time} -quiet")
    else:
        command = (f"module load {module_soft} && echo 0 | gmx_mpi trjconv -s {tpr_file} -f {xtc_file} -o {save_gro_file} "
                   f"-dump {frame_time} -quiet")

    # Use subprocess.run for safer external command execution
    try:
        # Avoid ``shell=True`` and rely on subprocess.run for better security
        process = subprocess.run(['bash', '-c', command], capture_output=True, text=True, check=True)
        print(f"Output: {process.stdout}")
        return process.stdout
    except subprocess.CalledProcessError as e:
        # Error handling: print stderr and return ``None``
        print(f"Error executing command: {e.stderr}")
        return None


# Convert an RDKit ``mol`` object into a NetworkX graph
def mol_to_networkx_rdkit(mol, include_h=True):
    """
    Convert an RDKit ``mol`` object into a NetworkX graph.
    """
    G = nx.Graph()
    for atom in mol.GetAtoms():
        if not include_h and atom.GetSymbol() == 'H':
            continue
        G.add_node(atom.GetIdx(), element=atom.GetSymbol())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type=str(bond.GetBondType()))
    return G


# Convert an Open Babel ``mol`` object into a NetworkX graph
def mol_to_networkx_ob(mol, include_h=True):
    """
    Convert an Open Babel ``mol`` object into a NetworkX graph.
    """
    G = nx.Graph()

    # Add atom nodes and change indices from one-based to zero-based
    for atom in ob.OBMolAtomIter(mol):
        atomic_num = atom.GetAtomicNum()  # Retrieve the atomic number
        if 1 <= atomic_num <= len(const.PERIODIC_TABLE):
            element = const.PERIODIC_TABLE[atomic_num - 1]
        else:
            element = 'Unknown'
        if not include_h and element == 'H':
            continue
        # Add the node with the adjusted index
        G.add_node(atom.GetIdx() - 1, element=element)

    # Add bond edges with indices converted from one-based to zero-based
    for bond in ob.OBMolBondIter(mol):
        bond_order = bond.GetBondOrder()
        # Normalize the bond type representation
        if bond_order == 1:
            bond_type = 'SINGLE'
        elif bond_order == 2:
            bond_type = 'DOUBLE'
        elif bond_order == 3:
            bond_type = 'TRIPLE'
        else:
            bond_type = 'SINGLE'  # Default handling
        # ``GetBeginAtomIdx`` and ``GetEndAtomIdx`` are one-based in Open Babel
        G.add_edge(bond.GetBeginAtomIdx() - 1, bond.GetEndAtomIdx() - 1, bond_type=str(bond_type))

    return G


def get_atom_mapping(mol1, mol2, include_h=True):
    """
    Retrieve the atom correspondence between two ``mol`` objects.
    Returns a mapping dict from ``mol1`` node indices to ``mol2`` node indices.
    """
    # Convert ``mol1`` and ``mol2`` to NetworkX graphs
    G1 = mol_to_networkx_rdkit(mol1, include_h=include_h)  # NetworkX graph from RDKit mol
    G2 = mol_to_networkx_ob(mol2, include_h=include_h)  # NetworkX graph from Open Babel mol

    # Debug output for graph information
    print(f"G1 nodes: {G1.nodes(data=True)}")
    print(f"G2 nodes: {G2.nodes(data=True)}")
    print(f"G1 edges: {G1.edges(data=True)}")
    print(f"G2 edges: {G2.edges(data=True)}")

    # Define node matching based on atomic elements
    def node_match(n1, n2):
        return (n1['element'] == n2['element']) and (n1.get('charge', 0) == n2.get('charge', 0))

    def bond_match(b1, b2):
        return b1['bond_type'] == b2['bond_type']

    # Create the graph isomorphism matcher
    gm = isomorphism.GraphMatcher(G1, G2, node_match=node_match, edge_match=bond_match)
    if gm.is_isomorphic():
        # Return a possible mapping dict from G1 nodes to G2 nodes
        mapping = gm.mapping
        return mapping
    else:
        return None

def reorder_atoms(mol_3D, mapping):
    """
    Reorder ``mol_3D`` so that its atom order matches the mapping from
    ``mol1_idx`` to ``mol2_idx``.
    """
    # Create the reverse mapping: mol2_idx -> mol1_idx
    reverse_mapping = {v: k for k, v in mapping.items()}

    # Ensure mol2 does not contain more atoms than mol1
    num_atoms = mol_3D.GetNumAtoms()
    new_order = []
    for i in range(num_atoms):
        if i in reverse_mapping:
            new_order.append(reverse_mapping[i])
        else:
            # If mol2 has no corresponding atom, keep the original order
            new_order.append(i)

    # Reorder atoms in ``mol_3D``
    reordered_mol_3D = Chem.RenumberAtoms(mol_3D, new_order)

    return reordered_mol_3D

def distance_matrix(coord1, coord2=None):
    coord1 = np.array(coord1)
    coord2 = np.array(coord2) if coord2 is not None else coord1
    return np.sqrt(np.sum((coord1[:, np.newaxis, :] - coord2[np.newaxis, :, :])**2, axis=-1))







