"""
Polymer model building tools.

Developed by: Tan Shendong
Date: 2024.01.18
"""


import random
from pathlib import Path

from PEMD import io
from rdkit import Chem
from PEMD.model import polymer
from rdkit.Chem import Descriptors
from PEMD.model import model_lib


def gen_copolymer_3D(smiles_A,
                     smiles_B,
                     *,
                     name: str | None = None,
                     mode: str | None = None,
                     length: int | None = None,
                     frac_A: float = 0.5,
                     block_sizes: list[int] | None = None,
                     sequence: list[str] | None = None,
                     left_cap_smiles: str | None = None,
                     right_cap_smiles: str | None = None):
    """Generate a 3D copolymer model."""

    if sequence is None:
        if mode == "homopolymer":
            if length is None:
                raise ValueError("length is required for homopolymer mode")
            sequence = ['A'] * length
        elif mode == "random":
            if length is None:
                raise ValueError("length is required for random mode")
            sequence = [
                'A' if random.random() < frac_A else 'B'
                for _ in range(length)
            ]
        elif mode == "alternating":
            if length is None:
                raise ValueError("length is required for alternating mode")
            sequence = ['A' if i % 2 == 0 else 'B' for i in range(length)]
        elif mode == "block":
            if not block_sizes:
                raise ValueError("block_sizes is required for block mode")
            sequence = []
            for i, blk in enumerate(block_sizes):
                mon = 'A' if i % 2 == 0 else 'B'
                sequence += [mon] * blk
        else:
            raise ValueError("mode must be provided when sequence is None")

    return polymer.gen_sequence_copolymer_3D(
        name,
        smiles_A,
        smiles_B,
        sequence,
        left_cap_smiles=left_cap_smiles,
        right_cap_smiles=right_cap_smiles,
    )

def mol_to_pdb(work_dir, mol, name, resname, pdb_filename):
    work_path = Path(work_dir)
    pdb_file = work_path / pdb_filename
    Chem.MolToXYZFile(mol, 'mid.xyz', confId=0)
    io.convert_xyz_to_pdb('mid.xyz', pdb_file, name, resname)
    Path("mid.xyz").unlink(missing_ok=True)


def calc_poly_chains(num_Li_salt , conc_Li_salt, mass_per_chain):

    # calculate the mol of LiTFSI salt
    avogadro_number = 6.022e23  # unit 1/mol
    mol_Li_salt = num_Li_salt / avogadro_number # mol

    # calculate the total mass of the polymer
    total_mass_polymer =  mol_Li_salt / (conc_Li_salt / 1000)  # g

    # calculate the number of polymer chains
    num_chains = (total_mass_polymer*avogadro_number) / mass_per_chain  # no unit; mass_per_chain input unit g/mol

    return int(num_chains)


def calc_poly_length(total_mass_polymer, smiles_repeating_unit, smiles_leftcap, smiles_rightcap, ):
    # remove [*] from the repeating unit SMILES, add hydrogens, and calculate the molecular weight
    simplified_smiles_repeating_unit = smiles_repeating_unit.replace('[*]', '')
    molecule_repeating_unit = Chem.MolFromSmiles(simplified_smiles_repeating_unit)
    mol_weight_repeating_unit = Descriptors.MolWt(molecule_repeating_unit) - 2 * 1.008

    # remove [*] from the end group SMILES, add hydrogens, and calculate the molecular weight
    simplified_smiles_rightcap = smiles_rightcap.replace('[*]', '')
    simplified_smiles_leftcap = smiles_leftcap.replace('[*]', '')
    molecule_rightcap = Chem.MolFromSmiles(simplified_smiles_rightcap)
    molecule_leftcap = Chem.MolFromSmiles(simplified_smiles_leftcap)
    mol_weight_end_group = Descriptors.MolWt(molecule_rightcap) + Descriptors.MolWt(molecule_leftcap) - 2 * 1.008

    # calculate the mass of the polymer chain
    mass_polymer_chain = total_mass_polymer - mol_weight_end_group

    # calculate the number of repeating units in the polymer chain
    length = round(mass_polymer_chain / mol_weight_repeating_unit)

    return length


def gen_poly_smiles(poly_name, repeating_unit, length, leftcap, rightcap,):
    # Generate the SMILES representation of the polymer.
    (
        dum1,
        dum2,
        atom1,
        atom2,
    ) = polymer.Init_info(
        poly_name,
        repeating_unit,
    )

    smiles_poly = model_lib.gen_oligomer_smiles(
        poly_name,
        dum1,
        dum2,
        atom1,
        atom2,
        repeating_unit,
        length,
        leftcap,
        rightcap,
    )

    Path(f"{poly_name}.xyz").unlink(missing_ok=True)

    return smiles_poly


