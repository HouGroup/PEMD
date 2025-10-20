"""
Polymer model building tools.

Developed by: Tan Shendong
Date: 2024.01.18
"""


import random
from pathlib import Path

from rdkit.Chem import AllChem
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

def mol_to_pdb(
    work_dir,
    mol: Chem.Mol,
    name: str,
    resname: str,
    pdb_filename: str,
    *,
    conf_id: int = 0,
    chain_id: str = "A",
    residue_number: int = 1,
    hetatm: bool = False,
    also_write_sdf: bool = False,
    also_write_mol2: bool = False,
):
    """
    直接把 RDKit Mol 写成 PDB，并显式写出 CONECT（保存键连）。
    同时可选写出 SDF/MOL2 来保留键级/芳香性等完整信息。
    """
    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)
    pdb_file = work_path / pdb_filename

    # 1) 确保有 3D 构象（LigParGen/MD 通常需要）
    m = Chem.Mol(mol)  # 复制，避免修改原对象
    if m.GetNumConformers() == 0:
        m = Chem.AddHs(m, addCoords=True)
        AllChem.EmbedMolecule(m, AllChem.ETKDG())
        try:
            AllChem.UFFOptimizeMolecule(m)
        except Exception:
            pass  # 有些体系 UFF 不稳定，忽略即可
    # 设置分子名
    m.SetProp("_Name", name)

    # 2) 给每个原子设置 PDB Residue/Atom 信息（PDB 字段更规范）
    resname = (resname or "MOL")[:3].upper()
    for i, atom in enumerate(m.GetAtoms()):
        # PDB 原子名字段宽度为 4；右对齐更标准
        atom_name = (atom.GetSymbol() + str((i+1) % 10000)).rjust(4)
        info = Chem.AtomPDBResidueInfo(
            atomName=atom_name,
            residueName=resname,
            residueNumber=residue_number,
            chainId=(chain_id or "A"),
            isHeteroAtom=bool(hetatm),
        )
        atom.SetMonomerInfo(info)

    # 3) 直接由 RDKit 写 PDB（包含坐标、残基、链等）
    Chem.MolToPDBFile(m, str(pdb_file), confId=conf_id)

    # 4) 确保写出 CONECT（显式保存“有哪些键”）
    #    某些 RDKit 版本/风味不一定自动写全，这里手动补齐。
    text = Path(pdb_file).read_text()
    if "CONECT" not in text:
        with open(pdb_file, "a") as fh:
            for b in m.GetBonds():
                i = b.GetBeginAtomIdx() + 1  # PDB 原子序号从 1 开始
                j = b.GetEndAtomIdx() + 1
                fh.write(f"CONECT{i:>5}{j:>5}\n")
        # PDB 规范以 END 结尾（RDKit 已写 END，若你补了 CONECT，最好再补一个 END）
        with open(pdb_file, "a") as fh:
            fh.write("END\n")

    # 5) 可选：同时写 SDF（完整保留键级/芳香性）或 MOL2（某些工具喜欢）
    if also_write_sdf:
        sdf_path = work_path / (Path(pdb_filename).with_suffix(".sdf").name)
        Chem.MolToMolFile(m, str(sdf_path), confId=conf_id)
    if also_write_mol2:
        mol2_path = work_path / (Path(pdb_filename).with_suffix(".mol2").name)
        try:
            Chem.MolToMol2File(m, str(mol2_path), confId=conf_id)
        except Exception:
            pass  # 旧版 RDKit 可能没有 Mol2 写出，忽略即可

    return str(pdb_file)


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


