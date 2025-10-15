# Copyright (c) 2024. PEMD developers. All rights reserved.
# Distributed under the terms of the MIT License.

# ******************************************************************************
# simulation.qm module
# ******************************************************************************

import glob
import pandas as pd

from rdkit import Chem
from pathlib import Path
from rdkit.Chem.AllChem import (
    EmbedMultipleConfs,
    MMFFGetMoleculeProperties,
    MMFFGetMoleculeForceField,
)

from PEMD.simulation import sim_lib
from PEMD.model import model_lib
from PEMD.simulation.xtb import PEMDXtb
from PEMD.simulation.gaussian import PEMDGaussian
from PEMD.simulation.multiwfn import PEMDMultiwfn


# Input: smiles (str)
# Output: a xyz file
# Description: Generates multiple conformers for a molecule from a SMILES string, optimizes them using the MMFF94
# force field, and saves the optimized conformers to a single XYZ file.
def gen_conf_rdkit(
    work_dir: Path | str,
    max_conformers: int = 1000,
    top_n_MMFF: int = 100,
    *,
    pdb_file: Path | str,
    smiles: str,
):

    # Generate multiple conformers
    work_path = Path(work_dir)
    mol = None
    if pdb_file:
        pdb_path = work_path / pdb_file
        mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=False)
    elif smiles:
        mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    ids = EmbedMultipleConfs(mol, numConfs=max_conformers, randomSeed=20)
    props = MMFFGetMoleculeProperties(mol)

    # Minimize the energy of each conformer and store the energy
    minimized_conformers = []
    for conf_id in ids:
        ff = MMFFGetMoleculeForceField(mol, props, confId=conf_id)
        status = ff.Minimize()
        if status != 0:
            print(f"Conformer {conf_id} optimization did not converge. Status code: {status}")
        energy = ff.CalcEnergy()
        minimized_conformers.append((conf_id, energy))

    print(f"[1] Generated {len(minimized_conformers)} conformers using RDKit.\n")

    # Sort the conformers by energy and select the top N conformers
    minimized_conformers.sort(key=lambda x: x[1])
    top_conformers = minimized_conformers[:top_n_MMFF]

    # merge the top conformers to a single xyz file
    output_file = f"MMFF_top{top_n_MMFF}.xyz"
    output_path = work_path / output_file
    with open(output_path, 'w') as merged_xyz:
        for idx, (conf_id, energy) in enumerate(top_conformers):
            conf = mol.GetConformer(conf_id)
            atoms = mol.GetAtoms()
            num_atoms = mol.GetNumAtoms()
            merged_xyz.write(f"{num_atoms}\n")
            merged_xyz.write(f"Conformer {idx + 1}, Energy: {energy:.4f} kcal/mol\n")
            for atom in atoms:
                pos = conf.GetAtomPosition(atom.GetIdx())
                element = atom.GetSymbol()
                merged_xyz.write(f"{element} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n")

    print(f"[2] Saved {len(top_conformers)} MMFF-minimized, lowest-energy conformers to '{output_file}'.\n")

    return output_file

def conf_xtb(
    work_dir: Path | str,
    xyz_file: str,
    top_n_xtb: int = 8,
    charge: float = 0,
    mult: int = 1,
    gfn: str = 'gfn2',
    optimize: bool = True,
):
    work_path = Path(work_dir)
    xtb_dir = work_path / f"XTB_dir"
    xtb_dir.mkdir(parents=True, exist_ok=True)

    full_xyz = Path(work_dir) / xyz_file
    structures = sim_lib.read_xyz_file(str(full_xyz))

    energy_list = []

    for idx, structure in enumerate(structures):
        comment = structure['comment']
        atoms = structure['atoms']
        conf_xyz = xtb_dir / f"conf_{idx}.xyz"
        with open(conf_xyz, 'w') as f:
            f.write(f"{structure['num_atoms']}\n")
            f.write(f"{comment}\n")
            for atom in atoms:
                f.write(f"{atom}\n")

        outfile_headname = f'conf_{idx}'

        xtb_calculator = PEMDXtb(
            work_dir=xtb_dir,
            chg=charge,
            mult=mult,
            gfn=gfn
        )
        result = xtb_calculator.run_local(
            xyz_filename=conf_xyz,
            outfile_headname=outfile_headname,
            optimize=optimize
        )

        if not optimize:
            if isinstance(result, dict):
                energy_info = result.get('energy_info')
                if energy_info and 'total_energy' in energy_info:
                    energy = energy_info['total_energy']
                    energy_list.append({
                        'idx': idx,
                        'energy': energy,
                        'filename': f'conf_{idx}.xyz'
                    })
                else:
                    print(f"Failed to extract energy for structure conf_{idx}.xyz.")
            else:
                print(f"Calculation failed for structure conf_{idx}.xyz.")
        else:
            energy_info = result.get('energy_info')
            if energy_info and 'total_energy' in energy_info:
                energy = energy_info['total_energy']
                energy_list.append({
                    'idx': idx,
                    'energy': energy,
                    'filename': f'conf_{idx}.xtbopt.xyz'
                })
            else:
                print(f"Failed to extract optimized energy for structure conf_{idx}.xyz.")

    # print("XTB run locally successfully!")

    if not energy_list:
        print("No energy values were successfully extracted.")
        return None

    sorted_energies = sorted(energy_list, key=lambda x: x['energy'])
    top_structures = sorted_energies[:top_n_xtb]

    output_path = work_path / f"xtb_top{top_n_xtb}.xyz"
    with open(output_path, 'w') as out:
        for r in top_structures:
            src = xtb_dir / r['filename']
            if src.exists():
                out.write(src.read_text())
            else:
                print(f"File {src} not found.")

    print(f"[3] Saved {len(top_structures)} lowest-energy xtb structures to 'xtb_top{top_n_xtb}.xyz'.\n")

    return output_path


# input: a xyz file
# output: a xyz file
# description:
def qm_gaussian(
        work_dir: Path | str,
        xyz_file: str,
        gjf_filename: str,
        *,
        charge: float = 0,
        mult: int = 1,
        function: str = 'B3LYP',
        basis_set: str = '6-31+g(d,p)',
        epsilon: float = 5.0,
        core: int = 64,
        memory: str = '128GB',
        chk: bool = False,
        optimize: bool = True,
        multi_step: bool = False,
        max_attempts: int = 1,
        toxyz: bool = False,
        top_n_qm: int | None = None,
):

    work_path = Path(work_dir)
    qm_dir = work_path / 'QM_dir'
    qm_dir.mkdir(parents=True, exist_ok=True)

    xyz_path = Path(work_dir) / xyz_file
    structures = sim_lib.read_xyz_file(xyz_path)

    success_indices: list[int] = []

    for idx, structure in enumerate(structures):
        filename = f'{gjf_filename}_{idx}.gjf'
        Gau = PEMDGaussian(
            work_dir=qm_dir,
            filename=filename,
            core=core,
            mem=memory,
            chg=charge,
            mult=mult,
            function=function,
            basis_set=basis_set,
            epsilon=epsilon,
            optimize=optimize,
            multi_step=multi_step,
            max_attempts=max_attempts,
        )

        state, log_filename = Gau.run_local(
            structure=structure,
            resp=False,
            chk=chk,
        )

        if state == 'success':
            Gau.logger.info(f"Optimization succeeded for {filename}.")
            success_indices.append(idx)
        else:
            Gau.logger.error(f"Optimization failed for {filename}.")

    if toxyz or top_n_qm is not None:
        n_success = len(success_indices)
        if n_success == 0:
            print("[4] No successful Gaussian jobs. Skip exporting xyz.")
            return None

        if (top_n_qm is None) or (top_n_qm in (0, -1)) or (top_n_qm >= n_success):
            n_keep = n_success
            out_name = "gaussian_all.xyz"
        else:
            n_keep = max(1, int(top_n_qm))
            out_name = f"gaussian_top{n_keep}.xyz"

        output_filepath = Path(work_dir) / out_name
        sim_lib.order_energy_gaussian(
            qm_dir,
            gjf_filename,
            n_keep,
            output_filepath,
        )

        print(f"[4] Saved {n_keep}/{n_success} lowest-energy Gaussian structures to {out_name}.")
        return out_name

    return None


def calc_resp_gaussian(
        work_dir: Path | str,
        xyz_file: str,
        charge: float = 0,
        mult: int = 1,
        function: str = 'B3LYP',
        basis_set: str = '6-311+g(d,p)',
        epsilon: float = 5.0,
        core: int = 32,
        memory: str = '64GB',
):
    # Build the resp_dir.
    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)
    resp_path = work_path / f"resp_dir"
    resp_path.mkdir(exist_ok=True)

    # Read xyz file as a list of structures.
    structures = sim_lib.read_xyz_file(xyz_file)

    # Generate Gaussian input files of selected conformers.
    for idx, structure in enumerate(structures):
        filename = f"conf_{idx}.gjf"
        Gau = PEMDGaussian(
            work_dir=resp_path,
            filename=filename,
            core=core,
            mem=memory,
            chg=charge,
            mult=mult,
            function=function,
            basis_set=basis_set,
            epsilon=epsilon,
            multi_step=False,  # Disable multi-step for RESP calculations
            max_attempts=1,  # Only attempt once
        )

        state, log_filename = Gau.run_local(
            structure=structure,
            resp=True,
            chk=False,
        )

        if state == 'failed':
            Gau.logger.error(f"RESP calculation failed for {filename}.")


def RESP_fit_Multiwfn(
    work_dir: Path | str,
    method: str = 'resp2',
    delta: float = 0.5
):
    # Build the resp_dir.
    work_path = Path(work_dir)
    resp_path = work_path / f"resp_dir"
    resp_path.mkdir(parents=True, exist_ok=True)

    # Find chk files and convert them to fchk files.
    chk_pattern = resp_path / 'SP*.chk'
    chk_files = glob.glob(str(chk_pattern))
    for chk_file in chk_files:
        model_lib.convert_chk_to_fchk(chk_file)

    # Calculate RESP charges using Multiwfn.
    PEMDMultiwfn(str(resp_path)).resp_run_local(method)

    # Read charges data of solvation state.
    solv_chg_df = pd.DataFrame()
    solv_chg_files = glob.glob(str(resp_path / 'SP_solv_conf*.chg'))
    # Calculate average charges of solvation state.
    for file in solv_chg_files:
        data = pd.read_csv(file, sep=r'\s+', names=['atom', 'X', 'Y', 'Z', 'charge'])
        data['position'] = data.index
        solv_chg_df = pd.concat([solv_chg_df, data], ignore_index=True)
    average_charges_solv = solv_chg_df.groupby('position')['charge'].mean().reset_index()

    # If using RESP2 method, calculate weighted charge of both solvation and gas states.
    if method == 'resp2':
        # Read charges data of gas state.
        gas_chg_df = pd.DataFrame()
        gas_chg_files = glob.glob(str(resp_path / 'SP_gas_conf*.chg'))
        # Calculate average charges of gas state.
        for file in gas_chg_files:
            data = pd.read_csv(file, sep=r'\s+', names=['atom', 'X', 'Y', 'Z', 'charge'])
            data['position'] = data.index
            gas_chg_df = pd.concat([gas_chg_df, data], ignore_index=True)
        average_charges_gas = gas_chg_df.groupby('position')['charge'].mean().reset_index()
        # Combine the average charges of solvation and gas states, calculated by weight.
        average_charges = average_charges_solv.copy()
        average_charges['charge'] = average_charges_solv['charge'] * delta + average_charges_gas['charge'] * (1 - delta)
    else:
        # If using RESP method, just calculate average charges of solvation state.
        average_charges = average_charges_solv

    # Extract atomic types and retain the position for mapping.
    reference_file = solv_chg_files[0]
    ref_data = pd.read_csv(reference_file, sep=r'\s+', names=['atom', 'X', 'Y', 'Z', 'charge'])
    atom_types = ref_data['atom']
    average_charges['atom'] = atom_types.values
    # Retain 'position' to map charges to atoms
    average_charges = average_charges[['atom', 'charge']]

    # Save to csv file.
    csv_path = resp_path / f"{method}_average_chg.csv"
    average_charges.to_csv(csv_path, index=False)

    return average_charges







