# Copyright (c) 2024. PEMD developers. All rights reserved.
# Distributed under the terms of the MIT License.

# ******************************************************************************
# simulation.qm module
# ******************************************************************************

import glob
import math
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
    mol: Chem.Mol | None = None,
):

    # Generate multiple conformers
    work_path = Path(work_dir)

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

    return output_file              # 你原来的模块名

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
    xtb_dir = work_path / "XTB_dir"
    xtb_dir.mkdir(parents=True, exist_ok=True)

    full_xyz = work_path / xyz_file
    structures = sim_lib.read_xyz_file(str(full_xyz))

    energy_list = []

    for idx, structure in enumerate(structures):
        num_atoms = structure["num_atoms"]
        atoms     = structure["atoms"]
        ID        = structure.get("id")
        E0        = structure.get("energy")
        success0  = structure.get("success")

        # 根据新的字段拼一个注释行（也可以按你自己喜好改）
        comment_fields = []
        if ID is not None:
            comment_fields.append(f"id={ID}")
        if E0 is not None:
            comment_fields.append(f"E0={E0:.6f}")
        if success0 is not None:
            comment_fields.append(f"success={success0}")
        # 如果啥都没有，就至少给个标识
        comment = " ; ".join(comment_fields) if comment_fields else f"conf_{idx}"

        conf_xyz = xtb_dir / f"conf_{idx}.xyz"
        with open(conf_xyz, "w") as f:
            f.write(f"{num_atoms}\n")
            f.write(f"{comment}\n")
            for atom in atoms:
                f.write(f"{atom}\n")

        outfile_headname = f"conf_{idx}"

        xtb_calculator = PEMDXtb(
            work_dir=xtb_dir,
            chg=charge,
            mult=mult,
            gfn=gfn,
        )

        result = xtb_calculator.run_local(
            xyz_filename=conf_xyz,
            outfile_headname=outfile_headname,
            optimize=optimize,
        )

        # === 收集能量信息 ===
        if not optimize:
            if isinstance(result, dict):
                energy_info = result.get("energy_info")
                if energy_info and "total_energy" in energy_info:
                    energy = energy_info["total_energy"]
                    energy_list.append(
                        {
                            "idx": idx,
                            "energy": energy,
                            "filename": f"conf_{idx}.xyz",
                        }
                    )
                else:
                    print(f"Failed to extract energy for structure conf_{idx}.xyz.")
            else:
                print(f"Calculation failed for structure conf_{idx}.xyz.")
        else:
            if not isinstance(result, dict):
                print(f"Calculation failed for structure conf_{idx}.xyz.")
                continue
            energy_info = result.get("energy_info")
            if energy_info and "total_energy" in energy_info:
                energy = energy_info["total_energy"]
                energy_list.append(
                    {
                        "idx": idx,
                        "energy": energy,
                        "filename": f"conf_{idx}.xtbopt.xyz",
                        # 如果你想后面还能知道原始 id/energy/success，也可以一起存进去
                        # "id": ID,
                        # "init_energy": E0,
                        # "init_success": success0,
                    }
                )
            else:
                print(f"Failed to extract optimized energy for structure conf_{idx}.xyz.")

    if not energy_list:
        print("No energy values were successfully extracted.")
        return None

    # 按 XTB 计算出来的能量排序，取最低的 top_n_xtb 个
    sorted_energies = sorted(energy_list, key=lambda x: x["energy"])
    top_structures = sorted_energies[:top_n_xtb]

    output_path = work_path / f"xtb_top{top_n_xtb}.xyz"
    with open(output_path, "w") as out:
        for r in top_structures:
            src = xtb_dir / r["filename"]
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
        epsilon: float | None = None,
        core: int = 64,
        memory: str = '128GB',
        chk: bool = False,
        optimize: bool = True,
        multi_step: bool = False,
        max_attempts: int = 1,
        toxyz: bool = False,
        top_n_qm: int | None = None,
        dedup: bool = False,
        dedup_mode: str = 'energy',   # 'energy' | 'rmsd' | 'energy_and_rmsd' | 'energy_or_rmsd'
        energy_window: float = 0.10,  # kcal/mol
        rmsd_window: float = 0.30,    # Å
        continue_on_fail: bool = False,
        continue_on_success: bool = False,
        save_log: bool = True,
):
    """
    说明：
    - 仅当 (toxyz==True) 或 (top_n_qm is not None) 时，才会收集最终结构、执行去重与导出。
    - 能量阈值 energy_window 以 kcal/mol 传入；内部会自动换算至 Hartree（/627.509474）。
    - 去重模式：
        'energy'            仅能量阈值
        'rmsd'              仅 RMSD 阈值（重原子）
        'energy_and_rmsd'   两个都满足才视为重复
        'energy_or_rmsd'    任一满足即视为重复
    """
    work_path = Path(work_dir)
    qm_dir = work_path / 'QM_dir'
    qm_dir.mkdir(parents=True, exist_ok=True)

    xyz_path = Path(work_dir) / xyz_file
    structures = sim_lib.read_xyz_file(xyz_path)

    for i, st in enumerate(structures):
        if st.get('id') is None:
            st['id'] = i
        if 'success' not in st:
            st['success'] = None

    need_collect = bool(toxyz or (top_n_qm is not None))

    success_indices: list[int] = []
    final_structures: list[dict] = []

    if continue_on_success:
        indices = [st.get('id') for i, st in enumerate(structures) if st.get('success') is True]
    elif continue_on_fail:
        indices = [st.get('id') for i, st in enumerate(structures) if st.get('success') is False]
    else:
        indices = [st.get('id') for i, st in enumerate(structures)]

    for idx in indices:

        structure = next((st for st in structures if st.get('id') == idx), None)
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

        if need_collect:
            prefix = Path(filename).stem
            log_file = Path(qm_dir) / f"{prefix}.log"
            energy = None
            atoms = None
            if log_file.exists():
                energy = sim_lib.read_energy_from_gaussian(str(log_file))           # 假定返回 Hartree
                atoms = sim_lib.read_final_structure_from_gaussian(str(log_file))   # ['C x y z', ...]
            if not atoms:
                atoms = structure.get('atoms')
            if atoms:
                final_structures.append({
                    'num_atoms': len(atoms),
                    'atoms': atoms,
                    'id': idx,
                    'energy': energy,                     # Hartree 或 None
                    'success': (state == 'success'),
                })

        if not save_log:
            gjf_path = qm_dir / f'{gjf_filename}_{idx}.gjf'
            log_path = qm_dir / log_filename
            mid_log_path = qm_dir / f'{gjf_filename}_{idx}.gjf.log'
            if gjf_path.exists():
                gjf_path.unlink()
            if log_path.exists():
                log_path.unlink()
            if mid_log_path.exists():
                mid_log_path.unlink()

    # ================== 导出 / 去重（仅在 need_collect 时执行） ==================
    if need_collect:
        # 稳定排序：成功优先 → 能量 → id
        def _sort_key(item):
            e = item.get('energy')
            e_val = float('inf') if (e is None or (isinstance(e, float) and math.isnan(e))) else float(e)
            return (not item.get('success', False), e_val, item['id'])
        final_structures.sort(key=_sort_key)

        # 去重（可选）
        selected = final_structures
        if dedup and len(final_structures) > 1:
            energy_tol_in_input_unit = energy_window / 627.509474  # kcal/mol → Hartree
            selected = sim_lib.dedup_structures(
                final_structures,
                mode=dedup_mode,                          # 'energy' | 'rmsd' | 'energy_and_rmsd' | 'energy_or_rmsd'
                energy_tol_in_input_unit=energy_tol_in_input_unit,
                rmsd_tol=rmsd_window,
                rmsd_heavy_only=True,
            )

        # 处理 top_n（若指定）
        if (top_n_qm is not None) and (top_n_qm not in (0, -1)):
            # 按能量升序截取前 n；能量缺失视为 +∞
            selected.sort(key=lambda it: float('inf') if it.get('energy') is None else float(it['energy']))
            n_keep = max(1, min(int(top_n_qm), len(selected)))
            selected = selected[:n_keep]
            out_name = f"gaussian_top{n_keep}.xyz"
        else:
            out_name = "gaussian_all.xyz"

        # 写出 XYZ
        output_path = work_path / out_name
        with open(output_path, 'w') as outfile:
            for item in selected:
                outfile.write(f"{item['num_atoms']}\n")
                e = item.get('energy', None)
                if isinstance(e, (int, float)) and not (isinstance(e, float) and math.isnan(e)):
                    energy_str = f"{float(e):.10f}"
                else:
                    energy_str = "None"
                success_str = 'True' if item.get('success') else 'False'
                id_str = item['id']
                outfile.write(f"id = {id_str}; energy = {energy_str}; success = {success_str}\n")
                for atom_line in item['atoms']:
                    outfile.write(f"{atom_line}\n")

        print(f"[4] Saved final Gaussian structures to {out_name} "
              f"(collected={len(final_structures)}, kept={len(selected)}, "
              f"dedup={'on' if dedup else 'off'}, mode={dedup_mode}).")
        return out_name

    # ================== 不导出、不去重（保持原行为） ==================
    print("[4] Skip exporting xyz (toxyz=False and top_n_qm is None).")
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







