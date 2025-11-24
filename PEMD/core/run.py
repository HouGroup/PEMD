# Copyright (c) 2024. PEMD developers. All rights reserved.
# Distributed under the terms of the MIT License.

# ******************************************************************************
# Module Docstring
# ******************************************************************************


import os
import json

from pathlib import Path
from typing import List, Dict, Any
import PEMD.core.output_lib as lib
from dataclasses import dataclass, field

from PEMD.simulation.uma import uma
from PEMD.simulation.qm import (
    gen_conf_rdkit,
    conf_xtb,
    qm_gaussian,
    calc_resp_gaussian,
    RESP_fit_Multiwfn,
)
from PEMD.simulation.md import (
    relax_poly_chain,
    annealing,
    Tg,
    run_gmx_prod
)


@dataclass
class QMRun:
    work_dir: Path
    name: str
    smiles: str

    @staticmethod
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
        toxyz: bool = True,
        top_n_qm: int | None = None,
        dedup: bool = False,
        dedup_mode: str = 'energy',
        energy_window: float = 0.1, # kcal/mol
        rmsd_window: float = 0.3,   # angstrom
        continue_on_fail: bool = False,
        continue_on_success: bool = False,
        save_log: bool = True,
    ):
        lib.print_input('Quantum Chemistry Calculations')
        return qm_gaussian(
            work_dir=work_dir,
            xyz_file=xyz_file,
            gjf_filename=gjf_filename,
            charge=charge,
            mult=mult,
            function=function,
            basis_set=basis_set,
            epsilon=epsilon,
            core=core,
            memory=memory,
            chk=chk,
            optimize=optimize,
            multi_step=multi_step,
            max_attempts=max_attempts,
            toxyz=toxyz,
            top_n_qm=top_n_qm,
            dedup=dedup,
            dedup_mode=dedup_mode,
            energy_window=energy_window,
            rmsd_window=rmsd_window,
            continue_on_fail=continue_on_fail,
            continue_on_success=continue_on_success,
            save_log=save_log,
        )


    @staticmethod
    def uma(
        work_dir: Path | str,
        xyz_file: str,
        *,
        charge: float = 0,
        mult: int = 1,
        model_name: str = "uma-m-1p1",
        task_name: str = "omol",
        max_steps: int = 500,
        fmax: float = 0.03,
        device: str | None = None,
        opt_log: bool = False,
        use_local_uma: bool = False,
        local_checkpoint: Path | str | None = None,
        hf_cache_dir: Path | str | None = None,
        hf_offline: bool = False,
        top_n_uma: int | None = None,
        continue_on_fail: bool = False,
        continue_on_success: bool = False,
    ):
        lib.print_input('UMA Geometry Optimization')
        return uma(
            work_dir,
            xyz_file,
            top_n_uma=top_n_uma,
            charge=charge,
            mult=mult,
            model_name=model_name,
            task_name=task_name,
            max_steps=max_steps,
            fmax=fmax,
            device=device,
            opt_log=opt_log,
            use_local_uma=use_local_uma,
            local_checkpoint=local_checkpoint,
            hf_cache_dir=hf_cache_dir,
            hf_offline=hf_offline,
            continue_on_fail=continue_on_fail,
            continue_on_success=continue_on_success,
        )


    @staticmethod
    def conformer_search(
        work_dir: Path | str,
        *,
        smiles: str | None = None,
        pdb_file: str | None = None,
        mol: Any | None = None,
        max_conformers: int | None = None,
        top_n_MMFF: int | None = None,
        top_n_uma: int | None = None,
        top_n_xtb: int | None = None,
        top_n_qm: int | None = None,
        charge: float = 0,
        mult: int = 1,
        gfn: str = 'gfn2',
        function: str = 'b3lyp',
        basis_set: str = '6-31g*',
        epsilon: float | None = None,
        core: int = 32,
        memory: str = '64GB',
        max_steps_uma: int = 2000,
        fmax_uma: float = 0.0025,
        use_local_uma: bool = False,
        local_checkpoint: Path | str | None = None,
        hf_cache_dir: Path | str | None = None,
        hf_offline: bool = False,
    ):
        lib.print_input('Conformer Search')

        # Generate conformers using RDKit
        xyz_file_MMFF = gen_conf_rdkit(
            work_dir=work_dir,
            max_conformers=max_conformers,
            top_n_MMFF=top_n_MMFF,
            smiles=smiles,
            pdb_file=pdb_file,
            mol=mol,
        )

        xyz_file_xtb_uma = None
        # Optimize conformers using XTB
        if top_n_xtb:
            xyz_file_xtb_uma = conf_xtb(
                work_dir,
                xyz_file_MMFF,
                top_n_xtb=top_n_xtb,
                charge=charge,
                mult=mult,
                gfn=gfn,
                optimize=True
            )

        # Optimize conformers using UMA
        if top_n_uma:
            xyz_file_xtb_uma = uma(
                work_dir,
                xyz_file_MMFF,
                top_n_uma=top_n_uma,
                charge=charge,
                mult=mult,
                model_name="uma-m-1p1",
                task_name="omol",
                max_steps=max_steps_uma,
                fmax=fmax_uma,
                device=None,
                opt_log=True,
                use_local_uma=use_local_uma,
                local_checkpoint=local_checkpoint,
                hf_cache_dir=hf_cache_dir,
                hf_offline=hf_offline,
            )

        xyz_file_gaussian = None
        # Optimize conformers using Gaussian
        if top_n_qm:
            xyz_file_gaussian = qm_gaussian(
                work_dir,
                xyz_file_xtb_uma,
                gjf_filename="conf",
                charge=charge,
                mult= mult,
                function=function,
                basis_set=basis_set,
                epsilon=epsilon,
                core=core,
                memory=memory,
                optimize=True,
                multi_step=True,
                max_attempts=2,
                toxyz=True,
                top_n_qm=top_n_qm,
                save_log = False
            )

        lib.print_output(f'Conformer Search Done')

        if xyz_file_gaussian:
            return xyz_file_gaussian
        elif xyz_file_xtb_uma:
            return xyz_file_xtb_uma

    @staticmethod
    def resp_chg_fitting(
        work_dir: Path | str,
        xyz_file: str,
        charge: float = 0,
        mult: int = 1,
        function: str = 'b3lyp',
        basis_set: str = '6-311+g(d,p)',
        epsilon: float = 5.0,
        core: int = 32,
        memory: str = '64GB',
        method: str = 'resp2',
    ):
        lib.print_input('RESP Charge Fitting')

        calc_resp_gaussian(
            work_dir,
            xyz_file,
            charge,
            mult,
            function,
            basis_set,
            epsilon,
            core,
            memory,
        )

        df_chg = RESP_fit_Multiwfn(
            work_dir,
            method,
            delta=0.5
        )

        lib.print_output(f'RESP Charge Fitting Done')

        return df_chg


@dataclass
class MDRun:
    work_dir: Path | str
    molecules: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_json(cls, work_dir, json_file):

        json_path = os.path.join(work_dir, json_file)
        with open(json_path, 'r', encoding='utf-8') as file:
            model_info = json.load(file)

        molecules = []
        for key, value in model_info.items():
            name = value["name"]
            number = value["numbers"]
            resname = value["resname"]

            molecule = {
                "name": name,
                "number": number,
                "resname": resname,
            }
            molecules.append(molecule)

        return cls(work_dir, molecules,)


    @staticmethod
    def relax_poly_chain(
        work_dir: Path | str,
        name: str,
        resname: str,
        pdb_file: str,
        temperature: int = 1000,
        gpu: bool = False,
    ):

        return relax_poly_chain(
            work_dir,
            name,
            resname,
            pdb_file,
            temperature,
            gpu,
        )

    @staticmethod
    def relax_poly_chain_from_json(
        work_dir: Path | str,
        json_file: str,
        pdb_file: str,
        temperature: int = 1000,
        gpu: bool = False,
    ):
        work_dir = Path(work_dir)
        json_path = os.path.join(work_dir, json_file)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        poly = data["polymer"]

        return relax_poly_chain(
            work_dir,
            name = poly["name"],
            resname = poly["resname"],
            pdb_file = pdb_file,
            temperature = temperature,
            gpu = gpu,
        )


    @staticmethod
    def annealing(
        work_dir: Path | str,
        molecules: List[Dict[str, Any]],
        temperature: int = 298,
        T_high_increase: int = 300,
        anneal_rate: float = 0.05,
        anneal_npoints: int = 5,
        packmol_pdb: str = "pack_cell.pdb",
        gpu: bool = False,
    ):
        lib.print_input('Molecular Dynamics Simulation')

        annealing(
            work_dir,
            molecules,
            temperature,
            T_high_increase,
            anneal_rate,
            anneal_npoints,
            packmol_pdb,
            gpu,
        )


    @classmethod
    def annealing_from_json(
        cls,
        work_dir: Path | str,
        json_file: str,
        temperature: int = 298,
        T_high_increase: int = 300,
        anneal_rate: float = 0.05,
        anneal_npoints: int = 5,
        packmol_pdb: str = "pack_cell.pdb",
        gpu: bool = False,
    ):

        work_dir = Path(work_dir)
        instance = cls.from_json(work_dir, json_file)

        annealing(
            work_dir,
            instance.molecules,
            temperature,
            T_high_increase,
            anneal_rate,
            anneal_npoints,
            packmol_pdb,
            gpu,
        )

    @staticmethod
    def Tg(
        work_dir: Path | str,
        molecules: List[Dict[str, Any]],
        T_init: int = 500,
        T_final: int = 200,
        delta_T: int = 20,    # K
        eq_time: int = 2,    # ns
        anneal_rate: float = 0.05,
        gpu=False
    ):

        Tg(
            work_dir,
            molecules,
            T_init,
            T_final,
            delta_T,    # K
            eq_time,
            anneal_rate,
            gpu
        )

    @classmethod
    def Tg_from_json(
        cls,
        work_dir: Path | str,
        json_file: str,
        T_init: int = 500,
        T_final: int = 200,
        delta_T: int = 20,
        eq_time: int = 2,
        anneal_rate: float = 0.05,
        gpu=False
    ):

        work_dir = Path(work_dir)
        instance = cls.from_json(work_dir, json_file)

        Tg(
            work_dir,
            instance.molecules,
            T_init,
            T_final,
            delta_T,
            eq_time,
            anneal_rate = anneal_rate,
            gpu = gpu
        )


    @staticmethod
    def production(
        work_dir: Path | str,
        molecules: List[Dict[str, Any]],
        temperature: int = 298,
        nstep_ns: int = 200,   # 200 ns
        gpu=False
    ):

        run_gmx_prod(
            work_dir,
            molecules,
            temperature,
            nstep_ns,
            gpu
        )


    @classmethod
    def production_from_json(
        cls,
        work_dir: Path | str,
        json_file: str,
        temperature: int = 298,
        nstep_ns: int = 200,   # 200 ns
        gpu=False
    ):

        work_dir = Path(work_dir)
        instance = cls.from_json(work_dir, json_file)

        run_gmx_prod(
            instance.work_dir,
            instance.molecules,
            temperature,
            nstep_ns,
            gpu
        )

        lib.print_output(f'Molecular Dynamics Simulation Done')
















