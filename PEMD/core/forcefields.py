# Copyright (c) 2024. PEMD developers. All rights reserved.
# Distributed under the terms of the MIT License.

# ******************************************************************************
# core.forcefields Module
# ******************************************************************************


import json
import pandas as pd

from rdkit import Chem
from pathlib import Path
from dataclasses import dataclass
import PEMD.core.output_lib as lib


from PEMD.forcefields.ff import (
    get_oplsaa_xml,
    get_xml_ligpargen,
    get_oplsaa_ligpargen,
    gen_ff_from_data
)

from PEMD.forcefields.ff import (
    apply_chg_to_poly,
    apply_chg_to_molecule
)

from PEMD.model.build import (
    gen_copolymer_3D,
    mol_to_pdb,
)


@dataclass
class Forcefield:
    work_dir: Path | str
    name: str
    resname: str
    scale: float
    charge: float
    repeating_unit: str | None = None
    leftcap: str | None = None
    rightcap: str | None = None
    length_short: int = 0
    length_long: int = 0
    smiles: str | None = None
    terminal_cap: str | None = None

    @classmethod
    def from_json(cls, work_dir, json_file, mol_type="polymer"):
        work_dir = Path(work_dir)
        json_path = work_dir / json_file

        try:
            with open(json_path, 'r', encoding='utf-8') as file:
                model_info = json.load(file)
        except FileNotFoundError:
            print(f"Error: JSON file {json_file} not found in {work_dir}.")
            return None
        except json.JSONDecodeError:
            print(f"Error: JSON file {json_file} is not a valid JSON.")
            return None

        data = model_info.get(mol_type)
        if data is None:
            raise ValueError(
                f"'{mol_type}' section not found in '{json_file}'."
            )

        name = data.get("name", "")
        resname = data.get("resname", "")
        scale = data.get("scale")
        charge = data.get("charge")

        if mol_type == "polymer":
            repeating_unit = data.get("repeating_unit", "")
            leftcap = data.get("left_cap", "")
            rightcap = data.get("right_cap", "")
            length = data.get("length", [0, 0])
            length_short = length[0] if len(length) > 0 else 0
            length_long = length[1] if len(length) > 1 else 0
            return cls(
                work_dir,
                name,
                resname,
                scale,
                charge,
                repeating_unit,
                leftcap,
                rightcap,
                length_short,
                length_long,
            )

        smiles = data.get("smiles", "")
        return cls(
            work_dir,
            name,
            resname,
            scale,
            charge,
            smiles=smiles
        )


    @staticmethod
    def oplsaa(
        work_dir: Path,
        name: str = "PE",
        ff_source: str = "ligpargen",
        *,
        resname: str = "MOL",
        resp_csv: str | None = None,
        resp_df: pd.DataFrame | None = None,
        polymer: bool = False,
        length_short: int = 3,
        smiles: str | None = None,
        left_cap: str | None = None,
        right_cap: str | None = None,
        end_repeating: int = 1,
        scale: float = 1.0,
        charge: float = 0,
        mol_long: Chem.Mol | None = None,
    ):
        lib.print_input('OPLS-AA Force Field Generation')

        if ff_source == "ligpargen":
            # chg_df = None
            if polymer:
                mol_short = gen_copolymer_3D(
                    smiles_A=smiles,
                    smiles_B=smiles,
                    name=name,
                    mode="homopolymer",
                    length=length_short,
                    left_cap_smiles=left_cap,
                    right_cap_smiles=right_cap
                )

                mol_to_pdb(
                    work_dir=work_dir,
                    mol=mol_short,
                    name=name,
                    resname=resname,
                    pdb_filename=f"{name}_short.pdb",
                )

                chg_df = get_xml_ligpargen(
                    work_dir,
                    name,
                    resname,
                    pdb_file=f"{name}_short.pdb",
                    charge=charge,
                    charge_model='CM1A-LBCC',
                )

                if resp_csv:
                    chg_df = pd.read_csv(resp_csv)
                    chg_df.insert(0, 'position', chg_df.index)
                elif resp_df is not None:
                    chg_df = resp_df.copy()
                    chg_df.insert(0, 'position', chg_df.index)

                mol_to_pdb(
                    work_dir=work_dir,
                    mol=mol_long,
                    name=name,
                    resname=resname,
                    pdb_filename=f"{name}_long.pdb",
                )

                bonditp_filename = get_oplsaa_xml(
                    work_dir,
                    name,
                    pdb_file = f"{name}_long.pdb",
                )

                apply_chg_to_poly(
                    work_dir,
                    mol_short,
                    mol_long,
                    itp_file=bonditp_filename,
                    resp_chg_df=chg_df,
                    repeating_unit=smiles,
                    end_repeating=end_repeating,
                    scale=scale,
                    charge=charge,
                )

            else:
                bonditp_filename = get_oplsaa_ligpargen(
                    work_dir,
                    name,
                    resname,
                    charge,
                    smiles,
                    charge_model='CM1A-LBCC',
                )

                if resp_csv:
                    chg_df = pd.read_csv(resp_csv)
                    apply_chg_to_molecule(
                        work_dir,
                        itp_file=bonditp_filename,
                        chg_df=chg_df,
                        scale= scale,
                        charge=charge,
                    )

                elif resp_df is not None:
                    apply_chg_to_molecule(
                        work_dir,
                        itp_file=bonditp_filename,
                        chg_df=resp_df,
                        scale=scale,
                        charge=charge,
                    )

        elif ff_source == "database":
            gen_ff_from_data(
                work_dir,
                name,
                scale,
                charge,
            )

        lib.print_output(f'OPLS-AA Force Field Generation')


    @classmethod
    def oplsaa_from_json(
            cls,
            work_dir: Path | str,
            json_file: str,
            mol_type: str,
            *,
            ff_source: str = "ligpargen",
            resp_csv: str | None = None,
            resp_df: pd.DataFrame | None = None,
            mol_long: Chem.Mol | None = None,
            end_repeating: int = 1,
    ):
        instance = cls.from_json(work_dir, json_file, mol_type, )

        if mol_type == "polymer":
            polymer = True
        else:
            polymer = False

        cls.oplsaa(
            work_dir=work_dir,
            name=instance.name,
            ff_source=ff_source,
            resname=instance.resname,
            resp_csv=resp_csv,
            resp_df=resp_df,
            polymer=polymer,
            length_short=instance.length_short,
            left_cap=instance.leftcap,
            right_cap=instance.rightcap,
            scale=instance.scale,
            charge=instance.charge,
            smiles=instance.repeating_unit if polymer else instance.smiles,
            end_repeating=end_repeating,
            mol_long=mol_long,
        )





