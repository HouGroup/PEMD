# Copyright (c) 2024. PEMD developers. All rights reserved.
# Distributed under the terms of the MIT License.

# ******************************************************************************
# core.analysis module
# ******************************************************************************


import os
import math
import pandas as pd
import numpy as np
import MDAnalysis as mda
import PEMD.core.output_lib as lib

from tqdm.auto import tqdm
from cclib.io import ccopen
from functools import partial
from functools import lru_cache
from multiprocessing import Pool
from typing import Optional, Dict

from PEMD.analysis.conductivity import calc_cond_msd, calc_conductivity
from PEMD.analysis.transfer_number import calc_transfer_number
from PEMD.analysis.msd import (
    calc_slope_msd,
    create_position_arrays,
    calc_Lii_self,
    calc_Lii,
    calc_self_diffusion_coeff,
    plot_msd
)
from PEMD.analysis.polymer_ion_dynamics import (
    process_traj,
    calc_tau3,
    calc_delta_n_square,
    calc_tau1,
    ms_endtoend_distance,
    fit_rouse_model,
    calc_msd_M2
)
from PEMD.analysis.coordination import (
    calc_rdf_coord,
    obtain_rdf_coord,
    plot_rdf_coordination,
    num_of_neighbor,
    pdb2mol,
    get_cluster_index,
    find_poly_match_subindex,
    get_cluster_withcap,
    merge_xyz_files,
    analyze_coordination_structure,
    calc_population_parallel
)
from PEMD.analysis.residence_time import (
    calc_neigh_corr,
    fit_residence_time
)
from PEMD.analysis.energy import esw, homo_lumo_energy


class PEMDAnalysis:

    def __init__(
        self,
        run_wrap: mda.Universe,
        run_unwrap: mda.Universe,
        run_start: int,
        run_end: int,
        dt: float,
        dt_collection: int,
        temperature: float,
        select_dict: Optional[Dict[str, str]] = None,
        cation_name: str = "cation",
        anion_name: str = "anion",
        polymer_name: str = "polymer",
    ):

        self.run_wrap = run_wrap
        self.run_unwrap = run_unwrap
        self.run_start = run_start
        self.run_end = run_end
        self.dt = dt
        self.dt_collection = dt_collection
        self.temp = temperature
        self.times = self.times_range(self.run_end)
        self.select_dict = select_dict or {}
        self.cation_name = cation_name
        self.anion_name = anion_name
        self.polymer_name = polymer_name
        self.cations_unwrap = run_unwrap.select_atoms(self.select_dict.get(cation_name))
        self.anions_unwrap = run_unwrap.select_atoms(self.select_dict.get(anion_name))
        self.polymers_unwrap = run_unwrap.select_atoms(self.select_dict.get(polymer_name))
        self.volume = self.run_unwrap.coord.volume
        self.box_size = self.run_unwrap.dimensions[0]
        self.num_cation = len(self.cations_unwrap)
        self.num_o_polymer = len(self.polymers_unwrap)
        self.num_chain = len(np.unique(self.polymers_unwrap.resids))
        self.num_o_chain = int(self.num_o_polymer // self.num_chain)


    @classmethod
    def from_gromacs(
        cls,
        work_dir: str,
        tpr_file: str,
        wrap_xtc_file: str,
        unwrap_xtc_file: str,
        *,
        select_dict: Optional[Dict[str, str]] = None,
        cation_name: str = "cation",
        anion_name: str = "anion",
        polymer_name: str = "polymer",
        run_start: int = 0,
        run_end: int = -1,
        dt: float = 0.0,
        dt_collection: int = 1,
        temperature: float = 300.0,
    ):

        tpr_path = os.path.join(work_dir, tpr_file)
        wrap_xtc_path = os.path.join(work_dir, wrap_xtc_file)
        unwrap_xtc_path = os.path.join(work_dir, unwrap_xtc_file)

        run_wrap = mda.Universe(tpr_path, wrap_xtc_path)
        run_unwrap = mda.Universe(tpr_path, unwrap_xtc_path)

        return cls(
            run_wrap=run_wrap,
            run_unwrap=run_unwrap,
            run_start=run_start,
            run_end=run_end,
            dt=dt,
            dt_collection=dt_collection,
            temperature=temperature,
            select_dict=select_dict,
            cation_name=cation_name,
            anion_name=anion_name,
            polymer_name=polymer_name,
        )


    def times_range(self, end_time):

        t_total = end_time - self.run_start
        return np.arange(
            0,
            t_total * self.dt * self.dt_collection,
            self.dt * self.dt_collection,
            dtype=float,
        )


    def get_cond_array(self):

        return calc_cond_msd(
            self.run_unwrap,
            self.cations_unwrap,
            self.anions_unwrap,
            self.run_start,
        )


    def get_slope_msd(self, msd_array, interval_time=10000, step_size=10):

        slope, time_range = calc_slope_msd(
            self.times,
            msd_array,
            self.dt_collection,
            self.dt,
            interval_time,
            step_size
        )
        return slope, time_range


    @lru_cache(maxsize=128)
    def conductivity(self, save_csv=False, plot=False, ):

        msd_array = self.get_cond_array()
        slope, time_ranges = self.get_slope_msd(msd_array)

        if save_csv:
            df = pd.DataFrame({
                "time": self.times,
                "msd": msd_array
            })
            df.to_csv('msd.csv', index=False)

        if plot:
            plot_msd(
                msd_array,
                self.times,
                self.dt_collection,
                self.dt,
                time_ranges=time_ranges,
            )

        return calc_conductivity(
            slope,
            self.volume,
            self.temp
        )


    @lru_cache(maxsize=128)
    def get_ions_positions_array(self, mols_unwrap=None):

        mols_positions = create_position_arrays(
            self.run_unwrap,
            mols_unwrap,
            self.times,
            self.run_start,
        )
        return mols_positions


    def get_Lii_self_array(self, atom_positions):

        n_atoms = np.shape(atom_positions)[1]
        return calc_Lii_self(atom_positions, self.times) / n_atoms


    # calculate the self diffusion coefficient
    def diffusion_coefficient(self, atoms, save_csv=False, plot=False, ):

        atoms_unwrap = self.run_unwrap.select_atoms(atoms)
        atoms_positions = self.get_ions_positions_array(atoms_unwrap)

        msd_array = self.get_Lii_self_array(atoms_positions)
        slope, time_ranges = self.get_slope_msd(msd_array)

        if save_csv:
            df = pd.DataFrame({
                "time": self.times,
                "msd": msd_array
            })
            df.to_csv('msd.csv', index=False)

        if plot:
            plot_msd(
                msd_array,
                self.times,
                self.dt_collection,
                self.dt,
                time_ranges=time_ranges,
            )

        D = calc_self_diffusion_coeff(slope)

        return D


    # calculate the transfer number
    def transfer_number(self):

        cations_positions = self.get_ions_positions_array(self.cations_unwrap)
        anions_positions = self.get_ions_positions_array(self.anions_unwrap)

        slope_plusplus, time_range_plusplus = self.get_slope_msd(calc_Lii(cations_positions))
        slope_minusminus, time_range_minusminus = self.get_slope_msd(calc_Lii(anions_positions))

        return calc_transfer_number(
            slope_plusplus,
            slope_minusminus,
            self.temp,
            self.volume,
            self.conductivity()
        )


    def get_rdf_coordination_array(self, group1_name, group2_name):

        group1 = self.run_wrap.select_atoms(group1_name)
        group2 = self.run_wrap.select_atoms(group2_name)
        bins, rdf, coord_number = calc_rdf_coord(
            group1,
            group2,
            self.volume
        )
        return bins, rdf, coord_number


    def coordination_number(self, group1_name, group2_name, save_csv=False, plot=False):

        bins, rdf, coord_number = self.get_rdf_coordination_array(group1_name, group2_name)

        if save_csv:
            df = pd.DataFrame({
                "bins": bins,
                "rdf": rdf,
                "coordination_number": coord_number
            })
            df.to_csv('coord.csv', index=False)

        if plot:
            plot_rdf_coordination(
                bins,
                rdf,
                coord_number,
            )

        y_coord = obtain_rdf_coord(bins, rdf, coord_number)[1]

        return y_coord


    @lru_cache(maxsize=128)
    def get_cutoff_radius(self, group1_name, group2_name):

        bins, rdf, coord_number = self.get_rdf_coordination_array(group1_name, group2_name)
        x_val = obtain_rdf_coord(bins, rdf, coord_number)[0]

        return x_val


    @lru_cache(maxsize=128)
    def get_poly_array(self):

        return process_traj(
            self.run_unwrap,
            self.times,
            self.run_start,
            self.run_end,
            self.num_cation,
            self.num_o_polymer,
            self.get_cutoff_radius(self.cation_name, self.polymer_name),
            self.cations_unwrap,
            self.polymers_unwrap,
            self.box_size,
            self.num_chain,
            self.num_o_chain
        )


    # calculate the tau3
    @lru_cache(maxsize=128)
    def tau3(self):

        return calc_tau3(
            self.dt,
            self.dt_collection,
            self.num_cation,
            self.run_start,
            self.run_end,
            self.get_poly_array()[1]
        )


    def get_delta_n_square_array(self, time_window,):

        # msd = []
        poly_o_n = self.get_poly_array()[0]
        poly_n = self.get_poly_array()[1]

        partial_calc_delta_n_square = partial(calc_delta_n_square, poly_o_n=poly_o_n, poly_n=poly_n,
                                              run_start=self.run_start, run_end=self.run_end)

        with Pool() as pool:
            msd = list(tqdm(pool.imap(partial_calc_delta_n_square, range(time_window)), total=time_window))
        return np.array(msd)


    # calculate the tau1
    def tau1(self, time_window, save_csv=False, plot=False):

        times = self.times_range(time_window)
        msd_array = self.get_delta_n_square_array(time_window)

        if save_csv:
            df = pd.DataFrame({
                "time": times,
                "msd": msd_array
            })
            df.to_csv('msd.csv', index=False)

        if plot:
            plot_msd(
                msd_array,
                times,
                self.dt_collection,
                self.dt,
            )

        return calc_tau1(
            self.tau3(),
            times,
            msd_array,
            self.num_o_chain
        )


    @lru_cache(maxsize=128)
    def get_ms_endtoend_distance_array(self):

        return ms_endtoend_distance(
            self.run_unwrap,
            self.num_chain,
            self.polymers_unwrap,
            self.box_size,
            self.run_start,
            self.run_end,
        )


    def get_oe_msd_array(self):

        return self.get_Lii_self_array(
            self.get_poly_array()[3]
        )


    # calculate the tauR
    def tauR(self, save_csv=False, plot=False):

        msd_array = self.get_oe_msd_array()

        if save_csv:
            df = pd.DataFrame({
                "time": self.times,
                "msd": msd_array
            })
            df.to_csv('msd.csv', index=False)

        if plot:
            plot_msd(
                msd_array,
                self.times,
                self.dt_collection,
                self.dt,
            )

        return fit_rouse_model(
            self.get_ms_endtoend_distance_array(),
            self.times,
            msd_array,
            self.num_o_chain
        )


    def get_msd_M2_array(self, time_window):

        poly_o_n = self.get_poly_array()[0]
        bound_o_n = self.get_poly_array()[2]
        poly_o_positions = self.get_poly_array()[3]

        partial_calc_msd_M2 = partial(calc_msd_M2, poly_o_positions=poly_o_positions, poly_o_n=poly_o_n,
                                      bound_o_n=bound_o_n, run_start=self.run_start, run_end=self.run_end)

        with Pool() as pool:
            msd = list(tqdm(pool.imap(partial_calc_msd_M2, range(time_window)), total=time_window))
        return np.array(msd)


    # calculate the tau2
    def tau2(self, time_window, save_csv=False, plot=False):

        times = self.times_range(time_window)
        msd_array = self.get_msd_M2_array(time_window)

        if save_csv:
            df = pd.DataFrame({
                "time": times,
                "msd": msd_array
            })
            df.to_csv('msd.csv', index=False)

        if plot:
            plot_msd(
                msd_array,
                times,
                self.dt_collection,
                self.dt,
            )

        return fit_rouse_model(
            self.get_ms_endtoend_distance_array(),
            times,
            msd_array,
            self.num_o_chain
        )

    def calc_neigh_corr(self, distance_dict, center_atom: str = "cation",):

        return calc_neigh_corr(
            self.run_wrap,
            distance_dict,
            self.select_dict,
            self.run_start,
            self.run_end,
            center_atom=center_atom,
        )

    def fit_residence_time(self, acf_avg, cutoff_time, save_csv=False, plot=False):

        return fit_residence_time(
            self.times,
            acf_avg,
            cutoff_time,
            self.dt*self.dt_collection,
            save_csv,
            plot
        )


    def coordination_type(self, run_start, run_end, distance_dict, center_atom = "cation", counter_atom = "anion", plot = False):

        distance = distance_dict.get(counter_atom)

        return analyze_coordination_structure(
            self.run_wrap,
            run_start,
            run_end,
            self.select_dict,
            distance,
            center_atom,
            counter_atom,
            plot
        )


    def ion_cluster_population(self, run_start, run_end, center_atom = "cation", counter_atom = "anion", core = 4, plot = False):

        select_cations = self.select_dict.get(center_atom)
        select_anions = self.select_dict.get(counter_atom)

        calc_population_parallel(
            self.run_wrap,
            run_start,
            run_end,
            select_cations,
            select_anions,
            core,
            plot,
        )


    @staticmethod
    def write_cluster(
        work_dir: str,
        tpr_file: str,
        wrap_xtc_file: str,
        center_atom_name: str,
        distance_dict: dict[str, float],
        select_dict: dict[str, str],
        run_start: int = 0,
        run_end: int = 80000,
        structure_code: int = 1,
        max_number: int = 100,
        write: bool = True,
        write_freq: float = 0.01,
    ):

        tpr_path = os.path.join(work_dir, tpr_file)
        wrap_xtc_path = os.path.join(work_dir, wrap_xtc_file)
        nvt_run = mda.Universe(tpr_path, wrap_xtc_path)

        num_of_neighbor(
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
        )


    @staticmethod
    def write_cluster_polymer(
        work_dir: str,
        tpr_file: str,
        wrap_xtc_file: str,
        center_atom_name: str,
        distance_dict: dict[str, float],
        select_dict: dict[str, str],
        poly_name: str,
        repeating_unit: str,
        length: int = 3,
        run_start: int = 0,
        run_end: int = 80000,
        structure_code: int = 1,
        max_number: int = 100,
        write_freq: float = 0.01,
    ):
        lib.print_input("Extract Cluster Structures")

        PEMDAnalysis.write_cluster(
            work_dir,
            tpr_file,
            wrap_xtc_file,
            center_atom_name,
            distance_dict,
            select_dict,
            run_start,
            run_end,
            structure_code,
            max_number,
            write = True,
            write_freq = write_freq,
        )

        cluster_dir = os.path.join(work_dir, "cluster_dir")
        pdb_files = sorted([f for f in os.listdir(cluster_dir) if f.endswith(".pdb")])

        success, failed = 0, 0
        for i, pdb_file in enumerate(pdb_files):

            try:
                mol = pdb2mol(
                    cluster_dir,
                    pdb_file
                )

                (
                    c_idx,
                    center_atom_idx,
                    selected_atom_idx,
                    other_atom_idx,
                ) = get_cluster_index(
                    mol,
                    center_atom_name,
                    select_dict,
                    distance_dict,
                    poly_name,
                    repeating_unit,
                    length,
                )

                (
                    match_list,
                    start_atom,
                    end_atom,
                    mol
                ) = find_poly_match_subindex(
                    poly_name,
                    repeating_unit,
                    length,
                    mol,
                    selected_atom_idx,
                    c_idx,
                )

                outxyz_file = f'num_{i}_frag.xyz'
                get_cluster_withcap(
                    cluster_dir,
                    mol,
                    match_list,
                    center_atom_idx,
                    other_atom_idx,
                    start_atom,
                    end_atom,
                    outxyz_file
                )
                success += 1

            except Exception as e:
                # print(f"Error processing file {pdb_file}: {e}")
                continue

        lib.print_output(f'Extracted {success} / {max_number} cluster structures')

        return merge_xyz_files(cluster_dir)

    @staticmethod
    def esw(
        work_dir: str,
        logfile_init: str,
        *,
        logfile_oxid: str | None = None,
        logfile_red: str | None = None,
        output: str = 'eox',            # 'eox' | 'ered' | 'all'
        unit: str = 'hartree',
        errors: list[str] | None = None
    ):
        HARTREE_PER_EV = 1.0 / 27.211386245988

        def _safe_read(path: str, label: str):
            try:
                data = ccopen(path).parse()
                G = getattr(data, "freeenergy", None)
                if G is not None:
                    return float(G)
                scf = getattr(data, "scfenergies", None)
                if scf:
                    return float(scf[-1]) * HARTREE_PER_EV
                raise ValueError("no freeenergy or scfenergies found")
            except Exception as e:
                print("calculate ESW error:", e)
                if errors is not None:
                    errors.append(f"[ESW]{label}: {path} -> {e}")
                return None

        logfilepath_init = os.path.join(work_dir, logfile_init)
        G_init = _safe_read(logfilepath_init, "init")
        if G_init is None:
            return (math.nan, math.nan, math.nan) if output == 'all' else math.nan

        logfilepath_oxid = os.path.join(work_dir, logfile_oxid) if logfile_oxid else None
        logfilepath_red  = os.path.join(work_dir, logfile_red)  if logfile_red  else None
        G_oxid = _safe_read(logfilepath_oxid, "oxid") if logfile_oxid else None
        G_red  = _safe_read(logfilepath_red,  "red")  if logfile_red  else None

        if output == 'eox':
            return esw(G_init, G_oxid=G_oxid, G_red=None, output='eox', unit=unit) if G_oxid is not None else math.nan
        if output == 'ered':
            return esw(G_init, G_oxid=None, G_red=G_red, output='ered', unit=unit) if G_red is not None else math.nan
        if output == 'all':
            eox  = esw(G_init, G_oxid=G_oxid, G_red=None, output='eox', unit=unit) if G_oxid is not None else math.nan
            ered = esw(G_init, G_oxid=None, G_red=G_red, output='ered', unit=unit) if G_red  is not None else math.nan
            w    = esw(G_init, G_oxid=G_oxid, G_red=G_red, output='esw', unit=unit)  if (G_oxid is not None and G_red is not None) else math.nan
            return eox, ered, w

        raise ValueError("output only supports 'eox', 'ered', or 'all'")


    @staticmethod
    def homo_lumo_energy(work_dir, log_filename):

        return homo_lumo_energy(
            work_dir,
            log_filename
        )









