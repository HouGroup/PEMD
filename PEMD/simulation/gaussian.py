# Copyright (c) 2024. PEMD developers. All rights reserved.
# Distributed under the terms of the MIT License.

# ******************************************************************************
# Module Docstring
# ******************************************************************************

import os
import subprocess
import logging
from PEMD.simulation import sim_lib

class PEMDGaussian:
    def __init__(
        self,
        work_dir,
        filename,
        core=32,
        mem='64GB',
        chg=0,
        mult=1,
        function='B3LYP',
        basis_set='6-31+g(d,p)',
        epsilon=None,
        optimize=True,
        multi_step=False,  # Decide whether to perform multi-step calculations
        max_attempts=3,    # Maximum number of attempts
    ):
        self.work_dir = work_dir
        self.filename = filename
        self.core = core
        self.mem = mem
        self.chg = chg
        self.mult = mult
        self.function = function
        self.basis_set = basis_set
        self.epsilon = epsilon
        self.optimize = optimize
        self.multi_step = multi_step
        self.max_attempts = max_attempts

        os.makedirs(self.work_dir, exist_ok=True)

        self.logger = logging.getLogger(f"PEMDGaussian_{filename}")
        if not self.logger.handlers:
            handler = logging.FileHandler(os.path.join(self.work_dir, f'{filename}.log'))
            formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def generate_input_file(self, structure=None, chk=None, gaucontinue=False, additional_options=None, use_cartesian=False):
        if not structure or 'atoms' not in structure:
            self.logger.error("Invalid structure provided. 'atoms' key is missing.")
            raise ValueError("Invalid structure provided. 'atoms' key is missing.")

        prefix, _ = os.path.splitext(self.filename)

        file_contents = f"%nprocshared={self.core}\n"
        file_contents += f"%mem={self.mem}\n"

        if chk:
            file_contents += f"%chk={os.path.join(self.work_dir, prefix + '_opt.chk')}\n"

        use_pcm = self.epsilon is not None

        if self.optimize:

            tight_flag = "tight" if self.function.upper() == "M062X" else None

            opt_params = []
            if tight_flag:
                opt_params.append(tight_flag)
            if use_cartesian:
                opt_params.append("cartesian")

            if gaucontinue:
                opt_params += ["maxstep=2", "notrust", "maxcyc=100", "gdiis"]
            else:
                maxcyc = 40 if tight_flag is None else 100
                opt_params.append(f"maxcyc={maxcyc}")

            route_section = (
                f"# opt({','.join(opt_params)}) freq "
                f"{self.function} {self.basis_set}"
            )
            if use_pcm:
                route_section += " scrf=(pcm,solvent=generic,read)"
            route_section += " nosymm"
        else:
            route_section = (
                f"# {self.function} {self.basis_set} "
                f"scrf=(pcm,solvent=generic,read)"
            )

        # Automatically add D3BJ empirical dispersion when using B3LYP
        if 'B3LYP' in self.function.upper():
            if 'EmpiricalDispersion' not in route_section:
                route_section += ' EmpiricalDispersion=GD3BJ'

        if additional_options:
            route_section += f" {additional_options}"

        file_contents += route_section + "\n\n"

        file_contents += 'qm calculation\n\n'
        file_contents += f'{self.chg} {self.mult}\n'  # Charge and multiplicity

        for atom in structure['atoms']:
            atom_parts = atom.split()
            if len(atom_parts) >= 4:
                try:
                    x = float(atom_parts[1])
                    y = float(atom_parts[2])
                    z = float(atom_parts[3])
                    file_contents += f"{atom_parts[0]:<2} {x:>15.6f} {y:>15.6f} {z:>15.6f}\n"
                except ValueError:
                    self.logger.warning(f"Invalid coordinate values in atom line: {atom}")
                    continue
            else:
                self.logger.warning(f"Invalid atom line: {atom}")
                continue

        file_contents += '\n'
        if use_pcm:
            file_contents += f"eps={self.epsilon}\n"
            file_contents += "epsinf=2.1\n"
            file_contents += '\n\n'
        else:
            file_contents += '\n'

        file_path = os.path.join(self.work_dir, self.filename)
        try:
            with open(file_path, 'w') as file:
                file.write(file_contents)
            self.logger.info(f"Generated Gaussian input file: {file_path}")
        except IOError as e:
            self.logger.error(f"Failed to write Gaussian input file: {file_path}. Error: {e}")
            raise e

    def generate_input_file_resp(self, structure):
        if not structure or 'atoms' not in structure:
            self.logger.error("Invalid structure provided for RESP calculation. 'atoms' key is missing.")
            raise ValueError("Invalid structure provided for RESP calculation. 'atoms' key is missing.")

        file_prefix, _ = os.path.splitext(self.filename)
        file_contents = f"%nprocshared={self.core}\n"
        file_contents += f"%mem={self.mem}\n"
        file_contents += f"%chk={os.path.join(self.work_dir, 'resp_' + file_prefix + '.chk')}\n"
        file_contents += f"# opt B3LYP TZVP em=GD3BJ scrf=(pcm,solvent=generic,read)\n\n"
        file_contents += 'resp calculation\n\n'
        file_contents += f'{self.chg} {self.mult}\n'

        for atom in structure['atoms']:
            atom_parts = atom.split()
            if len(atom_parts) >= 4:
                try:
                    x = float(atom_parts[1])
                    y = float(atom_parts[2])
                    z = float(atom_parts[3])
                    file_contents += f"{atom_parts[0]:<2} {x:>15.6f} {y:>15.6f} {z:>15.6f}\n"
                except ValueError:
                    self.logger.warning(f"Invalid coordinate values in atom line: {atom}")
                    continue
            else:
                self.logger.warning(f"Invalid atom line: {atom}")
                continue

        file_contents += '\n'
        file_contents += f"eps={self.epsilon}\n"
        file_contents += '\n\n'

        file_contents += "--link1--\n"
        file_contents += f"nprocshared={self.core}\n"
        file_contents += f"%mem={self.mem}\n"
        file_contents += f"%oldchk={os.path.join(self.work_dir, 'resp_' + file_prefix + '.chk')}\n"
        file_contents += f"%chk={os.path.join(self.work_dir, 'SP_gas_' + file_prefix + '.chk')}\n"
        file_contents += f"# {self.function} {self.basis_set} em=GD3BJ geom=allcheck\n\n"

        file_contents += "--link1--\n"
        file_contents += f"nprocshared={self.core}\n"
        file_contents += f"%mem={self.mem}\n"
        file_contents += f"%oldchk={os.path.join(self.work_dir, 'resp_' + file_prefix + '.chk')}\n"
        file_contents += f"%chk={os.path.join(self.work_dir, 'SP_solv_' + file_prefix + '.chk')}\n"
        file_contents += f"# {self.function} {self.basis_set} em=GD3BJ scrf=(pcm,solvent=generic,read) geom=allcheck\n\n"
        file_contents += f"eps={self.epsilon}\n"
        file_contents += '\n\n'

        file_path = os.path.join(self.work_dir, self.filename)
        try:
            with open(file_path, 'w') as file:
                file.write(file_contents)
            self.logger.info(f"Generated RESP Gaussian input file: {file_path}")
        except IOError as e:
            self.logger.error(f"Failed to write RESP Gaussian input file: {file_path}. Error: {e}")
            raise e

    def execute_gaussian(self, log_file_path, command):
        self.logger.info(f"Executing Gaussian command: {' '.join(command)}")
        try:
            with open(log_file_path, 'w') as log_file:
                subprocess.run(
                    command,
                    check=True,
                    text=True,
                    stdout=log_file,
                    stderr=subprocess.STDOUT  # Redirect stderr to stdout
                )
            self.logger.info(f"Gaussian calculation completed successfully. Log file: {log_file_path}")
        except FileNotFoundError:
            self.logger.error(f"Gaussian executable not found. Command attempted: {' '.join(command)}")
            raise
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Gaussian calculation failed with return code {e.returncode}. See log file: {log_file_path}")
            raise e
        except Exception as e:
            self.logger.error(f"An unexpected error occurred while executing Gaussian: {e}")
            raise e

    def parse_log_for_errors(self, log_file_path):
        error_codes = []
        specific_errors = False
        specific_error_messages = [
            "Error in internal coordinate system",
            "Linear angle in Bend",
            "Linear angle in Tors",
            "FormBX had a problem."
        ]
        try:
            with open(log_file_path, 'r') as log_file_read:
                for line in log_file_read:
                    if "Error termination" in line:
                        if 'l9999.exe' in line.lower():
                            error_codes.append('l9999')
                        if 'l502.exe' in line.lower():
                            error_codes.append('l502')
                        if 'l103.exe' in line.lower():
                            error_codes.append('l103')
                    for msg in specific_error_messages:
                        if msg in line:
                            specific_errors = True
        except FileNotFoundError:
            self.logger.error(f"Log file {log_file_path} not found.")
        self.logger.info(f"Detected error codes: {error_codes}, specific errors: {specific_errors}")
        return error_codes, specific_errors

    def run_local(self, structure, resp=False, chk=None):
        prefix, _ = os.path.splitext(self.filename)
        log_file_path = os.path.join(self.work_dir, f"{prefix}.log")
        input_file = os.path.join(self.work_dir, self.filename)

        if resp:
            self.logger.info("Generating RESP Gaussian input file.")
            try:
                self.generate_input_file_resp(structure=structure)
            except Exception as e:
                self.logger.error(f"Failed to generate RESP input file for {self.filename}: {e}")
                return 'failed', f'{prefix}.log'

            command = ["g16", input_file]  # Use command list
            try:
                self.execute_gaussian(log_file_path, command)
                self.logger.info("RESP Gaussian calculation completed successfully.")
                return 'success', f'{prefix}.log'
            except subprocess.CalledProcessError:
                self.logger.error("RESP Gaussian calculation failed.")
                try:
                    with open(log_file_path, 'r') as log_file_read:
                        error_lines = [line.strip() for line in log_file_read if "Error termination" in line]
                    self.logger.error("Gaussian RESP error output:")
                    for line in error_lines:
                        self.logger.error(line)
                except FileNotFoundError:
                    self.logger.error(f"Log file {log_file_path} not found.")
                return 'failed', f'{prefix}.log'
        else:
            self.logger.info("Generating initial Gaussian input file for standard calculation.")
            try:
                self.generate_input_file(structure=structure, chk=chk, gaucontinue=False, additional_options=None)
            except Exception as e:
                self.logger.error(f"Failed to generate input file for {self.filename}: {e}")
                return 'failed', f'{prefix}.log'

            attempt = 0
            if not self.multi_step:
                attempts = 1
            else:
                attempts = self.max_attempts

            use_cartesian_flag = False  # Flag to track if cartesian has been added

            while attempt < attempts:
                command = ["g16", input_file]  # Use command list
                self.logger.info(f"Running Gaussian calculation (Attempt {attempt + 1}): {' '.join(command)}")

                try:
                    self.execute_gaussian(log_file_path, command)
                    self.logger.info("Gaussian calculation completed successfully.")
                    return 'success', f'{prefix}.log'
                except subprocess.CalledProcessError:
                    self.logger.error(f"Error executing Gaussian on attempt {attempt + 1}.")

                    try:
                        atoms = sim_lib.read_final_structure_from_gaussian(log_file_path)
                        if not atoms:
                            self.logger.error(f"No atoms read from log file {log_file_path}. Cannot continue optimization.")
                            return 'failed', f'{prefix}.log'
                        structure_current = {"atoms": atoms}
                        self.logger.info(f"Updated structure for next attempt: {structure_current}")
                    except Exception as e:
                        self.logger.error(f"Error reading final structure from log file {log_file_path}: {e}")
                        return 'failed', f'{prefix}.log'

                    error_codes, specific_errors = self.parse_log_for_errors(log_file_path)

                    if self.multi_step:
                        if 'l9999' in error_codes:
                            self.logger.info("Detected l9999 error. Modifying input for continuation.")
                            try:
                                self.generate_input_file(structure=structure_current, chk=chk, gaucontinue=True, additional_options=None)
                            except Exception as e:
                                self.logger.error(f"Failed to modify input file for {self.filename}: {e}")
                                return 'failed', f'{prefix}.log'
                        # elif 'l502' in error_codes:
                        #     self.logger.info("Detected l502 error. Adding scf=xqc to the route section.")
                        #     try:
                        #         self.generate_input_file(structure=structure_current, chk=chk, gaucontinue=False, additional_options=" scf=xqc")
                        #     except Exception as e:
                        #         self.logger.error(f"Failed to add scf=xqc to input file for {self.filename}: {e}")
                        #         return 'failed', f'{prefix}.log'
                        elif 'l103' in error_codes and specific_errors and not use_cartesian_flag:
                            self.logger.info("Detected l103 error with specific messages. Adding cartesian to opt.")
                            try:
                                self.generate_input_file(
                                    structure=structure_current,
                                    chk=chk,
                                    gaucontinue=False,
                                    additional_options=None,
                                    use_cartesian=True
                                )
                                use_cartesian_flag = True  # Ensure cartesian is added only once
                            except Exception as e:
                                self.logger.error(f"Failed to modify input file with cartesian for {self.filename}: {e}")
                                return 'failed', f'{prefix}.log'
                        else:
                            self.logger.error("Unrecognized or non-recoverable error encountered.")
                            return 'failed', f'{prefix}.log'
                    else:
                        self.logger.error("Multi-step calculation is disabled. Calculation failed.")
                        return 'failed', f'{prefix}.log'

                    attempt += 1

            self.logger.error("Maximum number of attempts reached. Calculation failed.")
            return 'failed', f'{prefix}.log'



