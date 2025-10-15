import os
import subprocess


class PEMDGROMACS:
    def __init__(
        self,
        work_dir,
        molecules,
        temperature,
        gpu,
    ):
        self.work_dir = work_dir
        os.makedirs(self.work_dir, exist_ok=True)
        self.molecules = molecules
        self.temperature = temperature
        self.commands = []
        self.gpu = gpu

        self.compounds = [molecule['name'] for molecule in self.molecules]
        self.resnames = [molecule['resname'] for molecule in self.molecules]
        self.numbers = [molecule['number'] for molecule in self.molecules]

    def gen_top_file(self, top_filename = 'topol.top'):

        self.top_filename = top_filename
        top_filepath = os.path.join(self.work_dir, top_filename)

        file_contents = "; gromcs generation top file\n"
        file_contents += "; Created by PEMD\n\n"

        file_contents += "[ defaults ]\n"
        file_contents += ";nbfunc  comb-rule   gen-pairs   fudgeLJ  fudgeQQ\n"
        file_contents += "1        3           yes         0.5      0.5\n\n"

        file_contents += ";LOAD atomtypes\n"
        file_contents += "[ atomtypes ]\n"

        for com in self.compounds:
            file_contents += f'#include "{com}_nonbonded.itp"\n'
        file_contents += "\n"
        for com in self.compounds:
            file_contents += f'#include "{com}_bonded.itp"\n'
        file_contents += "\n"

        file_contents += "[ system ]\n"
        file_contents += ";name "
        for i in self.compounds:
            file_contents += f"{i}"

        file_contents += "\n\n"

        file_contents += "[ molecules ]\n"
        for res, num in zip(self.resnames, self.numbers):
            file_contents += f"{res} {num}\n"

        file_contents += "\n"

        with open(top_filepath, 'w') as file:
            file.write(file_contents)
        print(f"Top file generation successful：{top_filename}\n")

    def gen_em_mdp_file(self, filename = 'em.mdp'):

        filepath = os.path.join(self.work_dir, filename)

        file_contents = "; em.mdp - used as input into grompp to generate em.tpr\n"
        file_contents += "; Created by PEMD\n\n"

        file_contents += "integrator      = steep\n"
        file_contents += "nsteps          = 50000\n"
        file_contents += "emtol           = 1000.0\n"
        file_contents += "emstep          = 0.01\n\n"

        file_contents += ("; Parameters describing how to find the neighbors of each atom and how to calculate the "
                          "interactions\n")
        file_contents += "nstlist         = 1\n"
        file_contents += "cutoff-scheme   = Verlet\n"
        file_contents += "ns_type         = grid\n"
        file_contents += "rlist           = 1.0\n"
        file_contents += "coulombtype     = PME\n"
        file_contents += "rcoulomb        = 1.0\n"
        file_contents += "rvdw            = 1.0\n"
        file_contents += "pbc             = xyz\n"

        file_contents += "; output control is on\n"
        file_contents += "energygrps      = System\n"

        # write to file
        with open(filepath, 'w') as file:
            file.write(file_contents)
        print(f"Minimization mdp file generation successful：{filename}\n")

    def gen_nvt_mdp_file(self, nsteps_nvt = 200000, filename = 'nvt.mdp', ):

        filepath = os.path.join(self.work_dir, filename)

        file_contents = "; nvt.mdp - used as input into grompp to generate nvt.tpr\n"
        file_contents += "; Created by PEMD\n\n"

        file_contents += "; RUN CONTROL PARAMETERS\n"
        file_contents += "integrator            = md\n"
        file_contents += "dt                    = 0.001 \n"
        file_contents += f"nsteps                = {nsteps_nvt}\n"
        file_contents += "comm-mode             = Linear\n\n"

        file_contents += "; OUTPUT CONTROL OPTIONS\n"
        file_contents += "nstxout               = 0\n"
        file_contents += "nstvout               = 0\n"
        file_contents += "nstfout               = 0\n"
        file_contents += "nstlog                = 5000\n"
        file_contents += "nstenergy             = 5000\n"
        file_contents += "nstxout-compressed    = 5000\n\n"

        file_contents += "; NEIGHBORSEARCHING PARAMETERS\n"
        file_contents += "cutoff-scheme         = verlet\n"
        file_contents += "ns_type               = grid\n"
        file_contents += "nstlist               = 20\n"
        file_contents += "rlist                 = 1.4\n"
        file_contents += "rcoulomb              = 1.4\n"
        file_contents += "rvdw                  = 1.4\n"
        file_contents += "verlet-buffer-tolerance = 0.005\n\n"

        file_contents += "; OPTIONS FOR ELECTROSTATICS AND VDW\n"
        file_contents += "coulombtype           = PME\n"
        file_contents += "vdw_type              = PME\n"
        file_contents += "fourierspacing        = 0.15\n"
        file_contents += "pme_order             = 4\n"
        file_contents += "ewald_rtol            = 1e-05\n\n"

        file_contents += "; OPTIONS FOR WEAK COUPLING ALGORITHMS\n"
        file_contents += "tcoupl                = v-rescale\n"
        file_contents += "tc-grps               = System\n"
        file_contents += "tau_t                 = 1.0\n"
        file_contents += f"ref_t                 = {self.temperature}\n"
        file_contents += "Pcoupl                = no\n"
        file_contents += "Pcoupltype            = isotropic\n"
        file_contents += "tau_p                 = 1.0\n"
        file_contents += "compressibility       = 4.5e-5\n"
        file_contents += "ref_p                 = 1.0\n\n"

        file_contents += "; GENERATE VELOCITIES FOR STARTUP RUN\n"
        file_contents += "gen_vel               = no\n\n"

        file_contents += "; OPTIONS FOR BONDS\n"
        file_contents += "constraints           = hbonds\n"
        file_contents += "constraint_algorithm  = lincs\n"
        file_contents += "unconstrained_start   = no\n"
        file_contents += "shake_tol             = 0.00001\n"
        file_contents += "lincs_order           = 4\n"
        file_contents += "lincs_warnangle       = 30\n"
        file_contents += "morse                 = no\n"
        file_contents += "lincs_iter            = 2\n"

        with open(filepath, 'w') as file:
            file.write(file_contents)
        print(f"NVT mdp file generation successful：{filename}\n")

    def gen_npt_mdp_file(self, nsteps_npt = 5000000, filename = 'npt.mdp', ):

        filepath = os.path.join(self.work_dir, filename)

        file_contents = "; npt.mdp - used as input into grompp to generate npt.tpr\n"
        file_contents += "; Created by PEMD\n\n"

        file_contents += "; RUN CONTROL PARAMETERS\n"
        file_contents += "integrator            = md\n"
        file_contents += "dt                    = 0.001 \n"
        file_contents += f"nsteps                = {nsteps_npt}\n"
        file_contents += "comm-mode             = Linear\n\n"

        file_contents += "; OUTPUT CONTROL OPTIONS\n"
        file_contents += "nstxout               = 0\n"
        file_contents += "nstvout               = 0\n"
        file_contents += "nstfout               = 0\n"
        file_contents += "nstlog                = 5000\n"
        file_contents += "nstenergy             = 5000\n"
        file_contents += "nstxout-compressed    = 5000\n\n"

        file_contents += "; NEIGHBORSEARCHING PARAMETERS\n"
        file_contents += "cutoff-scheme         = verlet\n"
        file_contents += "ns_type               = grid\n"
        file_contents += "nstlist               = 20\n"
        file_contents += "rlist                 = 1.4\n"
        file_contents += "rcoulomb              = 1.4\n"
        file_contents += "rvdw                  = 1.4\n"
        file_contents += "verlet-buffer-tolerance = 0.005\n\n"

        file_contents += "; OPTIONS FOR ELECTROSTATICS AND VDW\n"
        file_contents += "coulombtype           = PME\n"
        file_contents += "vdw_type              = PME\n"
        file_contents += "fourierspacing        = 0.15\n"
        file_contents += "pme_order             = 4\n"
        file_contents += "ewald_rtol            = 1e-05\n\n"

        file_contents += "; OPTIONS FOR WEAK COUPLING ALGORITHMS\n"
        file_contents += "tcoupl                = v-rescale\n"
        file_contents += "tc-grps               = System\n"
        file_contents += "tau_t                 = 1.0\n"
        file_contents += f"ref_t                 = {self.temperature}\n"
        file_contents += "Pcoupl                = Berendsen\n"
        file_contents += "Pcoupltype            = isotropic\n"
        file_contents += "tau_p                 = 1.0\n"
        file_contents += "compressibility       = 4.5e-5\n"
        file_contents += "ref_p                 = 1.0\n\n"

        file_contents += "; GENERATE VELOCITIES FOR STARTUP RUN\n"
        file_contents += "gen_vel               = no\n\n"

        file_contents += "; OPTIONS FOR BONDS\n"
        file_contents += "constraints           = hbonds\n"
        file_contents += "constraint_algorithm  = lincs\n"
        file_contents += "unconstrained_start   = no\n"
        file_contents += "shake_tol             = 0.00001\n"
        file_contents += "lincs_order           = 4\n"
        file_contents += "lincs_warnangle       = 30\n"
        file_contents += "morse                 = no\n"
        file_contents += "lincs_iter            = 2\n"

        with open(filepath, 'w') as file:
            file.write(file_contents)
        print(f"NPT mdp file generation successful：{filename}\n")

    def gen_npt_anneal_mdp_file(self, T_high_increase = 500, anneal_rate = 0.05, anneal_npoints = 5, filename='npt_anneal.mdp'):

        filepath = os.path.join(self.work_dir, filename)

        # Setup annealing
        T_high = self.temperature + T_high_increase
        annealing_time_steps = int((T_high - self.temperature) / anneal_rate)
        nsteps_annealing = (1000 * 2 + 2 * annealing_time_steps) * 1000
        annealing_time = f'0 1000 {1000 + 1 * annealing_time_steps} {1000 + 2 * annealing_time_steps} {1000 * 2 + 2 * annealing_time_steps}'
        annealing_temp = f'{self.temperature} {self.temperature} {T_high} {self.temperature} {self.temperature}'

        file_contents = "; npt_anneal.mdp - used as input into grompp to generate npt_anneal.tpr\n"
        file_contents += "; Created by PEMD\n\n"

        file_contents += "; RUN CONTROL PARAMETERS\n"
        file_contents += "integrator            = md\n"
        file_contents += "dt                    = 0.001 \n"
        file_contents += f"nsteps                = {nsteps_annealing}\n"
        file_contents += "comm-mode             = Linear\n\n"

        file_contents += "; OUTPUT CONTROL OPTIONS\n"
        file_contents += "nstxout               = 0\n"
        file_contents += "nstvout               = 0\n"
        file_contents += "nstfout               = 0\n"
        file_contents += "nstlog                = 5000\n"
        file_contents += "nstenergy             = 5000\n"
        file_contents += "nstxout-compressed    = 5000\n\n"

        file_contents += "; NEIGHBORSEARCHING PARAMETERS\n"
        file_contents += "cutoff-scheme         = verlet\n"
        file_contents += "ns_type               = grid\n"
        file_contents += "nstlist               = 20\n"
        file_contents += "rlist                 = 1.4\n"
        file_contents += "rcoulomb              = 1.4\n"
        file_contents += "rvdw                  = 1.4\n"
        file_contents += "verlet-buffer-tolerance = 0.005\n\n"

        file_contents += "; OPTIONS FOR ELECTROSTATICS AND VDW\n"
        file_contents += "coulombtype           = PME\n"
        file_contents += "vdw_type              = PME\n"
        file_contents += "fourierspacing        = 0.15\n"
        file_contents += "pme_order             = 4\n"
        file_contents += "ewald_rtol            = 1e-05\n\n"

        file_contents += "; OPTIONS FOR WEAK COUPLING ALGORITHMS\n"
        file_contents += "tcoupl                = v-rescale\n"
        file_contents += "tc-grps               = System\n"
        file_contents += "tau_t                 = 1.0\n"
        file_contents += f"ref_t                 = {self.temperature}\n"
        file_contents += "Pcoupl                = Berendsen\n"
        file_contents += "Pcoupltype            = isotropic\n"
        file_contents += "tau_p                 = 1.0\n"
        file_contents += "compressibility       = 4.5e-5\n"
        file_contents += "ref_p                 = 1.0\n\n"

        file_contents += "; Simulated annealing\n"
        file_contents += "annealing             = single\n"
        file_contents += f"annealing-npoints     = {anneal_npoints}\n"
        file_contents += f"annealing-time        = {annealing_time}\n"
        file_contents += f"annealing-temp        = {annealing_temp}\n\n"

        file_contents += "; GENERATE VELOCITIES FOR STARTUP RUN\n"
        file_contents += "gen_vel               = no\n\n"

        file_contents += "; OPTIONS FOR BONDS\n"
        file_contents += "constraints           = hbonds\n"
        file_contents += "constraint_algorithm  = lincs\n"
        file_contents += "unconstrained_start   = no\n"
        file_contents += "shake_tol             = 0.00001\n"
        file_contents += "lincs_order           = 4\n"
        file_contents += "lincs_warnangle       = 30\n"
        file_contents += "morse                 = no\n"
        file_contents += "lincs_iter            = 2\n"

        with open(filepath, 'w') as file:
            file.write(file_contents)
        print(f"NPT anneal mdp file generation successful：{filename}\n")

    def commands_pdbtogro(self, packmol_pdb, *,  box_length: float | None = None,):

        if box_length is None:
            pdb_files = []
            for com in self.compounds:
                filepath = os.path.join(self.work_dir, f"{com}.pdb")
                pdb_files.append(filepath)

            packmol_pdbpath = os.path.join(self.work_dir, packmol_pdb)
            with open(packmol_pdbpath, "r") as f:
                for line in f:
                    if line.startswith("CRYST1"):
                        box_length = float(line[6:15]) / 10.0

        if self.gpu == True:
            self.commands = [
                f"gmx editconf -f {self.work_dir}/{packmol_pdb} -o {self.work_dir}/conf.gro -box {box_length} {box_length} {box_length}",
            ]
        else:
            self.commands = [
                f"gmx_mpi editconf -f {self.work_dir}/{packmol_pdb} -o {self.work_dir}/conf.gro -box {box_length} {box_length} {box_length}",
            ]
        return self

    def commands_grotopdb(self, gro_filename, pdb_filename):

        if self.gpu == True:
            self.commands = [
                f"gmx editconf -f {self.work_dir}/{gro_filename} -o {self.work_dir}/{pdb_filename}",
            ]
        else:
            self.commands = [
                f"gmx_mpi editconf -f {self.work_dir}/{gro_filename} -o {self.work_dir}/{pdb_filename}",
            ]
        return self

    def commands_em(self, input_gro,):

        if self.gpu == True:
            self.commands = [
                f"gmx grompp -f {self.work_dir}/em.mdp -c {self.work_dir}/{input_gro} -p {self.work_dir}/{self.top_filename} -o {self.work_dir}/em.tpr -maxwarn 1",
                f"gmx mdrun -v -deffnm {self.work_dir}/em -ntmpi 1 -ntomp 5",
            ]
        else:
            self.commands = [
                f"gmx_mpi grompp -f {self.work_dir}/em.mdp -c {self.work_dir}/{input_gro} -p {self.work_dir}/{self.top_filename} -o {self.work_dir}/em.tpr -maxwarn 1",
                f"gmx_mpi mdrun -v -deffnm {self.work_dir}/em -ntomp 64",
            ]
        return self

    def commands_nvt(self, input_gro, output_str,):

        if self.gpu == True:
            self.commands = [
                f"gmx grompp -f {self.work_dir}/{output_str}.mdp -c {self.work_dir}/{input_gro} -p {self.work_dir}/{self.top_filename} -o {self.work_dir}/{output_str}.tpr -maxwarn 1",
                f"gmx mdrun -v -deffnm {self.work_dir}/{output_str} -ntmpi 1 -ntomp 5",
            ]
        else:
            self.commands = [
                f"gmx_mpi grompp -f {self.work_dir}/{output_str}.mdp -c {self.work_dir}/{input_gro} -p {self.work_dir}/{self.top_filename} -o {self.work_dir}/{output_str}.tpr -maxwarn 1",
                f"gmx_mpi mdrun -v -deffnm {self.work_dir}/{output_str} -ntomp 64",
            ]
        return self

    def commands_nvt_product(self, input_gro, output_str, ):

        if self.gpu == True:
            self.commands = [
                f"gmx grompp -f {self.work_dir}/{output_str}.mdp -c {self.work_dir}/{input_gro} -p {self.work_dir}/{self.top_filename} -o {self.work_dir}/{output_str}.tpr -maxwarn 1",
                f"gmx mdrun -v -deffnm {self.work_dir}/{output_str} -ntmpi 1 -ntomp 5",
            ]
        else:
            self.commands = [
                f"gmx_mpi grompp -f {self.work_dir}/{output_str}.mdp -c {self.work_dir}/{input_gro} -p {self.work_dir}/{self.top_filename} -o {self.work_dir}/{output_str}.tpr -maxwarn 1",
                f"mpirun gmx_mpi mdrun -v -deffnm {self.work_dir}/{output_str}",
            ]
        return self

    def commands_npt(self, input_gro, output_str, ):

        if self.gpu == True:
            self.commands = [
                f"gmx grompp -f {self.work_dir}/{output_str}.mdp -c {self.work_dir}/{input_gro} -p {self.work_dir}/{self.top_filename} -o {self.work_dir}/{output_str}.tpr -maxwarn 1",
                f"gmx mdrun -v -deffnm {self.work_dir}/{output_str} -ntmpi 1 -ntomp 5",
            ]
        else:
            self.commands = [
                f"gmx_mpi grompp -f {self.work_dir}/{output_str}.mdp -c {self.work_dir}/{input_gro} -p {self.work_dir}/{self.top_filename} -o {self.work_dir}/{output_str}.tpr -maxwarn 1",
                f"gmx_mpi mdrun -v -deffnm {self.work_dir}/{output_str} -ntomp 64",
            ]
        return self

    def commands_npt_anneal(self, input_gro, ):

        if self.gpu == True:
            self.commands = [
                f"gmx grompp -f {self.work_dir}/npt_anneal.mdp -c {self.work_dir}/{input_gro} -p {self.work_dir}/{self.top_filename} -o {self.work_dir}/npt_anneal.tpr -maxwarn 1",
                f"gmx mdrun -v -deffnm {self.work_dir}/npt_anneal -ntmpi 1 -ntomp 5",
            ]
        else:
            self.commands = [
                f"gmx_mpi grompp -f {self.work_dir}/npt_anneal.mdp -c {self.work_dir}/{input_gro} -p {self.work_dir}/{self.top_filename} -o {self.work_dir}/npt_anneal.tpr -maxwarn 1",
                f"gmx_mpi mdrun -v -deffnm {self.work_dir}/npt_anneal -ntomp 64",
            ]
        return self

    def commands_wraptounwrap(self, output_str, ):

        if self.gpu == True:
            self.commands = [
                f'echo 0 | gmx trjconv -s {self.work_dir}/{output_str}.tpr -f {self.work_dir}/{output_str}.xtc -o {self.work_dir}/{output_str}_unwrap.xtc -pbc nojump -ur compact'
            ]
        else:
            self.commands = [
                f'echo 0 | gmx_mpi trjconv -s {self.work_dir}/{output_str}.tpr -f {self.work_dir}/{output_str}.xtc -o {self.work_dir}/{output_str}_unwrap.xtc -pbc nojump -ur compact'
            ]
        return self

    def commands_extract_volume(self, edr_file, output_file='volume.xvg', ):

        if self.gpu == True:
            self.commands = [
                f"echo 21 | gmx energy -f {self.work_dir}/{edr_file} -o {self.work_dir}/{output_file}"
            ]
        else:
            self.commands = [
                f"echo 21 | gmx_mpi energy -f {self.work_dir}/{edr_file} -o {self.work_dir}/{output_file}"
            ]
        return self

    def commands_extract_structure(self, tpr_file, xtc_file, save_gro_file, frame_time, ):

        if self.gpu == True:
            self.commands = [
                f"echo 0 | gmx trjconv -s {self.work_dir}/{tpr_file} -f {self.work_dir}/{xtc_file} -o {self.work_dir}/{save_gro_file} -dump {frame_time} -quiet"
            ]
        else:
            self.commands = [
                f"echo 0 | gmx_mpi trjconv -s {self.work_dir}/{tpr_file} -f {self.work_dir}/{xtc_file} -o {self.work_dir}/{save_gro_file} -dump {frame_time} -quiet"
            ]
        return self

    def run_local(self, commands=None):
        os.makedirs(self.work_dir, exist_ok=True)

        if commands is None:
            commands = self.commands

        for cmd in commands:
            try:
                print(f"Executing command: {cmd}")
                subprocess.run(cmd, shell=True, check=True, text=True, capture_output=True)
                print(f"Command executed successfully: {cmd}\n")
            except subprocess.CalledProcessError as e:
                print(f"Error executing command: {cmd}\n{e.stderr}")
                break

    # def gen_slurm(self, script_name, job_name, nodes, ntasks_per_node, partition):
    #     slurm_script = PEMDSlurm(
    #         self.work_dir,
    #         script_name,
    #     )
    #
    #     # Add each command in self.commands to the SLURM script
    #     for cmd in self.commands:
    #         slurm_script.add_command(cmd)
    #
    #     # Generate the SLURM script with the accumulated commands
    #     script_path = slurm_script.generate_script(
    #         job_name=job_name,
    #         nodes=nodes,
    #         ntasks_per_node=ntasks_per_node,
    #         partition=partition,
    #     )
    #
    #     print(f"SLURM script generated successfully: {script_path}")
    #     return script_path
