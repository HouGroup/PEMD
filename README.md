# Polymer Electrolyte Modeling and Discovery (PEMD)

Polymer Electrolyte Modeling and Discovery (PEMD) is a Python toolkit for building, simulating, and analyzing polymer-electrolyte systems. The package wraps together quantum-chemistry (QM), molecular-dynamics (MD), and post-processing workflows so that you can go from a SMILES description of a repeat unit all the way to transport-property analysis with a single, reproducible input file

> **Status:** research code under active development. Interfaces and data formats may evolve between releases.

## Table of Contents
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
- [Workflows](#workflows)
- [Analysis Toolkit](#analysis-toolkit)
- [Development](#development)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## Key Features
- **Polymer builders** – Generate homo- and co-polymer structures directly from JSON configuration files using `PEMD.core.model.PEMDModel`. Automated Packmol integration produces amorphous boxes with user-defined composition and density.
- **Force-field automation** – Build OPLS-AA force fields from LigParGen, RESP-fitted, or curated database parameters via `PEMD.core.forcefields.Forcefield`. Partial charges can be replaced with RESP charges obtained from PEMD's QM pipeline.
- **Quantum-chemistry workflows** – Perform conformer searches, XTB refinement, Gaussian single-point or optimization jobs, and RESP charge fitting through the `PEMD.core.run.QMRun` helpers and companion shell scripts.
- **MD production runs** – Launch annealing and production simulations (GROMACS) with consistent inputs through `PEMD.core.run.MDRun` using only the JSON specification of system composition.
- **High-throughput orchestration** – Reusable workflow templates (see `workflow/`) chain together structure building, parameterization, QM, and MD steps for large screening campaigns.
- **Analysis suite** – Property analysis modules (e.g., conductivity, diffusion, residence time, glass-transition temperature) under `PEMD.analysis` accelerate post-processing of simulation trajectories.

## Repository Structure
```
PEMD_DEV/
├── PEMD/                 # Core Python package (modeling, simulation, analysis)
├── workflow/             # Ready-to-run workflow templates and sample inputs
├── bin/                  # Helper shell utilities (e.g., RESP charge automation)
├── environment.yml       # Conda environment for full-featured installations
├── setup.py              # Package metadata for editable/production installs
└── README.md             # Project overview and instructions
```


## Getting Started

### Prerequisites
- Operating system: Linux (tested) or macOS. Windows users are encouraged to work within WSL2.
- [Conda](https://docs.conda.io/en/latest/miniconda.html) or [Mamba](https://mamba.readthedocs.io) for environment management.
- External simulation engines and quantum-chemistry tools (GROMACS, Gaussian, Multiwfn, XTB, Packmol) available in your runtime environment if you plan to execute the full workflows.

### Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-org>/PEMD.git
   cd PEMD
   ```
2. **Create the recommended environment**
   ```bash
   conda env create -f environment.yml
   conda activate pemd-test
   ```
3. **Install PEMD in editable mode**
   ```bash
   pip install -e .
   ```
   Editable installs make it easy to track custom modifications while leveraging the packaged entry points.

### Quick Start
Below is a minimal Python example that reproduces the molecular-dynamics pipeline shipped in `workflow/md.py`. It expects a PEMD-style JSON input (see [`workflow/md.json`](workflow/md.json)) describing the polymer chain, salts, and ion counts.

```python
from pathlib import Path
import shutil

from PEMD.core.model import PEMDModel
from PEMD.core.forcefields import Forcefield
from PEMD.core.run import MDRun

work_dir = Path("./demo_md")
work_dir.mkdir(exist_ok=True)

# Copy the example JSON or replace it with your own specification
shutil.copy("workflow/md.json", work_dir / "md.json")
json_file = "md.json"

# 1) Build polymer chains (short + long for charge fitting / production)
pdb_short, pdb_long = PEMDModel.homopolymer_from_json(work_dir, json_file)

# 2) Generate force-field files (LigParGen by default)
Forcefield.oplsaa_from_json(
    work_dir,
    json_file,
    mol_type="polymer",
    ff_source="ligpargen",
    pdb_file=pdb_long,
)
Forcefield.oplsaa_from_json(work_dir, json_file, mol_type="Li_cation", ff_source="database")
Forcefield.oplsaa_from_json(work_dir, json_file, mol_type="salt_anion", ff_source="database")

# 3) Pack the amorphous simulation box and run MD
PEMDModel.amorphous_cell_from_json(
    work_dir,
    json_file,
    density=0.8,
    add_length=25,
    packinp_name="pack.inp",
    packpdb_name="pack_cell.pdb",
)
MDRun.annealing_from_json(
    work_dir,
    json_file,
    temperature=298,
    T_high_increase=300,
    anneal_rate=0.05,
    anneal_npoints=5,
    packmol_pdb="pack_cell.pdb",
)
MDRun.production_from_json(work_dir, json_file, temperature=298, nstep_ns=200)
```

After the run completes, simulation outputs (topologies, trajectories, logs) reside under the working directory and can be analyzed with the `PEMD.analysis` modules.

## Workflows
The `workflow/` directory provides end-to-end templates that can be executed directly or adapted to your project:

| Script | Purpose |
| ------ | ------- |
| `workflow/md.py` | Full MD pipeline: polymer build → force field → annealing → production.
| `workflow/md_withRESP.py` | MD pipeline including RESP charge derivation from QM calculations.
| `workflow/esw.py` | Compute electrochemical stability windows.
| `workflow/frontier_orbitals.py` | Analyze frontier orbitals for small molecules or polymer fragments.

Each script assumes the presence of a PEMD-compatible JSON file (see `workflow/md.json`) and the necessary external executables in `PATH`. Treat them as well-documented starting points for your own automation.

## Analysis Toolkit
Modules under `PEMD.analysis` implement commonly used observables for polymer electrolytes:

- `conductivity.py`, `transfer_number.py` – Ionic conductivity and transport number calculations.
- `msd.py`, `polymer_ion_dynamics.py`, `residence_time.py` – Mean-squared displacement and ion residence time metrics.
- `coordination.py`, `energy.py`, `tg.py` – Coordination statistics, energetic analyses, and glass-transition estimates.

These utilities consume standard MD trajectories (e.g., XTC, DCD) and topology files. Consult in-code docstrings for details about expected input formats.

## Development
- **Formatting & linting:** please adhere to [PEP 8](https://peps.python.org/pep-0008/) conventions. Configure your editor to use black (line length 88) and isort if contributing substantial changes.
- **Testing:** add or update unit tests alongside new features whenever possible. Run `pytest` (and any custom workflow regression tests) before submitting a pull request.
- **Contributing:** fork the repository, create feature branches, and submit pull requests describing your changes. Bug reports and feature requests are welcome via the issue tracker.

## Citation
If PEMD contributes to your research, please cite:

> S. Tan, T. Hou*, et al., *PEMD: An open-source framework for high-throughput simulation and analysis of polymer electrolytes*, 2025.

(An updated citation with DOI will be provided upon publication.)

## Contact
For questions, feature requests, or collaboration opportunities, please contact the PEMD development team at [tsd23@mails.tsinghua.edu.cn].






