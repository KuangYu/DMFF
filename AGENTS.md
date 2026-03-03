# DMFF Architecture Overview for Agents

This document summarizes the architecture and workflows of the DMFF codebase to help agents and developers navigate the project for development, refactoring, testing, and debugging.

## 1. Project Overview

- **Purpose**
  - DMFF (Differentiable Molecular Force Field) is a JAX-based Python package for differentiable molecular force fields, focusing on molecular systems such as water, biomolecules, polymers, and small organic molecules (`README.md:5-9`).
  - It combines OpenMM-based topology/parameter handling with JAX/XLA-backed energy and gradient evaluation to support parameter optimization, hybrid ML/force-field models, and trajectory-based fitting (`README.md:7-10`, `docs/dev_guide/introduction.md:11-19`).

- **High-Level Architecture**
  - **Python package `dmff/`**
    - Public API surface is re-exported in `dmff/__init__.py:1-7` (settings, neighbor lists, generators, `Hamiltonian`, operators, MD tools).
    - Global numerical configuration (precision, JIT, debug) lives in `dmff/settings.py:1-19` and is applied at import time via `jax.config.update`.
  - **API / Frontend layer (`dmff/api/`)**
    - `Hamiltonian` (`dmff/api/hamiltonian.py:38-137`) loads one or more OpenMM-compatible XML force-field files via `XMLIO`, builds an internal `ParamSet`, and instantiates registered force generators for each `<Force>`.
    - `DMFFTopology` (`dmff/api/topology.py:41-418`) wraps OpenMM `Topology`, RDKit molecules, or SDF files into a unified topology object with bonds, residues, virtual sites, and periodic box vectors.
    - `DMFFTopology.buildCovMat` and `buildVSiteUpdateFunction` (`dmff/api/topology.py:391-519`) precompute covalent maps and JAX functions that update virtual-site coordinates prior to energy evaluation.
    - `Hamiltonian.createPotential` returns a `Potential` object (`dmff/api/hamiltonian.py:38-71, 104-128`) that aggregates per-term JAX energy functions and exposes `getPotentialFunc` to obtain a total-energy callable over positions, box, neighbor pairs, and parameters.
  - **Force-field generators and calculators**
    - Individual force families live in dedicated subpackages: `dmff/admp`, `dmff/classical`, `dmff/sgnn`, `dmff/eann`, `dmff/generators`, and `dmff/operators` (`README.md:51-60`, `docs/dev_guide/convention.md:17-22`).
    - Generators map XML-defined forces into parameter arrays and JAX-pure energy kernels ("calculators"), following the specs described in the developer guide (`docs/dev_guide/introduction.md:11-19`).
  - **Neighbor list and common utilities**
    - High-level neighbor lists are defined in `dmff/common/nblist.py:1-230`, with implementations backed by freud (`NeighborListFreud`) and an optional dpnblist backend (`NeighborListDp`).
    - Neighbor lists produce padded pair arrays augmented with a covalent-neighborhood tag derived from the covalent map (`dmff/common/nblist.py:31-35, 91-94, 152-155`).
    - A pure-Python neighbor list without freud/dpnblist support is provided via `NoCutoffNeighborList` and `NoPeriodicNeighborList` (`dmff/common/nblist.py:145-230`).
  - **Differentiable MD and optimization**
    - `dmff/difftraj.py:1-242` defines `Loss_Generator`, which wraps a user-defined observable `f_nout` and an energy function into a reversible velocity-Verlet integrator. It exposes a custom-JVP loss function whose gradients are propagated through MD trajectories using adjoint sensitivity.
    - Additional optimization and analysis utilities live in `dmff/optimize.py` and `dmff/mbar.py` (referenced by the user guide `docs/user_guide/4.6MBAR.md`, `docs/user_guide/4.6Optimization.md`).
  - **C++ / CUDA neighbor-list backend (`dmff/dpnblist/`)**
    - Implements a reusable neighbor-list library with cell, octree, and hash algorithms on CPU and optionally CUDA (`dmff/dpnblist/README.md:1-12`).
    - Built as a pybind11 extension module `dpnblist` via CMake (`dmff/dpnblist/CMakeLists.txt:1-48`), and consumed from Python in `dmff/common/nblist.py:19-27`.
  - **OpenMM-DMFF plugin backend (`backend/openmm_dmff_plugin/`)**
    - Provides an OpenMM `Force` that wraps a TensorFlow-exported DMFF model (`backend/openmm_dmff_plugin/README.md:1-6`).
    - Includes platform-specific C++/CUDA implementations under `backend/openmm_dmff_plugin/platforms/` and a Python package `OpenMMDMFFPlugin` for higher-level usage and tests.
  - **Documentation, examples, and tests**
    - Markdown documentation and mkdocs configuration live in `docs/` and `mkdocs.yml:1-47` (site nav, mkdocstrings, and math support).
    - End-to-end and tutorial examples are under `examples/` and referenced from `README.md:48-60` and user guide notebooks.
    - Python unit and integration tests live in `tests/` organized by module (`README.md:51`, `Makefile:1-31`). C++ tests for `dpnblist` and the OpenMM plugin live under `dmff/dpnblist/tests/` and `backend/openmm_dmff_plugin/*/tests/` respectively.

## 2. Build & Commands

### 2.1 Python package and environment

- **Environment creation and core dependencies** (from `docs/user_guide/2.installation.md:3-33`)
  - Create and activate a conda environment:
    - `conda create -n dmff python=3.9 --yes`
    - `conda activate dmff`
  - Install JAX (choose CPU or GPU wheel):
    - CPU: `pip install "jax[cpu]==0.4.14"`
    - GPU: `pip install "jax[cuda11_local]==0.4.14" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`
  - Install MD and optimization dependencies:
    - `conda install -c conda-forge mdtraj==1.9.7`
    - `pip install optax==0.1.3 jaxopt==0.8.1 pymbar==4.0.1` (exact versions vary across docs but are pinned in `docs/user_guide/2.installation.md:15-25`).
  - Install OpenMM and RDKit:
    - `conda install -c conda-forge openmm==7.7.0`
    - `conda install -c conda-forge rdkit`

- **Install DMFF from source** (`docs/user_guide/2.installation.md:35-40`)
  - Clone and install:
    - `git clone https://github.com/deepmodeling/DMFF.git`
    - `cd DMFF`
    - `pip install . --user`
  - The same `pip install .` command is used when developing locally (also reflected in `setup.py:34-58`).

- **Python test commands** (`Makefile:1-31`)
  - The top-level `test` target runs all Python tests:
    - `make test` (aggregates module-specific targets).
  - Module-scoped targets (all use `pytest --disable-warnings`):
    - `make test_admp` → `pytest --disable-warnings tests/test_admp`
    - `make test_classical` → `pytest --disable-warnings tests/test_classical`
    - `make test_common` → `pytest --disable-warnings tests/test_common`
    - `make test_difftraj` → `pytest --disable-warnings tests/test_difftraj`
    - `make test_dimer` → `pytest --disable-warnings tests/test_dimer`
    - `make test_frontend` → `pytest --disable-warnings tests/test_frontend`
    - `make test_mbar` → `pytest --disable-warnings tests/test_mbar`
    - `make test_sgnn` → `pytest --disable-warnings tests/test_sgnn`
    - `make test_energy` → `pytest --disable-warnings tests/test_energy.py`
    - `make test_utils` → `pytest --disable-warnings tests/test_utils.py`

- **Quick installation verification** (`docs/user_guide/2.installation.md:42-55`)
  - Import checks in Python:
    - `import dmff`
    - `import dmff.admp`
  - Run an example, e.g.:
    - `cd examples/water_fullpol`
    - `python run.py`

### 2.2 dpnblist neighbor-list backend

- **Build and install** (`dmff/dpnblist/README.md:4-9`)
  - From `dmff/dpnblist/`:
    - `pip install .`
  - This invokes CMake (`dmff/dpnblist/CMakeLists.txt:1-48`) to build the `dpnblist` pybind11 extension with optional CUDA support and installs it into the active Python environment.

- **Key build characteristics** (`dmff/dpnblist/CMakeLists.txt:9-65`)
  - If CUDA is available, GPU kernels are compiled (`hashSchAlgGPU.cu`, `octreeSchAlgGPU.cu`, `cellSchAlgGPU.cu`).
  - Otherwise, a CPU-only configuration is used (`pywrap_CPU.cpp`, `nbList_CPU.cpp`).
  - OpenMP and Python3 development headers are required and linked into the module.

### 2.3 OpenMM–DMFF plugin backend

- **Environment and dependencies** (`backend/openmm_dmff_plugin/README.md:9-37`)
  - Create a separate conda environment for the plugin:
    - `mkdir omm_dmff_working_dir && cd omm_dmff_working_dir`
    - `conda create -n dmff_omm -c conda-forge python=3.9 openmm cudatoolkit=11.6`
    - `conda activate dmff_omm`
  - Install TensorFlow C++ runtime and headers:
    - `conda install -y libtensorflow_cc=2.9.1 -c conda-forge`
    - Download TensorFlow sources (`v2.9.1`), then copy the `tensorflow/c` headers into `${CONDA_PREFIX}/include/tensorflow/`.
  - Install `cppflow` headers and apply the local patch `backend/openmm_dmff_plugin/tests/cppflow_empty_constructor.patch`.

- **Build and install the plugin** (`backend/openmm_dmff_plugin/README.md:41-56`)
  - Set environment variables and configure/build via CMake:
    - `export OPENMM_INSTALLED_DIR=$CONDA_PREFIX`
    - `export CPPFLOW_INSTALLED_DIR=$CONDA_PREFIX`
    - `export LIBTENSORFLOW_INSTALLED_DIR=$CONDA_PREFIX`
    - `cd DMFF/backend/openmm_dmff_plugin`
    - `mkdir build && cd build`
    - `cmake .. -DOPENMM_DIR=${OPENMM_INSTALLED_DIR} -DCPPFLOW_DIR=${CPPFLOW_INSTALLED_DIR} -DTENSORFLOW_DIR=${LIBTENSORFLOW_INSTALLED_DIR}`
    - `make && make install`
    - `make PythonInstall`

- **Plugin tests** (`backend/openmm_dmff_plugin/README.md:58-62`)
  - From the same environment, run:
    - `python -m OpenMMDMFFPlugin.tests.test_dmff_plugin_nve -n 100`
    - `python -m OpenMMDMFFPlugin.tests.test_dmff_plugin_nvt -n 100 --platform CUDA`

## 3. Code Style

- **Project layout** (`docs/dev_guide/convention.md:10-22`, `README.md:44-57`)
  - `dmff/`: main source tree (APIs, generators, operators, models, MD tools).
  - `docs/`: Markdown-based documentation and mkdocs config.
  - `examples/`: runnable, self-contained examples and notebooks.
  - `tests/`: unit and integration tests organized by feature area.
  - Under `dmff/`, each subpackage corresponds to a potential form or subsystem (e.g., `admp`, `classical`, `sgnn`, `eann`).

- **Docstrings and type hints** (`docs/dev_guide/convention.md:24-27`) 
  - Python docstrings follow **NumPy-style** conventions to integrate with Sphinx/napoleon and mkdocstrings.
  - Public APIs should include type annotations; docstrings document parameters, returns, raises, and examples in NumPy style.

- **API design patterns**
  - Calculators are expected to be **pure JAX functions** that take `(positions, box, pairs, params)` (plus optional aux data) and return energies; `Hamiltonian` wraps these into higher-level `Potential` objects (`dmff/api/hamiltonian.py:38-71, 114-127`).
  - Topology handling is centralized in `DMFFTopology`, which exposes methods such as `buildCovMat`, `buildVSiteUpdateFunction`, `addVSiteToPos`, and equivalent-atom detection utilities (`dmff/api/topology.py:391-519, 521-769`).

- **Testing and documentation expectations** (`docs/dev_guide/introduction.md:15-21`)
  - New force-field modules and calculators are expected to ship with unit tests and documentation describing the underlying theory and user interface.
  - The developer guide references a checklist before PR covering tests, formatting, and comments.

## 4. Testing

- **Python test layout**
  - Top-level tests live in `tests/`, grouped by module (e.g., `tests/test_admp/`, `tests/test_classical/`, `tests/test_common/`, `tests/test_difftraj/`, `tests/test_dimer/`, `tests/test_frontend/`, `tests/test_mbar/`, `tests/test_sgnn/`, and focused tests like `tests/test_energy.py`, `tests/test_utils.py`).
  - `tests/conftest.py` is present for shared pytest configuration.

- **Running Python tests**
  - Use the Makefile targets listed in **2.1** for full or module-specific suites (`Makefile:1-31`).
  - Under the hood, all targets call `pytest --disable-warnings` on the relevant test packages.

- **C++ / CUDA tests for dpnblist** (`dmff/dpnblist/CMakeLists.txt:67-76`)
  - When `CMAKE_BUILD_TYPE` matches `Debug`, CMake also builds a `dpnblist` library target and adds the `dmff/dpnblist/tests/` subtree to the build via `add_subdirectory(tests)`, enabling C++-level tests for the neighbor-list algorithms.

- **OpenMM plugin tests**
  - The plugin is validated by the Python tests described in **2.3**, which exercise both reference and CUDA platforms (`backend/openmm_dmff_plugin/README.md:58-62`).

- **Installation smoke tests** (`docs/user_guide/2.installation.md:42-55`)
  - Quick checks: module imports and example scripts under `examples/` (for example, `examples/water_fullpol/run.py`) to ensure DMFF and its backends are wired correctly.

## 5. Security

- **Scope and threat model**
  - DMFF primarily operates on scientific data (force-field XML files, PDB/SDF structures, trajectories) and does not include explicit authentication, authorization, or network access layers in the core package.
  - The repository does not define a DMFF-specific security policy; vendored dependencies like pybind11 carry their own security policy under `dmff/dpnblist/external/pybind11-2.11.1/SECURITY.md:1-13`.

- **Input handling**
  - Topology and parameter inputs are loaded from XML, PDB, and SDF/SMILES via OpenMM and RDKit (`dmff/api/topology.py:70-96, 130-158` and `dmff/api/hamiltonian.py:78-90`). Invalid or inconsistent inputs can raise exceptions during sanitization or molecule regularization (`dmff/api/topology.py:218-229, 348-389`).
  - Neighbor-list backends rely on freud or dpnblist; large systems and aggressive cutoffs can lead to very large neighbor lists and associated memory/compute costs (`dmff/common/nblist.py:96-123, 157-170, 204-223`).

- **Native extensions and external runtimes**
  - The dpnblist extension (`dmff/dpnblist`) and the OpenMM–DMFF plugin (`backend/openmm_dmff_plugin`) compile native code that runs in-process with Python, TensorFlow, and OpenMM.
  - When upgrading these components or their dependencies (CUDA, TensorFlow, OpenMM, pybind11), refer to upstream security advisories and version policies; this is especially relevant for deployments on shared HPC resources.

## 6. Configuration

- **Global numerical settings** (`dmff/settings.py:1-19`)
  - `PRECISION`: string, currently `'double'`; `update_jax_precision` maps this to `jax_enable_x64` at import time.
  - `DO_JIT`: boolean flag controlling whether JAX JIT compilation is used by higher-level code (mentioned in `docs/user_guide/2.installation.md:55-55` as affecting initial run-time due to compilation).
  - `DEBUG`: boolean, available for debug-related code paths; its use is module-specific.
  - These symbols are exported via `__all__` and re-exported from `dmff/__init__.py:1-7` so they can be adjusted from user code before heavy JAX tracing.

- **Neighbor-list backend selection** (`dmff/common/nblist.py:4-17, 81-140`)
  - If `freud` is installed, `NeighborListFreud` is available; if `dpnblist` is installed, `NeighborListDp` is available. Otherwise, the code falls back to pure-Python neighbor lists and emits warnings (`dmff/common/nblist.py:5-17`).
  - `NeighborList` is currently defined as an alias for the freud-based implementation (`dmff/common/nblist.py:81-142`).
  - Cutoffs (`rcut`) and padding behavior are configured per neighbor-list instance.

- **Topology and chemistry configuration** (`dmff/api/topology.py:41-61, 259-263`)
  - `DMFFTopology` can be constructed from OpenMM `Topology`, SDF files, or RDKit molecules, with optional residue names and formal charges.
  - Atom-level metadata (e.g., `FormalCharge`) is stored on atoms and reused when converting to RDKit molecules, enabling SMARTS-based pattern matching and atom-typing.

- **freud UI configuration** (`config/freud.ini:2-41`)
  - A standalone INI file configures layout, key bindings, DB filename, JSON indentation, sorting, and style for tools based on freud; it is not imported by the core DMFF package but may be used by auxiliary tooling.

- **OpenMM–DMFF plugin environment** (`backend/openmm_dmff_plugin/README.md:41-56`)
  - The plugin build is configured via environment variables pointing to the OpenMM, cppflow, and TensorFlow prefix directories:
    - `OPENMM_INSTALLED_DIR`
    - `CPPFLOW_INSTALLED_DIR`
    - `LIBTENSORFLOW_INSTALLED_DIR`
  - These control include/library discovery during the CMake configuration of the plugin.

