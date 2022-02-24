# CS6216 Project: Pyro Implementation of Nonparametric Hamiltonian Monte Carlo Method

This repo serves as a record of a course project. It contains a forked Pyro repo
as the submodule, and a package to facilitate testing.

## Setup

It is recommended to do this in a virtual environment (assuming you are using bash):

```bash
python -m venv ./my-venv && source ./my-venv/activate
```

Then install the forked pyro and the utility package in develop mode:

```bash
pip install -e pyro[extras]
pip install -e cs6216-nphmc-utilities
```

With `-e` flag, you don't have to re-install after making changes in the
installed packages.

`cs6216-nphmc-utilities` provides a command line interface `ndu` (NP-HMC
development utilities). Run `ndu --help` for usage.
