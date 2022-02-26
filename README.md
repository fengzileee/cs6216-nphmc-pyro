# CS6216 Project: Pyro Implementation of Nonparametric Hamiltonian Monte Carlo Method

This repo serves as a record of a course project. It contains a forked Pyro repo
as the submodule, and some scripts to facilitate development.

## Setup

Remember to update the submodule:

``` bash
git submodule update --init --recursive
```

It is recommended to do the setup in a virtual environment (assuming you are using bash):

```bash
python -m venv ./my-venv && source ./my-venv/activate
```

Then install the forked pyro and the utility package in develop mode:

```bash
pip install -e pyro[extras]
```

With `-e` flag, you don't have to re-install pyro after making changes in the
forked repo.

Install dependencies:

``` bash
pip install -r requirements.txt
```

## Random walk

The python module `walk.py` contains scripts for different purposes. To get help strings, run

``` bash
./walk.py --help  # if walk.py is executable
python walk.py --help
```

### Groundtruth from importance sampling and systematic resampling

Run enough repetitions of `./walk.py imp` to collect draws from importance sampling, then use
the draws to do systematic resampling `./walk.py gt`.

### MCMC methods

Do `./walk.py run nuts -r 1`. See `./walk.py run --help` for help. 

### Generate the plot

Do `./walk.py plot`.
