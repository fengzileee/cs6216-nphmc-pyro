#!/usr/bin/env python
"""A commandline interface for the random walk example.

Mostly adapted from https://github.com/fzaiser/nonparametric-hmc
"""
import pickle
import typing as T
from pathlib import Path

import typer
import torch
import pyro
import pyro.infer.mcmc as pyromcmc  # type: ignore

app = typer.Typer()


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@app.command()
def run(
    method: str,
    count: int = typer.Option(1000, "-c", help="Number of samples to draw."),
    distance_limit: float = typer.Option(
        10, "-b", help="Distance limit of random walk."
    ),
    eps: float = typer.Option(0.1, "-e", help="Step size in leapfrog integration."),
    num_steps: int = typer.Option(
        50, "-l", help="Numer of steps in leapfrog integration."
    ),
    output_dir: str = typer.Option("./samples_walk", "-d", help="Data directory."),
    output_file: T.Optional[str] = typer.Option(None, "-f", help="Data file name."),
):
    def walk_model():
        start = pyro.sample("start", pyro.distributions.Uniform(0, 3))
        t = 0
        position = start
        distance = torch.tensor(0.0)
        while position > 0 and position < distance_limit:
            step = pyro.sample(f"step_{t}", pyro.distributions.Uniform(-1, 1))
            distance = distance + torch.abs(step)
            position = position + step
            t = t + 1
        pyro.sample("obs", pyro.distributions.Normal(1.1, 1.0), obs=distance)
        return start.item()

    output_dir = Path(output_dir).expanduser().resolve() / method
    if method == "nuts":
        info = ""
        kernel = pyromcmc.NUTS(walk_model)
    elif method == "hmc":
        info = f"_{eps}_{num_steps}"
        kernel = pyromcmc.HMC(
            walk_model,
            step_size=eps,
            num_steps=num_steps,
            adapt_step_size=False,
        )
    elif method == "nphmc":
        raise NotImplementedError(f"{method} not implemented!")
    else:
        raise RuntimeError(f"Unknown method {method}")

    mcmc = pyromcmc.MCMC(kernel, num_samples=count, warmup_steps=count // 10)
    mcmc.run()

    samples = mcmc.get_samples()
    raw_samples = [value.item() for value in samples["start"]]

    output_dir.mkdir(parents=True, exist_ok=True)
    if output_file is None:
        output_file = f"walk_{count}{info}.pickle"
    with open(output_dir / output_file, "wb") as f:
        pickle.dump(raw_samples, f)
    mcmc.summary()


def main():
    app()


if __name__ == "__main__":
    main()
