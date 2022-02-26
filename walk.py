#!/usr/bin/env python
"""A commandline interface for the random walk example.

Mostly adapted from https://github.com/fzaiser/nonparametric-hmc
"""
import pickle
import glob
import time
import math
from pathlib import Path

import typer

app = typer.Typer()

import random
import string


def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = "".join(random.choice(letters) for i in range(length))
    return result_str


def walk_model():
    import pyro
    import torch

    start = pyro.sample("start", pyro.distributions.Uniform(0, 3))
    t = 0
    position = start
    distance = torch.tensor(0.0)
    while position > 0 and position < 10:
        step = pyro.sample(f"step_{t}", pyro.distributions.Uniform(-1, 1))
        distance = distance + torch.abs(step)
        position = position + step
        t = t + 1
    pyro.sample("obs", pyro.distributions.Normal(1.1, 1.0), obs=distance)
    return start.item()


@app.command()
def run(
    method: str,
    count: int = typer.Option(1000, "-c", help="Number of samples to draw."),
    eps: float = typer.Option(0.1, "-e", help="Step size in leapfrog integration."),
    num_steps: int = typer.Option(
        50, "-l", help="Numer of steps in leapfrog integration."
    ),
    output_dir: str = typer.Option("./samples_walk", "-d", help="Data directory."),
    rep: int = typer.Option(1, "-r", help="Number of repetitions of sampling"),
):
    """Generate samples using different MCMC methods."""
    import pyro.infer.mcmc as pyromcmc  # type: ignore

    output_dir = Path(output_dir).expanduser().resolve() / method

    for _ in range(rep):
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
        random_str = get_random_string(6)
        output_file = f"walk_{random_str}_{count}{info}.pickle"
        with open(output_dir / output_file, "wb") as f:
            pickle.dump(raw_samples, f)
        mcmc.summary()


def systematic_resampling(log_weights, values):
    import torch

    mx = max(log_weights)
    weight_sum = sum(math.exp(log_weight - mx) for log_weight in log_weights)
    u_n = torch.distributions.Uniform(0, 1).sample().item()
    sum_acc = 0.0
    resamples = []
    for (log_weight, value) in zip(log_weights, values):
        weight = math.exp(log_weight - mx) * len(values) / weight_sum
        sum_acc += weight
        while u_n < sum_acc:
            u_n += 1
            resamples.append(value)
    return resamples


@app.command()
def imp(
    output_dir: str = typer.Option(
        "./importance_draws_walk/",
        "-d",
        help="Path to store the importance sampling draws.",
    ),
    rep: int = typer.Option(1, "-r", help="Number of repetitions."),
):
    """Repeatedly draw 10,000 samples using importance sampling.

    For each repetition, dump a pickle file of the format Tuple[List[float],
    List[float]]. The first list contains the log weights and the second
    contains the drawn sample values.
    """
    from pyro.infer.importance import Importance

    count = 10000
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i in range(rep):
        print(f"t = {round(time.time() - t0)} sec.\t Start importance sampling {i+1}/{rep}")
        output_path = output_dir / f"groundtruth_{get_random_string(10)}.pickle"
        importance = Importance(walk_model, guide=None, num_samples=count)
        importance.run()

        log_weights = [w.item() for w in importance.log_weights]
        values = [t.nodes["start"]["value"].item() for t in importance.exec_traces]

        with open(output_path, "wb") as f:
            pickle.dump([log_weights, values], f)


@app.command()
def gt(
    draw_dir: str = typer.Option(
        "./importance_draws_walk/",
        "-d",
        help="Path to store the groundtruth samples.",
    ),
    output_dir: str = typer.Option(
        "./samples_walk/groundtruth",
        "-o",
        help="Path to store the groundtruth samples.",
    ),
):
    """Sample the groundtruth distribution by systematic resampling."""
    draw_dir = Path(draw_dir)
    files = glob.glob((draw_dir / "*.pickle").as_posix())
    log_weights = []
    values = []
    for f in files:
        with open(f, "rb") as _f:
            data = pickle.load(_f)
            log_weights.extend(data[0])
            values.extend(data[1])
    resamples = systematic_resampling(log_weights, values)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "groundtruth.pickle"
    with open(output_path, "wb") as f:
        pickle.dump(resamples, f)


@app.command()
def plot(
    output_path: str = typer.Option("./output", "-o", help="Output image path."),
    samples_dir: str = typer.Option(
        "./samples_walk", "-s", help="Directory containing the samples."
    ),
):
    """Generate the plots of the groundtruth and all methods."""
    import pandas
    import seaborn as sns

    data = {}
    samples_dir = Path(samples_dir).expanduser().resolve()
    for label in ["hmc", "nuts", "nphmc", "groundtruth"]:
        files = glob.glob((samples_dir / label / "*.pickle").as_posix())
        if len(files) == 0:
            continue
        label_data = []
        for f in files:
            with open(f, "rb") as _f:
                label_data.append(pickle.load(_f))
        label_data = sum(label_data, [])
        data[label] = label_data

    for k, v in data.items():
        print(k, len(v))
    data_tuples = []
    for label in data:
        data_tuples.extend([(label, v) for v in data[label]])

    x_label = "starting point"
    dataframe = pandas.DataFrame(data_tuples, columns=["method", x_label])
    plot = sns.displot(
        data=dataframe,
        x=x_label,
        hue="method",
        kind="kde",
        common_norm=False,
        facet_kws={"legend_out": False},
        # palette=palette,
        aspect=1,
        height=4,
    )
    plot.set_ylabels(label="posterior density")
    plot.savefig("walk-kde.png", bbox_inches="tight")


def main():
    app()


if __name__ == "__main__":
    main()
