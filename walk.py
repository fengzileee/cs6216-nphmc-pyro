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

    # ZIRUI: add is_cont in the kwargs in the pyro.sample()
    start = pyro.sample("start", pyro.distributions.Uniform(0, 3), is_cont=False)
    t = 0
    position = start
    distance = torch.tensor(0.0)
    while position > 0 and position < 10:
        step = pyro.sample(
            f"step_{t}", pyro.distributions.Uniform(-1, 1), is_cont=False
        )
        distance = distance + torch.abs(step)
        position = position + step
        t = t + 1
    pyro.sample("obs", pyro.distributions.Normal(1.1, 0.1), obs=distance, is_cont=False)
    return start.item()


def geometric_model():
    import pyro
    import torch

    t = 0
    ret = torch.tensor(1.0)
    # prob of continue flipping
    prob = pyro.sample("prob", pyro.distributions.Uniform(0.25, 0.75), is_cont=False)
    while True:
        draw = pyro.sample(f"draw_{t}", pyro.distributions.Uniform(0, 1), is_cont=False)
        if draw.item() < prob.item():
            ret = ret + torch.tensor(1)
            t = t + 1
        else:
            break
    pyro.sample(
        "obs", pyro.distributions.Delta(torch.tensor(5.0)), obs=ret, is_cont=False
    )
    return prob.item()


def get_model(model_name):
    if model_name == "random_walk":
        return walk_model, "start"
    elif model_name == "geometric":
        return geometric_model, "prob"
    else:
        raise NotImplementedError(f"We don't have model {model_name}")


@app.command()
def run(
    method: str,
    output_dir: str = typer.Option(
        "./output", "-o", help="Path to the output data folder."
    ),
    model_name: str = typer.Option(
        "random_walk", "-m", help="Name of the probabilistic program model."
    ),
    count: int = typer.Option(1000, "-c", help="Number of samples to draw."),
    eps: float = typer.Option(0.1, "-e", help="Step size in leapfrog integration."),
    num_steps: int = typer.Option(
        50, "-l", help="Numer of steps in leapfrog integration."
    ),
    rep: int = typer.Option(1, "-r", help="Number of repetitions of sampling"),
):
    """Generate samples using different MCMC methods."""
    import pyro.infer.mcmc as pyromcmc  # type: ignore

    model, target_node_name = get_model(model_name)
    output_dir = (
        Path(output_dir).expanduser().resolve() / model_name / "samples" / method
    )

    for _ in range(rep):
        if method == "nuts":
            info = ""
            kernel = pyromcmc.NUTS(model)
        elif method == "hmc":
            info = f"_{eps}_{num_steps}"
            kernel = pyromcmc.HMC(
                model,
                step_size=eps,
                num_steps=num_steps,
                adapt_step_size=False,
            )
        elif method == "npdhmc":
            info = f"_{eps}_{num_steps}"
            kernel = pyromcmc.NPDHMC(
                model,
                step_size=eps,
                num_steps=num_steps,
                adapt_step_size=False,
            )
        else:
            raise RuntimeError(f"Unknown method {method}")

        mcmc = pyromcmc.MCMC(kernel, num_samples=count, warmup_steps=count // 10)
        mcmc.run()

        samples = mcmc.get_samples()
        raw_samples = [value.item() for value in samples[target_node_name]]

        output_dir.mkdir(parents=True, exist_ok=True)
        random_str = get_random_string(6)
        output_file = f"{model_name}_{random_str}_{count}{info}.pickle"
        with open(output_dir / output_file, "wb") as f:
            pickle.dump(raw_samples, f)
        try:
            mcmc.summary()
        except Exception:
            pass


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
    model_name: str = typer.Option(
        "random_walk", "-m", help="Name of the probabilistic program model."
    ),
    output_dir: str = typer.Option(
        "./output", "-o", help="Path to the output data folder."
    ),
    rep: int = typer.Option(1, "-r", help="Number of repetitions."),
    count: int = typer.Option(10000, "-c", help="Number of draws."),
):
    """Repeatedly draw 10,000 samples using importance sampling.

    For each repetition, dump a pickle file of the format Tuple[List[float],
    List[float]]. The first list contains the log weights and the second
    contains the drawn sample values.
    """
    from pyro.infer.importance import Importance

    model, target_node_name = get_model(model_name)
    output_dir = (
        Path(output_dir).expanduser().resolve() / model_name / "importance_draws"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i in range(rep):
        print(
            f"t = {round(time.time() - t0)} sec.\t"
            f"Start importance sampling {i+1}/{rep}. Count = {count}."
        )
        output_path = output_dir / f"groundtruth_{get_random_string(10)}.pickle"
        importance = Importance(model, guide=None, num_samples=count)
        importance.run()

        log_weights = [w.item() for w in importance.log_weights]
        values = [
            t.nodes[target_node_name]["value"].item() for t in importance.exec_traces
        ]

        with open(output_path, "wb") as f:
            pickle.dump([log_weights, values], f)


@app.command()
def gt(
    model_name: str = typer.Option(
        "random_walk", "-m", help="Name of the probabilistic program model."
    ),
    output_dir: str = typer.Option(
        "./output", "-o", help="Path to the output data folder."
    ),
):
    """Sample the groundtruth distribution by systematic resampling."""
    output_dir = Path(output_dir).resolve().expanduser()
    model_dir = output_dir / model_name
    draw_dir = model_dir / "importance_draws"
    if not draw_dir.exists():
        raise ValueError(f"Importance draws directory {draw_dir} does not exist!")

    files = glob.glob((draw_dir / "*.pickle").as_posix())
    log_weights = []
    values = []
    for f in files:
        with open(f, "rb") as _f:
            data = pickle.load(_f)
            log_weights.extend(data[0])
            values.extend(data[1])
    resamples = systematic_resampling(log_weights, values)
    gt_dir = model_dir / "samples" / "groundtruth"
    gt_dir.mkdir(parents=True, exist_ok=True)
    gt_path = gt_dir / "groundtruth.pickle"
    with open(gt_path, "wb") as f:
        pickle.dump(resamples, f)


def _plot_averaged_distributions(model_name: str, output_dir: Path):
    import pandas
    import seaborn as sns

    print(f"Generating averaged distribution plot for {model_name}")
    data = {}
    samples_dir = output_dir / model_name / "samples"
    for label in ["hmc", "nuts", "npdhmc", "groundtruth"]:
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
    image_save_path = output_dir / model_name / f"{model_name}_all.png"
    plot.set_ylabels(label="posterior density")
    plot.savefig(image_save_path.as_posix(), bbox_inches="tight")


def _plot_distribution_of_distributions(
    model_name: str, method_name: str, output_dir: Path, variacne_thresh: float = 1e-2
):
    import pandas
    import seaborn as sns
    import numpy as np

    print(f"Generating {method_name} distribution plot for {model_name}")
    data = {}
    samples_dir = output_dir / model_name / "samples" / method_name
    files = glob.glob((samples_dir / "*.pickle").as_posix())
    if len(files) == 0:
        return
    data = []
    for f in files:
        with open(f, "rb") as _f:
            data.append(pickle.load(_f))

    print(f"{method_name} has {len(data)} repetitions.")

    data_tuples = []
    for i, rep in enumerate(data):
        variance = np.var(rep)
        if variance < variacne_thresh:
            continue
        data_tuples.extend([(str(i + 1), v) for v in rep])

    x_label = "starting point"
    dataframe = pandas.DataFrame(data_tuples, columns=["rep", x_label])
    plot = sns.displot(
        data=dataframe,
        x=x_label,
        hue="rep",
        kind="kde",
        common_norm=False,
        facet_kws={"legend_out": False},
        # palette=palette,
        aspect=1,
        height=4,
    )
    image_save_path = output_dir / model_name / f"{model_name}_{method_name}.png"
    plot.set_ylabels(label="posterior density")
    plot.savefig(image_save_path.as_posix(), bbox_inches="tight")


@app.command()
def plot(output_dir: str = typer.Option("./output", "-o", help="Output data path.")):
    """Generate the plots of the groundtruth and all methods in all models."""
    output_dir = Path(output_dir).resolve().expanduser()
    var_thresh_dict = {'random_walk': 1e-2, 'geometric': 1e-6}
    for model_name, var_thresh in var_thresh_dict.items():
        _plot_averaged_distributions(model_name, output_dir)
        for method_name in ["hmc", "nuts", "npdhmc"]:
            _plot_distribution_of_distributions(model_name, method_name, output_dir, var_thresh)


def main():
    app()


if __name__ == "__main__":
    main()
