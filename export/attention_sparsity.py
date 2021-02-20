import argparse
import os
import pickle

import numpy as np
import pandas as pd

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Quantify attention sparsity")
parser.add_argument(
    "--persistent-dir",
    action="store",
    default=os.path.realpath(os.path.join(thisdir, "..")),
    type=str,
    help="Directory where all persistent data will be stored",
)
parser.add_argument(
    "--mass", action="store", default=0.95, type=float, help="The percentage of mass"
)


def _get_n_tokens_in_top_mass(alpha, mass=0.95):
    # Compute the number of tokens that make up the largest `mass`% of the attention
    # mass for each example (i.e. the smallest number of tokens that have a cumulative
    # mass of atleast `mass`.
    alpha = sorted(alpha, reverse=True)

    running_cum_mass = 0.0
    n_tokens = 0
    for value in alpha:
        running_cum_mass += value
        n_tokens += 1
        if running_cum_mass > mass:
            break

    return n_tokens


def _get_pretty_mean_and_std(row):
    print(row)
    mean = row["mean"]
    std = row["std"]
    pretty_str = f"${mean:.2f} \\pm {std:.2f}$"
    return pretty_str


if __name__ == "__main__":
    args = parser.parse_args()

    dataset_mapping = pd.DataFrame(
        [
            {"dataset_name": "sst", "dataset_pretty_name": "SST"},
            {"dataset_name": "snli", "dataset_pretty_name": "SNLI"},
            {"dataset_name": "imdb", "dataset_pretty_name": "IMDB"},
            {"dataset_name": "babi-1", "dataset_pretty_name": "bAbI-1"},
            {"dataset_name": "babi-2", "dataset_pretty_name": "bAbI-2"},
            {"dataset_name": "babi-3", "dataset_pretty_name": "bAbI-3"},
        ]
    )

    with open(f"{args.persistent_dir}/cache/encoded/alphas.pkl", "rb") as f:
        attention_distributions = pickle.load(f)

    results = []
    for experiment_id, values in attention_distributions.items():
        dataset_name = values["dataset_name"]
        seed = values["seed"]
        alphas = values["alphas"]

        # Compute number of tokens that make up of the top {args.mass}% of the mass
        token_counts = [
            _get_n_tokens_in_top_mass(alpha, mass=args.mass) for alpha in alphas
        ]
        # Average over examples in dataset
        average_n_tokens = np.mean(token_counts)

        results.append(
            {
                "dataset_name": dataset_name,
                "seed": seed,
                "average_n_tokens": average_n_tokens,
            }
        )

    df = pd.DataFrame(results)
    df = df.merge(dataset_mapping, on="dataset_name").drop("dataset_name", axis=1)
    # Average over seeds
    df = df.groupby("dataset_pretty_name").agg({"average_n_tokens": ["mean", "std"]})
    df = df.xs("average_n_tokens", axis=1, drop_level=True)
    df = df.reset_index()
    df["pretty_mean_and_std"] = df.apply(
        lambda row: _get_pretty_mean_and_std(row), axis=1
    )
    print(df)

    latex_str = df.to_latex(
        header=["Dataset", "Avg. num. tokens $\pm$ STD"],
        columns=["dataset_pretty_name", "pretty_mean_and_std"],
        float_format="%.2f",
        escape=False,
        index=False,
        index_names=False,
    )

    os.makedirs(f"{args.persistent_dir}/tables", exist_ok=True)
    with open(f"{args.persistent_dir}/tables/attention_sparsity.tex", "w") as f:
        f.write(latex_str)
