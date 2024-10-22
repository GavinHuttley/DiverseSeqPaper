from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import seaborn as sns

FIG_PATH = Path("../../figs")

def load_data():
    df = pd.read_csv(
        "out/results_with_ls.tsv",
        sep="\t",
        names=["k", "ss", "cpus", "time", "likelihood", "tree"],
    )
    l_df = pd.read_csv(
        "out/iqtree_with_ls.tsv",
        sep="\t",
        names=["time", "likelihood", "tree"],
    )
    return df, l_df


def plot_likelihood_vs_k(df, l_df, k_values, ss_values, colormap):
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, ss in enumerate(ss_values):
        sub_df = df[df["ss"] == ss]
        color = colormap(i / (len(ss_values) - 1))
        ax.plot(
            sub_df["k"],
            sub_df["likelihood"],
            label=f"sketch size={ss}",
            color=color,
            marker="o",
            markersize=8,
            markerfacecolor=color,
            markeredgewidth=1.5,
            linewidth=2,
        )

    ax.plot(
        k_values,
        l_df["likelihood"].repeat(len(k_values)),
        linestyle="--",
        color="black",
        linewidth=2,
        label="IQTree likelihood",
    )

    ax.set_xlabel("k", fontsize=14)
    ax.set_ylabel("Likelihood", fontsize=14)
    ax.set_title("Likelihood vs k for Different Sketch Sizes", fontsize=16)

    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), title="ss values", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG_PATH / "likelihood_vs_k_for_ss.png", bbox_inches="tight", dpi=300)


def plot_likelihood_vs_ss(df, l_df, k_values, ss_values, colormap, min_k_value):
    fig, ax = plt.subplots(figsize=(10, 6))

    filtered_k_values = [k for k in k_values if k >= min_k_value]

    for i, k in enumerate(filtered_k_values):
        sub_df = df[df["k"] == k]
        color = colormap(i / (len(filtered_k_values) - 1))
        ax.plot(
            sub_df["ss"],
            sub_df["likelihood"],
            label=f"k={k}",
            color=color,
            marker="o",
            markersize=8,
            markerfacecolor=color,
            markeredgewidth=1.5,
            linewidth=2,
        )

    ax.plot(
        ss_values,
        l_df["likelihood"].repeat(len(ss_values)),
        linestyle="--",
        color="black",
        linewidth=2,
        label="IQTree likelihood",
    )

    ax.set_xlabel("Sketch Size", fontsize=14)
    ax.set_ylabel("Likelihood", fontsize=14)
    ax.set_title(
        f"Likelihood vs Sketch Size for Different k Values (k >= {min_k_value})", fontsize=16
    )

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10, subs=[1.0, 2.5, 5.0]))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xticks(ss_values, minor=True)

    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), title="k values", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG_PATH / "likelihood_vs_ss_for_k.png", bbox_inches="tight", dpi=300)


def main():
    sns.set_context("paper")
    sns.set_style("darkgrid")

    df, l_df = load_data()

    k_values = sorted(df["k"].unique())
    ss_values = sorted(df["ss"].unique())
    colormap = sns.color_palette(palette="viridis", as_cmap=True)

    plot_likelihood_vs_k(df, l_df, k_values, ss_values, colormap)
    plot_likelihood_vs_ss(df, l_df, k_values, ss_values, colormap, min_k_value=8)


if __name__ == "__main__":
    main()
