import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from project_path import FIG_DIR


def load_and_prepare_data(file_path):
    data = pd.read_csv(file_path, sep="\t").dropna(subset=["k", "ss", "numseqs"])
    data[["k", "ss", "numseqs"]] = data[["k", "ss", "numseqs"]].astype(int)
    return data[data["command"] == "ctree"]


def check_replicates(grouped):
    if (grouped["count"] != 3).any():
        raise ValueError("Not all points have 3 replicates. Please check the data.")


def set_log_scale(ax, is_log):
    if is_log:
        ax.set_xscale("log")
        major_ticks = [500, 750, 1000, 2500, 5000, 7500, 10000]
        ax.set_xticks(major_ticks)
        ax.xaxis.set_major_formatter(plt.ScalarFormatter())
        ax.xaxis.set_minor_locator(ticker.FixedLocator([750, 2500, 7500]))
        ax.xaxis.set_minor_formatter(plt.NullFormatter())


def create_plot(grouped, x_col, y_col, hue_col, title, output_dir):
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("viridis", n_colors=len(grouped[hue_col].unique()))

    ax = sns.lineplot(
        data=grouped,
        x=x_col,
        y=y_col,
        hue=hue_col,
        palette=palette,
        marker="o",
        errorbar=None,
    )

    for hue_value in grouped[hue_col].unique():
        subset = grouped[grouped[hue_col] == hue_value]
        plt.errorbar(
            subset[x_col],
            subset[y_col],
            yerr=subset["std_time"],
            fmt="none",
            capsize=5,
            color=palette[list(grouped[hue_col].unique()).index(hue_value)],
        )

    set_log_scale(ax, x_col == "ss")

    x_label = (
        "Sketch Size"
        if x_col == "ss"
        else "Number of Sequences"
        if x_col == "numseqs"
        else x_col
    )
    y_label = "Average Time (s)"
    plt.title(
        title.replace("ss", "Sketch Size").replace("numseqs", "Number of Sequences")
    )
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(
        title=hue_col.replace("ss", "Sketch Size").replace(
            "numseqs", "Number of Sequences"
        )
    )
    plt.grid(True)

    plt.savefig(output_dir / f"time_vs_{x_col}.png")
    plt.close()


def create_directory(base_dir, value, prefix):
    dir_path = base_dir / f"{prefix}_{value}"
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def group_data(data, x_col, y_col, hue_col):
    return (
        data.groupby([x_col, hue_col])
        .agg(
            mean_time=("time(s)", "mean"),
            std_time=("time(s)", "std"),
            count=("time(s)", "size"),
        )
        .reset_index()
    )


def plot_for_k(data, base_output_dir):
    for k_value in data["k"].unique():
        subset = data[data["k"] == k_value]
        grouped = group_data(subset, "numseqs", "mean_time", "ss")
        check_replicates(grouped)

        k_output_dir = create_directory(base_output_dir, k_value, "k")
        create_plot(
            grouped,
            "ss",
            "mean_time",
            "numseqs",
            f"Time vs. Sketch Size (k = {k_value})",
            k_output_dir,
        )
        create_plot(
            grouped,
            "numseqs",
            "mean_time",
            "ss",
            f"Time vs. Number of Sequences (k = {k_value})",
            k_output_dir,
        )


def plot_for_ss(data, base_output_dir):
    for ss_value in data["ss"].unique():
        subset = data[data["ss"] == ss_value]
        grouped = group_data(subset, "numseqs", "mean_time", "k")
        check_replicates(grouped)

        ss_output_dir = create_directory(base_output_dir, ss_value, "ss")
        create_plot(
            grouped,
            "k",
            "mean_time",
            "numseqs",
            f"Time vs. k (Sketch Size = {ss_value})",
            ss_output_dir,
        )
        create_plot(
            grouped,
            "numseqs",
            "mean_time",
            "k",
            f"Time vs. Number of Sequences (Sketch Size = {ss_value})",
            ss_output_dir,
        )


def plot_for_numseqs(data, base_output_dir):
    for numseqs_value in data["numseqs"].unique():
        subset = data[data["numseqs"] == numseqs_value]
        grouped = group_data(subset, "k", "mean_time", "ss")
        check_replicates(grouped)

        numseqs_output_dir = create_directory(base_output_dir, numseqs_value, "numseqs")
        create_plot(
            grouped,
            "ss",
            "mean_time",
            "k",
            f"Time vs. Sketch Size (Number of Sequences = {numseqs_value})",
            numseqs_output_dir,
        )
        create_plot(
            grouped,
            "k",
            "mean_time",
            "ss",
            f"Time vs. k (Number of Sequences = {numseqs_value})",
            numseqs_output_dir,
        )


def generate_plots():
    data = load_and_prepare_data("ctree_time.tsv")
    base_output_dir = FIG_DIR / "ctree_times"
    base_output_dir.mkdir(exist_ok=True)
    # plot_for_k(data, base_output_dir)
    plot_for_ss(data, base_output_dir)
    # plot_for_numseqs(data, base_output_dir)


if __name__ == "__main__":
    generate_plots()
