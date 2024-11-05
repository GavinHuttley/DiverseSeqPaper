from pathlib import Path
import argparse
import cogent3
from diverse_seq import cli as dvs_cli
from diverse_seq import util as dvs_util
from click.testing import CliRunner
from benchmark import TempWorkingDir, TimeIt
from rich.progress import Progress

RUNNER = CliRunner()


def time_big_dataset():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str, help="dvseqs file")

    args = parser.parse_args()
    in_file = Path(args.file_path)

    assert in_file.exists()
    times = {"time": []}
    with Progress() as track:
        task = track.add_task("Running ctree", total=5)
        for _ in range(5):
            with TempWorkingDir() as temp_dir:
                out_tree_file = temp_dir / f"{in_file.name}.tre"
                args = f"-s {in_file} -o {out_tree_file} -k 12 --sketch-size 3000 -hp".split()
                with TimeIt() as timer:
                    r = RUNNER.invoke(dvs_cli.ctree, args, catch_exceptions=False)
                    assert r.exit_code == 0, r.output
                times["time"].append(timer.get_elapsed_time())
                track.update(task, advance=1, refresh=True)

    table = cogent3.make_table(data=times)
    outpath = f"benchmark_ctree_{in_file.stem}.tsv"
    table.write(outpath)
    dvs_util.print_colour(
        f"mean(sec)={table.columns["time"].mean():.2f}; stdev(sec)={table.columns["time"].std(ddof=1):.2f}",
        "blue",
    )
    dvs_util.print_colour(f"Wrote {outpath}!", "green")


if __name__ == "__main__":
    time_big_dataset()
