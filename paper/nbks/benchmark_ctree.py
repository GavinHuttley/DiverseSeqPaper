import os
import shutil
import tempfile
import time
from pathlib import Path

import click
from click.testing import CliRunner
from cogent3 import make_table
from rich import progress as rich_progress

from diverse_seq import cli as dvs_cli
from diverse_seq import util as dvs_util

RUNNER = CliRunner()


class TempWorkingDir:
    def __enter__(self):
        self.original_directory = os.getcwd()
        self.temp_directory = tempfile.mkdtemp()
        os.chdir(self.temp_directory)
        return Path(self.temp_directory)

    def __exit__(self, exc_type, exc_value, traceback):
        os.chdir(self.original_directory)
        shutil.rmtree(self.temp_directory)


class TimeIt:
    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time

    def get_elapsed_time(self):
        return self.elapsed_time


_click_command_opts = dict(
    no_args_is_help=True,
    context_settings={"show_default": True},
)


def run_prep(temp_dir, seqdir, dvs_file, suffix, num_seqs):
    # run the prep command
    args = f"-s {seqdir} -o {dvs_file} -sf {suffix} -L {num_seqs} -np 1 -hp".split()
    with TimeIt() as timer:
        r = RUNNER.invoke(dvs_cli.prep, args, catch_exceptions=False)
        assert r.exit_code == 0, r.output
    return timer.get_elapsed_time()


def run_ctree(seqfile, outpath, k, sketch_size):
    args = f"-s {seqfile} -o {outpath} -k {k} --sketch-size {sketch_size} -np 8 -hp".split()
    with TimeIt() as timer:
        r = RUNNER.invoke(dvs_cli.ctree, args, catch_exceptions=False)
        assert r.exit_code == 0, r.output
    return timer.get_elapsed_time()


@click.command(**_click_command_opts)
@click.argument("seqdir", type=Path)
@click.argument("outpath", type=Path)
@click.option("-c", "--command", type=click.Choice(["ctree"]), default="ctree")
@click.option("-s", "--suffix", type=str, default="gb")
def run(seqdir, suffix, outpath, command):
    seqdir = seqdir.absolute()
    reps = [1, 2, 3]
    kmer_sizes = list(range(6, 19, 2))
    sketch_sizes = [
        500,
        750,
        1000,
        2500,
        5000,
        7500,
        10000,
    ]

    num_seqs = [25, 50, 75, 100, 125, 150, 175, 200]
    results = []
    run_func = run_ctree
    with dvs_util.keep_running():
        with rich_progress.Progress(
            rich_progress.TextColumn("[progress.description]{task.description}"),
            rich_progress.BarColumn(),
            rich_progress.TaskProgressColumn(),
            rich_progress.TimeRemainingColumn(),
            rich_progress.TimeElapsedColumn(),
        ) as progress:
            repeats = progress.add_task("Doing reps", total=len(reps))
            for _ in reps:
                seqnum = progress.add_task("Doing num seqs", total=len(num_seqs))
                for num in num_seqs:
                    with TempWorkingDir() as temp_dir:
                        dvs_file = temp_dir / f"dvs_L{num}.dvseqs"
                        elapsed_time = run_prep(
                            temp_dir,
                            seqdir,
                            dvs_file,
                            suffix,
                            num,
                        )
                        results.append(("prep", num, None, None, elapsed_time))

                        kmers = progress.add_task(
                            "Doing kmers",
                            total=len(kmer_sizes),
                            transient=True,
                        )
                        for k in kmer_sizes:
                            sses = progress.add_task(
                                "Doing sketch sizes",
                                total=len(sketch_sizes),
                                transient=True,
                            )
                            for ss in sketch_sizes:
                                out_file = temp_dir / f"selected-{k}-{ss}.tsv"
                                elapsed_time = run_func(dvs_file, out_file, k, sses)
                                results.append((command, num, k, ss, elapsed_time))

                                progress.update(sses, advance=1, refresh=True)

                            progress.remove_task(sses)
                            progress.update(kmers, advance=1, refresh=True)

                        progress.remove_task(kmers)

                        progress.update(seqnum, advance=1, refresh=True)
                progress.remove_task(seqnum)
                progress.update(repeats, advance=1, refresh=True)

    table = make_table(
        header=("command", "numseqs", "k", "ss", "time(s)"), data=results
    )
    print(table)
    table.write(outpath)


if __name__ == "__main__":
    run()
