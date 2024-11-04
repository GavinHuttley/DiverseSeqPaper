from pathlib import Path
import time
import subprocess
import argparse


def time_big_dataset():
    parser = argparse.ArgumentParser()

    parser.add_argument("file_path", type=str, help="dvseqs file")

    args = parser.parse_args()
    in_file = Path(args.file_path)
    out_tree_file = Path(f"{in_file.name}.tre")
    out_time_file = Path(f"ctree_time_{in_file.name}.txt")

    assert in_file.exists()

    if out_tree_file.exists():
        out_tree_file.unlink()

    command = f"dvs ctree -s {in_file} -o {out_tree_file} -k 12 --sketch-size 2500 -np 8 -hp".split()
    print(" ".join(command))
    start = time.time()
    subprocess.check_output(command)
    end = time.time()

    assert out_tree_file.exists()
    out_tree_file.unlink()

    with out_time_file.open("w") as f:
        f.write(str(end - start))

    print("Completed Successfully")


if __name__ == "__main__":
    time_big_dataset()
