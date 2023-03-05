import os
import sys
import argparse
import pickle

import matplotlib.pyplot as plt


def parse(args):
    parser = argparse.ArgumentParser("Graph particle sim results.")

    parser.add_argument(
        "--files", type=str, nargs="+",
        help=""
    )

    args = parser.parse_args(args)
    return args


def load(files):
    all_data = {}
    for fn in files:
        name = os.path.basename(fn)
        name = name.split(".")[0]
        n = name.split("_")[-1]
        with open(fn, "rb") as f:
            data = pickle.load(f)
        all_data[n] = data
    return all_data


def graph(all_data):
    plt.figure(figsize = (5, 5))
    for n, data in all_data.items():
        plt.plot(data, label=f"{n} Particles")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend(loc="upper left")
    plt.xlabel("Average X Displacement Squared")
    plt.xlabel("Time")
    os.makedirs("data", exist_ok=True)
    fn = os.path.join("data", f"shifts")
    plt.savefig(fn)


def main(raw_args):
    args = parse(raw_args)
    data = load(args.files)
    graph(data)


if __name__ == "__main__":
    main(sys.argv[1:])
