import sys
import argparse

import simulation


def parse_args(args):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("Run Particle Simulation")
    parser.add_argument("--n_particles", type=int, default=10)
    parser.add_argument("--steps", type=int, default=128)
    args = parser.parse_args(args)
    return args


def run(args):
    """Initialize a new simulation."""
    return simulation.run(
        steps=args.steps,
        n=args.n_particles,
    )


def main(raw_args):
    """Run simulation."""
    args = parse_args(raw_args)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
