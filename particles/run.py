import sys
import argparse

from simulation import Simulation


def parse_args(args):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("Run Particle Simulation")
    parser.add_argument("--n_particles", type=int, default=8)
    parser.add_argument("--steps", type=int, default=128)
    args = parser.parse_args(args)
    return args


def init(args):
    """Initialize a new simulation."""
    return Simulation(args.n_particles)


def main(raw_args):
    """Run simulation."""
    args = parse_args(raw_args)
    sim = init(args)
    sim.run(args.steps)


if __name__ == "__main__":
    main(sys.argv[1:])
