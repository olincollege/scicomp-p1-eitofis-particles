import sys
import argparse

import simulation


def parse_args(args):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("Run Particle Simulation")

    parser.add_argument(
        "--n_particles", type=int, default=100,
        help="Number of particles in simulation.",
    )
    parser.add_argument(
        "--steps", type=int, default=512,
        help="Number of steps of simulation.",
    )
    parser.add_argument(
        "--size", type=int, default=256,
        help="Size of simulation environment.",
    )
    parser.add_argument(
        "--n_cells", type=int, default=64,
        help="Number of cells in uniform grid.",
    )
    parser.add_argument(
        "--dt", type=float, default=0.25,
        help="Size of timestep.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed."
    )

    args = parser.parse_args(args)
    return args


def run(args):
    """Initialize a new simulation."""
    return simulation.run(
        steps=args.steps,
        n=args.n_particles,
        size=args.size,
        n_cells=args.n_cells,
        dt=args.dt,
        seed=args.seed,
    )


def main(raw_args):
    """Run simulation."""
    args = parse_args(raw_args)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
