import sys
import argparse

import moderngl_window as mglw

from render import Renderer


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
    parser.add_argument(
        '--graph', action='store_true', default=False,
        help=(
            "Produce an interactive graph instead of rendering a gif. "
            "NOTE: Significantly less performant - will fail to plot "
            "for larger numbers of particles"
        )
    )

    args = parser.parse_args(args)
    return args


def run(args):
    """Run simulation."""
    Renderer._init_args = (
        args.n_particles,
        args.size,
        args.n_cells,
        args.dt,
        args.seed,
        args.graph,
    )
    mglw.run_window_config(Renderer, args=["-r", "True"])


def main(raw_args):
    """Main function."""
    args = parse_args(raw_args)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
