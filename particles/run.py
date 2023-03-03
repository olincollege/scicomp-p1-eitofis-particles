import sys
import argparse
from functools import partial

import moderngl
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
        "--steps", type=int, default=None,
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
        '--headless', action='store_true', default=False,
        help="Run simulation in headless mode."
    )
    parser.add_argument(
        '--save', type=str, default=None,
        help=(
            "Filename to save video to. Requires steps to be specified."
            " NOTE: When save is specified, no real-time output will be "
            "drawn."
        )
    )

    args = parser.parse_args(args)
    return args


def run(args):
    """Run simulation."""
    Renderer._init_args = (
        args.steps,
        args.n_particles,
        args.size,
        args.n_cells,
        args.dt,
        args.seed,
        args.save,
    )
    if args.headless:
        moderngl.create_standalone_context = partial(
            moderngl.create_standalone_context,
            backend="egl",
        )
        mglw.run_window_config(Renderer, args=["--window", "headless"])
    else:
        mglw.run_window_config(Renderer, args=["-r", "True"])


def main(raw_args):
    """Main function."""
    args = parse_args(raw_args)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
