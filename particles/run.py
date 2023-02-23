import sys
import argparse


def parse_args(args):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("Run Particle Simulation")
    args = parser.parse_args(args)
    return args


def init(args):
    """Initialize a new simulation."""
    pass


def main(raw_args):
    """Run simulation."""
    args = parse_args(raw_args)
    init(args)


if __name__ == "__main__":
    main(sys.argv[1:])
