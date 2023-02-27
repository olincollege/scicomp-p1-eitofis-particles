# Particle Simulation

## Overview

This project is for the Scientific Computing class at Olin College of Engineering. The goal for this project was to implement the simulation such that it could be GPU accelerated - as such, it is written entirely in JAX. Rendering of the results can be done in OpenGL or Plotly.

This project is incomplete. Further steps include:
  * Breaking up and optimizing simulation functions.
  * Implementing separation of overlapping particles.
  * Real-time rendering (?)

## Setup

1. Clone the repository.
2. (Recommended) Setup and activate a virtual environment:

```
python -m vevn venv
source venv/bin/activate
```

3. Install dependencies:

```
pip install -r requirements.txt
```

If a virtual env was created, it will have to be re-activated before running the code if the original terminal session is terminated.

By default, JAX's CPU version will be installed. To run the simulation of GPU, please follow JAX's [GPU installation guide](https://github.com/google/jax#pip-installation-gpu-cuda).

## Run

The simulation can be run with default parameters with:

```
python particles/run.py
```

To see which parameters can be tuned from the command line, see:

```
python particles/run.py --help
```

## Results

Currently, the simulation is not fully functional. Collisions can be simulated, but the overall speed of the simulation drastically increases overtime. Further, "sticking" behavior can be seen, as we currently do not separate particles on overlap.

![Simulation Gif](/assets/example.gif)
