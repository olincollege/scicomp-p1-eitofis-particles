import jax.numpy as jnp
from jax import random


def _init_particles(n, seed, size):
    ids = jnp.arange(0, n)
    key = random.PRNGKey(seed)
    key, rng = random.split(key)
    positions = random.uniform(rng, (2, n), maxval=size)
    _, rng = random.split(key)
    velocities = random.uniform(rng, (2, n), maxval=1)
    return ids, positions, velocities


def _build_grid(n_cells, cell_size, ids, positions):
    grid_positions = (positions // cell_size).astype(jnp.int32)
    cell_ids = grid_positions[0] + grid_positions[1] * n_cells

    cell_particle_ids = jnp.stack([cell_ids, ids], axis=-1)
    sort_idxs = cell_particle_ids[:, 0].argsort()
    cell_particle_ids = cell_particle_ids[sort_idxs]
    cell_particle_idxs = jnp.arange(cell_particle_ids.shape[0])

    is_cell_change = cell_particle_ids[:, 0][1:] != cell_particle_ids[:, 0][:-1]

    cell_start_ids = cell_particle_ids[1:, 0][is_cell_change]
    cell_start_idxs = cell_particle_idxs[1:][is_cell_change]
    grid_starts = jnp.zeros(n_cells ** 2, dtype=jnp.int32)
    grid_starts = grid_starts.at[cell_start_ids].set(cell_start_idxs)

    cell_end_ids = cell_particle_ids[:-1, 0][is_cell_change]
    cell_end_idxs = cell_particle_idxs[:-1][is_cell_change]
    grid_ends = jnp.zeros(n_cells ** 2, dtype=jnp.int32)
    grid_ends = grid_ends.at[cell_end_ids].set(cell_end_idxs)
    grid_ends = grid_ends.at[cell_end_ids].add(1)
    # Last cell that has start needs end manually set
    grid_ends = grid_ends.at[cell_start_ids[-1]].set(cell_particle_ids.shape[0])

    return cell_particle_ids, grid_starts, grid_ends


def _get_neighbors(cell_particle_ids, grid_starts, grid_ends):
    pass


def _get_broad_collisions(n, size, n_cells, cell_size, ids, pos):
    cell_particle_ids, grid_starts, grid_ends = _build_grid(
        n_cells, cell_size, ids, pos
    )
    _get_neighbors(cell_particle_ids, grid_starts, grid_ends)


def _get_narrow_collisions():
    pass


def _get_collisions(n, size, n_cells, cell_size, ids, pos):
    _get_broad_collisions(n, size, n_cells, cell_size, ids, pos)
    _get_narrow_collisions()


def _get_particle_collision_response():
    pass


def _get_wall_collision_response():
    pass


def _update_velocities(_):
    _get_particle_collision_response()
    _get_wall_collision_response()


def _move():
    pass


def _step(n, size, n_cells, cell_size, ids, pos, vel):
    _get_collisions(n, size, n_cells, cell_size, ids, pos)
    _update_velocities(vel)
    _move()


def run(steps, n, size=1024, n_cells=2, seed=42):
    cell_size = (size // n_cells) + (size % n_cells > 0)
    ids, pos, vel = _init_particles(n, seed, size)
    for _ in range(steps):
        _step(n, size, n_cells, cell_size, ids, pos, vel)
