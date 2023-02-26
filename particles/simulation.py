import jax
import jax.numpy as jnp
from jax.scipy import signal


def _init_particles(n, seed, size):
    ids = jnp.arange(0, n)
    key = jax.random.PRNGKey(seed)
    key, rng = jax.random.split(key)
    positions = jax.random.uniform(rng, (2, n), maxval=size)
    _, rng = jax.random.split(key)
    velocities = jax.random.uniform(rng, (2, n), maxval=1)
    return ids, positions, velocities


def _build_grid(n_cells, cell_size, ids, positions):
    grid_positions = (positions // cell_size).astype(jnp.int32) + 1
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


def _get_neighbors(n_cells, cell_particle_ids, grid_starts, grid_ends):
    cell_counts = grid_ends - grid_starts
    grid_cell_counts = jnp.reshape(cell_counts, (n_cells, n_cells))
    kernel = jnp.ones((3, 3), dtype=jnp.int32)
    neighbor_counts = signal.convolve2d(
        grid_cell_counts[1:-1, 1:-1], kernel, mode="same"
    ).astype(jnp.int32)
    max_neighbors = jnp.max(neighbor_counts).item()

    def _map_func(cell_id):
        _idxs = jnp.array([
            cell_id - 1 - n_cells,
            cell_id - n_cells,
            cell_id + 1 - n_cells,
            cell_id - 1,
            cell_id,
            cell_id + 1,
            cell_id - 1 + n_cells,
            cell_id + n_cells,
            cell_id + 1 + n_cells,
        ])
        neighbors = jnp.zeros(max_neighbors, dtype=jnp.int32)
        idx = 0
        def _body_func(i, state):
            neighbors, idx = state
            def _body_func(i, state):
                neighbors, idx = state
                neighbors = neighbors.at[idx].set(cell_particle_ids[i, 1])
                return neighbors, idx + 1
            start = grid_starts[_idxs[i]]
            end = grid_ends[_idxs[i]]
            neighbors, idx = jax.lax.fori_loop(start, end, _body_func, (neighbors, idx))
            return neighbors, idx
        neighbors, _ = jax.lax.fori_loop(0, 9, _body_func, (neighbors, idx))
        return neighbors

    neighbors = jax.vmap(_map_func)(cell_particle_ids[:, 0])
    return neighbors


def _get_broad_collisions(n_cells, cell_size, ids, pos):
    cell_particle_ids, grid_starts, grid_ends = _build_grid(
        n_cells, cell_size, ids, pos
    )
    neighbors = _get_neighbors(n_cells, cell_particle_ids, grid_starts, grid_ends)
    return cell_particle_ids, neighbors


def _get_narrow_collisions():
    pass


def _get_collisions(n_cells, cell_size, ids, pos):
    _get_broad_collisions(n_cells, cell_size, ids, pos)
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


def _step(n_cells, cell_size, ids, pos, vel):
    _get_collisions(n_cells, cell_size, ids, pos)
    _update_velocities(vel)
    _move()


def run(steps, n, size=1024, n_cells=2, seed=42):
    cell_size = (size // n_cells) + (size % n_cells > 0)
    n_cells = n_cells + 2  # Add outer padding to grid
    ids, pos, vel = _init_particles(n, seed, size)
    for _ in range(steps):
        _step(n_cells, cell_size, ids, pos, vel)
