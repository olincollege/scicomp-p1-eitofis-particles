import time
from functools import partial

import jax
import jax.numpy as jnp
from tqdm import tqdm

from graph import graph
from render import render


def _init_particles(n, seed, size):
    """Initialize particle positions and velocities.

    Args:
        n: Number of particles.
        size: Size of environment in each direction.
        seed: Random seed.

    Returns:
        ids: Array of shape (n), all particle ids.
        positions: Array of shape (2, n), all particle positions.
        velocities: Array of shape (2, n), all particle velocities.
    """
    ids = jnp.arange(0, n)
    key = jax.random.PRNGKey(seed)
    key, rng = jax.random.split(key)
    n_row_col = int(jnp.sqrt(n))
    x, y = jnp.meshgrid(
        jnp.linspace(1, size - 1, n_row_col),
        jnp.linspace(1, size - 1, n_row_col),
    )
    positions = jnp.stack((x.flatten(), y.flatten()), axis=0)

    _, rng = jax.random.split(key)
    velocities = jax.random.uniform(rng, (2, n), minval=-1, maxval=1)
    return ids, positions, velocities


def _build_grid(n_particles, n_cells, cell_size, max_per_cell, ids, positions):
    """Construct uniform grid.

    Args:
        n_cells: Number of uniform grid cells in each direction.
        cell_size: Size of single uniform grid cell.
        ids: Array of shape (n), particle ids.
        positions: Array of shape (2, n), particle positions.

    Returns:
        cell_particle_ids: Array of shape(2, n), mapping of uniform grid cell id
            to particle id. Sorted by cell id.
        grid_starts: Array of shape (n_cells), for each cell contains the first
            index in which the cell appears in cell_particle_ids.
        grid_ends: Array of shape (n_cells), for each cell contains the last + 1
            index in which the cell appears in cell_particle_ids.
    """
    grid_positions = (positions // cell_size).astype(jnp.int32) + 1
    cell_ids = grid_positions[0] + grid_positions[1] * n_cells
    cell_particle_ids = jnp.stack([cell_ids, ids], axis=-1)

    def _map_func(cell_id):
        return jnp.sum(jnp.where(cell_ids == cell_id, 1, 0))
    cell_counts = jax.vmap(_map_func)(jnp.arange(n_cells ** 2))
    cell_missing = max_per_cell - cell_counts

    cell_missing_mask = jnp.zeros((cell_counts.shape[0], max_per_cell))
    cell_missing_mask = cell_missing_mask + jnp.arange(0, cell_missing_mask.shape[1])
    cell_missing_mask = cell_missing_mask < cell_missing[:, None]

    pad_cells = jnp.zeros((cell_counts.shape[0], max_per_cell), dtype=jnp.int32)
    pad_cells = pad_cells + jnp.arange(0, pad_cells.shape[0])[:, None]

    missing = ((n_cells ** 2) * max_per_cell) - n_particles
    pad_cells = jnp.where(cell_missing_mask, pad_cells, -1).flatten()
    pad_cells = jnp.sort(pad_cells)[-missing:]

    pad_particles = jnp.zeros_like(pad_cells, dtype=jnp.int32) - 1
    pad_cell_particle_ids = jnp.stack([pad_cells, pad_particles], axis=-1)

    padded_cell_particle_ids = jnp.concatenate(
        (cell_particle_ids, pad_cell_particle_ids), axis=0
    )
    sort_idxs = padded_cell_particle_ids[:, 0].argsort()
    padded_cell_ids = padded_cell_particle_ids[:, 1][sort_idxs]

    grid = padded_cell_ids.reshape((n_cells * n_cells, max_per_cell))
    return cell_particle_ids, grid


def _get_neighbors(n_cells, cell_particle_ids, grid):
    """Gather neighbors from uniform grid.

    Args:
        n_cells: Number of uniform grid cells in each direction.
        cell_particle_ids: Array of shape(2, n), mapping of uniform grid cell id
            to particle id. Sorted by cell id.
        grid_starts: Array of shape (n_cells), for each cell contains the first
            index in which the cell appears in cell_particle_ids.
        grid_ends: Array of shape (n_cells), for each cell contains the last + 1
            index in which the cell appears in cell_particle_ids.

    Returns:
        neighbors: Array of shape (n, max_neighbors), contains each particles
            neighbors with position corresponding to cell_particle_ids.
        neighbors_mask: Array of shape (n, max_neighbors), contains mask where 1
            for real value and 0 for padding, with position corresponding to
            cell_particle_ids
    """
    def _map_func(cell_id):
        neighbor_cells = jnp.concatenate((
            grid[cell_id - 1 - n_cells],
            grid[cell_id - n_cells],
            grid[cell_id + 1 - n_cells],
            grid[cell_id - 1],
            grid[cell_id],
            grid[cell_id + 1],
            grid[cell_id - 1 + n_cells],
            grid[cell_id + n_cells],
            grid[cell_id + 1 + n_cells],
        ), axis=0)
        return neighbor_cells
    neighbors = jax.vmap(_map_func)(cell_particle_ids[:, 0])
    neighbors_mask = jnp.where(neighbors == -1, 0, 1)
    return neighbors, neighbors_mask


def _get_broad_collisions(n, n_cells, cell_size, max_per_cell, ids, pos):
    """Compute broad particle collisions.

    Args:
        n_cells: Number of uniform grid cells in each direction.
        cell_size: Size of single uniform grid cell.
        ids: Array of shape (n), particle ids.
        pos: Array of shape (2, n), particle positions.

    Returns:
        particle_ids: Array of shape (n), contains particle ids no longer in
            sorted order.
        neighbors: Array of shape (n, max_neighbors), contains each particles
            neighbors with position corresponding to particle_ids.
        neighbors_mask: Array of shape (n, max_neighbors), contains mask where 1
            for real value and 0 for padding, with position corresponding to
            particle_ids.
    """
    cell_particle_ids, grid = _build_grid(n, n_cells, cell_size, max_per_cell, ids, pos)
    neighbors, neighbor_mask = _get_neighbors(n_cells, cell_particle_ids, grid)
    return neighbors, neighbor_mask


def _get_narrow_collisions(particle_ids, positions, neighbors, neighbor_mask):
    """Compute exact particle collisions.

    Args:
        positions: Array of shape (2, n), particle positions.
        particle_ids: Array of shape (n), contains particle ids no longer in
            sorted order.
        neighbors: Array of shape (n, max_neighbors), contains each particles
            neighbors with position corresponding to particle_ids.
        neighbors_mask: Array of shape (n, max_neighbors), contains mask where 1
            for real value and 0 for padding, with position corresponding to
            particle_ids.

    Returns:
        collisions: Array of shape(n, max_neighbors), mask that is 1 if paticles
            collide, 0 otherwise, with position corresponding to particle_ids
    """
    def _map_func(particle_id, neighbors, mask):
        same_id_mask = neighbors != particle_id
        mask = mask & same_id_mask

        current_position = positions[:, particle_id]
        current_radius = 1
        neighbor_positions = positions[:, neighbors]
        neighbor_radii = jnp.ones_like(neighbors)

        distances = (
            (current_position[0] - neighbor_positions[0]) ** 2 +
            (current_position[1] - neighbor_positions[1]) ** 2
        )
        collision_distance = (current_radius + neighbor_radii) ** 2
        collides = distances <= collision_distance

        mask = mask & collides
        return mask

    collisions = jax.vmap(_map_func)(particle_ids, neighbors, neighbor_mask)
    return collisions


def _get_collisions(n, n_cells, cell_size, max_per_cell, ids, pos):
    """Compute particle collisions.

    Args:
        n_cells: Number of uniform grid cells in each direction.
        cell_size: Size of single uniform grid cell.
        ids: Array of shape (n), particle ids.
        pos: Array of shape (2, n), particle positions.

    Returns:
        neighbors: Array of shape (n, max_neighbors), contains each particles
            neighbors with position corresponding to sorted particle ids
        collisions: Array of shape(n, max_neighbors), mask that is 1 if paticles
            collide, 0 otherwise, with position corresponding to sorted particle
            ids.
    """
    neighbors, neighbor_mask = _get_broad_collisions(
        n, n_cells, cell_size, max_per_cell, ids, pos
    )
    collisions = _get_narrow_collisions(ids, pos, neighbors, neighbor_mask)
    return neighbors, collisions


def _dot(v1, v2):
    """Elementwise dot product."""
    return jnp.sum(v1 * v2, axis=0)


def _get_particle_collision_response(
    positions, velocities, particle_ids, neighbors, collisions
):
    """Compute particle collision resonses.

    Args:
        positions: Array of shape (2, n), particle positions.
        velocities: Array of shape (2, n), particle velocities.
        particle_ids: Array of shape (n), particle ids.
        neighbors: Array of shape (n, max_neighbors), contains each particles
            neighbors with position corresponding to sorted particle_ids.
        collisions: Array of shape(n, max_neighbors), mask that is 1 if paticles
            collide, 0 otherwise, with position corresponding to particle_ids.

    Returns:
        velocity_changes: Array of shape (2, n), changes in particle velocity
            with position corresponding to particle_ids.
    """
    def _map_func(particle_id, neighbors, collisions):
        particle_position = positions[:, particle_id]
        particle_velocity = velocities[:, particle_id]
        particle_mass = 1
        neighbor_positions = positions[:, neighbors]
        neighbor_velocities = velocities[:, neighbors]
        neighbor_masses = jnp.ones_like(neighbors)

        collision_normals = neighbor_positions - particle_position[:, None]
        magnitudes = jnp.linalg.norm(collision_normals, axis=0)
        collision_normals = jnp.nan_to_num(collision_normals / magnitudes)

        relative_velocities = neighbor_velocities - particle_velocity[:, None]

        numerator = _dot(-2 * relative_velocities, collision_normals)
        denominator = _dot(
            collision_normals,
            collision_normals * (1 / neighbor_masses + 1 / particle_mass)
        )
        impulse = jnp.nan_to_num(numerator / denominator)

        velocity_changes = (impulse / particle_mass) * -1
        velocity_changes = velocity_changes * collisions
        velocity_changes = velocity_changes * collision_normals
        velocity_changes = jnp.sum(velocity_changes, axis=-1)
        return velocity_changes

    velocity_changes = jax.vmap(_map_func)(
        particle_ids, neighbors, collisions
    )
    velocity_changes = jnp.transpose(velocity_changes)
    return velocity_changes


def _get_wall_collision_response(size, positions, velocities):
    """Compute wall collision resonses.

    Args:
        positions: Array of shape (2, n), particle positions.
        velocities: Array of shape (2, n), particle velocities.

    Returns:
        velocity_changes: Array of shape (2, n), changes in particle velocity
            with position corresponding to particle_ids.
    """
    radii = jnp.ones(positions.shape[1])

    oob_x = ((positions[0, :] - radii) <= 0) | ((positions[0, :] + radii) >= size)
    oob_y = ((positions[1, :] - radii) <= 0) | ((positions[1, :] + radii) >= size)
    oob = jnp.stack([oob_x, oob_y], axis=0)

    inverse_velocities = velocities * -2
    velocity_changes = inverse_velocities * oob
    return velocity_changes


def _update_velocities(size, pos, vel, particle_ids, neighbors, collisions):
    """Compute collision responses and update velocities.

    Args:
        size: Size of environment in each direction.
        pos: Array of shape (2, n), particle positions.
        vel: Array of shape (2, n), particle velocities.
        particle_ids: Array of shape (n), particle ids.
        neighbors: Array of shape (n, max_neighbors), contains each particles
            neighbors with position corresponding to sorted particle_ids.
        collisions: Array of shape(n, max_neighbors), mask that is 1 if paticles
            collide, 0 otherwise, with position corresponding to particle_ids.

    Returns:
        vel: Array of shape (2, n), new velocity.
    """
    velocity_changes = _get_particle_collision_response(
        pos, vel, particle_ids, neighbors, collisions
    )
    vel = vel + velocity_changes
    velocity_changes = _get_wall_collision_response(size, pos, vel)
    vel = vel + velocity_changes
    return vel


def _move(positions, velocities, dt):
    """Move particles."""
    positions = positions + velocities * dt
    return positions


@partial(jax.jit, static_argnames=['n', 'size', 'n_cells', 'cell_size', 'max_per_cell'])
def _step(n, size, n_cells, cell_size, max_per_cell, ids, pos, vel, dt):
    """Take single step of the simulation."""
    neighbors, collisions = _get_collisions(n, n_cells, cell_size, max_per_cell, ids, pos)
    vel = _update_velocities(size, pos, vel, ids, neighbors, collisions)
    pos = _move(pos, vel, dt)
    return pos, vel


def run(steps, n, size, n_cells, dt, seed, plot):
    """Initialize and run the simulation.

    Args:
        steps: Number of simulation steps.
        n: Number of particles.
        size: Size of environment in each direction.
        n_cells: Number of uniform grid cells in each direction.
        dt: Timestep size.
        seed: Random seed.
        plot: Whether to plot or render. If True, plot.
    """
    n = int(jnp.sqrt(n))
    n = n * n

    cell_size = (size // n_cells) + (size % n_cells > 0)
    n_cells = n_cells + 2  # Add outer padding to grid
    min_radii = 1
    max_per_cell = int((cell_size ** 2) // (jnp.pi * min_radii ** 2) + 1) * 1
    ids, pos, vel = _init_particles(n, seed, size)

    all_pos = [pos]
    print("Starting simulation...")
    for _ in tqdm(range(steps)):
        pos, vel = _step(n, size, n_cells, cell_size, max_per_cell, ids, pos, vel, dt)
        all_pos.append(pos)
    if plot:
        graph(size, all_pos)
    else:
        render(size, all_pos)
