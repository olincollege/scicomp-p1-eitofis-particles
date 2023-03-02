import time
from functools import partial

import jax
import jax.numpy as jnp


def _resolve_wall_movements(size, positions):
    """Compute wall collision resonses.

    Args:
        positions: Array of shape (2, n), particle positions.
        velocities: Array of shape (2, n), particle velocities.

    Returns:
        velocity_changes: Array of shape (2, n), changes in particle velocity
            with position corresponding to particle_ids.
    """
    position_changes_x = jnp.zeros_like(positions[0])
    position_changes_y = jnp.zeros_like(positions[1])

    oob_left_x = (positions[0, :] <= 0)
    oob_right_x = (positions[0, :] >= size)
    position_changes_x = (
        ((-positions[0] + 1) * oob_left_x) +
        ((size - positions[0] - 1) * oob_right_x)
    )

    oob_up_y = (positions[1, :] <= 0)
    oob_down_y = (positions[1, :] >= size)
    position_changes_y = (
        ((-positions[1] + 1) * oob_up_y) +
        ((size - positions[1] - 1) * oob_down_y)
    )

    position_changes = jnp.stack([position_changes_x, position_changes_y], axis=0)
    return positions + position_changes


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
        jnp.linspace(2, size - 2, n_row_col),
        jnp.linspace(2, size - 2, n_row_col),
    )
    positions = jnp.stack((x.flatten(), y.flatten()), axis=0)

    key, rng = jax.random.split(key)
    velocities = jax.random.uniform(rng, (2, n), minval=-0, maxval=0)
    return ids, positions, velocities, key


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


def _get_overlap_movements(positions, particle_ids, neighbors, neighbor_mask):
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
    def _map_func(particle_id, neighbors, mask):
        particle_position = positions[:, particle_id]
        particle_radius = 1
        neighbor_positions = positions[:, neighbors]
        neighbor_radii = jnp.ones_like(neighbors)

        collision_normals = neighbor_positions - particle_position[:, None]
        magnitudes = jnp.linalg.norm(collision_normals, axis=0)
        collision_normals = jnp.nan_to_num(collision_normals / magnitudes)
        tangent_dists = neighbor_radii + particle_radius
        overlap_dists = jnp.clip(tangent_dists - magnitudes, a_min=0) * mask

        position_changes = -(collision_normals * overlap_dists) / 2
        # first_change = (position_changes[0] > 0).argmax()
        # position_change = position_changes[:, first_change]
        position_change = jnp.sum(position_changes, axis=-1)

        return position_change

    position_changes = jax.vmap(_map_func)(particle_ids, neighbors, neighbor_mask)
    position_changes = jnp.transpose(position_changes)
    return position_changes


def _resolve_overlap_movements(pos, ids, neighbors, neighbor_mask):
    def _body_func(_, pos):
        position_changes = _get_overlap_movements(pos, ids, neighbors, neighbor_mask)
        return pos + position_changes
    pos = jax.lax.fori_loop(0, 8, _body_func, pos)
    return pos


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

        # Collision Response
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
        velocity_change = jnp.sum(velocity_changes, axis=-1)

        return velocity_change

    velocity_changes = jax.vmap(_map_func)(
        particle_ids, neighbors, collisions
    )
    velocity_changes = jnp.transpose(velocity_changes)
    return velocities + velocity_changes


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

    moving_left_x = velocities[0, :] < 0
    moving_right_x = velocities[0, :] > 0
    oob_left_x = ((positions[0, :] - radii) <= 0)
    oob_right_x = ((positions[0, :] + radii) >= size)
    oob_x = (oob_left_x * moving_left_x) | (oob_right_x * moving_right_x)

    moving_up_y = velocities[1, :] < 0
    moving_down_y = velocities[1, :] > 0
    oob_up_y = ((positions[1, :] - radii) <= 0)
    oob_down_y = ((positions[1, :] + radii) >= size)
    oob_y = (oob_up_y & moving_up_y) | (oob_down_y & moving_down_y)

    oob = jnp.stack([oob_x, oob_y], axis=0)
    inverse_velocities = velocities * -2
    velocity_changes = inverse_velocities * oob

    return velocities + velocity_changes


def _get_brownian_motion(velocities, key):
    key, rng = jax.random.split(key)
    sigma = 0.25
    mu = 0
    velocity_changes = jax.random.normal(rng, velocities.shape)
    velocity_changes = velocity_changes * sigma + mu
    return velocities + velocity_changes, key


def _move(positions, velocities, dt):
    """Move particles."""
    positions = positions + velocities * dt
    return positions


@partial(jax.jit, static_argnames=['n', 'size', 'n_cells', 'cell_size', 'max_per_cell'])
def step(n, size, n_cells, cell_size, max_per_cell, ids, pos, vel, dt, key):
    """Take single step of the simulation."""
    pos = _resolve_wall_movements(size, pos)
    cell_particle_ids, grid = _build_grid(n, n_cells, cell_size, max_per_cell, ids, pos)
    neighbors, neighbor_mask = _get_neighbors(n_cells, cell_particle_ids, grid)
    pos = _resolve_overlap_movements(pos, ids, neighbors, neighbor_mask)
    collisions = _get_narrow_collisions(ids, pos, neighbors, neighbor_mask)
    vel = _get_particle_collision_response(pos, vel, ids, neighbors, collisions)
    vel = _get_wall_collision_response(size, pos, vel)
    vel, key = _get_brownian_motion(vel, key)
    vel = vel * 0.9
    pos = _move(pos, vel, dt)
    return pos, vel, key


def init_simulation(n, size, n_cells, seed):
    cell_size = (size // n_cells) + (size % n_cells > 0)
    n_cells = n_cells + 2  # Add outer padding to grid
    min_radii = 1
    max_per_cell = int((cell_size ** 2) // (jnp.pi * min_radii ** 2) + 1) * 1
    assert (max_per_cell * n_cells ** 2) > n, "Not enough cells!"
    ids, pos, vel, key = _init_particles(n, seed, size)
    return cell_size, n_cells, max_per_cell, ids, pos, vel, key
