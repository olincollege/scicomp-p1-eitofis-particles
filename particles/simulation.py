import time

import jax
import jax.numpy as jnp
from jax.scipy import signal

from graph import graph
from render import render


def _init_particles(n, seed, size):
    ids = jnp.arange(0, n)
    key = jax.random.PRNGKey(seed)
    key, rng = jax.random.split(key)
    positions = jax.random.uniform(rng, (2, n), minval=1, maxval=size - 1)
    _, rng = jax.random.split(key)
    velocities = jax.random.uniform(rng, (2, n), minval=-1, maxval=1)
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
        grid_cell_counts, kernel, mode="same"
    ).astype(jnp.int32)
    max_neighbors = jnp.max(neighbor_counts).item()

    neighbor_counts = jnp.reshape(neighbor_counts, n_cells ** 2)
    particle_counts = neighbor_counts[cell_particle_ids[:, 0]]
    neighbor_mask = jnp.zeros((particle_counts.shape[0], max_neighbors))
    neighbor_mask = neighbor_mask + jnp.arange(0, max_neighbors)
    neighbor_mask = neighbor_mask < particle_counts[:, None]

    def _map_func(cell_id):
        neighbor_cells = jnp.array([
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
            start = grid_starts[neighbor_cells[i]]
            end = grid_ends[neighbor_cells[i]]
            neighbors, idx = jax.lax.fori_loop(start, end, _body_func, (neighbors, idx))
            return neighbors, idx
        neighbors, _ = jax.lax.fori_loop(0, 9, _body_func, (neighbors, idx))

        return neighbors
    neighbors = jax.vmap(_map_func)(cell_particle_ids[:, 0])
    return neighbors, neighbor_mask


def _get_broad_collisions(n_cells, cell_size, ids, pos):
    cell_particle_ids, grid_starts, grid_ends = _build_grid(
        n_cells, cell_size, ids, pos
    )
    neighbors, neighbor_mask = _get_neighbors(
        n_cells, cell_particle_ids, grid_starts, grid_ends
    )
    particle_ids = cell_particle_ids[:, 1]
    return particle_ids, neighbors, neighbor_mask


def _get_narrow_collisions(positions, particle_ids, neighbors, neighbor_mask):
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


def _get_collisions(n_cells, cell_size, ids, pos):
    particle_ids, neighbors, neighbor_mask = _get_broad_collisions(
        n_cells, cell_size, ids, pos
    )
    collisions = _get_narrow_collisions(pos, particle_ids, neighbors, neighbor_mask)
    idxs = jnp.argsort(particle_ids)
    neighbors = neighbors[idxs]
    collisions = collisions[idxs]
    return neighbors, collisions


def _dot(v1, v2):
    return jnp.sum(v1 * v2, axis=0)


def _get_particle_collision_response(
    positions, velocities, particle_ids, neighbors, collisions
):
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
    radii = jnp.ones(positions.shape[1])

    oob_x = ((positions[0, :] - radii) <= 0) | ((positions[0, :] + radii) >= size)
    oob_y = ((positions[1, :] - radii) <= 0) | ((positions[1, :] + radii) >= size)
    oob = jnp.stack([oob_x, oob_y], axis=0)

    inverse_velocities = velocities * -2
    velocity_changes = inverse_velocities * oob
    return velocity_changes


def _update_velocities(size, pos, vel, particle_ids, neighbors, collisions):
    velocity_changes = _get_particle_collision_response(
        pos, vel, particle_ids, neighbors, collisions
    )
    vel = vel + velocity_changes
    velocity_changes = _get_wall_collision_response(size, pos, vel)
    vel = vel + velocity_changes
    return vel


def _move(positions, velocities, dt):
    positions = positions + velocities * dt
    return positions


def _step(size, n_cells, cell_size, ids, pos, vel, dt):
    neighbors, collisions = _get_collisions(n_cells, cell_size, ids, pos)
    vel = _update_velocities(size, pos, vel, ids, neighbors, collisions)
    pos = _move(pos, vel, dt)
    return pos, vel


def run(steps, n, size, n_cells, dt, seed, plot):
    cell_size = (size // n_cells) + (size % n_cells > 0)
    n_cells = n_cells + 2  # Add outer padding to grid
    ids, pos, vel = _init_particles(n, seed, size)

    all_pos = [pos]
    for i in range(steps):
        start = time.time()
        pos, vel = _step(size, n_cells, cell_size, ids, pos, vel, dt)
        all_pos.append(pos)
        end = time.time()
        print(i)
        print(f"Time: {end - start}")

    if plot:
        graph(size, all_pos)
    else:
        render(size, all_pos)
