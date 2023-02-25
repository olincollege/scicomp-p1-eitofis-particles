import jax.numpy as jnp
from jax import random

class Simulation:
    def __init__(self, n, size=1024, n_cells=2, seed=42):
        self.seed = seed
        self.size = size
        self.n_cells = n_cells
        self.cell_size = (size // n_cells) + (size % n_cells > 0)
        self._init_particles(n, seed)

    def _init_particles(self, n, seed):
        self.ids = jnp.arange(0, n)

        key = random.PRNGKey(seed)
        key, rng = random.split(key)
        self.positions = random.uniform(rng, (2, n), maxval=self.size)
        _, rng = random.split(key)
        self.velocities = random.uniform(rng, (2, n), maxval=1)

    def run(self, n):
        for _ in range(n):
            self.step()

    def step(self):
        self._get_collisions()
        self._update_velocities()
        self._move()

    def _build_grid(self):
        grid_positions = (self.positions // self.cell_size).astype(jnp.int32)
        cell_ids = grid_positions[0] + grid_positions[1] * self.n_cells

        cell_particle_ids = jnp.stack([cell_ids, self.ids], axis=-1)
        sort_idxs = cell_particle_ids[:, 0].argsort()
        cell_particle_ids = cell_particle_ids[sort_idxs]
        cell_particle_idxs = jnp.arange(cell_particle_ids.shape[0])

        is_cell_change = cell_particle_ids[:, 0][1:] != cell_particle_ids[:, 0][:-1]

        cell_start_ids = cell_particle_ids[1:, 0][is_cell_change]
        cell_start_idxs = cell_particle_idxs[1:][is_cell_change]
        grid_starts = jnp.zeros(self.n_cells ** 2, dtype=jnp.int32)
        grid_starts = grid_starts.at[cell_start_ids].set(cell_start_idxs)

        cell_end_ids = cell_particle_ids[:-1, 0][is_cell_change]
        cell_end_idxs = cell_particle_idxs[:-1][is_cell_change]
        grid_ends = jnp.zeros(self.n_cells ** 2, dtype=jnp.int32)
        grid_ends = grid_ends.at[cell_end_ids].set(cell_end_idxs)
        grid_ends = grid_ends.at[cell_end_ids].add(1)
        # Last cell that has start needs end manually set
        grid_ends = grid_ends.at[cell_start_ids[-1]].set(cell_particle_ids.shape[0])

        return cell_particle_ids, grid_starts, grid_ends

    def _get_neighbors(self, cell_particle_ids, grid_starts, grid_ends):
        pass

    def _get_broad_collisions(self):
        cell_particle_ids, grid_starts, grid_ends = self._build_grid()
        self._get_neighbors(cell_particle_ids, grid_starts, grid_ends)

    def _get_narrow_collisions(self):
        pass

    def _get_collisions(self):
        self._get_broad_collisions()
        self._get_narrow_collisions()

    def _get_particle_collision_response(self):
        pass

    def _get_wall_collision_response(self):
        pass

    def _update_velocities(self):
        self._get_particle_collision_response()
        self._get_wall_collision_response()

    def _move(self):
        pass
