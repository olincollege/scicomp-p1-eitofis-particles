import jax.numpy as jnp

class Simulation:
    def __init__(self, n, size=1024, n_cells=16):
        self.grid_size = (size // n_cells) + (size % n_cells > 0)
        self._init_particles(n)

    def _init_particles(self, n):
        self.ids = jnp.arange(0, n)
        self.positions = jnp.zeros((2, n))
        self.velocities = jnp.zeros((2, n))

    def step(self, n):
        for _ in range(n):
            self._get_collisions()
            self._update_velocities()
            self._move()

    def _build_grid(self):
        grid_positions = self.positions // self.grid_size
        cell_ids = grid_positions[0] + grid_positions[1] * self.grid_size

        cell_particle_ids = jnp.stack([cell_ids, self.ids], axis=-1)
        cell_particle_ids = cell_particle_ids.sort(axis=0)
        cell_particle_idxs = jnp.arange(cell_particle_ids.shape[0])

        is_cell_change = cell_particle_ids[:, 0][1:] != cell_particle_ids[:, 0][:-1]

        cell_start_ids = cell_particle_ids[:-1, 0][is_cell_change]
        cell_start_idxs = cell_particle_idxs[:-1][is_cell_change]
        grid_starts = jnp.zeros(self.grid_size ** 2)
        grid_starts[cell_start_ids] = cell_start_idxs

        cell_end_ids = cell_particle_ids[1:, 0][is_cell_change]
        cell_end_idxs = cell_particle_idxs[1:][is_cell_change]
        grid_ends = jnp.zeros(self.grid_size ** 2) - 1
        grid_ends[cell_end_ids] = cell_end_idxs

        return cell_particle_ids, grid_starts, grid_ends

    def _get_neighbors(self):
        pass

    def _get_broad_collisions(self):
        cell_particle_ids, grid_starts, grid_ends = self._build_grid()
        self._get_neighbors()

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
