class Simulation:
    def __init__(self):
        pass

    def step(self, n):
        for _ in range(n):
            self._get_collisions()
            self._update_velocities()
            self._move()

    def _build_grid(self):
        pass

    def _get_neighbors(self):
        pass

    def _get_broad_collisions(self):
        self._build_grid()
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
