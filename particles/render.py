import os

import moderngl_window as mglw
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm
import cv2
from PIL import Image

from simulation import step, init_simulation


class Renderer(mglw.WindowConfig):
    gl_version = (3, 3)
    window_size = (1024, 1024)
    aspect_ratio = 1

    def __init__(self, **kwargs):
        """Initialize the simulation.

        Args:
            n: Number of particles.
            size: Size of environment in each direction.
            n_cells: Number of uniform grid cells in each direction.
            dt: Timestep size.
            seed: Random seed.
            plot: Whether to plot or render. If True, plot.
        """
        super().__init__(**kwargs)

        steps, n, size, n_cells, dt, seed, save, mps = Renderer._init_args
        self.steps = steps
        self.n = int(jnp.sqrt(n)) ** 2
        self.size = size
        self.n_cells = n_cells
        self.dt = dt
        self.seed = seed
        self.max_per_cell = mps
        self.save = save

        self._init_renderer()
        self._init_sim()
        self._init_video_writer()

        self.step = 0


    def _init_renderer(self):
        self.program = self._make_program()
        self.vbo, self.ibo, self.cbo, self.sbo = self._make_buffers()
        self.vao = self.ctx.vertex_array(
            self.program,
            [
                (self.vbo, '2f', 'in_vert'),
                (self.cbo, '2f/i', 'in_pos'),
                (self.sbo, '1f/i', 'in_vel'),
            ],
            index_buffer=self.ibo,
            index_element_size=4,
        )

    def _init_sim(self):
        cell_size, n_cells, max_per_cell, ids, pos, vel = init_simulation(
            self.n, self.size, self.n_cells, self.seed, self.max_per_cell
        )
        self.cell_size = cell_size
        self.n_cells = n_cells
        self.max_per_cell = max_per_cell
        self.ids = ids
        self.pos = pos
        self.vel = vel
        self.initial_pos = pos

    def _init_video_writer(self):
        if self.save is None:
            self.fbo = self.wnd.fbo
            self.vw = None
            return
        assert self.steps is not None, "Cannot save video without steps specified!"
        os.makedirs("data", exist_ok=True)
        fp = os.path.join("data", self.save)
        fps = 60.0
        fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        self.vw = cv2.VideoWriter(fp, fourcc, fps, self.window_size)
        self.fbo = self.ctx.simple_framebuffer(self.window_size)

    def _render(self):
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.pos, self.vel = step(
            self.n,
            self.size,
            self.n_cells,
            self.cell_size,
            self.max_per_cell,
            self.ids,
            self.pos,
            self.vel,
            self.dt,
        )
        pos = jnp.transpose(self.pos).flatten() - (self.size / 2)
        self.cbo.write(np.array(pos).astype("f4"))
        vel = jnp.linalg.norm(self.vel, axis=0) / 1
        self.sbo.write(np.array(vel).astype("f4"))
        self.vao.render(instances=self.n)
        return pos, vel

    def render(self, *_):
        if self.step == 0:
            print("\nRunning simulation...")
            self.pbar = tqdm(total=self.steps)

        if self.vw:
            self.fbo.use()

        pos, vel = self._render()

        # print(f"Total velocity: {jnp.sum(jnp.linalg.norm(vel))}")

        if self.vw is not None:
            raw = self.fbo.read()
            img = Image.frombytes(
                'RGB', self.fbo.size, raw, 'raw', 'RGB'
            )
            self.vw.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))

        if self.steps is not None:
            if self.step >= self.steps:
                if self.vw is not None:
                    self.vw.release()
                self.pbar.close()
                self.wnd.destroy()
                exit()

        self.step += 1
        self.pbar.update(1)

    def _make_program(self):
        program = self.ctx.program(
            vertex_shader="""
                #version 330

                in vec2 in_vert;
                in vec2 in_pos;
                in float in_vel;

                uniform float scale;

                out vec2 pos;
                out float vel;
                out float limit;

                void main() {
                    gl_Position = vec4((in_vert + in_pos) / scale, 0.0, 1.0);
                    pos = in_vert;
                    vel = in_vel;
                    limit = 1;
                }
            """,
            fragment_shader="""
                #version 330

                in vec2 pos;
                in float vel;
                in float limit;

                out vec3 f_color;

                void main() {
                    float rsq = dot(pos, pos);
                    if (rsq > limit)
                        discard;
                    float _vel = clamp(vel, 0.0, 1);
                    f_color = vec3(_vel, 0.0, 1 - _vel);
                }
            """,
        )
        program["scale"].value = (self.size / 2) + 1
        return program

    def _make_buffers(self):
        vertices = np.array([
            1.0, 1.0,
            1.0, -1.0,
            -1.0, 1.0,
            -1.0, -1.0,
        ], dtype="f4")
        vbo = self.ctx.buffer(vertices)

        indicies = np.array([
            0, 1, 3,
            0, 2, 3,
        ], dtype="i4")
        ibo = self.ctx.buffer(indicies)

        cbo = self.ctx.buffer(reserve=(self.n * 2 * 4))
        sbo = self.ctx.buffer(reserve=(self.n * 4))

        return vbo, ibo, cbo, sbo
