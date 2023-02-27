import moderngl
import numpy as np
import jax.numpy as jnp

from PIL import Image


def _make_program(ctx, size):
    program = ctx.program(
        vertex_shader='''
            #version 330

            in vec2 in_vert;
            in vec2 in_pos;

            uniform float scale;

            out vec2 pos;
            out float limit;

            void main() {
                gl_Position = vec4((in_vert + in_pos) / scale, 0.0, 1.0);
                pos = in_vert;
                limit = 1;
            }
        ''',
        fragment_shader='''
            #version 330

            in vec2 pos;
            in float limit;

            out vec3 f_color;

            void main() {
                float rsq = dot(pos, pos);
                if (rsq > limit)
                    discard;
                f_color = vec3(1.0, 1.0, 1.0);
            }
        ''',
    )
    program['scale'].value = (size / 2) + 1
    return program


def _make_buffers(ctx, n):
    vertices = np.array([
        1.0, 1.0,
        1.0, -1.0,
        -1.0, 1.0,
        -1.0, -1.0,
    ], dtype="f4")
    vbo = ctx.buffer(vertices)

    indicies = np.array([
        0, 1, 3,
        0, 2, 3,
    ], dtype="i4")
    ibo = ctx.buffer(indicies)

    cbo = ctx.buffer(reserve=(n * 2 * 4))

    return vbo, ibo, cbo


def _make_images(ctx, cbo, vao, n, size, all_pos):
    fbo = ctx.simple_framebuffer((1024, 1024))
    fbo.use()

    images = []
    for pos in all_pos:
        pos = jnp.transpose(pos).flatten() - (size / 2)
        cbo.write(np.array(pos).astype("f4"))
        fbo.clear(0.0, 0.0, 0.0, 1.0)
        vao.render(instances=n)
        img = Image.frombytes('RGB', fbo.size, fbo.read(), 'raw', 'RGB', 0, -1)
        images.append(img)
    return images


def _make_gif(images):
    images[0].save(
        'simulation.gif',
        save_all=True,
        optimize=False,
        append_images=images[1:],
        loop=0,
    )


def render(size, all_pos):
    print("Starting render...")
    n = len(all_pos[0][0])

    ctx = moderngl.create_standalone_context()
    program = _make_program(ctx, size)
    vbo, ibo, cbo = _make_buffers(ctx, n)
    vao = ctx.vertex_array(
        program,
        [
            (vbo, '2f', 'in_vert'),
            (cbo, '2f/i', 'in_pos'),
        ],
        index_buffer=ibo,
        index_element_size=4,
    )

    images = _make_images(ctx, cbo, vao, n, size, all_pos)
    _make_gif(images)
