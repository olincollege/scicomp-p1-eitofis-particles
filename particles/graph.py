import jax.numpy as jnp
import plotly.graph_objects as go


def _build_base_shape(size):
    return dict(
        type="rect",
        x0=0,
        y0=0,
        x1=size,
        y1=size,
        line=dict(
            color="Black",
            width=2,
        ),
        fillcolor="#fefefe"
    )


def _build_shapes(size, pos):
    shapes = [_build_base_shape(size)]
    for x, y in jnp.transpose(pos):
        r = 1
        shapes.append(
            dict(
                type="circle",
                xref="x",
                yref="y",
                x0=x - r,
                y0=y - r,
                x1=x + r,
                y1=y + r,
                line_color="LightSeaGreen",
                fillcolor="LightSeaGreen",
            )
        )
    return shapes


def _build_frames(size, all_pos):
    frames = [
        go.Frame(layout=dict(shapes=_build_shapes(size, pos)), name=str(i))
        for i, pos in enumerate(all_pos)
    ]
    return frames


def _build_buttons():
    return [dict(
        type="buttons",
        # Buttons
        buttons=[
            dict(
                label="Play",
                method="animate",
                args=[None, dict(
                    frame=dict(duration=10, redraw=False),
                    fromcurrent=True,
                    transition=dict(duration=10, easing="linear"),
                )],
            ),
            dict(
                label="Pause",
                method="animate",
                args=[[None], dict(
                    frame=dict(duration=0, redraw=False),
                    mode="immediate",
                    transition=dict(duration=0),
                )],
            )
        ],
        # Styling
        showactive=False,
        direction="left",
        pad=dict(r=10, t=87),
        x=0.1,
        xanchor="right",
        y=0,
        yanchor="top",
    )]


def _build_sliders(steps):
    slider = dict(
        active=0,
        yanchor="top",
        xanchor="left",
        currentvalue=dict(
            font=dict(size=20),
            prefix="Timestep: ",
            visible=True,
            xanchor="right",
        ),
        transition=dict(duration=10, easing="linear"),
        pad=dict(b=10, t=50),
        len=0.9,
        x=0.1,
        y=0,
        steps=[
            dict(
                args=[
                    [str(i)],
                    dict(
                        frame=dict(duration=10, redraw=False),
                        mode="immediate",
                        transition=dict(duration=10),
                    ),
                ],
                label=str(i),
                method="animate"
            ) for i in steps
        ],
    )
    return [slider]



def _build_layout(size, all_pos):
    return go.Layout(
        shapes=_build_shapes(size, all_pos[0]),
        xaxis=dict(range=[0, size], autorange=False),
        yaxis=dict(range=[0, size], autorange=False),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        updatemenus=_build_buttons(),
        sliders=_build_sliders(range(0, len(all_pos), 1)),
    )


def _build_fig(frames, layout):
    fig = go.Figure(
        layout=layout,
        frames=frames,
    )
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
        visible=False,
    )
    fig.update_xaxes(
        visible=False,
    )
    return fig


def graph(size, all_pos):
    print("Starting plotting...")
    frames = _build_frames(size, all_pos)
    layout = _build_layout(size, all_pos)
    fig = _build_fig(frames, layout)
    fig.show()
