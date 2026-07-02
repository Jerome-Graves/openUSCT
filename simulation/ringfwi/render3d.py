"""3D rendering of sound-speed volumes as isosurfaces.

Turns a 3D velocity model (true or reconstructed) into surface renders: the
specimen boundary as a translucent shell and any flaw as a solid blob. Provides
a static matplotlib figure (for reports) and an interactive Plotly figure (for
the GUI). Isosurfaces are extracted with marching cubes.

A flaw is a low-speed region inside the specimen, but the couplant outside the
specimen is also low speed. To isolate the flaw, the couplant is filled up to
the specimen speed before extracting the flaw isosurface, so the surface wraps
only the flaw.
"""

from __future__ import annotations

import numpy as np


def _mesh(vol, level, h):
    from skimage.measure import marching_cubes

    if not (vol.min() < level < vol.max()):
        return None
    verts, faces, _, _ = marching_cubes(vol, level=level, spacing=(h, h, h))
    return verts, faces


def surfaces(vol, h, c_couplant, c_specimen, c_flaw=None):
    """Return [(verts, faces, colour, opacity, name)] for a velocity volume."""
    out = []
    lvl_spec = (c_couplant + c_specimen) / 2.0
    m = _mesh(vol, lvl_spec, h)
    if m is not None:
        out.append((m[0], m[1], "#8fbf8f", 0.15, "specimen"))

    if c_flaw is not None and abs(c_flaw - c_specimen) > 50:
        # Fill the couplant (everything at or below the specimen threshold) up to
        # the specimen speed, so only the flaw remains as a low/high anomaly.
        filled = np.where(vol <= lvl_spec, c_specimen, vol)
        lvl_flaw = (c_flaw + c_specimen) / 2.0
        colour = "#3060c0" if c_flaw < c_specimen else "#c03030"
        m = _mesh(filled, lvl_flaw, h)
        if m is not None:
            out.append((m[0], m[1], colour, 0.92, "flaw"))
    return out


def add_isosurfaces(ax, surfs, vol_shape, h, title=""):
    """Add precomputed isosurface meshes to a matplotlib 3D axis."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    for verts, faces, colour, alpha, _name in surfs:
        xyz = verts[:, ::-1] * 1e3          # (z,y,x) -> (x,y,z), metres -> mm
        tri = Poly3DCollection(xyz[faces], alpha=alpha)
        tri.set_facecolor(colour)
        tri.set_edgecolor("none")
        ax.add_collection3d(tri)
    span = np.array(vol_shape) * h * 1e3
    ax.set_xlim(0, span[2]); ax.set_ylim(0, span[1]); ax.set_zlim(0, span[0])
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.set_zlabel("z (mm)")
    ax.set_box_aspect(span[::-1])
    ax.set_title(title)


def matplotlib_render(vol, h, c_couplant, c_specimen, c_flaw=None, title=""):
    """Static 3D isosurface figure (matplotlib)."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(4.8, 4.8))
    ax = fig.add_subplot(111, projection="3d")
    add_isosurfaces(ax, surfaces(vol, h, c_couplant, c_specimen, c_flaw), vol.shape, h, title)
    return fig


def add_elements_plotly(fig, positions_m):
    """Overlay transducer element positions (absolute (x, y, z) metres) as points."""
    import plotly.graph_objects as go

    p = np.asarray(positions_m, float) * 1e3
    fig.add_trace(go.Scatter3d(
        x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers",
        marker=dict(size=3.2, color="#00e5ff", symbol="circle"), name="elements"))
    return fig


def array3d_figure(geom, el_shape="point", width_m=0.0, height_m=None, title=""):
    """Interactive 3D render of the array and its transducer elements (Plotly).

    Draws the array surface (the ring circle, or one circle per ring of a
    cylinder) and every element as its physical 3D shape: a rectangular patch
    ``width_m`` x ``height_m`` tangent to the array surface, an elliptical disc,
    or a point marker. Width runs circumferentially, height axially. A 2D ring
    is drawn as its physical 3D counterpart (elements standing on the ring
    plane), since real transducers have elevation even when the model is 2D.
    """
    import plotly.graph_objects as go

    pos = np.asarray(geom.element_positions, float)      # centred coords, metres
    if pos.shape[1] == 2:                                # lift 2D ring into 3D
        pos = np.column_stack([pos, np.zeros(len(pos))])
    if height_m is None:
        height_m = width_m
    pmm = pos * 1e3
    w2, h2 = width_m * 1e3 / 2.0, height_m * 1e3 / 2.0

    fig = go.Figure()
    # Array surface: one circle per distinct ring height.
    r_mm = geom.radius_m * 1e3
    th = np.linspace(0, 2 * np.pi, 121)
    for z0 in np.unique(np.round(pmm[:, 2], 6)):
        fig.add_trace(go.Scatter3d(
            x=r_mm * np.cos(th), y=r_mm * np.sin(th), z=np.full_like(th, z0),
            mode="lines", line=dict(color="#888888", width=2, dash="dash"),
            showlegend=False, hoverinfo="skip"))

    if el_shape == "point" or width_m <= 0:
        fig.add_trace(go.Scatter3d(
            x=pmm[:, 0], y=pmm[:, 1], z=pmm[:, 2], mode="markers",
            marker=dict(size=4, color="#00b8d4"), name="elements"))
    else:
        for k, p in enumerate(pmm):
            nrm = np.array([p[0], p[1], 0.0])
            nn = np.linalg.norm(nrm)
            nrm = nrm / nn if nn > 0 else np.array([1.0, 0.0, 0.0])
            t1 = np.array([-nrm[1], nrm[0], 0.0])        # circumferential
            t2 = np.array([0.0, 0.0, 1.0])               # axial
            if el_shape == "disc":
                ang = np.linspace(0, 2 * np.pi, 20, endpoint=False)
                ring_pts = [p + w2 * np.cos(a) * t1 + h2 * np.sin(a) * t2 for a in ang]
                verts = np.array([p] + ring_pts)
                m = len(ring_pts)
                i = np.zeros(m, int); j = np.arange(1, m + 1); kk = np.roll(j, -1)
            else:                                        # rect
                verts = np.array([p - w2 * t1 - h2 * t2, p + w2 * t1 - h2 * t2,
                                  p + w2 * t1 + h2 * t2, p - w2 * t1 + h2 * t2])
                i = np.array([0, 0]); j = np.array([1, 2]); kk = np.array([2, 3])
            fig.add_trace(go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2], i=i, j=j, k=kk,
                color="#00b8d4", opacity=0.95, flatshading=True,
                name=f"element {k}", showlegend=False))

    span = max(r_mm + max(w2, h2), 1.0)
    fig.update_layout(
        title=title, scene_aspectmode="data",
        scene=dict(xaxis_title="x (mm)", yaxis_title="y (mm)", zaxis_title="z (mm)",
                   xaxis_range=[-1.15 * span, 1.15 * span],
                   yaxis_range=[-1.15 * span, 1.15 * span]),
        margin=dict(l=0, r=0, t=30, b=0), height=430)
    return fig


def grain_colors(grain_values, vmin=0.0, vmax=90.0, cmap_name="twilight"):
    """Map per-grain scalar values to plotly rgb strings via a matplotlib cmap."""
    from matplotlib import colormaps

    cmap = colormaps[cmap_name]
    out = {}
    for k, v in enumerate(np.asarray(grain_values, float)):
        t = (v - vmin) / (vmax - vmin + 1e-30)
        r, g, b, _ = cmap(min(max(t, 0.0), 1.0))
        out[k] = f"rgb({int(255 * r)},{int(255 * g)},{int(255 * b)})"
    return out


def polycrystal_figure(labels, grain_values, h, title="", vmin=0.0, vmax=90.0,
                       cmap_name="twilight", melt_mask=None):
    """Interactive 3D render of a Voronoi polycrystal (Plotly).

    Each grain is extracted with marching cubes and coloured by its scalar
    ``grain_values[k]`` (for example the c-axis colatitude in degrees) mapped
    through a matplotlib colormap. An optional ``melt_mask`` renders a fluid
    pocket as a translucent blue body.
    """
    import plotly.graph_objects as go
    from matplotlib import colormaps
    from skimage.measure import marching_cubes

    colors = grain_colors(grain_values, vmin, vmax, cmap_name)
    fig = go.Figure()
    for k in np.unique(labels[labels >= 0]):
        mask = np.pad((labels == k).astype(float), 1)      # close border surfaces
        try:
            verts, faces, _, _ = marching_cubes(mask, 0.5, spacing=(h, h, h))
        except (ValueError, RuntimeError):
            continue
        verts = verts - h                                   # undo the padding shift
        fig.add_trace(go.Mesh3d(
            x=verts[:, 2] * 1e3, y=verts[:, 1] * 1e3, z=verts[:, 0] * 1e3,
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=colors[int(k)],
            opacity=1.0, flatshading=True, name=f"grain {k}",
            hovertext=f"grain {k}: {float(grain_values[k]):.0f}", showlegend=False))
    if melt_mask is not None and melt_mask.any():
        mask = np.pad(melt_mask.astype(float), 1)
        m = _mesh(mask, 0.5, h)
        if m is not None:
            verts, faces = m
            verts = verts - h
            fig.add_trace(go.Mesh3d(
                x=verts[:, 2] * 1e3, y=verts[:, 1] * 1e3, z=verts[:, 0] * 1e3,
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color="#3060c0", opacity=0.55, flatshading=True, name="fluid pocket",
                showlegend=False))
    fig.update_layout(
        title=title, scene_aspectmode="data",
        scene=dict(xaxis_title="x (mm)", yaxis_title="y (mm)", zaxis_title="z (mm)"),
        margin=dict(l=0, r=0, t=30, b=0), height=460)
    return fig


def plotly_figure(vol, h, c_couplant, c_specimen, c_flaw=None, title=""):
    """Interactive 3D isosurface figure (Plotly)."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for verts, faces, colour, alpha, name in surfaces(vol, h, c_couplant, c_specimen, c_flaw):
        z, y, x = verts[:, 0] * 1e3, verts[:, 1] * 1e3, verts[:, 2] * 1e3
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z, i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=colour, opacity=alpha, name=name, showscale=False, flatshading=True))
    fig.update_layout(
        title=title, scene_aspectmode="data",
        scene=dict(xaxis_title="x (mm)", yaxis_title="y (mm)", zaxis_title="z (mm)"),
        margin=dict(l=0, r=0, t=30, b=0), height=430)
    return fig
