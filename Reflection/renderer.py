"""
Render MinTh.vtu fields as PNG images for visual inspection.

The VTU file (VTK XML Unstructured Grid) is read via meshio.  Every scalar
field in point_data and cell_data is rendered as a matplotlib triangulation
plot.  PNGs are saved under results/<name>/ (one level above the VTU).
"""

from pathlib import Path
from typing import Optional

import meshio


def render_vtu_to_png(
    vtu_path: Path, output_dir: Optional[Path] = None, max_fields: int = 16
) -> str:
    """
    Render every scalar field in a VTU file as a PNG image.

    Args:
        vtu_path: Path to the .vtu file.
        output_dir: Directory for PNG output (default: parent of the
                    ``results/`` subdirectory containing the VTU).
        max_fields: Max number of fields to render.

    Returns:
        A text summary describing the rendered fields and their
        statistics, suitable for inclusion in the LLM context.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri
    import numpy as np

    if output_dir is None:
        # Save under results/<name>/ instead of results/<name>/results/
        output_dir = vtu_path.parent.parent

    # --- read mesh ---
    try:
        mesh = meshio.read(str(vtu_path))
    except Exception as e:
        return f"[renderer] Failed to read {vtu_path.name}: {e}"

    points = mesh.points  # (n_points, 2) or (n_points, 3)
    if points is None or points.shape[0] < 3:
        return f"[renderer] {vtu_path.name}: empty or degenerate mesh"

    is_3d = points.shape[1] == 3

    # --- find triangle cells, or extract surface from tetra for 3D ---
    tri_cells = None
    tri_indices = []  # which cell blocks are triangles
    for i, cell_block in enumerate(mesh.cells):
        if cell_block.type in ("triangle", "triangle6"):
            tri_indices.append(i)
            if tri_cells is None:
                tri_cells = cell_block.data
            else:
                tri_cells = np.vstack([tri_cells, cell_block.data])
        elif cell_block.type in ("tetra", "tetra10") and tri_cells is None:
            # 3D tetra mesh: extract surface faces (every 3-node
            # face of each tetra) as approximate triangles for 2D
            # rendering.  This is a projection, not a true 3D render.
            tet_data = cell_block.data
            # Faces of a tetra: nodes (0,1,2), (0,1,3), (0,2,3), (1,2,3)
            faces = np.vstack(
                [
                    tet_data[:, [0, 1, 2]],
                    tet_data[:, [0, 1, 3]],
                    tet_data[:, [0, 2, 3]],
                    tet_data[:, [1, 2, 3]],
                ]
            )
            tri_cells = faces
            tri_indices.append(i)

    if tri_cells is None:
        if mesh.cells:
            block = mesh.cells[0]
            if block.type in ("tetra", "tetra10"):
                # Last-resort tetra surface extraction
                td = block.data
                tri_cells = np.vstack(
                    [
                        td[:, [0, 1, 2]],
                        td[:, [0, 1, 3]],
                        td[:, [0, 2, 3]],
                        td[:, [1, 2, 3]],
                    ]
                )
            else:
                tri_cells = block.data
            tri_indices = [0]
        else:
            return f"[renderer] {vtu_path.name}: no cells found"

    # --- collect scalar fields ---
    fields = {}

    # Point data
    for name, values in mesh.point_data.items():
        values = np.asarray(values)
        if values.ndim == 1:
            fields[f"point/{name}"] = values
        elif values.ndim == 2:
            if values.shape[1] == 1:
                fields[f"point/{name}"] = values[:, 0]
            elif values.shape[1] <= 3:
                for comp in range(values.shape[1]):
                    fields[f"point/{name}[{comp}]"] = values[:, comp]

    # Cell data — only use triangle blocks so lengths match tri_cells
    for name, values in mesh.cell_data.items():
        if isinstance(values, (list, tuple)):
            parts = []
            for i in tri_indices:
                if i < len(values):
                    parts.append(np.asarray(values[i]).ravel())
            if parts:
                try:
                    values = np.concatenate(parts)
                except Exception:
                    continue
            else:
                continue
        else:
            values = np.asarray(values).ravel()
        if len(values) == 0:
            continue
        fields[f"cell/{name}"] = values

    if not fields:
        return f"[renderer] {vtu_path.name}: no scalar fields found"

    # --- render each field ---
    stem = vtu_path.stem
    rendered = []
    count = 0
    x, y = points[:, 0], points[:, 1]

    for field_name, values in sorted(fields.items()):
        if count >= max_fields:
            break

        safe_name = field_name.replace("/", "_").replace("[", "_").replace("]", "")
        png_path = output_dir / f"{stem}_{safe_name}.png"

        try:
            fig, ax = plt.subplots(figsize=(6, 5))
            if field_name.startswith("point/") and len(values) == len(points):
                triang = tri.Triangulation(x, y, tri_cells)
                tpc = ax.tripcolor(triang, values, shading="flat", cmap="viridis")
            elif field_name.startswith("cell/") and len(values) == tri_cells.shape[0]:
                tpc = ax.tripcolor(
                    tri.Triangulation(x, y, tri_cells),
                    facecolors=values,
                    shading="flat",
                    cmap="viridis",
                )
            else:
                plt.close(fig)
                rendered.append(
                    f"  {png_path.name}: SKIP — size mismatch (values={len(values)}, points={len(points)}, "
                    f"cells={tri_cells.shape[0]})"
                )
                continue

            fig.colorbar(tpc, ax=ax, label=field_name)
            ax.set_aspect("equal")
            ax.set_title(f"{stem} — {field_name}")
            if is_3d:
                ax.set_xlabel("x")
                ax.set_ylabel("y")

            fig.savefig(png_path, dpi=100, bbox_inches="tight")
            plt.close(fig)

            vmin, vmax = float(values.min()), float(values.max())
            rendered.append(
                f"  {png_path.name}: {field_name} (range [{vmin:.4g}, {vmax:.4g}], {len(values)} values)"
            )
            count += 1
        except Exception as e:
            rendered.append(f"  {png_path.name}: render failed — {e}")

    # --- build summary ---
    summary = (
        f"VTU render summary for {vtu_path.name}:\n  points: {points.shape[0]}, cells: {tri_cells.shape[0]}, "
        f"dim: {'3D' if is_3d else '2D'}\n" + "\n".join(rendered)
    )
    return summary


def render_and_summarize(vtu_path: Path) -> str:
    """
    Convenience: render the MinTh.vtu and return a text summary.

    If the VTU does not exist, returns an empty string.
    """
    if not vtu_path.exists():
        return ""
    return render_vtu_to_png(vtu_path)
