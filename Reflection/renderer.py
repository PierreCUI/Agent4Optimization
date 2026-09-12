"""
Render MinTh.vtu fields as PNG images for visual inspection.

The VTU file (VTK XML Unstructured Grid) is read via meshio.  Every scalar
field in point_data and cell_data is rendered as a matplotlib triangulation
plot.  PNGs are saved under results/<name>/ (one level above the VTU).

Additionally, ``render_iteration_labels_composite`` merges the Label
field of every iteration snapshot ``result*Th<N>.vtu`` (N numeric —
MinTh is excluded by the pattern) into one composite PNG, in iteration
order, for the multimodal reflection loop.
"""

import re
from pathlib import Path
from typing import Optional

# The classifier owns this constant (dependency-free module); the
# renderer imports it back so a rewording can never silently disable
# the classifier's hard filter.  meshio is imported inside the render
# functions — this module must stay importable in environments without
# it (the deterministic reflection logic does not render).
from .classifier import MINTH_RENDER_SUMMARY_HEADER


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
    # savefig cannot create directories — ensure the target exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- read mesh ---
    import meshio  # noqa: E402  (deferred — keeps this module importable without it)

    try:
        mesh = meshio.read(str(vtu_path))
    except Exception as e:
        return f"[renderer] Failed to read {vtu_path.name}: {e}"

    points = mesh.points  # (n_points, 2) or (n_points, 3)
    if points is None or points.shape[0] < 3:
        return f"[renderer] {vtu_path.name}: empty or degenerate mesh"

    # FreeFem's savevtk writes 3D coordinates (x, y, 0) even for 2D
    # meshes — a CONSTANT z is still a 2D problem.  The summary's "dim:"
    # field reaches the LLM judgement context, so a flat z must not be
    # reported as "3D".
    is_3d = points.shape[1] == 3 and float(np.ptp(points[:, 2])) > 1e-9

    # --- find triangle cells, or extract surface from tetra for 3D ---
    tri_cells = None
    tri_indices = []  # which cell blocks are triangles
    tetra_count = 0  # number of source tetrahedra (for cell-field tiling)
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
            tetra_count = tet_data.shape[0]
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
                tetra_count = td.shape[0]
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

    # 3D cell data lives per-tetra (n values); rendering draws per-FACE
    # (4 faces per tetra) — repeat each cell value onto its 4 faces so
    # the face count always matches the triangulation.
    if tetra_count > 0:
        for name in list(fields):
            if name.startswith("cell/") and len(fields[name]) == tetra_count:
                fields[name] = np.repeat(fields[name], 4)

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

        fig = None
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
                    f"  {png_path.name}: SKIP — size mismatch "
                    f"(values={len(values)}, points={len(points)}, "
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
            # A savefig/field error must not leak the figure (matplotlib
            # keeps it alive, and reflection renders up to hundreds).
            if fig is not None:
                plt.close(fig)
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


def render_iteration_labels_composite(
    results_dir: Path, output_dir: Path, max_frames: int = 16
) -> tuple[Optional[Path], str]:
    """
    Merge the Label field of every iteration snapshot into ONE image.

    Finds all ``result*Th<N>.vtu`` files under *results_dir* (N a
    NUMBER — ``*MinTh.vtu`` never matches because N is required, so the
    final design is excluded), sorts them by N, renders each snapshot's
    ``Label`` cell field as a small quad and stacks them into a single
    composite PNG saved to *output_dir*.

    Returns ``(composite_path, summary_text)``; ``(None, "")`` when no
    iteration snapshots exist.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
    import matplotlib.tri as tri  # noqa: E402
    import numpy as np  # noqa: E402

    snapshots = []
    for p in sorted(results_dir.glob("result*Th*.vtu")):
        m = re.search(r"Th(\d+)\.vtu$", p.name)
        if m:
            snapshots.append((int(m.group(1)), p))
    if not snapshots:
        return None, ""
    snapshots.sort(key=lambda x: x[0])
    if len(snapshots) > max_frames:
        snapshots = snapshots[-max_frames:]  # keep the most recent frames

    n = len(snapshots)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.4 * rows))
    axes = np.atleast_1d(axes).ravel()

    import meshio  # noqa: E402  (deferred — keeps this module importable without it)

    for ax, (idx, vtu_path) in zip(axes, snapshots):
        try:
            mesh = meshio.read(str(vtu_path))
            points = mesh.points
            if points.ndim != 2 or points.shape[1] < 2:
                raise ValueError("unexpected point array")
            tri_cells = None
            tetra_count = 0
            for cb in mesh.cells:
                if cb.type in ("triangle", "triangle6"):
                    tri_cells = cb.data
                    break
                if cb.type in ("tetra", "tetra10"):
                    td = cb.data
                    tetra_count = td.shape[0]
                    tri_cells = np.vstack(
                        [
                            td[:, [0, 1, 2]],
                            td[:, [0, 1, 3]],
                            td[:, [0, 2, 3]],
                            td[:, [1, 2, 3]],
                        ]
                    )
                    break
            if tri_cells is None:
                raise ValueError("no triangle cells")
            # Label field: cell data first (Name 'Label'/'label'), else point data
            values = None
            for name in ("Label", "label"):
                if name in mesh.cell_data:
                    vals = mesh.cell_data[name]
                    if isinstance(vals, (list, tuple)):
                        vals = vals[0]
                    values = np.asarray(vals).ravel()
                    break
            if values is None:
                raise ValueError("no Label field")
            if len(values) != tri_cells.shape[0]:
                if tetra_count > 0 and len(values) == tetra_count:
                    # per-tetra label → per-face (4 faces per tetra)
                    values = np.repeat(values, 4)
                else:
                    # cell_data may cover only one cell block — pad by
                    # assigning the first matching-length block
                    values = values[: tri_cells.shape[0]]
                if len(values) != tri_cells.shape[0]:
                    raise ValueError("label length mismatch")
            ax.tripcolor(
                tri.Triangulation(points[:, 0], points[:, 1], tri_cells),
                facecolors=values,
                shading="flat",
                cmap="viridis",
            )
        except Exception as e:
            ax.text(0.5, 0.5, f"Th{idx}: render failed\n{e}", ha="center", va="center",
                    fontsize=7, transform=ax.transAxes)
        ax.set_aspect("equal")
        ax.set_title(f"iter {idx}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    # hide unused axes
    for ax in axes[n:]:
        ax.axis("off")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "iteration_labels_composite.png"
    fig.suptitle(
        f"Design snapshots (Label) — iterations {snapshots[0][0]}..{snapshots[-1][0]}, "
        f"{n} frame(s), in order",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)

    summary = (
        f"COMPOSITE: {n} design snapshots (Label field) of iterations "
        f"{snapshots[0][0]}..{snapshots[-1][0]} merged in order into "
        f"{out_path.name} — the model sees it as part of this message; "
        f"use it to judge the design evolution."
    )
    return out_path, summary
