"""
Render MinTh.vtu fields as PNG images for visual inspection.

The VTU file (VTK XML Unstructured Grid) is read via meshio.  Every scalar
field in point_data and cell_data is rendered as a matplotlib triangulation
plot.  PNGs are saved under results/<name>/ (one level above the VTU).

Additionally, ``render_snapshot_label_png`` renders the Label field of
ONE iteration snapshot ``result*Th<N>.vtu`` (N numeric — MinTh is
excluded by the pattern) as a single PNG; the reflection loop renders
every snapshot individually and passes the images one by one to the
multimodal model (no composite).
"""

from pathlib import Path
from typing import Optional

from configs import (
    RENDER_DPI,
    RENDER_FIGSIZE,
    RENDER_SNAPSHOT_DPI,
    RENDER_SNAPSHOT_FIGSIZE,
)

# meshio is imported inside the render functions — this module must stay
# importable in environments without it (the deterministic reflection
# logic does not render).  MINTH_RENDER_SUMMARY_HEADER lives in
# classifier.py; the reflector (not the renderer) imports it to embed a
# render summary into the context text the classifier then hard-filters on.

# z-tolerance for the flat-2D test (FreeFEM writes 3D coordinates with a
# constant z even for 2D meshes)
_PLANAR_Z_EPS = 1e-9


def render_vtu_to_png(
    vtu_path: Path, output_dir: Optional[Path] = None, max_fields: int = None
) -> str:
    """
    Render every scalar field in a VTU file as a PNG image.

    Args:
        vtu_path: Path to the .vtu file.
        output_dir: Directory for PNG output (default: parent of the
                    ``results/`` subdirectory containing the VTU).
        max_fields: Max number of fields to render (default:
                    ``configs.RENDER_MAX_FIELDS``).

    Returns:
        A text summary describing the rendered fields and their
        statistics, suitable for inclusion in the LLM context.
    """
    if max_fields is None:
        from configs import RENDER_MAX_FIELDS

        max_fields = RENDER_MAX_FIELDS
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
    import meshio  # deferred — keeps this module importable without it

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
    is_3d = points.shape[1] == 3 and float(np.ptp(points[:, 2])) > _PLANAR_Z_EPS

    # --- collect triangle cells ---
    tri_cells = None
    tri_indices = []  # which cell blocks are triangles
    for i, cell_block in enumerate(mesh.cells):
        if cell_block.type in ("triangle", "triangle6"):
            tri_indices.append(i)
            if tri_cells is None:
                tri_cells = cell_block.data
            else:
                tri_cells = np.vstack([tri_cells, cell_block.data])

    if tri_cells is None:
        return f"[renderer] {vtu_path.name}: no triangle cells found"

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

        fig = None
        try:
            fig, ax = plt.subplots(figsize=RENDER_FIGSIZE)
            vmin, vmax = float(values.min()), float(values.max())
            # a uniform field (vmin == vmax) degenerates matplotlib's
            # normalization to the colormap's lowest color — widen the
            # range so the true value's color is drawn instead
            if vmin == vmax:
                vmin, vmax = vmin - 0.5, vmax + 0.5
            if field_name.startswith("point/") and len(values) == len(points):
                triang = tri.Triangulation(x, y, tri_cells)
                tpc = ax.tripcolor(triang, values, shading="flat",
                                   cmap="viridis", vmin=vmin, vmax=vmax)
            elif field_name.startswith("cell/") and len(values) == tri_cells.shape[0]:
                tpc = ax.tripcolor(
                    tri.Triangulation(x, y, tri_cells),
                    facecolors=values,
                    shading="flat",
                    cmap="viridis",
                    vmin=vmin,
                    vmax=vmax,
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

            fig.savefig(png_path, dpi=RENDER_DPI, bbox_inches="tight")
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


# ======================================================================
# ParaView (pvbatch) rendering — the GUI look: Label field with the
# default Cool-to-Warm preset (blue interior, red boundary lines)
# ======================================================================

def find_pvbatch() -> str:
    """Path of the pvbatch executable, or '' when ParaView is absent."""
    import glob
    import os
    import shutil

    cand = os.environ.get("PARAVIEW_BATCH", "")
    if cand and os.path.exists(cand):
        return cand
    # the GUI is a Windows install under C:/Program Files/ParaView <ver>/
    if os.name == "nt":
        hits = sorted(glob.glob("C:/Program Files/ParaView*/bin/pvbatch.exe"))
        if hits:
            return hits[-1]  # newest version
    w = shutil.which("pvbatch")
    return w or ""


def render_labels_paraview_batch(jobs: list) -> list:
    """Render (vtu_path, out_png) pairs with ONE pvbatch invocation —
    the ParaView GUI look (color by Label, Cool to Warm).  Returns the
    list of job indices whose output was NOT produced (the caller
    falls back to the matplotlib renderer for those)."""
    if not jobs:
        return []
    pvbatch = find_pvbatch()
    if not pvbatch:
        return list(range(len(jobs)))
    import json
    import os
    import subprocess
    import tempfile

    script = Path(__file__).with_name("pv_render.py")
    if not script.exists():
        return list(range(len(jobs)))
    with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False,
            encoding="utf-8") as fh:
        json.dump([[str(v), str(o)] for v, o in jobs], fh)
        jobs_file = fh.name
    try:
        proc = subprocess.run(
            [pvbatch, str(script), jobs_file],
            capture_output=True, text=True, timeout=3600,
            errors="replace")
    except Exception:
        proc = None
    finally:
        try:
            os.unlink(jobs_file)
        except OSError:
            pass
    failed = [i for i, (_, out) in enumerate(jobs)
              if not Path(out).exists()]
    if failed and proc is not None:
        print(f"[pv] pvbatch rc={proc.returncode}: "
              f"{proc.stderr.strip()[-400:]}", flush=True)
    return failed


def render_snapshot_label_png(vtu_path: Path, out_path: Path) -> str:
    """Render the Label field of ONE design snapshot as a PNG.

    The Label cell field (solid/void indicator; Label=3 = solid) of
    *vtu_path* is rendered as a tripcolor quad saved to *out_path*.
    Raises on unrenderable input (missing Label field, no triangle
    cells) — the caller decides how to degrade.  Returns the output
    path as a string.

    The reflection loop renders every iteration snapshot INDIVIDUALLY
    and passes the images one by one to the multimodal model (no
    composite) — the model reads each epoch's design at full size.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri
    import numpy as np

    import meshio  # deferred — keeps this module importable without it

    mesh = meshio.read(str(vtu_path))
    points = mesh.points
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("unexpected point array")
    tri_cells = None
    for cb in mesh.cells:
        if cb.type in ("triangle", "triangle6"):
            tri_cells = cb.data
            break
    if tri_cells is None:
        raise ValueError("no triangle cells")
    # Label field: cell data (Name 'Label'/'label'), else point data
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
        # cell_data may cover only one cell block — pad by assigning the
        # first matching-length block
        values = values[: tri_cells.shape[0]]
        if len(values) != tri_cells.shape[0]:
            raise ValueError("label length mismatch")

    fig, ax = plt.subplots(figsize=RENDER_SNAPSHOT_FIGSIZE)
    ax.tripcolor(
        tri.Triangulation(points[:, 0], points[:, 1], tri_cells),
        facecolors=values,
        shading="flat",
        cmap="viridis",
        # Label is a discrete indicator (0=void .. 3=solid) — a FIXED
        # normalization keeps a uniform design (all 0 or all 3) in its
        # true color instead of collapsing to the colormap's lowest
        # color (matplotlib's degenerate-norm fallback)
        norm=plt.Normalize(vmin=0, vmax=3),
    )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=RENDER_SNAPSHOT_DPI)
    plt.close(fig)
    return str(out_path)
