# pvbatch script — render MinTh.vtu design screenshots the way the
# ParaView GUI shows them: color by the Label cell field with the
# default Cool-to-Warm (blue -> red) preset, auto data range, white
# background, no scalar bar.
#
# Usage:  pvbatch.exe pv_render.py jobs.json
#   jobs.json = [["<vtu path>", "<out png path>"], ...]
import json
import sys

from paraview.simple import (
    ColorBy, Delete, GetActiveCamera, GetActiveViewOrCreate,
    GetColorTransferFunction, GetScalarBar, Render, ResetCamera,
    SaveScreenshot, Show, XMLUnstructuredGridReader,
)

jobs = json.load(open(sys.argv[1], encoding="utf-8"))

view = GetActiveViewOrCreate("RenderView")
view.ViewSize = [440, 528]
GetActiveCamera().SetParallelProjection(True)

n = 0
for vtu, out in jobs:
    try:
        r = XMLUnstructuredGridReader(FileName=[vtu])
        disp = Show(r, view)
        disp.Representation = "Surface"
        ColorBy(disp, ("CELLS", "Label"))
        lut = GetColorTransferFunction("Label")
        lut.ApplyPreset("Cool to Warm", True)
        GetScalarBar(lut, view).Visibility = 0
        r.UpdatePipeline()
        # the GUI's auto range: the array's full range over the dataset
        # (triangles + boundary lines combined — lines carry values
        # up to 10)
        dr = (r.GetDataInformation().GetCellDataInformation()
              .GetArrayInformation("Label"))
        if dr is not None:
            rg = dr.GetComponentRange(0)
            if rg[0] is not None and rg[0] != rg[1]:
                lut.RescaleTransferFunction(rg[0], rg[1])
        ResetCamera()
        Render()
        SaveScreenshot(out, view, ImageResolution=[440, 528])
        n += 1
    except Exception as e:
        # ONE corrupt/unreadable vtu must not kill the whole batch —
        # the caller falls back for the missing outputs
        print(f"  [pv] skip {out}: {e}", flush=True)
    finally:
        try:
            Delete(r)
        except Exception:
            pass
        try:
            del r
        except Exception:
            pass
    if n % 25 == 0:
        print(f"  [pv] rendered {n}/{len(jobs)}", flush=True)

print(f"[pv] done: {n} images", flush=True)
