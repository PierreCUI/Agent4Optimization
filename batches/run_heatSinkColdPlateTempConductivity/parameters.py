"""parameters.py — Cartesian sweep families for run_heatSinkColdPlateTempConductivity."""

CASE = 'run_heatSinkColdPlateTempConductivity'

# Sweep grid: hmin x hmax x diffusion x v0 x mesh_n x p_degree
# (3^6 = 729 configs).  Values centred on the current main.py /
# mesh.txt values (one smaller / current / one larger).
HMIN = [0.003, 0.005, 0.008]
HMAX = [0.04, 0.05, 0.07]
DIFFUSION = [0.1, 0.2, 0.4]
V0 = [5e-6, 1e-5, 2e-5]
MESH_N = [30, 50, 75]
P_DEGREE = ['P1', 'P2', 'P3']
