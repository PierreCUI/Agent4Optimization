from autofreefem import *
from autofreefem.units import *

name = 'heat sink water wall'

Th = Domain('D', 'Th')
u = ScalarField('u', 'P2', Th, '0.', [4], kelvin)
testu = ScalarField(r'\delta u', 'P2', Th)

k0 = Constant('k_0', 200, 'thermal conductivity (aluminium)', Watt / meter / kelvin)
h = Constant('h', 25, 'convective heat transfer coefficient (air)', Watt / meter ** 2 / kelvin)
u_amb = Constant('u_{amb}', 298, 'ambient air temperature', kelvin)
hH = Constant('h_H', 25, 'convective heat transfer coefficient (hot-fluid side)', Watt / meter ** 2 / kelvin)
uH = Constant('u_H', 338, 'hot-fluid temperature', kelvin)
Afac = Constant(r'\ell', 100, 'Lagrange multiplier', Watt * kelvin / meter ** 3)

X = CharacteristicFunction(r'\raisebox{\depth}{$\chi$}', 1, '0.00013')

# objective: the Robin load enters through the equivalent surface
# heat-flux weight qb = hH * (uH - u_amb) = 25 * 40 = 1000 W/m^2, the
# same numerical weight as the Neumann flux q = 1000 W/m^2 of the
# benchmark cold-plate case
qb = Expression('q_b', hH * (uH - u_amb))
J = Expression('J', dx(Afac * X, Th) + dsx(qb * u, Th, 2))

lag = Lagrangian([u], [testu], J,
    dx(k0 * X * inner(grad(u), grad(testu)), Th)
    + dsx(h * (u - u_amb) * testu, Th, 3)
    + dsx(hH * (u - uH) * testu, Th, 2),
    check_units=True)

lag.setup_optimization(name,
    mesh='mesh.txt',
    hmin=0.005,
    hmax=0.05,
    diffusion=0.2,
    v0=1e-05,
    iterations=100,
    phi='min(abs(y-0.9) - 0.1, max(abs(y-0.1) - 0.1, abs(x-0.5) - 0.25), abs(x-0.5) - 0.1)',
    boundaryLabels=['3'],
    fixedLabels=['1', '2', '4'],
    show_weak_material=False)
