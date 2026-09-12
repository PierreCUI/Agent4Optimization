from autofreefem import *
from autofreefem.units import *

name = 'heat sink cold plate temperature conductivity'

Th = Domain('D', 'Th')
u = ScalarField('u', 'P2', Th, '0.', [4], kelvin)
testu = ScalarField(r'\delta u', 'P2', Th)

k0 = Constant('k_0', 200, 'thermal conductivity (aluminium, at u_amb)', Watt / meter / kelvin)
kair = Constant('k_{air}', 0.026, 'thermal conductivity (air)', Watt / meter / kelvin)
beta = Constant(r'\beta', 5e-4, 'linear temperature coefficient of conductivity', 1 / kelvin)
h = Constant('h', 25, 'convective heat transfer coefficient (air)', Watt / meter ** 2 / kelvin)
u_amb = Constant('u_{amb}', 298, 'ambient air temperature', kelvin)
q = Constant('q', 1000.0, 'heat flux (chip)', Watt / meter ** 2)
Afac = Constant(r'\ell', 2000, 'volume weight (initial estimate, calibrated by pipeline)', Watt * kelvin / meter ** 3)

X = CharacteristicFunction(r'\raisebox{\depth}{$\chi$}', 1, '0.00013')

# temperature-dependent conductivity k(u) = k0 * (1 + beta*(u - u_amb))
# — the u-dependence makes the state problem NONLINEAR; the Newton
# linearisation is derived automatically
kofu = Expression('k(u)', k0 * (1 + beta * (u - u_amb)))

J = Expression('J', dx(Afac * X, Th) + dsx(q * u, Th, 2))

lag = Lagrangian([u], [testu], J,
    dx(X * kofu * inner(grad(u), grad(testu)), Th)
    + dsx(h * u * testu, Th, 3)
    - dsx(q * testu, Th, 2)
    - dsx(h * u_amb * testu, Th, 3),
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
