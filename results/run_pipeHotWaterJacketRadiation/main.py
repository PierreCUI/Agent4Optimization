from autofreefem import *
from autofreefem.units import *

name = 'pipe hot water jacket radiation'

Th = Domain('D', 'Th')

# The inner circle (label 2) is a DESIGN boundary swept by the
# district-heating water (Robin h_H = 1000, u_H = 353 K).  The outer
# circle (label 3) is FIXED: convection h = 25 plus Stefan-Boltzmann
# surface radiation q_rad = eps*sig*(u^4 - u_amb^4) — the radiation
# term makes the problem NONLINEAR; the Newton linearisation is
# derived automatically.
u = ScalarField('u', P2, Th, unit=kelvin)
testu = ScalarField(r'\delta u', P2, Th)

# constants
k0 = Constant('k_0', 200, 'thermal conductivity (aluminium)', Watt / meter / kelvin)
kair = Constant('k_{air}', 0.026, 'thermal conductivity (air)', Watt / meter / kelvin)
h = Constant('h', 25, 'convective heat transfer coefficient (environment)', Watt / meter ** 2 / kelvin)
uamb = Constant('u_{amb}', 298, 'ambient temperature', kelvin)
hH = Constant('h_H', 1000, 'convective heat transfer coefficient (hot-water side)', Watt / meter ** 2 / kelvin)
uH = Constant('u_H', 353, 'district-heating water temperature', kelvin)
eps = Constant(r'\varepsilon', 0.8, 'surface emissivity (oxidised aluminium)', '-')
sig = Constant(r'\sigma', 5.67e-8, 'Stefan-Boltzmann constant', Watt / meter ** 2 / kelvin ** 4)
qb = Expression('q_b', hH * (uH - uamb))
Afac = Constant(r'\ell', 4.64159e+06, 'volume weight (initial estimate, calibrated by pipeline)', Watt * kelvin / meter ** 3)

# characteristic function
X = CharacteristicFunction(r'\raisebox{\depth}{$\chi$}', 1, '0.00013')

# objective: minimise the casing-temperature term qb*(u_amb-u) ->
# maximise the heat rejected through the casing (label 3)
J = Expression('J', dx(Afac * X, Th) + dsx(qb * (uamb - u), Th, 3))

# weak form of the state equation: Robin heating on the design inner
# wall (label 2), Robin convection AND Stefan-Boltzmann radiation on
# the fixed casing (label 3)
weak = (dx(k0 * X * inner(grad(u), grad(testu)), Th)
        + dsx(hH * (u - uH) * testu, Th, 2)
        + dsx(h * (u - uamb) * testu, Th, 3)
        + dsx(eps * sig * (u ** 4 - uamb ** 4) * testu, Th, 3))

lag = Lagrangian([u], [testu], J, weak, check_units=True)

# seed: the FULL annulus with TWELVE circular holes cut on the
# mid-radius circle r = 0.325 at 30-degree steps (clock layout).  Each
# hole radius is chosen so that the angle between its TWO TANGENTS
# THROUGH THE ORIGIN is 25 degrees: rc = 0.325*sin(12.5 deg) = 0.07034.
# FreeFEM's max takes at most 3 arguments, so the twelve hole
# indicators are nested into ternary max trees.
SEED = ('max(max(max(0.07034 - sqrt((x-0.325)^2 + y^2), '
        '0.07034 - sqrt((x-0.28146)^2 + (y-0.1625)^2), '
        '0.07034 - sqrt((x-0.1625)^2 + (y-0.28146)^2)), '
        'max(0.07034 - sqrt(x^2 + (y-0.325)^2), '
        '0.07034 - sqrt((x+0.1625)^2 + (y-0.28146)^2), '
        '0.07034 - sqrt((x+0.28146)^2 + (y-0.1625)^2))), '
        'max(max(0.07034 - sqrt((x+0.325)^2 + y^2), '
        '0.07034 - sqrt((x+0.28146)^2 + (y+0.1625)^2), '
        '0.07034 - sqrt((x+0.1625)^2 + (y+0.28146)^2)), '
        'max(0.07034 - sqrt(x^2 + (y+0.325)^2), '
        '0.07034 - sqrt((x-0.1625)^2 + (y+0.28146)^2), '
        '0.07034 - sqrt((x-0.28146)^2 + (y+0.1625)^2))))')

lag.setup_optimization(name,
                       mesh='mesh.txt',
                       hmin=0.005,
                       hmax=0.05,
                       diffusion=0.2,
                       v0=1.e-5,
                       iterations=100,
                       phi=SEED,
                       boundaryLabels=['2'],
                       fixedLabels=['3'],
                       show_weak_material=False)
