from autofreefem import *
from autofreefem.units import *

name = 'pipe cooling water jacket'

Th = Domain('D', 'Th')

# field and test function
u = ScalarField('u', P2, Th, unit=kelvin)
testu = ScalarField(r'\delta u', P2, Th)

# constants
k0 = Constant('k_0', 200, 'thermal conductivity (aluminium)', Watt / meter / kelvin)
kair = Constant('k_{air}', 0.026, 'thermal conductivity (air)', Watt / meter / kelvin)
h = Constant('h', 25, 'convective heat transfer coefficient (environment)', Watt / meter ** 2 / kelvin)
uamb = Constant('u_{amb}', 298, 'ambient temperature', kelvin)
hC = Constant('h_C', 1000, 'convective heat transfer coefficient (cooling-water side)', Watt / meter ** 2 / kelvin)
uC = Constant('u_C', 283, 'cooling-water temperature', kelvin)
qb = Expression('q_b', hC * (uamb - uC))
Afac = Constant(r'\ell', 250000, 'Lagrange multiplier', Watt * kelvin / meter ** 3)

# characteristic function
X = CharacteristicFunction(r'\raisebox{\depth}{$\chi$}', 1, '1.3e-4')

# objective: minimise the casing-temperature term qb*u -> maximise the
# heat absorbed from the environment through the casing (label 3) and
# delivered to the cooling water
J = Expression('J', dx(Afac * X, Th) + dsx(qb * u, Th, 3))

# weak form of the state equation: Robin cooling on the design inner
# wall (label 2), Robin heating on the fixed casing (label 3)
weak = (dx(k0 * X * inner(grad(u), grad(testu)), Th)
        + dsx(hC * (u - uC) * testu, Th, 2)
        + dsx(h * (u - uamb) * testu, Th, 3))

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
