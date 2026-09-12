from autofreefem import *
from autofreefem.units import *

name = 'pipe hot water jacket'

# boundaries
bnd_o = Boundary(r'\Gamma_o', 'ThL', '3')
bnd_i = Boundary(r'\Gamma_i', 'ThL', '2')
Th = Domain('D', 'Th', bnd_o, bnd_i)

# field and test function
u = ScalarField('u', P2, Th, unit=kelvin)
testu = ScalarField(r'\delta u', P2, Th)

# constants
R = Constant('R', 0.5, 'outer radius of the pipe cross-section', meter)
r0 = Constant('r_0', 0.1, 'initial inner radius', meter)
k0 = Constant('k_0', 200, 'thermal conductivity (aluminium)', Watt / meter / kelvin)
h = Constant('h', 25, 'convective heat transfer coefficient (environment)', Watt / meter ** 2 / kelvin)
u_amb = Constant('u_amb', 298, 'ambient temperature', kelvin)
h_H = Constant('h_H', 1000, 'convective heat transfer coefficient (hot-water side)', Watt / meter ** 2 / kelvin)
u_H = Constant('u_H', 353, 'district-heating water temperature', kelvin)
Afac = Constant(r'\ell', 5e+06, 'Lagrange multiplier', Watt * kelvin / meter ** 3)
q_b = Expression('q_b', h_H * (u_H - u_amb))

# characteristic function
X = CharacteristicFunction(r'\raisebox{\depth}{$\chi$}', 1, '0.00013')

# objective
J = Expression('J', dx(Afac * X, Th) + dsx(q_b * (u_amb - u), Th, 0))

# state equation weak form
weak = (dx(X * k0 * inner(grad(u), grad(testu)), Th)
        + dsx(h_H * u * testu, Th, 1)
        + dsx(h * u * testu, Th, 0)
        - dsx(h_H * u_H * testu, Th, 1)
        - dsx(h * u_amb * testu, Th, 0))

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
