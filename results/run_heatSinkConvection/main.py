from autofreefem import *
from autofreefem.units import *

name = 'heat sink convection'

Th = Domain('D', 'Th')

# definition of the field and the test function
u = ScalarField('u', P2, Th, unit=kelvin)
testu = ScalarField(r'\delta u', P2, Th)

# definition of constants
k0 = Constant('k_0', 200, 'thermal conductivity (aluminium)', Watt / meter / kelvin)
kair = Constant('k_{air}', 0.026, 'thermal conductivity (air)', Watt / meter / kelvin)
h = Constant('h', 25, 'convective heat transfer coefficient', Watt / meter ** 2 / kelvin)
uamb = Constant('u_{amb}', 298, 'ambient temperature', kelvin)
q = Constant('q', 1000, 'heat flux', Watt / meter ** 2)
Afac = Constant(r'\ell', 5000.0, 'Lagrange multiplier', Watt * kelvin / meter ** 3)

# characteristic function
X = CharacteristicFunction(r'\raisebox{\depth}{$\chi$}', 1, '1.3e-4')

# objective functional
J = Expression('J', dx(Afac * X, Th) + dsx(q * u, Th, 2))

# weak form of the state equation
weak = (dx(k0 * X * inner(grad(u), grad(testu)), Th)
        + dsx(h * (u - uamb) * testu, Th, 3)
        - dsx(q * testu, Th, 2))

lag = Lagrangian([u], [testu], J, weak, check_units=True)

lag.setup_optimization(name,
                       mesh='mesh.txt',
                       hmin=0.005,
                       hmax=0.05,
                       diffusion=0.2,
                       v0=1.e-5,
                       iterations=100,
                       phi='0.1 - sqrt((x-0.5)^2+(y-0.5)^2)',
                       boundaryLabels=['3'],
                       fixedLabels=['1', '2'],
                       show_weak_material=False)
