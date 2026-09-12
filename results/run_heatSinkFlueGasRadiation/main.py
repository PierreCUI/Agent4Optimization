from autofreefem import *
from autofreefem.units import *

name = 'heat sink flue gas radiation'

Th = Domain('D', 'Th')

# definition of the field and the test function
u = ScalarField('u', P2, Th, unit=kelvin)
testu = ScalarField(r'\delta u', P2, Th)

# definition of constants
k0 = Constant('k_0', 200, 'thermal conductivity (aluminium)', Watt / meter / kelvin)
kair = Constant('k_{air}', 0.026, 'thermal conductivity (air)', Watt / meter / kelvin)
h = Constant('h', 25, 'convective heat transfer coefficient (air side)', Watt / meter ** 2 / kelvin)
uamb = Constant('u_{amb}', 298, 'ambient temperature', kelvin)
hH = Constant('h_H', 25, 'convective heat transfer coefficient (flue-gas side)', Watt / meter ** 2 / kelvin)
uH = Constant('u_H', 338, 'flue-gas temperature', kelvin)
epsg = Constant(r'\varepsilon_g', 0.9, 'effective emissivity (gas-filled duct)', '-')
sig = Constant(r'\sigma', 5.67e-8, 'Stefan-Boltzmann constant', Watt / meter ** 2 / kelvin ** 4)
Afac = Constant(r'\ell', 2000, 'volume weight (initial estimate, calibrated by pipeline)', Watt * kelvin / meter ** 3)

# characteristic function
X = CharacteristicFunction(r'\raisebox{\depth}{$\chi$}', 1, '1.3e-4')

# objective functional: the Robin load enters through the equivalent
# surface heat-flux weight qb = hH * (uH - uamb) = 25 * 40 = 1000 W/m^2
qb = Expression('q_b', hH * (uH - uamb))
J = Expression('J', dx(Afac * X, Th) + dsx(qb * u, Th, 2))

# weak form of the state equation: Robin convection AND gas radiation
# q_rad = epsg*sig*(uH^4 - u^4) absorbed from the hot gas on the
# flue-gas segment (label 2, fixed), Robin cooling on the three
# air-side edges (label 3, design).  The u^4 dependence makes the
# problem NONLINEAR; the Newton linearisation is derived automatically.
weak = (dx(k0 * X * inner(grad(u), grad(testu)), Th)
        + dsx(h * (u - uamb) * testu, Th, 3)
        + dsx(hH * (u - uH) * testu, Th, 2)
        + dsx(epsg * sig * (u ** 4 - uH ** 4) * testu, Th, 2))

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
