from autofreefem import *
from autofreefem.units import *
from sympy import Symbol

name = 'thermo elastic'

Th = Domain('D', 'Th')

u = VectorField('u', P2, Th, unit=meter)
testu = VectorField(r'\delta u', P2, Th)
T = ScalarField('T', P2, Th, unit=kelvin)
testT = ScalarField(r'\delta T', P2, Th)

nu = Constant(r'\nu', 0.23, "Poisson's ratio", '-')
E0 = Constant('E_0', 32e6, "Young's modulus", Newton / meter ** 2)
rho = Constant(r'\rho', 2500, 'density', kilogram / meter ** 3)
fac = Constant(r'\gamma', 100, 'displacement cost factor', Newton / meter ** 2)
g = Constant('g', 9.81, 'gravitational acceleration', meter / second ** 2)
Afac = Constant(r'\ell', 1, 'material cost', Newton / meter ** 2)
betaT = Constant(r'\beta_T', 1e11, 'penalty parameter', Watt / kelvin / meter ** 2)
beta = Constant(r'\beta', 1e11, 'penalty parameter', Newton / meter ** 3)
k0 = Constant('k_0', 1.25, 'thermal conductivity', Watt / meter / kelvin)
alpha0 = Constant(r'\alpha_T', 36e-6, 'thermal expansion coefficient', kelvin ** -1)
q = Constant('q', 5000, 'vertical load component', Newton / meter ** 2)

X = CharacteristicFunction(r'\raisebox{\depth}{$\chi$}', 1, 0)
E = CharacteristicFunction('E', E0, E0 / 1000)
thermalConductivity = CharacteristicFunction('k', k0, k0 / 1000)
alpha = CharacteristicFunction(r'\alpha_T', alpha0, alpha0 / 1000)

mu = Expression(r'\mu', E / (2 * (1 + nu)))
lam = Expression(r'\lambda', E * nu / ((1 - 2 * nu) * (1 + nu)))

strain = Expression(r'\pmb\varepsilon', (grad(u) + transpose(grad(u))) / 2)
strainTest = Expression(r'\pmb\varepsilon', (grad(testu) + transpose(grad(testu))) / 2)
strain_eff = Expression(r'\pmb\varepsilon_e', strain - identity() * T * alpha)
stress = Expression(r'\pmb\sigma', lam * tr(strain_eff) * identity() + 2 * mu * strain_eff)

y = Symbol('y')
Tboundary = Parameter('g_T', 30 * y, kelvin)

J = Expression('J', dx(Afac * X, Th) + dsx(beta * inner(u, ex()) ** 2, Th, 2) + dsx(beta * inner(u, ex()) ** 2, Th, 4) - dsx(fac * inner(u, ey()), Th, 3))

lag = Lagrangian([T, u], [testT, testu], J,
                 dx(thermalConductivity * inner(grad(T), grad(testT)), Th)
                 + dsx(betaT * T * testT, Th, 4) - dsx(betaT * Tboundary * testT, Th, 4)
                 + dx(inner2(stress, transpose(strainTest)), Th)
                 + dx(X * rho * g * inner(ey(), testu), Th)
                 + dsx(q * inner(ey(), testu), Th, 3)
                 + dsx(beta * inner(u, testu), Th, 2)
                 + dsx(beta * inner(u, testu), Th, 4),
                 check_units=True)

lag.setup_optimization(name,
                       mesh='mesh.txt',
                       hmin=0.02, hmax=0.1,
                       diffusion=1. / 40000., v0=5.1e-5,
                       iterations=100,
                       phi='-0.1 - cos(pi * x) * cos(pi * 3 * y)*(y<0.9)*(y>0.05) + (y<0.05)',
                       boundaryLabels=[], fixedLabels=[],
                       show_weak_material=False,
                       maxLineSearch=5)
