from autofreefem import *
from autofreefem.units import *
from sympy import Symbol, symbols
import subprocess

#
name = 'thermo-elastic'
bndD = Boundary(r'\Gamma_D', 'bndD', '2/4')
bndN = Boundary(r'\Gamma_N', 'bndN', '3')
bndLeft = Boundary(r'\Gamma_\ell', 'bndL', '4')
bndRight = Boundary(r'\Gamma_r', 'bndR', '2')
bndT = Boundary(r'\Gamma_{T}', 'bndT', '3/2/4')
Th = Domain('D', 'Th', bndD, bndN, bndLeft, bndRight, bndT)
# definition of fields and the test function
# elasticity part
u = VectorField('u', P2, Th, unit=meter)
testu = VectorField(r'\delta u', P2, Th)
# thermal part
T = ScalarField('T', P2, Th, '0', [1, 3], kelvin)
testT = ScalarField(r'\delta T', P2, Th)
# definition of constants
q = Constant('q', 5000, 'vert. load comp.', Newton / meter ** 2)
nu = Constant(r'\nu', 0.23, 'Poisson\'s ratio', '-')
E0 = Constant('E_0', 32 * 1e6, 'Young\'s modulus', Newton / meter ** 2)
alpha0 = Constant(r'\alpha_T', 36 * 1e-6, 'ther. expansion coeff.', kelvin ** -1)
k0 = Constant('k_0', 1.25, 'ther. conductivity', Watt / meter / kelvin)
Afac = Constant(r'\ell', 1e0, 'material cost', Newton / meter ** 2)
penalty = Constant(r'\beta', 1e11, 'penalty parameter', Newton / meter ** 3)
penaltyT = Constant(r'\beta_{T}', 1e11, 'penalty parameter', Watt / kelvin / meter ** 2)
rho = Constant(r'\rho', 2500, 'density', kilogram / meter ** 3)
g = Constant('g', 9.81, 'gravitational acc.', meter / second ** 2)
fac = Constant(r'\gamma', 1e2, 'disp. cost factor', Newton / meter ** 2)
X = CharacteristicFunction(r'\raisebox{\depth}{$\chi$}', 1, 0)
E = CharacteristicFunction('E', E0, E0 / 1000)
thermalConductivity = CharacteristicFunction('k', k0, k0 / 1000)
alpha = CharacteristicFunction(r'\alpha_T', alpha0, alpha0 / 1000)
mu = Expression(r'\mu', E / (2 * (1 + nu)))
lam = Expression(r'\lambda', E * nu / ((1 - 2 * nu) * (1 + nu)))
# Expressions
strain = Expression(r'\bm\varepsilon', (grad(u) + transpose(grad(u))) / 2)
strainTest = Expression(r'\pmb\varepsilon', (grad(testu) + transpose(grad(testu))) / 2)
stress = (lam * tr(strain) * identity() + 2 * mu * strain)
# objective function J
J = Expression('J', -dsx(fac * inner(u, ey()), Th, 1) + dsx(penalty * inner(u, ex()) ** 2, Th, 0) + dx(X * Afac, Th))
#
phi0 = '-0.1 - cos(pi * x) * cos(pi * 3 * y)*(y<0.9)*(y>0.05) + (y<0.05)'
lag = Lagrangian([u], [testu], J,
                 dx(inner2(stress, transpose(strainTest)), Th)
                 + dx(X * rho * g * inner(ey(), testu), Th)
                 + dsx(q * inner(ey(), testu), Th, 1)
                 + dsx(penalty * inner(u, testu), Th, 0),
                 check_units=True)
lag.setup_optimization(name, mesh='meshThermoElastic.txt', show_weak_material=False,
                       hmin=0.02, hmax=0.1, diffusion=1. / 40000., v0=5.2e-5, iterations=100,
                       phi=phi0,
                       boundaryLabels=[], fixedLabels=[],
                       maxLineSearch=5,
                       objective_split=[['A', dx(X, Th)],
                                        ['kA', dx(X * Afac, Th)],
                                        ['horiFl', dsx(penalty * inner(u, ex()), Th, 2)],
                                        ['horiFr', dsx(penalty * inner(u, ex()), Th, 3)],
                                        ['Fx', dsx(penalty * inner(u, ex()) ** 2, Th, 0)],
                                        ['upper', dsx(-fac * inner(u, ey()), Th, 1)],
                                        ['G', dx(X * rho * g, Th)], ])
# subprocess.run(['FreeFem++','optimize_thermoelastic.edp'])
##
y = Symbol('y')
Tboundary = Parameter('g_T', 30 * y, kelvin)
T = ScalarField('T', P2, Th, unit=kelvin)
strain_eff = Expression(r'\pmb\varepsilon_e', strain - identity() * T * alpha)
stressThermal = Expression(r'\pmb \sigma', lam * tr(strain_eff) * identity() + 2 * mu * strain_eff)
lag = Lagrangian([T, u], [testT, testu], J,
                 dx(inner2(stressThermal, transpose(strainTest)), Th)
                 + dx(thermalConductivity * inner(grad(T), grad(testT)), Th)
                 + dx(X * rho * g * inner(ey(), testu), Th)
                 + dsx(q * inner(ey(), testu), Th, 1)
                 + dsx(penalty * inner(u, testu), Th, 0)
                 + dsx(penaltyT * T * testT, Th, 4) - dsx(penaltyT * Tboundary * testT, Th, 4),
                 check_units=True
                 )
lag.setup_optimization(name + '30b', mesh='meshThermoElastic.txt', show_weak_material=False,
                       hmin=0.02, hmax=0.1, diffusion=1. / 40000., v0=5.1e-5, iterations=100,
                       phi=phi0,
                       boundaryLabels=[], fixedLabels=[],
                       objective_split=[['A', dx(X, Th)],
                                        ['kA', dx(X * Afac, Th)],
                                        ['horiFl', dsx(penalty * inner(u, ex()), Th, 2)],
                                        ['horiFr', dsx(penalty * inner(u, ex()), Th, 3)],
                                        ['Fx', dsx(penalty * inner(u, ex()) ** 2, Th, 0)],
                                        ['upper', dsx(-fac * inner(u, ey()), Th, 1)],
                                        ['G', dx(X * rho * g, Th)], ])
# subprocess.run(['FreeFem++','optimize_thermoelastic30b.edp'])
##
T = ScalarField('T', P2, Th, '-30', [1, 3], kelvin)
strain_eff = Expression(r'\pmb\varepsilon_e', strain - identity() * T * alpha)
stressThermal = (lam * tr(strain_eff) * identity() + 2 * mu * strain_eff)
lag = Lagrangian([T, u], [testT, testu], J,
                 dx(inner2(stressThermal, transpose(strainTest)), Th)
                 + dx(thermalConductivity * inner(grad(T), grad(testT)), Th)
                 + dx(X * rho * g * inner(ey(), testu), Th)
                 + dsx(q * inner(ey(), testu), Th, 1)
                 + dsx(penalty * inner(u, testu), Th, 0),
                 check_units=True)
lag.setup_optimization(name + '30c', mesh='meshThermoElastic.txt', show_weak_material=False,
                       hmin=0.02, hmax=0.1, diffusion=1. / 40000., v0=5.0e-5, iterations=100,
                       phi=phi0,
                       boundaryLabels=['1', '2', '3', '4'], fixedLabels=[],
                       objective_split=[['A', dx(X, Th)],
                                        ['kA', dx(X * Afac, Th)],
                                        ['horiFl', dsx(penalty * inner(u, ex()), Th, 2)],
                                        ['horiFr', dsx(penalty * inner(u, ex()), Th, 3)],
                                        ['Fx', dsx(penalty * inner(u, ex()) ** 2, Th, 0)],
                                        ['upper', dsx(-fac * inner(u, ey()), Th, 1)],
                                        ['G', dx(X * rho * g, Th)], ])
# subprocess.run(['FreeFem++', 'optimize_thermoelastic30c.edp'])
