from autofreefem import *
from autofreefem.units import *

name = 'nonlinear diffusion'

Th = Domain('D', 'Th')

# temperature field and test function
u = ScalarField('u', P2, Th, '0.', [3], kelvin)
testu = ScalarField(r'\delta u', P2, Th)

# constants
f = Constant('f', -10, 'source', Watt / meter ** 3)
Afac = Constant(r'\ell', 100, 'Lagrange multiplier', Watt * kelvin / meter ** 3)
alpha = Constant(r'\alpha', 0.01, 'factor', 1 / kelvin ** 2)
lam0 = Constant(r'\lambda_0', 1, 'conductivity', Watt / (meter * kelvin))

# characteristic function
X = CharacteristicFunction(r'\raisebox{\depth}{$\chi$}', 1, '1/10')

# constitutive law and objective
lam = Expression(r'\lambda', lam0 * (1 + alpha * u ** 2))
J = Expression('J', dx(X * lam * inner(grad(u), grad(u)), Th) + dx(Afac * X, Th))

# Lagrangian with nonlinear state equation
lag = Lagrangian([u], [testu], J,
                 dx(X * lam * inner(grad(u), grad(testu)), Th) - dx(f * testu, Th),
                 check_units=True)

# optimization setup
lag.setup_optimization(name,
                       mesh='mesh.txt',
                       hmin=0.01 / 8,
                       hmax=0.1 / 8,
                       diffusion=1. / 40000.,
                       v0=1.e-6,
                       iterations=100,
                       phi='-0.1-sin(pi*8*(x+0.6))*cos(pi*8*y)',
                       boundaryLabels=['1'],
                       fixedLabels=['3'])
