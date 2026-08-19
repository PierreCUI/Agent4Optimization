from autofreefem import *
from autofreefem.units import *

name = 'linear diffusion'
Th = Domain('D', 'Th')
# definition of the field and the test function
u = ScalarField('u', P2, Th, '0.', 3, kelvin)
testu = ScalarField(r'\delta u', P2, Th)
# definition of constants
f = Constant('f', -10, 'source', Watt / meter ** 3)
Afac = Constant(r'\ell', 100, 'Lagrange multiplier', Watt * kelvin / meter ** 3)
lam = Constant(r'\lambda', 1, 'diffusion coefficient', Watt / meter / kelvin)
# definition of the problem
X = CharacteristicFunction(r'\raisebox{\depth}{$\chi$}', 1, '1/10')
J = Expression('J', dx(X * lam * inner(grad(u), grad(u)), Th) + dx(Afac * X, Th))
lag = Lagrangian([u], [testu], J, dx(X * lam * inner(grad(u), grad(testu)), Th) - dx(f * testu, Th), check_units=True)
lag.setup_optimization(name, mesh='mesh.txt', hmin=0.01 / 8, hmax=0.1 / 8,
                       diffusion=2. / 40000., v0=1.e-6, iterations=10,
                       phi='-0.1-sin(pi*8*(x+0.6))*cos(pi*8*y)',
                       boundaryLabels=['1'], fixedLabels=['3'])
