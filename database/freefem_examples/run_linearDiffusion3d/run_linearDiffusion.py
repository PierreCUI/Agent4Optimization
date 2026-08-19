from autofreefem import *
from autofreefem.units import *

#
name = 'linear diffusion'
Th = Domain('D', 'Th')
# definition of the field and the test function
u = ScalarField('u', P2, Th, 0, 3, kelvin)
testu = ScalarField(r'\delta u', P2, Th)
# definition of constants
f = Constant('f', -10, 'source', Watt / meter ** 3)
Afac = Constant(r'\ell', 100, 'Lagrange multiplier', Watt * kelvin / meter ** 3)
lam = Constant(r'\lambda', 1, 'diffusion coefficient', Watt / meter / kelvin)
# definition of the problem
X = CharacteristicFunction(r'\raisebox{\depth}{$\chi$}', 1, '1/10')
J = Expression('J', dx(X * lam * inner(grad(u), grad(u)), Th) + dx(Afac * X, Th))
lag = Lagrangian([u], [testu], J, dx(X * lam * inner(grad(u), grad(testu)), Th) - dx(f * testu, Th), check_units=True,
                 dimensions=3)
lag.setup_optimization(name, mesh='meshLaplace3d.txt', hmin=0.05, hmax=0.2,
                       diffusion=2. / 10000., v0=1.e-3, iterations=100, phi='y-0.5',
                       boundaryLabels=['1', '2', '3', '4', '5', '6'], fixedLabels=[])
