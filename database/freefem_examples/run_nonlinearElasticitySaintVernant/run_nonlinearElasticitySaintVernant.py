from autofreefem import *

name = 'non-linearElasticitySaintVernant'
bndN = Boundary(r'\Gamma_N', 'ThL', '2')
Th = Domain('D', 'Th', bndN)
# definition of the displacement field and the test function
u = VectorField('u', P2, Th, '0.', 4)  # left and right fixed
v = VectorField('v', P2, Th, )
# definition of constants
f = Constant('f', -10, 'surface load', 'N/m')
nu = Constant(r'\nu', .3, 'Poisson\'s ratio', '-')
EE = Constant('E', 200., 'Young\'s modulus', 'N/m^2')
Afac = Constant(r'\ell', 0.25, 'Lagrange multiplier', '')
mu = Expression('\mu', EE / (2 * (1 + nu)))
lam = Expression('\lambda', EE * nu / ((1 - 2 * nu) * (1 + nu)))
X = CharacteristicFunction(r'\raisebox{\depth}{$\chi$}', 1, '1/100')
F = Expression('\mathbf F', identity() + grad(u))
E = Expression('\mathbf E', (inner(transpose(F), F) - identity()) / 2)
S = Expression('\mathbf S', X * (lam * identity() * inner2(E, identity()) + 2 * mu * E))
# definition of the problem
J = Expression('J', dx(inner2(E, S), Th) + dx(Afac * X, Th))
lag = Lagrangian([u], [v], J, dx(inner2(S, inner(transpose(F), grad(v))), Th) - dsx(f * inner(v, ey()), Th, 0))
lag.setup_optimization(name, mesh='meshelasticCantilever.txt',
                       hmin=0.01, hmax=0.02, diffusion=1. / 20000., v0=0.0002, iterations=100,
                       phi='-0.4 - sin(pi * 3 * (x+0.5)) * cos(pi * 6 * y)',
                       boundaryLabels=['1', '4'], fixedLabels=['2'], show_weak_material=False)
