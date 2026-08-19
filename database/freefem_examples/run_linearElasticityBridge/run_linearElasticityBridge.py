from autofreefem import *
from sympy import Symbol

#
name = 'linear Elasticity bridge'
Th = Domain('D', 'Th')
# definition of the displacement field and the test function
u = VectorField('u', P2, Th, '0.', [2, 4])  # left and right fixed
v = VectorField('v', P2, Th)
# definition of constants
f = Constant('f', -10, 'surface load', 'N/m')
nu = Constant(r'\nu', .3, 'Poisson\'s ratio', '-')
E = Constant('E', 1., 'Young\'s modulus', 'N/m^2')
Afac = Constant(r'\ell', 10, 'Lagrange multiplier', '')
mu = Expression(r'\mu', E / (2 * (1 + nu)))
lam = Expression(r'\lambda', E * nu / ((1 - 2 * nu) * (1 + nu)))
#
f = Constant('f', -5, 'surface load', 'N/m')
mu = Constant(r'\mu', 8., 'Poisson\'s ratio', '-')
lam = Constant(r'\lambda', 1., 'Young\'s modulus', 'N/m^2')
Afac = Constant(r'\ell', 100, 'Lagrange multiplier', '')
#
Xapprox = CharacteristicFunction(r'\raisebox{\depth}{$\chi$}', 1, '1/1000')
# Xapprox = Symbol('X')
strain = Expression(r'\pmb \varepsilon', (grad(u) + transpose(grad(u))) / 2)
# definition of the problem
J = Expression('J', dx(Xapprox * (lam * div(u) * div(u) + 2 * mu * inner2(strain, transpose(strain))), Th) + dx(
    Afac * Xapprox, Th))
lag = Lagrangian([u], [v], J,
                 dx(lam * Xapprox * div(v) * inner2(strain, identity()) + 2 * mu * Xapprox * inner2(strain, grad(v)),
                    Th)
                 - dsx(f * inner(v, ey()), Th, 1))
# run in shape optimization mode
lag.setup_optimization(name, mesh='meshelasticBridge.txt', hmin=0.01, hmax=0.02,
                       diffusion=4. / 40000., v0=0.05, iterations=100,
                       phi='-0.1 - (sin(pi * 4 * (x-0.125)) * cos(pi * 4 * (y)))*(y>0.05)',
                       boundaryLabels=['1', '4'], fixedLabels=['2'])
