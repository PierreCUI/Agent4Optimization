from autofreefem import *
from autofreefem.units import *

name = 'non-linear Elasticity Saint-Venant'

bndN = Boundary(r'\Gamma_N', 'ThL', '2')
Th = Domain('D', 'Th', bndN)

# displacement field and test function
u = VectorField('u', P2, Th, '0.', 4, meter)
v = VectorField('v', P2, Th)

# constants
Afac = Constant(r'\ell', 0.25, 'Lagrange multiplier', Newton / meter**2)
nu = Constant(r'\nu', 0.3, "Poisson's ratio", '-')
E = Constant('E', 200., "Young's modulus", Newton / meter**2)
f = Constant('f', -10, 'surface load', Newton / meter)

# material contrast
X = CharacteristicFunction(r'\raisebox{\depth}{$\chi$}', 1, '1/100')

# constitutive expressions
mu = Expression(r'\mu', E / (2 * (1 + nu)))
lam = Expression(r'\lambda', E * nu / ((1 - 2 * nu) * (1 + nu)))
F = Expression(r'\mathbf F', identity() + grad(u))
Egreen = Expression(r'\mathbf E', (inner(transpose(F), F) - identity()) / 2)
S = Expression(r'\mathbf S', X * (lam * identity() * inner2(Egreen, identity()) + 2 * mu * Egreen))

# objective functional
J = Expression('J', dx(inner2(Egreen, S), Th) + dx(Afac * X, Th))

# Lagrangian with nonlinear Saint-Venant state equation
lag = Lagrangian([u], [v], J,
                 dx(inner2(S, inner(transpose(F), grad(v))), Th) - dsx(f * inner(v, ey()), Th, 0))

# topology optimization setup
lag.setup_optimization(name,
                       mesh='mesh.txt',
                       hmin=0.01,
                       hmax=0.02,
                       diffusion=1. / 20000.,
                       v0=0.0002,
                       iterations=100,
                       phi='-0.4 - sin(pi * 3 * (x+0.5)) * cos(pi * 6 * y)',
                       boundaryLabels=['1', '4'],
                       fixedLabels=['2'],
                       show_weak_material=False)
