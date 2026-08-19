from autofreefem import *
from autofreefem.units import *

name = 'non-linear Elasticity Neo-Hookean'

bndN = Boundary(r'\Gamma_N', 'ThL', '2')
Th = Domain('D', 'Th', bndN)

# displacement field and test function
u = VectorField('u', P2, Th, '0.', 4, meter)  # fixed on label 4
v = VectorField('v', P2, Th)

# constants
f = Constant('f', -10, 'surface load', Newton / meter)
nu = Constant(r'\nu', 0.3, "Poisson's ratio", '-')
EE = Constant('E', 200., "Young's modulus", Newton / meter ** 2)
Afac = Constant(r'\ell', 0.25, 'Lagrange multiplier', Newton / meter ** 2)

# material contrast
X = CharacteristicFunction(r'\raisebox{\depth}{$\chi$}', 1, '1/100')

# Neo-Hookean expressions
mu = Expression(r'\mu', EE / (2 * (1 + nu)))
lam = Expression(r'\lambda', EE * nu / ((1 - 2 * nu) * (1 + nu)))
F = Expression(r'\mathbf F', identity() + grad(u))
J = Expression('J', determinant(F))
C = Expression(r'\mathbf C', inner(transpose(F), F))
E = Expression(r'\mathbf E', (C - identity()) / 2)
S = Expression(r'\mathbf S', X * ((lam / 2 * (J ** 2 - 1) - mu) * inverse(C) + mu * identity()))

# objective functional
G = Expression('G', dx(inner2(E, S), Th) + dx(Afac * X, Th))

# Lagrangian with nonlinear state equation
lag = Lagrangian([u], [v], G,
                 dx(inner2(S, inner(transpose(F), grad(v))), Th) - dsx(f * inner(v, ey()), Th, 0))

# shape optimization setup
lag.setup_optimization(name,
                       mesh='mesh.txt',
                       hmin=0.01,
                       hmax=0.02,
                       diffusion=1. / 10000.,
                       v0=0.0011,
                       iterations=100,
                       phi='-0.4 - sin(pi * 3 * (x+0.5)) * cos(pi * 6 * (y))',
                       boundaryLabels=['1', '4'],
                       fixedLabels=['2'],
                       show_weak_material=False)
