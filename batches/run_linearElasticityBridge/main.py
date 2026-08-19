from autofreefem import *
from autofreefem.units import *

name = 'linear Elasticity bridge'

Th = Domain('D', 'Th')

# displacement field and test function
u = VectorField('u', P2, Th, '0.', [2, 4], meter)  # Dirichlet on supports 2 and 4
v = VectorField('v', P2, Th)

# constants
f = Constant('f', -5, 'applied surface load', Newton / meter)
mu = Constant(r'\mu', 8., 'shear modulus', Newton / meter ** 2)
lam = Constant(r'\lambda', 1., 'Lame parameter', Newton / meter ** 2)
Afac = Constant(r'\ell', 100., 'Lagrange multiplier', Newton / meter ** 2)

# material contrast
X = CharacteristicFunction(r'\raisebox{\depth}{$\chi$}', 1, '1/1000')

# strain and stress
strain = Expression(r'\pmb \varepsilon', (grad(u) + transpose(grad(u))) / 2)
stress = Expression(r'\pmb \sigma', X * (lam * div(u) * identity() + 2 * mu * strain))

# objective
J = Expression('J', dx(inner2(stress, strain), Th) + dx(Afac * X, Th))

# Lagrangian
lag = Lagrangian([u], [v], J,
                 dx(inner2(stress, grad(v)), Th) - dsx(f * inner(v, ey()), Th, 3))

lag.setup_optimization(name,
                       mesh='mesh.txt',
                       hmin=0.005, hmax=0.05,
                       diffusion=0.01, v0=0.0001,
                       iterations=10,
                       phi='min(min(0.8 - y, max(max(-x, x - 0.15), max(-y, y - 0.8)) - 0.01), max(max(3.85 - x, x - 4.0), max(-y, y - 0.8)) - 0.01)',
                       boundaryLabels=['1', '3'], fixedLabels=['2', '4'],
                       show_weak_material=False)
