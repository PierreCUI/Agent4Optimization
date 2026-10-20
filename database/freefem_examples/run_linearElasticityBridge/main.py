from autofreefem import *
from autofreefem.units import *

name = 'linear elasticity bridge'

Th = Domain('D', 'Th')

u = VectorField('u', P2, Th, '0.0', [2, 4], unit=meter)
v = VectorField('v', P2, Th)

f = Constant('f', -5.0, 'applied surface load', Newton / meter)
mu = Constant(r'\mu', 8.0, 'shear modulus', Newton / meter ** 2)
lam = Constant(r'\lambda', 1.0, 'Lame parameter', Newton / meter ** 2)
Afac = Constant(r'\ell', 100.0, 'Lagrange multiplier', Newton / meter ** 2)

X = CharacteristicFunction(r'\raisebox{\depth}{$\chi$}', 1, '1/1000')
strain = Expression(r'\pmb \varepsilon', (grad(u) + transpose(grad(u))) / 2)
stress = Expression(r'\pmb \sigma', X * (lam * div(u) * identity() + 2 * mu * strain))
J = Expression('J', dx(inner2(stress, strain), Th) + dx(Afac * X, Th))

lag = Lagrangian(
    [u], [v], J,
    dx(inner2(stress, grad(v)), Th) - dsx(f * inner(v, ey()), Th, 3),
    check_units=True,
)
lag.setup_optimization(
    name, mesh='mesh.txt',
    hmin=0.005, hmax=0.05, diffusion=0.01, v0=1.0e-4, iterations=100,
    phi='min(min(0.8 - y, max(max(-x, x - 0.15), max(-y, y - 0.8)) - 0.01), max(max(3.85 - x, x - 4.0), max(-y, y - 0.8)) - 0.01)',
    boundaryLabels=['1', '3'], fixedLabels=['2', '4'],
    show_weak_material=False,
)
