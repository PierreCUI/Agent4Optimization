from autofreefem import *

name = 'linear elasticity'

Th = Domain('D', 'Th')

u = VectorField('u', P2, Th, '0.0', 4, meter)
testu = VectorField(r'\delta u', P2, Th)

f = Constant('f', -10.0, 'vertical load component', Newton / meter ** 2)
nu = Constant(r'\nu', 0.3, "Poisson's ratio", '-')
E = Constant('E', 200.0, "Young's modulus", Newton / meter ** 2)
Afac = Constant(r'\ell', 0.25, 'Lagrange multiplier', Newton / meter ** 2)

X = CharacteristicFunction(r'\raisebox{\depth}{$\chi$}', 1, '1/100')
mu = Expression(r'\mu', E / (2 * (1 + nu)))
lam = Expression(r'\lambda', E * nu / ((1 - 2 * nu) * (1 + nu)))
strain = Expression(r'\pmb \varepsilon', (grad(u) + transpose(grad(u))) / 2)
stress = Expression(r'\pmb \sigma', X * (lam * tr(strain) * identity() + 2 * mu * strain))
J = Expression('J', dx(inner2(stress, strain), Th) + dx(Afac * X, Th))

lag = Lagrangian(
    [u], [testu], J,
    dx(inner2(stress, grad(testu)), Th) - dsx(f * inner(testu, ey()), Th, 2),
    check_units=True,
)
lag.setup_optimization(
    name, mesh='mesh.txt',
    hmin=0.01, hmax=0.02, diffusion=1.0e-4, v0=0.002, iterations=100,
    phi='-0.4 - sin(pi * 3 * (x + 0.5)) * cos(pi * 6 * y)',
    boundaryLabels=['1'], fixedLabels=['2', '4'],
    show_weak_material=False,
)
