from autofreefem import *

x = Parameter('x', 'x', '-')
y = Parameter('y', 'y', '-')

name = 'linear Elasticity Targetmesh'

Th = Domain('D', 'Th')

# displacement field and test function
u = VectorField('u', P2, Th, '0.', 4, meter)  # zero Dirichlet on label 4
testu = VectorField(r'\delta u', P2, Th)

# constants
c = Constant('c', 1e10, 'spring stiffness', '-')
f = Constant('f', -10, 'vertical load component', Newton / meter ** 2)
nu = Constant(r'\nu', 0.3, "Poisson's ratio", '-')
E = Constant('E', 2000., "Young's modulus", Newton / meter ** 2)

# characteristic functions
X = CharacteristicFunction(r'\raisebox{\depth}{$\chi$}', 1, '1/100')
X2 = CharacteristicFunction(r'\raisebox{\depth}{$\chi_2$}', 1, 0)

# expressions
mu = Expression(r'\mu', E / (2 * (1 + nu)))
lam = Expression(r'\lambda', E * nu / ((1 - 2 * nu) * (1 + nu)))
strain = Expression(r'\pmb \varepsilon', (grad(u) + transpose(grad(u))) / 2)
stress = Expression(r'\pmb \sigma', X * (lam * tr(strain) * identity() + 2 * mu * strain))

# target signed-distance function in deformed configuration
phiT = Parameter(r'\phi_T',
                 (((x + inner(u, ex())) - 1.0) ** 2 + ((y + inner(u, ey())) - 0.5) ** 2) ** 0.5 - 0.4,
                 '-')

# objective
J = Expression('J', dx(X2 * phiT * determinant(identity() + grad(u)), Th))

# Lagrangian: linear elasticity state with body force and interface spring
lag = Lagrangian([u], [testu], J,
                 dx(inner2(stress, grad(testu)), Th)
                 - dx(X2 * f * inner(testu, ey()), Th)
                 + dsx(inner(testu, c * u), Th, '999'),
                 check_units=True)

lag.setup_optimization(
    name,
    mesh='mesh.txt',
    hmin=0.02,
    hmax=0.1,
    diffusion=0.01,
    v0=1e-3,
    iterations=100,
    phi='max(abs(x-0.75)-0.75, abs(y-0.5)-0.35)',
    boundaryLabels=['1', '2', '3'],
    fixedLabels=['4'],
    show_weak_material=True)
