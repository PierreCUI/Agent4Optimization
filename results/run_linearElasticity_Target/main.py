from autofreefem import *
from autofreefem.units import *

name = 'linear Elasticity Target'

x = Parameter('x', 'x', '-')
y = Parameter('y', 'y', '-')

Th = Domain('D', 'Th')

# displacement field and test function
u = VectorField('u', P2, Th, '0.', 4, meter)  # zero Dirichlet on label 4
testu = VectorField(r'\delta u', P2, Th)

# constants
nu = Constant(r'\nu', 0.3, "Poisson's ratio", '-')
E = Constant('E', 2000., "Young's modulus", Newton / meter ** 2)
alpha = Constant(r'c', 1e10, 'spring stiffness', '-')
f = Constant('f', -10., 'vertical load component', Newton / meter ** 2)

# material interpolation
X = CharacteristicFunction(r'\raisebox{\depth}{$\chi$}', 1, '1/100')
X2 = CharacteristicFunction(r'\raisebox{\depth}{$\chi_2$}', 1, 0)

# kinematics and constitutive law
mu = Expression(r'\mu', E / (2 * (1 + nu)))
lam = Expression(r'\lambda', E * nu / ((1 - 2 * nu) * (1 + nu)))
strain = Expression(r'\pmb \varepsilon', (grad(u) + transpose(grad(u))) / 2)
stress = Expression(r'\pmb \sigma', X * (lam * tr(strain) * identity() + 2 * mu * strain))
F = Expression(r'\mathbf F', identity() + grad(u))

# target functional (deformation-dependent target region)
phiT = Expression(r'\phi_T',
                0.02 * ((x + inner(u, ex())) + 0.5) ** 2 + ((y + inner(u, ey())) - 0.5) ** 2 - 0.10)

J = Expression('J', dx(phiT * determinant(F), Th))

# linear elasticity state equation with volume load and interface spring
lag = Lagrangian([u], [testu], J,
                 dx(inner2(stress, grad(testu)), Th)
                 - dx(X2 * f * inner(testu, ey()), Th)
                 + dsx(inner(testu, alpha * u), Th, '999'),
                 check_units=True)

lag.setup_optimization(name,
                       mesh='mesh.txt',
                       hmin=0.005, hmax=0.02,
                       diffusion=0.01, v0=2e-4,
                       iterations=100,
                       phi='0.4 - sqrt((x-0.5)^2+(y-0.5)^2)',
                       boundaryLabels=['1','2','3'],
                       fixedLabels=['4'],
                       show_weak_material=True)
