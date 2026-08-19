from autofreefem import *

x = Parameter('X', 'x', '-')
y = Parameter('Y', 'y', '-')

name = 'linear Elasticity'
bndD = Boundary(r'\Gamma_D', 'ThL', '4')
Th = Domain('D', 'Th', bndD)

# displacement field and test function
u = VectorField('u', P2, Th, '0.', 4, meter)  # Dirichlet on label 4
testu = VectorField(r'\delta u', P2, Th)

# constants
alpha = Constant(r'c', 1e10, 'spring stiffness', Newton / meter ** 3)
f = Constant('f', -10, 'vertical load component', Newton / meter ** 2)
nu = Constant(r'\nu', 0.3, 'Poisson ratio', '-')
E = Constant('E', 2000., 'Young modulus', Newton / meter ** 2)

# characteristic functions
X = CharacteristicFunction(r'\raisebox{\depth}{$\chi$}', 1, '1/100')
X2 = CharacteristicFunction(r'\raisebox{\depth}{$\chi_2$}', 1, 0)

# expressions
mu = Expression(r'\mu', E / (2 * (1 + nu)))
lam = Expression(r'\lambda', E * nu / ((1 - 2 * nu) * (1 + nu)))
strain = Expression(r'\pmb \varepsilon', (grad(u) + transpose(grad(u))) / 2)
stress = Expression(r'\pmb \sigma', X * (lam * tr(strain) * identity() + 2 * mu * strain))
phiT = Parameter(r'\phi_T', 0.02 * ((x + inner(u, ex())) + 0.5) ** 2 + ((y + inner(u, ey())) - 0.5) ** 2 - 0.10, '-')

# objective and Lagrangian
J = Expression('J', dx(X2 * phiT * determinant(identity() + grad(u)), Th))
lag = Lagrangian([u], [testu], J,
                 dx(inner2(stress, grad(testu)), Th)
                 - dx(X2 * f * inner(testu, ey()), Th)
                 + dsx(inner(testu, alpha * u), Th, '4'),
                 check_units=True)

# optimization
lag.setup_optimization(name, mesh='mesh.txt',
                       hmin=0.01, hmax=0.02,
                       diffusion=1. / 10000., v0=0.01,
                       iterations=10,
                       phi='0.02*(x+0.5)**2+(y-0.5)**2-0.10',
                       boundaryLabels=['1', '2', '3'],
                       fixedLabels=['4'],
                       show_weak_material=True)
