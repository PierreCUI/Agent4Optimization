from autofreefem import *
from sympy import Symbol

# x = Symbol('x')
# y = Symbol('y')
x = Parameter('x', 'x', '-')
y = Parameter('y', 'y', '-')

name = 'linear Elasticity'
bndN = Boundary(r'\Gamma_N', 'ThL', '2')
Th = Domain('D', 'Th', bndN)
# definition of the displacement field and the test function
u = VectorField('u', P2, Th, '0.', 4, meter)  # left fixed
# u = VectorField('u', P2, Th, unit=meter)  # left fixed
testu = VectorField(r'\delta u', P2, Th)
# definition of constants
alpha = Constant(r'c', 1e10, 'spring stiffness', '-')
f = Constant('f', -10, 'vertical load component', Newton / meter ** 2)
nu = Constant(r'\nu', .3, 'Poisson\'s ratio', '-')
E = Constant('E', 2000., 'Young\'s modulus', Newton / meter ** 2)
Afac = Constant(r'\ell', 0.25, 'Lagrange multiplier', Newton / meter ** 2)
# definition of the material contrast
X = CharacteristicFunction(r'\raisebox{\depth}{$\chi$}', 1, '1/100')
X2 = CharacteristicFunction(r'\raisebox{\depth}{$\chi_2$}', 1, 0)
# definition of expressions
mu = Expression(r'\mu', E / (2 * (1 + nu)))
lam = Expression(r'\lambda', E * nu / ((1 - 2 * nu) * (1 + nu)))
strain = Expression(r'\pmb \varepsilon', (grad(u) + transpose(grad(u))) / 2)
stress = Expression(r'\pmb \sigma', X * (lam * tr(strain) * identity() + 2 * mu * strain))
# phiT = Parameter(r'\phi_T',(x-1)*(0.5-4*(y-0.5)**2),'-')
# phiT = Parameter(r'\phi_T',0.02*((x+inner(u,ex()))+0.5)**2+((y+inner(u,ey()))-0.5)**2-0.10,'-') # Kragträger (gerundet)
phiT = Parameter(r'\phi_T', (((x + inner(u, ex())) - 1.0) ** 2 + ((y + inner(u, ey())) - 0.5) ** 2) ** 0.5 - 0.4,
                 '-')  # Kreis in der Mitte
# phiT = 0.02*((x+inner(u,ex()))+0.5)**2+((y+inner(u,ey()))-0.5)**2-0.10

# definition of the shape optimization problem
J = Expression('J', dx(X2 * phiT * determinant(identity() + grad(u)), Th))
J = Expression('J', dx(X2 * phiT * determinant(identity() + grad(u)), Th))
lag = Lagrangian([u], [testu], J, dx(inner2(stress, grad(testu)), Th)
                 - dx(X2 * f * inner(testu, ey()), Th) + dsx(inner(testu, alpha * u), Th, '999'), check_units=True)
# lag.setup_optimization(name, mesh='meshelasticCantilever.txt',
lag.setup_optimization(name, mesh='meshelasticCantileverwithinnermesh.txt',
                       hmin=0.01, hmax=0.02, diffusion=1. / 10000., v0=0.002, iterations=100,
                       # phi='-0.4 - sin(pi * 3 * (x+0.5)) * cos(pi * 6 * y)', #Bereich mit Löchern
                       # phi = '0.02*(x+0.5)**2+(y-0.5)**2-0.10', #Kragträger
                       # phi = '(x-1.0)**2+(y-0.5)**2-0.4**2', #Kreis

                       boundaryLabels=['1', '2', '3', '4'], fixedLabels=['4'], show_weak_material=True)
