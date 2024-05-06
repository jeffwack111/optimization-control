import cvxpy as cp
import numpy as np
import scipy.linalg as la

nx = 10 #dimension of the state space  
nu = 4  #dimension of the control input space
nw = 2  #dimension of the disturbance input space
ny = 5  #dimension of the feedback output space
nz = 3  #dimension of the performance output space

#The following define the state space representation of the plant
A = np.random.randn(nx,nx)
B1 = np.random.randn(nx,nw)
B2 = np.random.randn(nx,nu)
C1 = np.random.randn(nz,nx)
D11 = np.random.randn(nz,nw)
D12 = np.random.randn(nz,nu)
C2 = np.random.randn(ny,nx)
D21 = np.random.randn(ny,nw)
D22 = np.random.randn(ny,nu)

#We will optimize over the following intermediate matrices 
#which will be used to produce the state space representation of the controller
An = cp.Variable((nx,nx))
Bn = cp.Variable((nx,ny))
Cn = cp.Variable((nu,nx))
Dn = cp.Variable((nu,ny))

X1 = cp.Variable((nx,nx),symmetric=True)
Y1 = cp.Variable((nx,nx),symmetric=True)
Z = cp.Variable((nz,nz),symmetric=True)

nu = cp.Variable()

#Constraints
constraints = [nu >= 0]

constraints += [cp.bmat([[A@Y1 + Y1@A.T + B2@Cn + Cn.T@B2.T, A + An.T + B2@Dn@C2,               B1 + B2@Dn@D21    ],
                         [(A + An.T + B2@Dn@C2).T,           X1@A + A.T@X1 + Bn@C2 + C2.T@Bn.T, X1@B1+Bn@D21      ],
                         [(B1 + B2@Dn@D21).T,                (X1@B1+Bn@D21).T,                  -1*np.identity(nw)]]) << 0]

constraints += [cp.bmat([[X1,                       np.identity(nx),            Y1@C1.T + Cn.T@D12.T  ],
                         [np.identity(nx),          Y1,                         C1.T + C2.T@Dn.T@D12.T],
                         [(Y1@C1.T + Cn.T@D12.T).T, (C1.T + C2.T@Dn.T@D12.T).T, Z                     ]]) >> 0]

constraints += [D11 + D12@Dn@D21 == 0]

constraints += [cp.bmat([[X1,              np.identity(nx)],
                         [np.identity(nx), Y1             ]]) >> 0]

constraints += [cp.trace(Z) <= nu]
#Now put it all into a cvxpy problem and solve
prob = cp.Problem(cp.Minimize(nu), constraints)
prob.solve(verbose=False)
#Now we can construct the controller

(P, L, U) = la.lu(np.identity(nx) - X1.value*Y1.value)