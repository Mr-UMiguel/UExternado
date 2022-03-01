import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import quadprog
import cvxopt

returns = pd.read_excel('./TP/datasets/retornos.xlsx')


mu = np.array(returns.agg(np.mean))
try:
    vcov, corr = np.matrix(returns.cov()), np.matrix(returns.corr())
except:
    raise ValueError("returns must be a pandas data frame")

# mu es un vector de retornos esperados de cada activo 
mu = np.array(returns.agg(np.mean))

# Varianza y desviación estándar de los activos
var = np.diag(vcov)
sigma = np.sqrt(var)

################################################################
# Construcción de los portafolios óptimos

# longitud del vector mu
n = len(mu)

# Creamos un vector de unos de longitud n
ones = np.ones(n)

# Solucionamos el ejercicio de optimización
x = mu @ np.linalg.inv(vcov) @ mu
y = mu @ np.linalg.inv(vcov) @ ones
z = ones @ np.linalg.inv(vcov) @ ones

d = x*z - y**2
g = (np.linalg.solve(vcov,ones) * np.array(x)-np.linalg.solve(vcov,mu)*np.array(y)) * 1/d
h = (np.linalg.solve(vcov,mu) * np.array(z)-np.linalg.solve(vcov,ones)*np.array(y)) * 1/d


# Simulamos 1000 portafolios
Rp = np.linspace(start=np.min(mu), stop=np.max(mu),num=1000)
wpo = np.zeros((1000,n))
sigmapo = np.zeros(1000)
rpo = np.zeros(1000)


wpmv = np.linalg.solve(vcov,ones) * 1/z
wpmv = np.squeeze(np.asarray(wpmv))
rpmv = wpmv @ mu
rpmv = np.asarray(rpmv).reshape(-1)
sigmapmv = np.sqrt(wpmv@vcov@np.transpose(wpmv))
sigmapmv = np.asarray(sigmapmv).reshape(-1)

rf = 0
er = mu - rf
zi = np.linalg.solve(vcov,er)
wpt = np.squeeze(np.asarray(zi / sum(zi)))
rpt = np.asarray(wpt @ mu).reshape(-1)

from cvxopt import matrix
from cvxopt import solvers

r_avg = matrix(mu)
sigma = matrix(vcov)

P = sigma
q = matrix(np.zeros((n,1)))

G = matrix(np.concatenate((
    -np.transpose(np.array(r_avg)),
    -np.identity(n)),0))

h = matrix(np.concatenate((
    -np.ones((1,1))*rpt,
    np.zeros((n,1))),0))


# A = matrix(1.0,(1,n))
# b = matrix(1.0)
diag = np.diag(np.full(5,1))
A = matrix(np.concatenate((r_avg,np.ones(n).reshape(n,1),diag),axis=1))
b = matrix(np.concatenate((rpmv,np.array([1]),np.zeros(n))))

sol =  solvers.qp(P, q, G, h, A=None, b=None)
woptimal = sol['x']


print(woptimal,sum(woptimal))
fig = plt.figure()
plt.bar(returns.columns,woptimal)
plt.show()




# def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None):
#     """
#     La programación cuadrática (QP) se escribe en su forma estándar como:

#     minimzar (1/2)(x^t)Px+(q^t)x

#     sujeto a  Gx <= h
#                 Ax = b
    
#     Dónde, x es el vector de variables a optimizar x1, ... , xn. La matriz P y el vector q
#     usualmente se usan para definir cualquier función cuadrática objetivo de esas variables x
#     mientras que las parejas matriz-vector (G,h) y (A,b) son las restricciónes de desigualdad 
#     e igualdad, respectivamente. 

#     El vector de desigualdad se aplica coordenada a coordenada, por ejemplo x >= 0 quiere decir que
#     cada coordenada de x es positiva

#     Además se asume, sin pérdida de generalidad, que la matriz P es simétrica.

#     fuente: 
#     https://scaron.info/blog/quadratic-programming-in-python.html  
    
#     """

#     qp_G = .5 * (P + P.T)   # make sure P is symmetric
#     qp_a = -q
#     if A is not None:
#         qp_C = np.matrix(np.concatenate((G.T,A),axis=1))
#         qp_b = np.hstack([b, h])
#         meq = A.shape[0]
#     else:  # no equality constraint
#         qp_C = -G.T
#         qp_b = -b
#         meq = 0

#     print(
#     qp_G,
#     "\n",
#     qp_a,
#     "\n",
#     qp_C,
#     "\n",
#     qp_b
#     )
#     return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]

# r_avg = np.array(mu).reshape(n,1)
# sigma = np.matrix(vcov)

# P = sigma
# # q = matrix(np.zeros((n,1)))
# q = np.zeros(n).reshape(-1)

# G = np.matrix(np.concatenate((
#     -np.transpose(np.array(r_avg)),
#     -np.ones(n).reshape(1,n)),axis=0))


# h = np.zeros(n).reshape(-1)

# A = np.diag(np.full(n,1))
# b = np.concatenate((-1*rpmv,np.array([1]).reshape(-1)),axis=0)

# res = quadprog_solve_qp(P, q, G=G, h=h, A=A, b=b)
# print(res,sum(res))

# fig = plt.figure()
# plt.bar(returns.columns,res)
# plt.show()