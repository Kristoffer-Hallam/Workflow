from __future__ import division
import numpy as np
from scipy.special import lpmv
import math
cos = np.cos
sin = np.sin

theta = np.linspace(-np.pi, np.pi, 3)
m = np.array([0, 1, 2])
n = np.array([0, 1, 2])
A = lpmv(m, n,cos(theta))

# def legendre_poly(n, m, colatitude):
#     r'''Calculates the Legendre polynomials through
#     equation 1-62 of Heiskanen and Moritz(1967)
    
#     input >
#     n, m:              int   - degree and order of Legendre polynomials
#     colatitude:        array - colatitude coordinates
#     '''
#     colat = np.deg2rad(colatitude)
#     t = cos(colat)
#     w1 = (0.5)**n
#     w2 = np.sqrt((1-t*t)**m)

#     if type((n-m)/2) == int:
#         r = int((n-m)/2)
#         print 1
#     else:
#         r = int((n-m-1)/2)
#         print 2
#     print 'r =', r
#     P = 0
#     for k in range(0,r):
#         print 'k =',  k
#         aux1 = math.factorial(2*n-2*k)*t**(n-m-2*k)
#         aux2 = (math.factorial(k)*math.factorial(n-k)*math.factorial(n-m-2*k))**(-1)
#         P += aux1*aux2*(-1)**k
#     Pnm = w1*w2*P
#     print w1, w2
#     return Pnm

print A