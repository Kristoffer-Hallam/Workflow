import numpy as np
# from ellipsoid import WGS84
from scripts import ellipsoid
#---------------------------------- Functions redefinition --------------------------------------
pi = np.pi
cos = np.cos
sin = np.sin
tan = np.tan

#-------------------------------------- Constants -----------------------------------------------
G = 6.673e-11 # Hofmann-Wellenhof and Moritz G = (6.6742+/-0.001)e-11 m^{3}kg^{-1}s^{-2}
SI2MGAL = 1.0e5

# class Norm_Grav(WGS84):
class Norm_Grav(ellipsoid.WGS84):

    def __init__(self):
        ellipsoid.WGS84.__init__(self)

    def somigliana(self, phi):
       r'''This function calculates the normal gravity by using
       the Somigliana's formula.
       
       input >
       phi: array containing the geodetic latitudes [degree]
       
       output >
       gamma: array containing the values of normal gravity
              on the surface of the elipsoid for each geodetic
              latitude [mGal]
       '''
       phi = np.asarray(phi)
       a2 = self.a*self.a
       b2 = self.b*self.b
       E = np.sqrt(a2 - b2)
       elinha = E/self.b
       bE = self.b/E
       Eb = E/self.b
       atg = np.arctan(Eb)
       q0 = 0.5*((1+3*(bE**2))*atg - (3*bE))
       q0linha = 3.0*(1+(bE**2))*(1-(bE*atg)) - 1
       m = (self.omega**2)*(a2)*self.b/self.GM
       aux = elinha*q0linha/q0
       gammaa = (self.GM/(self.a*self.b))*(1-m-(m/6.0)*aux)
       gammab = (self.GM/a2)*(1+(m/3.0)*aux)
       aux = np.deg2rad(phi)
       s2 = np.sin(aux)**2
       c2 = np.cos(aux)**2
       # the 10**5 converts from m/s**2 to mGal
       gamma = (10**5)*((self.a*gammaa*c2) + (self.b*gammab*s2))/np.sqrt((a2*c2) + (b2*s2))
       return gamma
       
    def closedform(self, h, phi):
       r'''This function calculates the normal gravity by using
       a closed-form formula.

       input:
       phi: array containing the geodetic latitudes [degree]
       h: array containing the normal heights [m]

       output:
       gamma: array containing the values of normal gravity
              on a chosen height relative to the surface of
              the elipsoid for each geodetic latitude [mGal]
       '''
       h = np.asarray(h)
       phi = np.asarray(phi)
       assert h.size == phi.size, 'Dimension mismatch'
       a2 = self.a*self.a
       b2 = self.b*self.b
       E = np.sqrt(a2 - b2)
       E2 = E**2
       bE = self.b/E
       Eb = E/self.b
       atanEb = np.arctan(Eb)
       phirad = np.deg2rad(phi)
       tanphi = np.tan(phirad)
       cosphi = np.cos(phirad)
       sinphi = np.sin(phirad)
       beta = np.arctan(self.b*tanphi/self.a)
       sinbeta = np.sin(beta)
       cosbeta = np.cos(beta)
       zl = self.b*sinbeta+h*sinphi
       rl = self.a*cosbeta+h*cosphi
       zl2 = zl**2
       rl2 = rl**2
       dll2 = rl2-zl2
       rll2 = rl2+zl2
       D = dll2/E2
       R = rll2/E2
       cosbetal = np.sqrt(0.5*(1+R) - np.sqrt(0.25*(1+R**2) - 0.5*D))
       cosbetal2 = cosbetal**2
       sinbetal2 = 1-cosbetal2
       bl = np.sqrt(rll2 - E2*cosbetal2)
       bl2 = bl**2
       blE = bl/E
       Ebl = E/bl
       atanEbl = np.arctan(Ebl)
       q0 = 0.5*((1+3*(bE**2))*atanEb - (3*bE))
       q0l = 3.0*(1+(blE**2))*(1-(blE*atanEbl)) - 1.
       W = np.sqrt((bl2+E2*sinbetal2)/(bl2+E2))

       gamma = self.GM/(bl2+E2) - cosbetal2*bl*self.omega**2
       gamma += (((self.omega**2)*a2*E*q0l)/((bl2+E2)*q0))*(0.5*sinbetal2 - 1./6.)
       # the 10**5 converts from m/s**2 to mGal
       gamma = (10**5)*gamma/W
       return gamma