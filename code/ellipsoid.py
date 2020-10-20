class WGS84():
    '''This class serves as reference coordinate system by passing
    the parameters of the reference elipsoid WGS84. These are the
    following:
    a:          float - semimajor axis [m]
    b:          float - semiminor axis [m]
    GM:         float - geocentric gravitational constant of the
                        Earth (including the atmosphere) [m**3/s**2]
    omega:      float - Earth's angular velocity [rad/s]
    e2:         float - first eccentricity squared
    k2:         float - second eccentricity squared
    '''
    def __init__(self):
        self.a = 6378137.0
        self.f = 1.0/298.257223563 # WGS84
        self.GM = 3986004.418*(10**8)
        self.omega = 7292115*(10**-11)
        self.b = self.a*(1-self.f)
        self.e2 = (self.a**2-self.b**2)/(self.a**2)
        self.k2 = 1-self.e2