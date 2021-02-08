import numpy as np
from coord import GGS

class GravConstants():
    G = 6.673e-11 # Hofmann-Wellenhof and Moritz G = (6.6742+/-0.001)e-11 m^{3}kg^{-1}s^{-2}
    SI2MGAL = 1.0e5

class Sphere():

    radius = 1.0

class EqLayer(GravConstants, Sphere, GGS):
    r'''
    Constructs an equivalent layer object.

    input  >
    longitude, latitude, height:     1D arrays  ---->  Curvilinear coordinates

    output >
    EqLayer object
    '''
    def __init__(self, longitude, latitude, height):
        # super().__init__(self)
        GGS.__init__(self)
        self.lon = np.asarray(longitude)
        self.lat = np.asarray(latitude)
        self.height = np.asarray(height)
        assert type(self.lon) == type(self.lat) == type(self.height), \
            'Input coordinate variables do not have the same type!'
        assert self.lon.size == self.lat.size == self.height.size, \
            'Input coordinate variables do not have the same size!'
        if self.lon.size == 1 or self.lat.size == 1 or self.height.size == 1:
            self.radius = np.asarray(self.radius)
        elif self.lon.size > 1 or self.lat.size > 1 or self.height.size > 1:
            self.radius = np.zeros_like(self.lon) + np.asarray(self.radius)
        else:
            self.radius = np.asarray(self.radius)

    def build_layer(self, x, y, z):
        r'''Function that builds the equivalent layer by assembling
        arrays to its columns

        input >
        x:      array - x coordinates of sources
        y:      array - y coordinates of sources
        z:      array - z coordinates of sources

        output >
        layer:      array - equivalent layer
        '''
        xlay = np.asarray(x)
        ylay = np.asarray(y)
        zlay = np.asarray(z)
        # ASSERTING SAME DIMENSIONS
        assert xlay.size == ylay.size == zlay.size, 'Dimension mismatch.'

        radius = np.zeros_like(xlay) + (3./(4.*np.pi))**(1/3.)
        density = np.ones_like(xlay)

        layer = np.zeros((xlay.size,5))
        layer[:,0] = xlay
        layer[:,1] = ylay
        layer[:,2] = zlay
        layer[:,3] = self.radius
        layer[:,4] = density
        return layer

    def designMat(self, x, y, z, R, model):
        r'''Calculates the design matrix (H) of the linear equation

        :math:  \delta g = H \rho .

        To obtain the H matrix in geodetic coordinates from the
        correspondent Cartesian coordinates.

        input >
        x:      array - x-position matrix of the observation point
        y:      array - y-position matrix of the observation point
        z:      array - z-position matrix of the observation point
        R:      list of numpy arrays 1D - list containing vectors with 
                    the elements of the rotation matrix evaluated at the
                    computation points. The vectors containd the elements
                    R11, R12, R13, R21, R22, R23, R31 and R32, 
                    respectively. The element R33 is equal to zero.    
        model:  array - lists of source Cartesian coordinates and physical 
                        property

        output >
        H:      array - design matrix
        '''
        assert x.shape == y.shape == z.shape,   'Dimension mismatch.'
        assert len(R) == 8, 'rotation matrix must contain 8 numpy arrays'
        for Ri in R:
            assert Ri.size == x.size, 'the vectors of R must have the same number \
                                        of elements as x, y and z'
        H = np.empty((x.size, len(model)))
        radius = model[0][3]
        for j, m in enumerate(model):
            # if j%500 == 0:
            #     print(j)
            A = x-m[0]
            B = y-m[1]
            C = z-m[2]
            r = np.sqrt(A*A + B*B + C*C)
            H[:,j] = (R[0]*A + R[1]*B + R[2]*C)/(r*r*r)
        H *= self.G*self.SI2MGAL*(4./3.)*np.pi*radius*radius*radius
        return H

    def continuation(self, *args):
        r'''Continuates the harmonic field to specified observation points.
        input >
        args:                 tuple     -> (obs_lon, obs_lat, obs_height, laydepth, param)
        Each args variable within the parenthesis is related to the observables.
        OBS: All variables inside the tuple args are required! They represent:

            obs_lon, obs_lat: 1D arrays -> horizontal coordinates of the continued field
            obs_height:       float     -> altitude of the continued field (must be positive)
            laydepth:         float     -> depth of point-masses (must be negative)
            param:            1D arrays -> equivalent layer coefficients
        '''
        lon = np.asarray(args[0])
        lat = np.asarray(args[1])
        # h = np.zeros_like(lon) + args[2]
        x, y, z = GGS().geodetic2cartesian(lon,lat,np.zeros_like(lon) + args[2])
        R = GGS().rotation_matrix(lon, lat)
        # xlay, ylay, zlay = func2(lon,lat,np.zeros_like(lon)+args[3])
        Layer = EqLayer(self.lon, self.lat, np.zeros_like(self.lon)+args[3])
        xlay, ylay, zlay = GGS().geodetic2cartesian(self.lon,self.lat, \
            np.zeros_like(self.lon)+args[3])
        lay = Layer.build_layer(xlay, ylay, zlay)
        T = Layer.designMat(x, y, z, R, lay)

        if args[2] < 0.:
            print('The transformation is a downward continuation')
        else:
            print('The transformation is an upward continuation')
        continued_data = np.dot(T, np.asarray(args[4]))
        return continued_data

# 1 element for each variable
# longitude = -55.
# latitude = -25.
# height = 1000.
# 3 element for each variable making vectors
# longitude = np.asarray([x for x in xrange(-55,-49,2)])
# latitude = np.asarray([x for x in xrange(-25,-19,2)])
# height = np.zeros_like(longitude) + 1000.

# Observation points object
# observ = GGS()
# x, y, z = observ.geodetic2cartesian(longitude,latitude,height)

# Equivalent layer object
# Lay = EqLayer(longitude,latitude,height)
# xlay, ylay, zlay = Lay.geodetic2cartesian(longitude,latitude,np.zeros_like(longitude)-5000.)
# layer = Lay.build_layer(xlay, ylay, zlay)

# H = Lay.designMat(x, y, z, observ.rotation_matrix(longitude, latitude), layer)
# print H
# print Lay.continuation(10.)