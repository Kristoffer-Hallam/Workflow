import numpy as np
from ellipsoid import WGS84
# import numpy.testing as npt
# from scipy.sparse import linalg, diags

#---------------------------------- Functions redefinition --------------------------------------
pi = np.pi
cos = np.cos
sin = np.sin
tan = np.tan
#-------------------------------------- Constants -----------------------------------------------
G = 6.673e-11 # Hofmann-Wellenhof and Moritz G = (6.6742+/-0.001)e-11 m^{3}kg^{-1}s^{-2}
SI2MGAL = 1.0e5

class GGS(WGS84):

    def __init__(self):
        WGS84.__init__(self)

    def prime_curvature(self, latitude):
        r'''Computes the prime vertical radius of curvature (N).

        input >
        latitude:   numpy 1D array - latitude of computation points [degrees].

        output >
        N:          numpy 1D array - prime vertical radius of curvature
                        computed at each latitude.
        '''

        # Squared first eccentricity
        # e2 = (self.a*self.a - self.b*self.b)/(self.a*self.a)

        lat = np.asarray(np.deg2rad(latitude))

        # assert (np.deg2rad(lat)>=np.deg2rad(-90.)).all(), 'Input variable does \
        #         not belong in the established interval OR IS NOT IN RADIANS.'
        # assert (np.deg2rad(lat)<=np.deg2rad(90.)).all(), 'Input variable does \
        #         not belong in the established interval OR IS NOT IN RADIANS.'

        # Squared sine
        sin2lat = sin(lat)
        sin2lat *= sin2lat

        N = self.a/np.sqrt(1. - self.e2*sin2lat)

        return N

    def meridian_curvature(self, latitude):
        r'''Computes the meridian radius of curvature (M).

        input >
        latitude:   numpy 1D array - latitude of computation points [degrees].

        output >
        M:          numpy 1D array - meridian radius of curvature
                        computed at each latitude.
        '''

        # Squared first eccentricity
        # e2 = (self.a*self.a - self.b*self.b)/(self.a*self.a)

        lat = np.asarray(np.deg2rad(latitude))

        # assert (np.deg2rad(lat)>=np.deg2rad(-90.)).all(), 'Input variable does \
        #         not belong in the established interval OR IS NOT IN RADIANS.'
        # assert (np.deg2rad(lat)<=np.deg2rad(90.)).all(), 'Input variable does \
        #         not belong in the established interval OR IS NOT IN RADIANS.'

        # Squared sine
        sin2lat = sin(lat)
        sin2lat *= sin2lat

        # auxiliary variable
        aux = np.sqrt(1. - self.e2*sin2lat)

        M = ( self.a*(1.-self.e2) )/(aux*aux*aux)

        return M

    def geodetic2cartesian(self, longitude, latitude, height):
        r"""
        Makes the transformation from :math:'(\lamb,\phi,\mathfrak{h})'
        to :math:'(x,y,z)'.

        input >

        longitude:   numpy 1D array - geodetic longitude vector [degrees].
        latitude:    numpy 1D array - geodetic latitude vector [degrees].
        height:      numpy 1D array - geometric height vector [m].

        output >

        x:           numpy 1D array - x-component of the Cartesian coordinate system [m].
        y:           numpy 1D array - y-component of the Cartesian coordinate system [m].
        z:           numpy 1D array - z-component of the Cartesian coordinate system [m].
        """

        h = np.asarray(height)
        lat = np.asarray(latitude)
        lon = np.asarray(longitude)

        assert lon.size == lat.size == h.size, 'Dimension mismatch'

        # Prime vertical radius of curvature
        N = GGS().prime_curvature(lat)

        # Conversion to radians
        lat = np.deg2rad(lat)
        lon = np.deg2rad(lon)

        aux = N + height

        x = aux*cos(lat)*cos(lon)
        y = aux*cos(lat)*sin(lon)
        z = ((1. - self.e2)*N + height)*sin(lat)
        return x, y, z

    def rotation_matrix(self, longitude, latitude):
        r'''Computes the elements of the rotation matrix. The rotation matrix
        calculated by this function agrees with the notation developed by
        Soler(1976).

        input >

        longitude:   numpy 1D array - geodetic longitude
                        vector of computation points  [degrees].
        latitude:    numpy 1D array - geodetic latitude
                        vector of computation points [degrees].

        output >
        R:           list of numpy 1D arrays - list of vectors
            containing the elements ij = 11, 12, 13, 21, 22, 23,
            31 and 32 of the rotation matrix evaluated at the
            computation points.
        R[0] = cos(lat)*cos(lon) 
        R[1] = cos(lat)*sin(lon) 
        R[2] = sin(lat)          
        R[3] = -sin(lat)*cos(lon)
        R[4] = -sin(lat)*sin(lon)
        R[5] = cos(lat)          
        R[6] = -sin(lon)         
        R[7] = cos(lon)          
        '''
        lat = np.asarray(latitude)
        lon = np.asarray(longitude)

        assert lat.size == lon.size, 'Dimension mismatch'

        # convert to radians
        lat = np.deg2rad(lat)
        lon = np.deg2rad(lon)

        R11 = cos(lat)*cos(lon)  # ok
        R12 = cos(lat)*sin(lon)  # ok
        R13 = sin(lat)           # ok
        R21 = -sin(lat)*cos(lon) # ok
        R22 = -sin(lat)*sin(lon) # ok
        R23 = cos(lat)           # ok
        R31 = -sin(lon)          # ok
        R32 = cos(lon)           # ok

        R = [R11, R12, R13, R21, R22, R23, R31, R32]
        return R

    def geodetic2spherical(self, longitude, latitude, height):
            r"""
            Makes the transformation from :math:'(\lamb,\phi,\mathfrak{h})'
            to :math:'('\lamb, \theta, r')'.

            input >

            longitude:   numpy 1D array - geodetic longitude vector [degrees].
            latitude:    numpy 1D array - geodetic latitude vector [degrees].
            height:      numpy 1D array - geometric height vector [m].

            output >

            theta:       numpy 1D array - spherical colatitude of the GSS [degrees].
            lamb:        numpy 1D array - spherical longitude of the GSS [degrees].
            r:           numpy 1D array - radial distance of the GSS [m].
            """

            h = np.asarray(height)
            lat = np.asarray(latitude)
            lon = np.asarray(longitude)

            assert lon.size == lat.size == h.size, 'Dimension mismatch'

            # Prime vertical radius of curvature
            N = GGS().prime_curvature(lat)

            # Conversion to radians
            lat = np.deg2rad(lat)
            lon = np.deg2rad(lon)

            aux = N + height

            # Cartesian coordinates
            x = aux*cos(lat)*cos(lon)
            y = aux*cos(lat)*sin(lon)
            z = ((1. - self.e2)*N + height)*sin(lat)

            # Spherical coordinates
            r = (x*x + y*y + z*z)**0.5
            theta = np.arccos(z*r**(-1))
            lamb = np.arctan(y/x)

            return lamb, theta, r

# print CGS_t0_GGS().e2
# print CGS_t0_GGS().geodetic2cartesian(-55.,-25.,1000.)
# print CGS_to_GGS().rotation_matrix(-55.,-25.)

# a = np.asarray(1.)
# print a.size