import numpy as np
import numpy.testing as npt
from scipy.sparse import linalg, diags

#---------------------------------- Functions redefinition --------------------------------------
pi = np.pi
cos = np.cos
sin = np.sin
tan = np.tan
#-------------------------------------- Constants -----------------------------------------------
G = 6.673e-11 # Hofmann-Wellenhof and Moritz G = (6.6742+/-0.001)e-11 m^{3}kg^{-1}s^{-2}
SI2MGAL = 1.0e5

def prime_curvature(a, b, latitude):
    '''Computes the prime vertical radius of curvature (N).
     
    input >
    a:          float - semimajor axis [m]
    b:          float - semiminor axis [m]
    latitude:   numpy 1D array - latitude of computation points [degrees].
    
    output >
    N:          numpy 1D array - prime vertical radius of curvature
                    computed at each latitude.
    '''
    
    assert a > b, 'major semiaxis must be greater than minor semiaxis'
    assert a > 0, 'major semiaxis must be nonnull'
    assert b > 0, 'minor semiaxis must be nonnull'
    
    # Squared first eccentricity
    e2 = (a*a - b*b)/(a*a)
    
    lat = np.asarray(np.deg2rad(latitude))
    
    assert (np.deg2rad(lat)>=np.deg2rad(-90.)).all(), 'Input variable does \
            not belong in the established interval OR IS NOT IN RADIANS.'
    assert (np.deg2rad(lat)<=np.deg2rad(90.)).all(), 'Input variable does \
            not belong in the established interval OR IS NOT IN RADIANS.'
    
    # Squared sine
    sin2lat = sin(lat)
    sin2lat *= sin2lat
    
    N = a/np.sqrt(1. - e2*sin2lat)
    
    return N

def meridian_curvature(a, b, latitude):
    '''Computes the meridian radius of curvature (M).
     
    input >
    a:          float - semimajor axis [m]
    b:          float - semiminor axis [m]
    latitude:   numpy 1D array - latitude of computation points [degrees].
    
    output >
    M:          numpy 1D array - meridian radius of curvature
                    computed at each latitude.
    '''
    
    assert a > b, 'major semiaxis must be greater than minor semiaxis'
    assert a > 0, 'major semiaxis must be nonnull'
    assert b > 0, 'minor semiaxis must be nonnull'
    
    # Squared first eccentricity
    e2 = (a*a - b*b)/(a*a)
    
    lat = np.asarray(np.deg2rad(latitude))
    
    assert (np.deg2rad(lat)>=np.deg2rad(-90.)).all(), 'Input variable does \
            not belong in the established interval OR IS NOT IN RADIANS.'
    assert (np.deg2rad(lat)<=np.deg2rad(90.)).all(), 'Input variable does \
            not belong in the established interval OR IS NOT IN RADIANS.'
    
    # Squared sine
    sin2lat = sin(lat)
    sin2lat *= sin2lat
    
    # auxiliary variable
    aux = np.sqrt(1. - e2*sin2lat)
    
    M = ( a*(1.-e2) )/(aux*aux*aux)
    
    return M

def geodetic2cartesian(height, latitude, longitude, major_semiaxis, minor_semiaxis):
    r"""
    Makes the transformation from :math:'(\mathfrak{h},\phi,\lamb)'
    to :math:'(x,y,z)'.
    
    input >
    
    height:      numpy 1D array - geometric height vector [m].
    latitude:    numpy 1D array - geodetic latitude vector [degrees].
    longitude:   numpy 1D array - geodetic longitude vector [degrees].
    major_semiaxis: float - major semiaxis of the reference ellipsoid [m].
    minor_semiaxis: float - minor semiaxis of the reference ellipsoid [m].
    
    output >

    x:           numpy 1D array - x-component of the Cartesian coordinate system [m].
    y:           numpy 1D array - y-component of the Cartesian coordinate system [m].
    z:           numpy 1D array - z-component of the Cartesian coordinate system [m].
    """
    
    h = np.asarray(height)
    lat = np.asarray(latitude)
    lon = np.asarray(longitude)
    
    assert h.size == lat.size == lon.size, 'Dimension mismatch'
    assert major_semiaxis > minor_semiaxis, 'major_axis must be greater than minor_axis'

    # Squared first eccentricity
    e2 = (major_semiaxis*major_semiaxis - minor_semiaxis*minor_semiaxis)/(major_semiaxis*major_semiaxis)
    
    # Prime vertical radius of curvature
    N = prime_curvature(major_semiaxis, minor_semiaxis, lat)

    # Conversion to radians
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    
    aux = N + height
    
    x = aux*cos(lat)*cos(lon)
    y = aux*cos(lat)*sin(lon)
    z = ((1. - e2)*N + height)*sin(lat)
    
    return x, y, z

def cartesian2geodetic_approx(X, Y, Z, major_semiaxis, minor_semiaxis):
    '''Convert geocentric Cartesian coordinates into geocentric geodetic
    coordinates by using an approximated formula (Hofmann-Wellenhof and
    Moritz, 2005).
    
    reference
    Hofmann-Wellenhof, B. and Moritz, H., 2005, Physical Geodesy. Springer
    
    input >
    X:           numpy 1D array or float - x-component of the Cartesian
            coordinates system [m].
    Y:           numpy 1D array or float - y-component of the Cartesian
            coordinates system [m].
    Z:           numpy 1D array or float - z-component of the Cartesian
            coordinates system [m].
    major_semiaxis: float - major semiaxis of the reference 
                    ellipsoid [m].
    minor_semiaxis: float - minor semiaxis of the reference 
                    ellipsoid [m].
    itmax:          int - maximum number of iterations in the 
                Hirvonen-Moritz algorithm.
                
    output >
    height:      numpy 1D array or float - geometric height vector [m].
    latitude:    numpy 1D array or float - geodetic latitude vector [deg].
    longitude:   numpy 1D array or float - geodetic longitude vector [deg].
    '''
    
    x = np.asarray(X)
    y = np.asarray(Y)
    z = np.asarray(Z)
    
    assert x.size == y.size == z.size, 'Dimension mismatch'
    assert major_semiaxis > minor_semiaxis, 'major_axis must be \
    greater than minor_axis'
    
    # horizontal distance
    p = np.sqrt(x*x + y*y)
    
    # null and non-null horizontal distances
    p_non_null = (p >= 1e-8)
    p_null = np.logical_not(p_non_null)
    
    lon = np.zeros_like(x)
    lat = np.zeros_like(x)
    height = np.zeros_like(x)    
    
    # define the coordinates for null horizontal distances
    lon[p_null] = 0.
    height[p_null] = np.abs(z[p_null]) - minor_semiaxis
    lat[p_null] = np.sign(z[p_null])*pi*0.5
    
    # Squared first eccentricity
    e2 = (major_semiaxis*major_semiaxis - minor_semiaxis*minor_semiaxis)/(major_semiaxis*major_semiaxis)
    
    # Squared second eccentricity
    elinha2 = (major_semiaxis*major_semiaxis - minor_semiaxis*minor_semiaxis)/(minor_semiaxis**2.)
    
    # auxiliary variable
    theta = np.arctan(z[p_null]*major_semiaxis/(p[p_non_null]*minor_semiaxis))
    sintheta = np.sin(theta)
    costheta = np.cos(theta)
    
    aux1 = z[p_non_null] + elinha2*minor_semiaxis*sintheta*sintheta*sintheta
    aux2 = p[p_non_null] + e2*major_semiaxis*costheta*costheta*costheta
    
    lat[p_non_null] = np.arctan2(aux1, aux2)
    lon[p_non_null] = np.arctan2(y[p_non_null], x[p_non_null])
    
    sinlat = np.sin(lat[p_non_null])
    N = major_semiaxis/np.sqrt(1. - e2*sinlat*sinlat)
    
    height[p_non_null] = p[p_non_null]/np.cos(lat[p_non_null]) - N
    
    # convert latitude and longitude from radians to degree
    latitude = np.rad2deg(lat)
    longitude = np.rad2deg(lon)    
    
    return height, latitude, longitude

def cartesianDist(x0, y0, z0, x, y, z):
    '''Calculates the Euclidian distance of two points in Cartesian coordinates.
    
     ----------------------------------------
    | INITIATE THE COORDINATES AS ARRAYS  !!!|
     ----------------------------------------
     
    input >    
    x0:   int or array - x-position vector of the source point    
    y0:   int or array - y-position vector of the source point
    z0:   int or array - z-position vector of the source point    
    x:    int or array - x-position vector of the observation point    
    y:    int or array - y-position vector of the observation point    
    z:    int or array - z-position vector of the observation point
    
    output >
    Rcart: array - vector containing all distance values in Cartesian
    coordinates.
    '''
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    
    assert x.shape == y.shape == z.shape,     'Dimension mismatch'
    
    x0 = np.asarray(x0)
    y0 = np.asarray(y0)
    z0 = np.asarray(z0)
    
    assert x0.shape == y0.shape == z0.shape,  'Dimension mismatch'
    Rcart = np.sqrt((x-x0)**2 + (y-y0)**2 + (z-z0)**2)
    assert np.alltrue(Rcart != 0), 'Distance in cartesian coordinates cannot be zero to avoid instability!'
    return Rcart

def cartKernel(x0, y0, z0, x, y, z):
    '''Calculates the kernel of gravitacional potential in geodetic
    coordinates. One can make, for instance, density equal to unity.
    
    input >
    x0:     array - x-position vector of the source point
    y0:     array - y-position vector of the source point
    z0:     array - z-position vector of the source point
    x:      array - x-position vector of the observation point
    y:      array - y-position vector of the observation point
    z:      array - z-position vector of the observation point
    
    output >
    1/R:     array - matrix containing the configuration kernel of the
    gravitacional potential.
    '''
    R = cartesianDist(x0, y0, z0, x, y, z)
    assert np.alltrue(R != 0), 'Distance in cartesian coordinates cannot be zero to avoid instability!'
    kernel = 1./R
    return kernel    

def geodeticDist(major_axis, minor_axis, N0, h0, latitude0, longitude0, N, h, latitude, longitude):
    '''Calculates the Euclidian distance of two points in geodetic coordinates.
    
     ----------------------------------------------------
    |   latitudes and longitudes must be in DEGREES!!!   |
     ----------------------------------------------------
    
    input >
    major_axis:     float - ellipsoid's equatorial radius
    minor_axis:     float - ellipsoid's polar radius
    N0:             array - value or list of numbers related
                        to the prime vertical radius of
                        curvature of source points
    h0, lat0, lon0: numpy arrays 1D or floats - coordinates of
                        the computation points referred to the
                        Geocentric Coordinate System. The values
                        are given in [meters, degrees, degrees]
    N:              array - value or list of numbers related to 
                        the prime vertical radius of curvature
                        of observation points
    h, lat, lon:    numpy arrays 1D or floats - coordinates of
                        the computation points referred to the
                        Geocentric Coordinate System. The values
                        are given in [meters, degrees, degrees]
    
    output >
    Rgeodetic:  array - vector containing all distance values in geodetic
    coordinates.
    '''
    assert major_axis > minor_axis, 'major semiaxis must be \
                                    greater than minor semiaxis'
    assert major_axis > 0, 'major semiaxis must be nonnull'
    assert minor_axis > 0, 'minor semiaxis must be nonnull'
    e2 = (major_axis*major_axis - minor_axis*minor_axis)/(major_axis*major_axis)
    
    h = np.asarray(h)
    lat = np.asarray(np.deg2rad(latitude))
    lon = np.asarray(np.deg2rad(longitude))

    assert h.size == lat.size == lon.size, 'Dimension mismatch'
    
    h0 = np.asarray(h0)
    lat0 = np.asarray(np.deg2rad(latitude0))
    lon0 = np.asarray(np.deg2rad(longitude0))
    
    assert np.alltrue(h0 <= 0.),   'Sources must be embedded inside \
    or on the surface of the Earth'
    assert h0.shape == lat0.shape == lon0.shape, 'Dimension mismatch'
    
    coslat = np.cos(lat)
    coslon = np.cos(lon)
    sinlat = np.sin(lat)
    sinlon = np.sin(lon)
    
    coslat0 = np.cos(lat0)
    coslon0 = np.cos(lon0)
    sinlat0 = np.sin(lat0)
    sinlon0 = np.sin(lon0)    
    
    A = (N+h)*coslat*coslon - (N0+h0)*coslat0*coslon0
    B = (N+h)*coslat*sinlon - (N0+h0)*coslat0*sinlon0
    C = ((1.-e2)*N+h)*sinlat - ((1.-e2)*N0+h0)*sinlat0
    Rgeodetic = np.sqrt(A*A + B*B + C*C)
    assert np.alltrue(Rgeodetic != 0), 'Distance in geodetic coordinates cannot be zero to avoid instability!'
    
    return Rgeodetic

def geodeticKernel(a, k2, h0, phi0, lamb0, h, phi, lamb):
    '''Calculates the kernel of gravitacional potential in geodetic
    coordinates. One can make, for instance, density equal to unity.
    
    input >
    h0:    float - geometric altitude position vector of the source point
    phi0:  float - geodetic latitude position vector of the source point
    lamb0: float - geodetic longitude position vector of the source point
    h:     array - geometric altitude position vector of the observation point
    phi:   array - geodetic latitude position vector of the observation point
    lamb:  array - geodetic longitude position vector of the observation point
    
    output >
    V:     array - matrix containing the gravitacional potential.
    '''
    Rgeod = geodeticDist(a, k2, h0, phi0, lamb0, h, phi, lamb)
    kernel = 1./Rgeod
    return kernel

def geodetic_dS(h, phi, lamb, spacing):
    '''Calculates the element of area in geodetic coordinates.
    
     ------------------------------------
    | Coodinate lat must be in radians!!!|
     ------------------------------------
     
    input >
    h:             array - list of height values    
    phi:           array - list of latitude values
    lamb:          array - list of longitude values.
    spacing:       int   - grid spacing
    
    output >
    A:             array - matrix containing the inverse of 
    elements of area.
    '''
    N, M = con.radii_curvature(phi)
    phimax = phi+spacing/2.
    phimin = phi-spacing/2.
    dphi = phimax - phimin
    lambmax = lamb+spacing/2.
    lambmin = lamb-spacing/2.
    dlamb = lambmax - lambmin
    dS = (M+h)*(N+h)*cos(phi)*dlamb*dphi
    A = G*SI2MGAL*2*pi*(1./dS)
    return A

def rotation_matrix(latitude, longitude):
    '''Computes the elements of the rotation matrix. The rotation matrix
    calculated by this function agrees with the notation developed by
    Soler(1976).

    input >

    latitude:    numpy 1D array - geodetic latitude
                    vector of computation points [degrees].
    longitude:   numpy 1D array - geodetic longitude
                    vector of computation points  [degrees].
                    
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