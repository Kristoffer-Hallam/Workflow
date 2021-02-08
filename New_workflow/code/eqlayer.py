import numpy as np
# from pickle_object import saving_data as pm
# from cartography import ellipsoid

#-------------------------------------- Constants -----------------------------------------------
G = 6.673e-11 # Hofmann-Wellenhof and Moritz G = (6.6742+/-0.001)e-11 m^{3}kg^{-1}s^{-2}
SI2MGAL = 1.0e5
a = 6378137.0
f = 1.0/298.257223563 # WGS84
GM = 3986004.418*(10**8)
omega = 7292115*(10**-11)
b = a*(1-f)
e2 = (a**2-b**2)/(a**2)
k2 = 1-e2
#---------------------------------- Functions redefinition --------------------------------------
pi = np.pi
cos = np.cos
sin = np.sin
tan = np.tan

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
    
    lat = np.asarray(latitude)
    
    assert (np.deg2rad(lat)>=np.deg2rad(-90.)).all(), 'Input variable does \
            not belong in the established interval OR IS NOT IN RADIANS.'
    assert (np.deg2rad(lat)<=np.deg2rad(90.)).all(), 'Input variable does \
            not belong in the established interval OR IS NOT IN RADIANS.'    
    
    # Squared sine
    sin2lat = np.sin(np.deg2rad(lat))
    sin2lat *= sin2lat
    
    N = a/np.sqrt(1. - e2*sin2lat)
    
    return N

def geodetic2cartesian(height, latitude, longitude,
            major_semiaxis, minor_semiaxis):
    r"""
    Makes the transformation from :math:'(\mathfrak{h},\phi,\lamb)'
    to :math:'(x,y,z)'.
    
    input >
    
    height:      numpy 1D array - geometric height 
                    vector [m].
    latitude:    numpy 1D array - geodetic latitude
                    vector [degrees].
    longitude:   numpy 1D array - geodetic longitude
                    vector [degrees].
    major_semiaxis: float - major semiaxis of the 
                    reference ellipsoid [m].
    minor_semiaxis: float - minor semiaxis of the 
                    reference ellipsoid [m].
    
    output >

    x:           numpy 1D array - x-component of the Cartesian 
            coordinate system [m].
    y:           numpy 1D array - y-component of the Cartesian 
            coordinate system [m].
    z:           numpy 1D array - z-component of the Cartesian 
            coordinate system [m].
    """
    
    h = np.asarray(height)
    lat = np.asarray(latitude)
    lon = np.asarray(longitude)
    
    assert h.size == lat.size == lon.size, 'Dimension mismatch'
    assert major_semiaxis > minor_semiaxis, 'major_axis must be \
    greater than minor_axis'
    
    # Prime vertical radius of curvature
    N = prime_curvature(major_semiaxis, minor_semiaxis, lat)
    
    # Conversion to radians
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    
    # Squared first eccentricity
    e2 = (major_semiaxis*major_semiaxis - minor_semiaxis*minor_semiaxis)/(major_semiaxis*major_semiaxis)
    
    aux = N + height
    
    x = aux*np.cos(lat)*np.cos(lon)
    y = aux*np.cos(lat)*np.sin(lon)
    z = ((1. - e2)*N + height)*np.sin(lat)
    
    return x, y, z

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

def build_layer(x, y, z):
    '''Function that builds the equivalent layer by assembling
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
    layer[:,3] = radius
    layer[:,4] = density
    return layer

def designMat(x, y, z, R, model):
    '''Calculates the design matrix (H) of the linear equation
    
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
        A = x-m[0]
        B = y-m[1]
        C = z-m[2]
        r = np.sqrt(A*A + B*B + C*C)
        H[:,j] = (R[0]*A + R[1]*B + R[2]*C)/(r*r*r)
    H *= G*SI2MGAL*(4./3.)*np.pi*radius*radius*radius
    return H

def L1_norm_estim(*args):#(residuals, param, G, it_max=3):
    '''Function that estimates the physical properties of the point-masses
    through L1 norm of the residuals
    
    :math:  \delta \mathbf{g} = \mathbf{H} \mathbf{\rho} .
    
    input >
    args:           tuple    - (residuals, param, G, it_max)
    where
    {
        residuals:  1D array - vector of residuals
        param:      1D array - vector of parameters obtained by least-squares
        G:          2D array - sensitivity matrix
        it_max:     int      - maximum iteraion number.
    }
    
    output >
    pcl:            1D array - vector of parameters obtained L1 norm of the residuals
    '''
    residuals = np.asarray(args[0])
    param = np.asarray(args[1])
    G = np.asarray(args[2])
    it_max = args[3]
    assert residuals.ndim == 1, 'Array is not a vector'
    assert residuals.ndim == param.ndim, 'Input data array is not compatible with input paramater array'
    assert G.ndim == 2, 'Sensitivity matrix is not a 2D array'
    assert type(it_max) == int, 'Maximum iteration value is not an integer'

    # residuals = np.asarray(residuals)
    # param = np.asarray(param)

    for i in range(it_max):
        res = residuals - np.dot(G, param)
        W = np.dot(G.T, np.diag(1./(np.abs(res)+1e-10)))
        pcl = np.linalg.solve(np.dot(W, G), np.dot(W, residuals))
    return pcl

def L1_norm_estim_regularized(*args):#(residuals, param, G, it_max=3):
    '''Function that estimates the physical properties of the point-masses
    through L1 norm of the residuals
    
    :math:  \delta \mathbf{g} = \mathbf{H} \mathbf{\rho} .
    
    input >
    args:           tuple    - (residuals, param, G, it_max)
    where
    {
        residuals:  1D array - vector of residuals
        param:      1D array - vector of parameters obtained by least-squares
        G:          2D array - sensitivity matrix
        it_max:     int      - maximum iteraion number.
    }
    
    output >
    pcl:            1D array - vector of parameters obtained L1 norm of the residuals
    '''
    residuals = np.asarray(args[0])
    param = np.asarray(args[1])
    G = np.asarray(args[2])
    it_max = args[3]
    mi = args[4]
    assert residuals.ndim == 1, 'Array is not a vector'
    assert residuals.ndim == param.ndim, 'Input data array is not compatible with input paramater array'
    assert G.ndim == 2, 'Sensitivity matrix is not a 2D array'
    assert type(it_max) == int, 'Iteration number is not an integer'
    assert mi < 1., 'Regularization parameter is not a small number'

    residuals = np.asarray(residuals)
    param = np.asarray(param)

    for i in range(it_max):
        res = residuals - np.dot(G, param)
        W = np.dot(G.T, np.diag(1./(np.abs(res)+1e-10)))
        W += (mi*np.trace(W)*np.identity(param.size))/param.size # --
        pcl = np.linalg.solve(np.dot(W, G), np.dot(W, residuals))
    return pcl

def L1_norm_estim_pond_regularized(*args):#residuals, param, G, it_max=3):
    '''Function that estimates the physical properties of the point-masses
    through L1 norm of the residuals
    
    :math:  \delta \mathbf{g} = \mathbf{H} \mathbf{\rho} .
    
    input >
    args:           tuple    - (residuals, param, G, it_max)
    where
    {
        residuals:  1D array - vector of residuals
        param:      1D array - vector of parameters obtained by least-squares
        G:          2D array - sensitivity matrix
        it_max:     int      - maximum iteraion number.
    }
    
    output >
    pcl:            1D array - vector of parameters obtained L1 norm of the residuals
    '''
    residuals = np.asarray(args[0])
    param = np.asarray(args[1])
    G = np.asarray(args[2])
    it_max = args[3]
    mi = args[4]
    assert residuals.ndim == 1, 'Array is not a vector'
    assert residuals.ndim == param.ndim, 'Input data array is not compatible with input paramater array'
    assert G.ndim == 2, 'Sensitivity matrix is not a 2D array'
    assert type(it_max) == int, 'Iteration number is not an integer'
    assert mi < 1., 'Regularization parameter is not a small number'

    residuals = np.asarray(residuals)
    param = np.asarray(param)

    for i in range(it_max):
        res = residuals - np.dot(G, param)
        W = np.dot(G.T, np.diag(1./(np.abs(res)+1e-10)))
        W += (mi*np.trace(W)*np.identity(param.size))/param.size # --
        pcl = np.linalg.solve(np.dot(W, G), np.dot(W, residuals))
    return pcl

def L2_norm_multi_estim(data, tau, mi, start_depth, it_max=10):
    '''Function that estimates the physical properties of several layers of
    point-masses through L2 norm of the residuals
    
    :math:  \delta \mathbf{g} = \mathbf{H} \mathbf{\rho} .
    
    input >
    data:           list     - data vector. Example: [[x, y, z, R], [lat, lon], disturb]
    ALERT: REMEMBER THAT THE LAST VARIABLE INSIDE data list IS THE PROPER DISTURBANCE OF
    THE ANOMALOUS MASS MODEL FOR SYNTHETIC SIMULATIONS. FOR REAL DATA, THE LONG WAVELENGTH
    MUST BE EXTRACTED FROM THE OBSERVATIONS!!!
    param:          1D array - parameter vector obtained by least-squares
    tau:            float    - estimate precision
    mi:             float    - regularization parameter
    start_depth:    float    - depth for the first layer
    it_max:         int      - maximum iteration number
    
    output >
    pcl:            1D array - vector of parameters obtained L1 norm of the residuals
    '''
    x, y, z, R = data[0]
    lat, lon = data[1]
    disturb = np.asarray(data[2])
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    lat, lon = np.asarray(lat), np.asarray(lon)

    assert x.size == y.size == z.size, 'Cartesian coordinate arrays dimension mismatch'
    assert lat.size == lon.size, 'Geodetic coordinate arrays dimension mismatch'
    assert x.size == lon.size, 'Geodetic and Cartesian coordinate arrays dimension mismatch'
    assert disturb.size == x.size, 'Data array size is not compatible to coordinate array'
    pc = []
    latc = []
    lonc = []
    hc = []

    indl = np.arange(lat.size)
    hcl = start_depth
    hlay = np.zeros_like(indl) - hcl
    xlay, ylay, zlay = geodetic2cartesian(hlay, lat, lon, a, b)
    layer = build_layer(xlay, ylay, zlay)
    A_l = designMat(x, y, z, R, layer)
    H = np.dot(A_l.T, A_l)
    f0 = np.trace(H/indl.size)
    H += mi*f0*np.identity(indl.size)
    p_l = np.linalg.solve(H, np.dot(A_l, disturb))
    rcl = disturb - np.dot(A_l, p_l)
    rc = rcl[:]
    pc.append(p_l)
    latc.append(lat)
    lonc.append(lon)
    hc.append(hcl)
    for i in range(it_max):
        print i, np.mean(np.abs(rc)/np.abs(disturb))
        # print np.abs(rc)/np.abs(disturb), np.max(np.abs(rc)/np.abs(disturb))
        # print np.where(np.abs(rc) > tau)[0].size
        # if (np.less_equal(np.abs(rc)/np.abs(disturb), np.zeros_like(rc)+tau)).all():
        # if np.max(np.abs(rc)/np.abs(disturb)) < tau:
        #     print 1
        #     break
        # else:
        indl = np.where(np.abs(rc) > tau)[0]
        print indl.size
        if indl.size == 0:
            break
        else:
            rcl = rc[indl]
            xcl = x[indl]
            ycl = y[indl]
            zcl = z[indl]
            Rcl = R[:,indl]
            latcl = lat[indl]
            loncl = lon[indl]
            hcl = 0.5*hcl
            hlay = np.zeros_like(indl) - hcl
            xlay, ylay, zlay = geodetic2cartesian(hlay, latcl, loncl, a, b)
            layer = build_layer(xlay, ylay, zlay)
            A_l = designMat(xcl, ycl, zcl, Rcl, layer)
            H = np.dot(A_l.T, A_l)
            f0 = np.trace(H/indl.size)
            H += mi*f0*np.identity(indl.size)
            p_l = np.linalg.solve(H, np.dot(A_l, rcl))
            rcl -= np.dot(A_l, p_l)
            rc[indl] = rcl
            pc.append(p_l)
            latc.append(latcl)
            lonc.append(loncl)
            hc.append(hcl)
    return pc, hc, latc, lonc

def svd_sed_depth_estim(*args,**kwargs):#coordinates, depth, path, save=False):
    '''Function that estimates the optimal depth of the point-mass layer
    according to SED's technique (Curtis, 2004)
    
    input >
    args   -  tuple          - (lon, lat, height, layer_depth)
    Each args variable within the parenthesis is related to the observables. OBS: All
        variables inside the tuple args are required! They represent:
        lon, lat, height: 1D arrays          -> observable coordinates
        layer_depth:      list of floats     -> possible depths of point-mass layer
    
    kwargs -  dictionaries   - (save)
    Each kwargs variable within the parenthesis is related either to the format of the map or
    to a object to be plotted inside the map. OBS: Only the *config* dict is necessary for plotting!
    Keep in mind that as the keyword-arguments (kwargs) are dictionaries, knowing their position is
    irrelevant. The keyword-arguments are stated as:
    save:                 boolean   -> Decide whether to save or not. Default is False
    
    output >
    sA:                 list - list of eigenvalues
    '''
    assert type(args[3]) == list, 'Fourth arguments is not a list'
    lon = np.asarray(args[0])
    lat = np.asarray(args[1])
    height = np.asarray(args[2])
    layer_depth = np.asarray(args[3])

    x, y, z = geodetic2cartesian(height, lat, lon, a, b)
    R = rotation_matrix(lat, lon)

    # Sets the layers
    A = []
    for i in range(len(layer_depth)):
        hlay = np.zeros(lat.size) - layer_depth[i]
        xlay, ylay, zlay = geodetic2cartesian(hlay, lat, lon, a, b)
        lay = build_layer(xlay, ylay, zlay)
        Acc_l = designMat(x, y, z, R, lay)
        A.append(np.dot(Acc_l.T, Acc_l))
    
    # Computes the eigenvalues
    sA = []
    for i in range(len(layer_depth)):
        if i%100==0:
            print i
        s = np.linalg.svd(A[i], compute_uv=False)
        sA.append(s)
    print len(sA)
    if 'save' in kwargs:
        assert type(kwargs['save']) == str, 'Keyword-argument save is not a string'
        if 'nb_name' in kwargs:
            assert type(kwargs['nb_name']) == str, 'Keyword-argument nb_name is not a string'
            pm.save_pickle([layer_depth, sA], kwargs['nb_name'], kwargs['save'])
    return sA

# def L2_norm_multi_estim(data, tau, mi, start_depth):
#     '''Function that estimates the physical properties of several layers of
#     point-masses through L2 norm of the residuals
    
#     :math:  \delta \mathbf{g} = \mathbf{H} \mathbf{\rho} .
    
#     input >
#     data:           list     - data vector. Example: [[x, y, z, R], [lat, lon], disturb]
#     ALERT: REMEMBER THAT THE LAST VARIABLE INSIDE data list IS THE PROPER DISTURBANCE OF
#     THE ANOMALOUS MASS MODEL FOR SYNTHETIC SIMULATIONS. FOR REAL DATA, THE LONG WAVELENGTH
#     MUST BE EXTRACTED FROM THE OBSERVATIONS!!!
#     param:          1D array - parameter vector obtained by least-squares
#     tau:            float    - estimate precision
#     mi:             float    - regularization parameter
#     start_depth:    float    - depth for the first layer
    
#     output >
#     pcl:            1D array - vector of parameters obtained L1 norm of the residuals
#     '''
#     x, y, z, R = data[0]
#     lat, lon = data[1]
#     disturb = np.asarray(data[2])
#     x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
#     lat, lon = np.asarray(lat), np.asarray(lon)

#     assert x.size == y.size == z.size, 'Cartesian coordinate arrays dimension mismatch'
#     assert lat.size == lon.size, 'Geodetic coordinate arrays dimension mismatch'
#     assert x.size == lon.size, 'Geodetic and Cartesian coordinate arrays dimension mismatch'
#     assert disturb.size == x.size, 'Data array size is not compatible to coordinate array'

#     # mi , tau are parameter inputs for this function !!!
    
#     pc = []
#     hc = []
#     xc = []
#     yc = []
#     zc = []
#     Rc = []
#     latc = []
#     lonc = []

#     indl = np.arange(disturb.size)
#     rcl = disturb[indl]
#     xcl = x[indl]
#     ycl = y[indl]
#     zcl = z[indl]
#     latcl = lat[indl]
#     loncl = lon[indl]
#     Rcl = R[:, indl]

#     hcl = np.copy(start_depth)
#     hlay = np.zeros(indl.size) - hcl
#     xlay, ylay, zlay = geodetic2cartesian(hlay, latcl, loncl, a, b)
#     lay = build_layer(xlay, ylay, zlay)

#     Acc_l = designMat(xcl, ycl, zcl, Rcl, lay)
#     H = np.dot(Acc_l.T, Acc_l)
#     H += (mi*np.trace(H)*np.identity(indl.size))/indl.size
#     pcl = np.linalg.solve(H, np.dot(Acc_l.T, rcl)) # FIRST ESTIMATIVE DONE BEFORE LOOP
#     rcl -= np.dot(Acc_l, pcl)
#     print rcl

#     pc.append(pcl)
#     hc.append(hcl)
#     xc.append(xcl)
#     yc.append(ycl)
#     zc.append(zcl)
#     Rc.append(Rcl)
#     latc.append(latcl)
#     lonc.append(loncl)
#     rc = rcl[:]
#     while (np.greater_equal(np.abs(rc)/np.abs(disturb[0]), np.zeros_like(rc)+tau)).all():
#         print np.mean(np.abs(rc)/np.abs(disturb))
#         indl = np.where(np.abs(rc) >= tau)[0]
#         print indl.size
#         rcl = rc[indl]
#         xcl = x[indl]
#         ycl = y[indl]
#         zcl = z[indl]
#         latcl = lat[indl]
#         loncl = lon[indl]
#         Rcl = R[:, indl]
#         hcl = 0.5*hcl
#         hlay = np.zeros(indl.size) - hcl
#         xlay, ylay, zlay = geodetic2cartesian(hlay, latcl, loncl, a, b)
#         lay = build_layer(xlay, ylay, zlay)
#         Acc_l = designMat(xcl, ycl, zcl, Rcl, lay)
#         H = np.dot(Acc_l.T,Acc_l)
#         H += (mi*np.trace(H)*np.identity(indl.size))/indl.size
#         pcl = np.linalg.solve(H, np.dot(Acc_l.T, rcl))
#         rcl -= np.dot(Acc_l, pcl)

#         pc.append(pcl)
#         hc.append(hcl)
#         xc.append(xcl)
#         yc.append(ycl)
#         zc.append(zcl)
#         Rc.append(Rcl)
#         latc.append(latcl)
#         lonc.append(loncl)
#         rc[indl] = rcl
#     return pc, hc, rc