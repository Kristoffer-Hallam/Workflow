import numpy as np
import numpy.testing as npt
from pytest import raises

from coord import GGS

def test_geodetic2cartesian_within_GGS_class_var_size():
    '''Checks compartibility of variable sizes'''
    d1 = np.empty(5)
    d2 = np.empty(4)
    d3 = np.empty(4)
    raises(AssertionError, GGS().geodetic2cartesian, d1, d2, d3)
    
    d1 = np.empty(4)
    d2 = np.empty(5)
    raises(AssertionError, GGS().geodetic2cartesian, d1, d2, d3)
    
    d2 = np.empty(4)
    d3 = np.empty(5)
    raises(AssertionError, GGS().geodetic2cartesian, d1, d2, d3)

def test_rotation_matrix_within_GGS_class_var_size():
    '''Checks compartibility of variable sizes'''
    d1 = np.empty(5)
    d2 = np.empty(4)
    raises(AssertionError, GGS().rotation_matrix, d1, d2)
    
    d1 = np.empty(4)
    d2 = np.empty(5)
    raises(AssertionError, GGS().rotation_matrix, d1, d2)

def test_rotation_matrix_within_GGS_class_value_check1():
    '''Checks compartibility of between given and calculated
    rotation matrix when lat and lon are equal to 0'''
    lat = 0.
    lon = 0.
    R_calc = GGS().rotation_matrix(lon,lat)
    R_obs = [1., 0., 0., 0., 0., 1., 0., 1.]
    npt.assert_almost_equal(R_calc,R_obs,decimal=20)

def test_rotation_matrix_within_GGS_class_value_check2():
    '''Checks compartibility of between given and calculated
    rotation matrix when lat and lon are equal to 90'''
    lat = 90.
    lon = 90.
    R_calc = GGS().rotation_matrix(lon,lat)
    R_obs = [0., 0., 1., 0., -1., 0., -1., 0.]
    npt.assert_almost_equal(R_calc,R_obs,decimal=15)