import numpy as np
import numpy.testing as npt
from pytest import raises

from equiv_layer import EqLayer
from coord import GGS

def test_EqLayer_class_var_size():
    '''Checks compartibility of variable sizes'''
    d1 = np.empty(5)
    d2 = np.empty(4)
    d3 = np.empty(4)
    raises(AssertionError, EqLayer, d1, d2, d3)
    
    d1 = np.empty(4)
    d2 = np.empty(5)
    raises(AssertionError, EqLayer, d1, d2, d3)
    
    d2 = np.empty(4)
    d3 = np.empty(5)
    raises(AssertionError, EqLayer, d1, d2, d3)

def test_build_layer_within_EqLayer_class_var_size():
    '''Checks compartibility of variable sizes'''
    
    # Creating Layer object
    d1 = np.empty(4)
    d2 = np.empty(4)
    d3 = np.empty(4)
    Lay = EqLayer(d1, d2, d3)
    
    # Applying build_layer function
    d1 = np.empty(5)
    d2 = np.empty(4)
    d3 = np.empty(4)
    raises(AssertionError, Lay.build_layer, d1, d2, d3)
    
    d1 = np.empty(4)
    d2 = np.empty(5)
    raises(AssertionError, Lay.build_layer, d1, d2, d3)
    
    d2 = np.empty(4)
    d3 = np.empty(5)
    raises(AssertionError, Lay.build_layer, d1, d2, d3)

def test_designMat_within_EqLayer_class_var_size():
    '''Checks compartibility of variable sizes'''
    
    # Creating Layer object
    lon = np.array([-60.,-50.,-40.])
    lat = np.array([-30.,-25.,-20.])
    h = np.zeros_like(lon)+3000.
    Lay = EqLayer(lon,lat,h)
    
    # Applying build_layer function
    d1 = np.empty(5)
    d2 = np.empty(4)
    d3 = np.empty(4)
    raises(AssertionError, Lay.build_layer, d1, d2, d3)
    
    d1 = np.empty(4)
    d2 = np.empty(5)
    raises(AssertionError, Lay.build_layer, d1, d2, d3)
    
    d2 = np.empty(4)
    d3 = np.empty(5)
    raises(AssertionError, Lay.build_layer, d1, d2, d3)

def test_designMat_within_EqLayer_class_check_value():
    '''Checks compartibility of calculated and given
    design matrices'''
    
    # Creating Observation points
    lon = np.array([-60.,-50.,-40.])
    lat = np.array([-30.,-25.,-20.])
    h = np.zeros_like(lon)+3000.
    x, y, z = GGS().geodetic2cartesian(lon,lat,h)


    Lay = EqLayer(lon,lat,np.zeros_like(lon)-200.)
    
    # Applying build_layer function
    d1 = np.empty(5)
    d2 = np.empty(4)
    d3 = np.empty(4)
    raises(AssertionError, Lay.build_layer, d1, d2, d3)
    
    d1 = np.empty(4)
    d2 = np.empty(5)
    raises(AssertionError, Lay.build_layer, d1, d2, d3)
    
    d2 = np.empty(4)
    d3 = np.empty(5)
    raises(AssertionError, Lay.build_layer, d1, d2, d3)