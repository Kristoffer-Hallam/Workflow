import numpy as numpy
from coord import CGS_to_GGS
from equiv_layer import EqLayer

longitude = -55.
latitude = -25.
height = 1000.

observ = CGS_to_GGS()
layer = EqLayer(longitude, latitude, height)
print layer.lon, layer.lat, layer.height

x, y, z = observ.geodetic2cartesian(longitude, latitude, height)
xlay, ylay, zlay = layer.geodetic2cartesian(longitude, latitude, -5000.)
# print x, y, z
# print xlay, ylay, zlay