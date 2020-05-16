from astropy.coordinates import SkyCoord, ITRS, EarthLocation, AltAz
from astropy import units as u
import numpy as np
from datetime import datetime
from scipy.spatial import distance

# !------------ Test 1 - GCRS to ITRS
from transformations import gcrs2irts_matrix, get_eops
eop = get_eops()

xyz1 = np.array([1285410, -4797210, 3994830], dtype=np.float64)

t=datetime(year = 2007, month = 4, day = 5, hour = 12, minute = 0, second = 0)
object = SkyCoord(x=xyz1[0] * u.m, y=xyz1[1] * u.m, z=xyz1[2] * u.m, frame='gcrs',
               representation_type='cartesian', obstime=t)# just for astropy
itrs = object.transform_to(ITRS)

test1_error = distance.euclidean(itrs.cartesian.xyz._to_value(u.m),
                                 gcrs2irts_matrix(eop,t) @ xyz1)
assert test1_error < 25, print("Failed Test 1: GCRS to ITRS transformation")
print("GCRS to ITRS other error in meters: ", test1_error)

# !------------ Test 2 - ITRS to LLA
from transformations import itrs2lla
xyz1 = np.array(itrs.cartesian.xyz._to_value(u.m), dtype=np.float64)
lla = EarthLocation.from_geocentric(x=xyz1[0]*u.m,y=xyz1[1]*u.m,z=xyz1[2]*u.m)
lat = lla.lat.to_value(u.rad)
lon = lla.lon.to_value(u.rad)
height = lla.height.to_value(u.m)
lla = [lon,lat,height]
test2_error = lla - np.asarray(itrs2lla(xyz1))
assert np.max(test2_error) < 0.0000001, print("Failed Test 2: ITRS to LLA transformation")
print("ITRS to LLA error in rads,rads,meters: ", test2_error)

# !------------ Test 3 - ITRS-ITRS to AzElDist
from transformations import itrs2azel
xyz1 = np.array([1285410, -4797210, 3994830], dtype=np.float64)
xyz2 = np.array([1202990, -4824940, 3999870], dtype=np.float64)

observer = EarthLocation.from_geocentric(x=xyz1[0]*u.m,y=xyz1[1]*u.m, z=xyz1[2]*u.m)
target = SkyCoord(x=xyz2[0] * u.m, y=xyz2[1] * u.m, z=xyz2[2] * u.m, frame='itrs',
               representation_type='cartesian', obstime=t)# just for astropy
AltAz_frame = AltAz(obstime=t, location=observer)
results = target.transform_to(AltAz_frame)

az1 = results.az.to_value(u.rad)
alt1 = results.alt.to_value(u.rad)
dist1 = results.distance.to_value(u.m)

aer = itrs2azel(xyz1,xyz2)

test3_error = [az1-aer[0],alt1-aer[1],dist1-aer[2]]

assert np.absolute(az1 - aer[0]) < 0.001, print("Failed Test 3a: ITRS-ITRS to Az transformation")
assert np.absolute(alt1 - aer[1]) < 0.001, print("Failed Test 3b: ITRS-ITRS to El transformation")
assert np.absolute(dist1 - aer[2]) < 0.001, print("Failed Test 3c: ITRS-ITRS to Dist transformation")
print("ITRS-ITRS to Az, El, Dist error in rads,rads,meters: ", test2_error)
