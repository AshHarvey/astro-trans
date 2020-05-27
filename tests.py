from astropy.coordinates import SkyCoord, ITRS, EarthLocation, AltAz
from astropy import units as u
import numpy as np
from datetime import datetime, timedelta
import time

from numpy.core._multiarray_umath import ndarray
from scipy.spatial import distance
from scipy.linalg import det

print("Running Test Cases...")

# !------------ Test 1 - GCRS to ITRS
from transformations import gcrs2irts_matrix_a, get_eops, gcrs2irts_matrix_b
eop = get_eops()

xyz1: ndarray = np.array([1285410, -4797210, 3994830], dtype=np.float64)

t=datetime(year = 2007, month = 4, day = 5, hour = 12, minute = 0, second = 0)
object = SkyCoord(x=xyz1[0] * u.m, y=xyz1[1] * u.m, z=xyz1[2] * u.m, frame='gcrs',
               representation_type='cartesian', obstime=t)# just for astropy
itrs = object.transform_to(ITRS)

test1a_error = distance.euclidean(itrs.cartesian.xyz._to_value(u.m),
                                 gcrs2irts_matrix_a(t, eop) @ xyz1)
test1b_error = distance.euclidean(itrs.cartesian.xyz._to_value(u.m),
                                  gcrs2irts_matrix_b(t, eop) @ xyz1)
assert test1a_error < 25, print("Failed Test 1: GCRS to ITRS transformation")
print("Test 1a: GCRS to ITRS (a) error in meters: ", test1a_error)
print("Test 1b: GCRS to ITRS (b) error in meters: ", test1b_error)

# !------------ Test 2a - ITRS to LLA
from transformations import itrs2lla
xyz1 = np.array(itrs.cartesian.xyz._to_value(u.m), dtype=np.float64)
lla = EarthLocation.from_geocentric(x=xyz1[0]*u.m, y=xyz1[1]*u.m, z=xyz1[2]*u.m)
lat = lla.lat.to_value(u.rad)
lon = lla.lon.to_value(u.rad)
height = lla.height.to_value(u.m)
lla = [lon, lat, height]
test2_error = lla - np.asarray(itrs2lla(xyz1))
assert np.max(test2_error) < 0.0000001, print("Failed Test 2a: ITRS to LLA transformation")
print("Test 2a: ITRS to LLA error in rads,rads,meters: ", test2_error)

# !------------ Test 2b - ITRS to LLA
from transformations import itrs2lla_py
xyz1 = np.array(itrs.cartesian.xyz._to_value(u.m), dtype=np.float64)
lla = EarthLocation.from_geocentric(x=xyz1[0]*u.m, y=xyz1[1]*u.m, z=xyz1[2]*u.m)
lat = lla.lat.to_value(u.rad)
lon = lla.lon.to_value(u.rad)
height = lla.height.to_value(u.m)
lla = [lon, lat, height]
test2_error = lla - np.asarray(itrs2lla_py(xyz1))
assert np.max(test2_error) < 0.0000001, print("Failed Test 2b: ITRS to LLA transformation")
print("Test 2b: ITRS to LLA (python) error in rads,rads,meters: ", test2_error)

# !------------ Test 3 - ITRS-ITRS to AzElSr
from transformations import itrs2azel, _itrs2azel
xyz1 = np.array([1285410, -4797210, 3994830], dtype=np.float64)
xyz2 = np.array([1202990, -4824940, 3999870], dtype=np.float64)

observer = EarthLocation.from_geocentric(x=xyz1[0]*u.m,y=xyz1[1]*u.m, z=xyz1[2]*u.m)
target = SkyCoord(x=xyz2[0] * u.m, y=xyz2[1] * u.m, z=xyz2[2] * u.m, frame='itrs',
               representation_type='cartesian', obstime=t)# just for astropy
AltAz_frame = AltAz(obstime=t, location=observer)
results = target.transform_to(AltAz_frame)

az1 = results.az.to_value(u.rad)
alt1 = results.alt.to_value(u.rad)
sr1 = results.distance.to_value(u.m)


aer = _itrs2azel(xyz1,xyz2)
aer2 = itrs2azel(xyz1,np.reshape(xyz2,(1,3)))[0]

test3_error = [az1-aer[0],alt1-aer[1],sr1-aer[2]]

assert np.allclose(aer,aer2), print("Failed Test 3a: _itrs2azel and itrs2azel do not match")
assert np.absolute(az1 - aer[0]) < 0.001, print("Failed Test 3a: ITRS-ITRS to Az transformation")
assert np.absolute(alt1 - aer[1]) < 0.001, print("Failed Test 3b: ITRS-ITRS to El transformation")
assert np.absolute(sr1 - aer[2]) < 0.001, print("Failed Test 3c: ITRS-ITRS to Srange transformation")
print("Test 3: ITRS-ITRS to Az, El, Srange error in rads,rads,meters: ", test2_error)

# !------------ Test 4 - ITRS-ITRS to AzElSr
t=datetime(year=2007, month=4, day=5, hour=12, minute=0, second=0)
t = [t]

start = time.time()
for i in range(2880):
    t.append(t[-1]+timedelta(seconds=30))
gcrs2irts_matrix_a(t, eop)
end = time.time()
print("Test 4a: Time to generate cel2ter transformation matrix with gcrs2irts_matrix_a for every 30 seconds for an "
      "entire day: ", end-start, " seconds")

start = time.time()
for i in range(2880):
    t.append(t[-1]+timedelta(seconds=30))
gcrs2irts_matrix_b(t, eop)
end = time.time()
print("Test 4b: Time to generate cel2ter transformation matrix with gcrs2irts_matrix_b for every 30 seconds for an "
      "entire day: ", end-start, " seconds")

# !------------ Test 5 - ITRS to GCRS SOFA cases
from transformations import gcrs2irts_matrix_a, gcrs2irts_matrix_b

t = datetime(year=2007, month=4, day=5, hour=12, minute=0, second=0)

Cel2Ter94 = np.asarray([[+0.973104317712772, +0.230363826174782, -0.000703163477127],
                        [-0.230363800391868, +0.973104570648022, +0.000118545116892],
                        [+0.000711560100206, +0.000046626645796, +0.999999745754058]])

Cel2Ter00aCIO = np.asarray([[+0.973104317697512, +0.230363826239227, -0.000703163482268],
                           [-0.230363800456136, +0.973104570632777, +0.000118545366806],
                           [+0.000711560162777, +0.000046626403835, +0.999999745754024]])

Cel2Ter00aEB = np.asarray([[+0.973104317697618, +0.230363826238780, -0.000703163482352],
                           [-0.230363800455689, +0.973104570632883, +0.000118545366826],
                           [+0.000711560162864, +0.000046626403835, +0.999999745754024]])

Cel2Ter06aCA = np.asarray([[+0.973104317697535, +0.230363826239128, -0.000703163482198],
                           [-0.230363800456037, +0.973104570632801, +0.000118545366625],
                           [+0.000711560162668, +0.000046626403995, +0.999999745754024]])

Cel2Ter06aXY = np.asarray([[+0.973104317697536, +0.230363826239128, -0.000703163481769],
                           [-0.230363800456036, +0.973104570632801, +0.000118545368117],
                           [+0.000711560162594, +0.000046626402444, +0.999999745754024]])

print("Test 5a: Cel2Ter06aXY vs Cel2Ter94, magnitude of error: ", det(Cel2Ter06aXY - Cel2Ter94))
print("Test 5b: Cel2Ter06aXY vs Cel2Ter00aCIO, magnitude of error: ", det(Cel2Ter06aXY - Cel2Ter00aCIO))
print("Test 5c: Cel2Ter06aXY vs Cel2Ter00aEB, magnitude of error: ", det(Cel2Ter06aXY - Cel2Ter00aEB))
print("Test 5d: Cel2Ter06aXY vs Cel2Ter06aCA, magnitude of error: ", det(Cel2Ter06aXY - Cel2Ter06aCA))
print("Test 5e: Cel2Ter06aXY vs gcrs2irts_matrix, magnitude of error: ", det(Cel2Ter06aXY - gcrs2irts_matrix_a(t, eop)))
print("Test 5f: Cel2Ter06aXY vs utc2cel06acio, magnitude of error: ", det(Cel2Ter06aXY - gcrs2irts_matrix_b(t, eop)))

# !------------ Test 6 - Az El end to end
from transformations import lla2itrs, _itrs2azel
from numba import njit

observer_lat = np.radians(38.828198)
observer_lon = np.radians(-77.305352)
observer_alt = np.float64(20.0) # in meters
observer_lla = np.array((observer_lon, observer_lat, observer_alt))
observer_itrs = lla2itrs(observer_lla)/1000 # meter -> kilometers

@njit
def hx(x_gcrs, trans_matrix, obs=observer_itrs):
    # measurement function - convert state into a measurement
    # where measurements are [azimuth, elevation]
    x_itrs = trans_matrix @ x_gcrs[:3]
    aer = _itrs2azel(observer_itrs, x_itrs)
    return aer


def hx2(x, t, obs_lat=observer_lat, obs_lon=observer_lon, obs_height=observer_alt):
    # measurement function - convert state into a measurement
    # where measurements are [azimuth, elevation]
    object = SkyCoord(x=x[0] * u.km, y=x[1] * u.km, z=x[2] * u.km, frame='gcrs',
                   representation_type='cartesian', obstime=t)
    obs = EarthLocation.from_geodetic(lon=obs_lon*u.rad, lat=obs_lat*u.rad, height=obs_height*u.m)
    AltAz_frame = AltAz(obstime=t, location=obs)
    results = object.transform_to(AltAz_frame)

    az = results.az.to_value(u.rad)
    alt = results.alt.to_value(u.rad)
    sr = results.distance.to_value(u.km)
    aer = np.array([az, alt, sr])
    return aer


gcrs_xyz = [[-400000.0,-400000.0,-400000.0],
            [-400000.0,-400000.0,400000.0],
            [-400000.0,400000.0,-400000.0],
            [-400000.0,400000.0,400000.0],
            [400000.0,-400000.0,400000.0],
            [400000.0,400000.0,-400000.0],
            [400000.0,-400000.0,-400000.0],
            [400000.0,400000.0,400000.0]]


t=datetime(year = 2007, month = 4, day = 5, hour = 12, minute = 0, second = 0)

trans_matrix = gcrs2irts_matrix_b(t, eop)

z1 = []
z2 = []
for xyz in gcrs_xyz:
    z1.append(hx(np.asarray(xyz), trans_matrix, obs=observer_itrs))
    z2.append(hx2(np.asarray(xyz), t, obs_lat=observer_lat, obs_lon=observer_lon, obs_height=observer_alt))

for z in z1:
    if z[0] < 0:
        z[0] += np.pi*2

hx_error = []
for i in range(len(z1)):
    hx_error.append(z2[i] - z1[i])

hx_error = np.asarray(hx_error)

error_az = np.degrees(hx_error[:,0])*60*60
error_el = np.degrees(hx_error[:,1])*60*60
error_sr = np.degrees(hx_error[:,2])*1000

print("Test 6a: hx error in azimuth (arc seconds) = ", np.mean(np.abs(error_az)), " (mean); individual errors: ", error_az)
print("Test 6b: hx error in elevation (arc seconds) = ", np.mean(np.abs(error_el)), " (mean); individual errors: ", error_el)
print("Test 6c: hx error in slant range (meters) = ", np.mean(np.abs(error_sr)), " (mean); individual errors: ", error_sr)
print("I am unsure if these errors are mine or AstroPy's given the above matched. Each sub-function checks out against "
      "AstroPy, but the end to end case has significantly more error...")
print("Done")
