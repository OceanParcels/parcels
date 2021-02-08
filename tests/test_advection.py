from parcels import FieldSet, ScipyParticle, JITParticle, ErrorCode
from parcels.particleset_vectorized import ParticleSet
from parcels import AdvectionEE, AdvectionRK4, AdvectionRK45, AdvectionRK4_3D
import numpy as np
import pytest
import math
from datetime import timedelta as delta


ptype = {'scipy': ScipyParticle, 'jit': JITParticle}
kernel = {'EE': AdvectionEE, 'RK4': AdvectionRK4, 'RK45': AdvectionRK45}

# Some constants
f = 1.e-4
u_0 = 0.3
u_g = 0.04
gamma = 1/(86400. * 2.89)
gamma_g = 1/(86400. * 28.9)


def lon(xdim=200):
    return np.linspace(-170, 170, xdim, dtype=np.float32)


@pytest.fixture(name="lon")
def lon_fixture(xdim=200):
    return lon(xdim=xdim)


def lat(ydim=100):
    return np.linspace(-80, 80, ydim, dtype=np.float32)


@pytest.fixture(name="lat")
def lat_fixture(ydim=100):
    return lat(ydim=ydim)


def depth(zdim=2):
    return np.linspace(0, 30, zdim, dtype=np.float32)


@pytest.fixture(name="depth")
def depth_fixture(zdim=2):
    return depth(zdim=zdim)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_advection_zonal(lon, lat, depth, mode, npart=10):
    """ Particles at high latitude move geographically faster due to
        the pole correction in `GeographicPolar`.
    """
    data2D = {'U': np.ones((lon.size, lat.size), dtype=np.float32),
              'V': np.zeros((lon.size, lat.size), dtype=np.float32)}
    data3D = {'U': np.ones((lon.size, lat.size, depth.size), dtype=np.float32),
              'V': np.zeros((lon.size, lat.size, depth.size), dtype=np.float32)}
    dimensions = {'lon': lon, 'lat': lat}
    fieldset2D = FieldSet.from_data(data2D, dimensions, mesh='spherical', transpose=True)
    assert fieldset2D.U.creation_log == 'from_data'

    pset2D = ParticleSet(fieldset2D, pclass=ptype[mode],
                         lon=np.zeros(npart) + 20.,
                         lat=np.linspace(0, 80, npart))
    pset2D.execute(AdvectionRK4, runtime=delta(hours=2), dt=delta(seconds=30))
    assert (np.diff(np.array([p.lon for p in pset2D])) > 1.e-4).all()

    dimensions['depth'] = depth
    fieldset3D = FieldSet.from_data(data3D, dimensions, mesh='spherical', transpose=True)
    pset3D = ParticleSet(fieldset3D, pclass=ptype[mode],
                         lon=np.zeros(npart) + 20.,
                         lat=np.linspace(0, 80, npart),
                         depth=np.zeros(npart) + 10.)
    pset3D.execute(AdvectionRK4, runtime=delta(hours=2), dt=delta(seconds=30))
    assert (np.diff(np.array([p.lon for p in pset3D])) > 1.e-4).all()


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_advection_meridional(lon, lat, mode, npart=10):
    """ Particles at high latitude move geographically faster due to
        the pole correction in `GeographicPolar`.
    """
    data = {'U': np.zeros((lon.size, lat.size), dtype=np.float32),
            'V': np.ones((lon.size, lat.size), dtype=np.float32)}
    dimensions = {'lon': lon, 'lat': lat}
    fieldset = FieldSet.from_data(data, dimensions, mesh='spherical', transpose=True)

    pset = ParticleSet(fieldset, pclass=ptype[mode],
                       lon=np.linspace(-60, 60, npart),
                       lat=np.linspace(0, 30, npart))
    delta_lat = np.diff(np.array([p.lat for p in pset]))
    pset.execute(AdvectionRK4, runtime=delta(hours=2), dt=delta(seconds=30))
    assert np.allclose(np.diff(np.array([p.lat for p in pset])), delta_lat, rtol=1.e-4)


@pytest.mark.parametrize('mode', ['jit', 'scipy'])
def test_advection_3D(mode, npart=11):
    """ 'Flat' 2D zonal flow that increases linearly with depth from 0 m/s to 1 m/s
    """
    xdim = ydim = zdim = 2
    dimensions = {'lon': np.linspace(0., 1e4, xdim, dtype=np.float32),
                  'lat': np.linspace(0., 1e4, ydim, dtype=np.float32),
                  'depth': np.linspace(0., 1., zdim, dtype=np.float32)}
    data = {'U': np.ones((xdim, ydim, zdim), dtype=np.float32),
            'V': np.zeros((xdim, ydim, zdim), dtype=np.float32)}
    data['U'][:, :, 0] = 0.
    fieldset = FieldSet.from_data(data, dimensions, mesh='flat', transpose=True)

    pset = ParticleSet(fieldset, pclass=ptype[mode],
                       lon=np.zeros(npart),
                       lat=np.zeros(npart) + 1e2,
                       depth=np.linspace(0, 1, npart))
    time = delta(hours=2).total_seconds()
    pset.execute(AdvectionRK4, runtime=time, dt=delta(seconds=30))
    assert np.allclose([p.depth*time for p in pset], [p.lon for p in pset], atol=1.e-1)


@pytest.mark.parametrize('mode', ['jit', 'scipy'])
@pytest.mark.parametrize('direction', ['up', 'down'])
@pytest.mark.parametrize('wErrorThroughSurface', [True, False])
def test_advection_3D_outofbounds(mode, direction, wErrorThroughSurface):
    xdim = ydim = zdim = 2
    dimensions = {'lon': np.linspace(0., 1, xdim, dtype=np.float32),
                  'lat': np.linspace(0., 1, ydim, dtype=np.float32),
                  'depth': np.linspace(0., 1, zdim, dtype=np.float32)}
    wfac = -1. if direction == 'up' else 1.
    data = {'U': 0.01*np.ones((xdim, ydim, zdim), dtype=np.float32),
            'V': np.zeros((xdim, ydim, zdim), dtype=np.float32),
            'W': wfac * np.ones((xdim, ydim, zdim), dtype=np.float32)}
    fieldset = FieldSet.from_data(data, dimensions, mesh='flat')

    def DeleteParticle(particle, fieldset, time):
        particle.delete()

    def SubmergeParticle(particle, fieldset, time):
        particle.depth = 0
        AdvectionRK4(particle, fieldset, time)  # perform a 2D advection because vertical flow will always push up in this case
        particle.time = time + particle.dt  # to not trigger kernels again, otherwise infinite loop
        particle.succeeded()

    recovery_dict = {ErrorCode.ErrorOutOfBounds: DeleteParticle}
    if wErrorThroughSurface:
        recovery_dict[ErrorCode.ErrorThroughSurface] = SubmergeParticle

    pset = ParticleSet(fieldset=fieldset, pclass=ptype[mode], lon=0.5, lat=0.5, depth=0.9)
    pset.execute(AdvectionRK4_3D, runtime=10., dt=1, recovery=recovery_dict)

    if direction == 'up' and wErrorThroughSurface:
        assert np.allclose(pset[0].lon, 0.6)
        assert np.allclose(pset[0].depth, 0)
    else:
        assert len(pset) == 0


def periodicfields(xdim, ydim, uvel, vvel):
    dimensions = {'lon': np.linspace(0., 1., xdim+1, dtype=np.float32)[1:],  # don't include both 0 and 1, for periodic b.c.
                  'lat': np.linspace(0., 1., ydim+1, dtype=np.float32)[1:]}

    data = {'U': uvel * np.ones((xdim, ydim), dtype=np.float32),
            'V': vvel * np.ones((xdim, ydim), dtype=np.float32)}
    return FieldSet.from_data(data, dimensions, mesh='spherical', transpose=True)


def periodicBC(particle, fieldset, time):
    particle.lon = math.fmod(particle.lon, 1)
    particle.lat = math.fmod(particle.lat, 1)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_advection_periodic_zonal(mode, xdim=100, ydim=100, halosize=3):
    fieldset = periodicfields(xdim, ydim, uvel=1., vvel=0.)
    fieldset.add_periodic_halo(zonal=True, halosize=halosize)
    assert(len(fieldset.U.lon) == xdim + 2 * halosize)

    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=[0.5], lat=[0.5])
    pset.execute(AdvectionRK4 + pset.Kernel(periodicBC), runtime=delta(hours=20), dt=delta(seconds=30))
    assert abs(pset[0].lon - 0.15) < 0.1


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_advection_periodic_meridional(mode, xdim=100, ydim=100):
    fieldset = periodicfields(xdim, ydim, uvel=0., vvel=1.)
    fieldset.add_periodic_halo(meridional=True)
    assert(len(fieldset.U.lat) == ydim + 10)  # default halo size is 5 grid points

    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=[0.5], lat=[0.5])
    pset.execute(AdvectionRK4 + pset.Kernel(periodicBC), runtime=delta(hours=20), dt=delta(seconds=30))
    assert abs(pset[0].lat - 0.15) < 0.1


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_advection_periodic_zonal_meridional(mode, xdim=100, ydim=100):
    fieldset = periodicfields(xdim, ydim, uvel=1., vvel=1.)
    fieldset.add_periodic_halo(zonal=True, meridional=True)
    assert(len(fieldset.U.lat) == ydim + 10)  # default halo size is 5 grid points
    assert(len(fieldset.U.lon) == xdim + 10)  # default halo size is 5 grid points
    assert np.allclose(np.diff(fieldset.U.lat), fieldset.U.lat[1]-fieldset.U.lat[0], rtol=0.001)
    assert np.allclose(np.diff(fieldset.U.lon), fieldset.U.lon[1]-fieldset.U.lon[0], rtol=0.001)

    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=[0.4], lat=[0.5])
    pset.execute(AdvectionRK4 + pset.Kernel(periodicBC), runtime=delta(hours=20), dt=delta(seconds=30))
    assert abs(pset[0].lon - 0.05) < 0.1
    assert abs(pset[0].lat - 0.15) < 0.1


def truth_stationary(x_0, y_0, t):
    lat = y_0 - u_0 / f * (1 - math.cos(f * t))
    lon = x_0 + u_0 / f * math.sin(f * t)
    return lon, lat


@pytest.fixture
def fieldset_stationary(xdim=100, ydim=100, maxtime=delta(hours=6)):
    """Generate a FieldSet encapsulating the flow field of a stationary eddy.

    Reference: N. Fabbroni, 2009, "Numerical simulations of passive
    tracers dispersion in the sea"
    """
    time = np.arange(0., maxtime.total_seconds()+1e-5, 60., dtype=np.float64)
    dimensions = {'lon': np.linspace(0, 25000, xdim, dtype=np.float32),
                  'lat': np.linspace(0, 25000, ydim, dtype=np.float32),
                  'time': time}
    data = {'U': np.ones((xdim, ydim, 1), dtype=np.float32) * u_0 * np.cos(f * time),
            'V': np.ones((xdim, ydim, 1), dtype=np.float32) * -u_0 * np.sin(f * time)}
    return FieldSet.from_data(data, dimensions, mesh='flat', transpose=True)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('method, rtol', [
    ('EE', 1e-2),
    ('RK4', 1e-5),
    ('RK45', 1e-5)])
def test_stationary_eddy(fieldset_stationary, mode, method, rtol, npart=1):
    fieldset = fieldset_stationary
    lon = np.linspace(12000, 21000, npart)
    lat = np.linspace(12500, 12500, npart)
    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=lon, lat=lat)
    endtime = delta(hours=6).total_seconds()
    pset.execute(kernel[method], dt=delta(minutes=3), endtime=endtime)
    exp_lon = [truth_stationary(x, y, endtime)[0] for x, y, in zip(lon, lat)]
    exp_lat = [truth_stationary(x, y, endtime)[1] for x, y, in zip(lon, lat)]
    assert np.allclose(np.array([p.lon for p in pset]), exp_lon, rtol=rtol)
    assert np.allclose(np.array([p.lat for p in pset]), exp_lat, rtol=rtol)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_stationary_eddy_vertical(mode, npart=1):
    lon = np.linspace(12000, 21000, npart)
    lat = np.linspace(10000, 20000, npart)
    depth = np.linspace(12500, 12500, npart)
    endtime = delta(hours=6).total_seconds()

    xdim = ydim = 100
    lon_data = np.linspace(0, 25000, xdim, dtype=np.float32)
    lat_data = np.linspace(0, 25000, ydim, dtype=np.float32)
    time_data = np.arange(0., 6*3600+1e-5, 60., dtype=np.float64)
    fld1 = np.ones((xdim, ydim, 1), dtype=np.float32) * u_0 * np.cos(f * time_data)
    fld2 = np.ones((xdim, ydim, 1), dtype=np.float32) * -u_0 * np.sin(f * time_data)
    fldzero = np.zeros((xdim, ydim, 1), dtype=np.float32) * time_data

    dimensions = {'lon': lon_data, 'lat': lat_data, 'time': time_data}
    data = {'U': fld1, 'V': fldzero, 'W': fld2}
    fieldset = FieldSet.from_data(data, dimensions, mesh='flat', transpose=True)

    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=lon,
                       lat=lat, depth=depth)
    pset.execute(AdvectionRK4_3D, dt=delta(minutes=3), endtime=endtime)
    exp_lon = [truth_stationary(x, z, endtime)[0] for x, z, in zip(lon, depth)]
    exp_depth = [truth_stationary(x, z, endtime)[1] for x, z, in zip(lon, depth)]
    assert np.allclose(np.array([p.lon for p in pset]), exp_lon, rtol=1e-5)
    assert np.allclose(np.array([p.lat for p in pset]), lat, rtol=1e-5)
    assert np.allclose(np.array([p.depth for p in pset]), exp_depth, rtol=1e-5)

    data = {'U': fldzero, 'V': fld2, 'W': fld1}
    fieldset = FieldSet.from_data(data, dimensions, mesh='flat', transpose=True)

    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=lon,
                       lat=lat, depth=depth)
    pset.execute(AdvectionRK4_3D, dt=delta(minutes=3), endtime=endtime)
    exp_depth = [truth_stationary(z, y, endtime)[0] for z, y, in zip(depth, lat)]
    exp_lat = [truth_stationary(z, y, endtime)[1] for z, y, in zip(depth, lat)]
    assert np.allclose(np.array([p.lon for p in pset]), lon, rtol=1e-5)
    assert np.allclose(np.array([p.lat for p in pset]), exp_lat, rtol=1e-5)
    assert np.allclose(np.array([p.depth for p in pset]), exp_depth, rtol=1e-5)


def truth_moving(x_0, y_0, t):
    lat = y_0 - (u_0 - u_g) / f * (1 - math.cos(f * t))
    lon = x_0 + u_g * t + (u_0 - u_g) / f * math.sin(f * t)
    return lon, lat


@pytest.fixture
def fieldset_moving(xdim=100, ydim=100, maxtime=delta(hours=6)):
    """Generate a FieldSet encapsulating the flow field of a moving eddy.

    Reference: N. Fabbroni, 2009, "Numerical simulations of passive
    tracers dispersion in the sea"
    """
    time = np.arange(0., maxtime.total_seconds()+1e-5, 60., dtype=np.float64)
    dimensions = {'lon': np.linspace(0, 25000, xdim, dtype=np.float32),
                  'lat': np.linspace(0, 25000, ydim, dtype=np.float32),
                  'time': time}
    data = {'U': np.ones((xdim, ydim, 1), dtype=np.float32) * u_g + (u_0 - u_g) * np.cos(f * time),
            'V': np.ones((xdim, ydim, 1), dtype=np.float32) * -(u_0 - u_g) * np.sin(f * time)}
    return FieldSet.from_data(data, dimensions, mesh='flat', transpose=True)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('method, rtol', [
    ('EE', 1e-2),
    ('RK4', 1e-5),
    ('RK45', 1e-5)])
def test_moving_eddy(fieldset_moving, mode, method, rtol, npart=1):
    fieldset = fieldset_moving
    lon = np.linspace(12000, 21000, npart)
    lat = np.linspace(12500, 12500, npart)
    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=lon, lat=lat)
    endtime = delta(hours=6).total_seconds()
    pset.execute(kernel[method], dt=delta(minutes=3), endtime=endtime)
    exp_lon = [truth_moving(x, y, endtime)[0] for x, y, in zip(lon, lat)]
    exp_lat = [truth_moving(x, y, endtime)[1] for x, y, in zip(lon, lat)]
    assert np.allclose(np.array([p.lon for p in pset]), exp_lon, rtol=rtol)
    assert np.allclose(np.array([p.lat for p in pset]), exp_lat, rtol=rtol)


def truth_decaying(x_0, y_0, t):
    lat = y_0 - ((u_0 - u_g) * f / (f ** 2 + gamma ** 2)
                 * (1 - np.exp(-gamma * t) * (np.cos(f * t) + gamma / f * np.sin(f * t))))
    lon = x_0 + (u_g / gamma_g * (1 - np.exp(-gamma_g * t))
                 + (u_0 - u_g) * f / (f ** 2 + gamma ** 2)
                 * (gamma / f + np.exp(-gamma * t)
                    * (math.sin(f * t) - gamma / f * math.cos(f * t))))
    return lon, lat


@pytest.fixture
def fieldset_decaying(xdim=100, ydim=100, maxtime=delta(hours=6)):
    """Generate a FieldSet encapsulating the flow field of a decaying eddy.

    Reference: N. Fabbroni, 2009, "Numerical simulations of passive
    tracers dispersion in the sea"
    """
    time = np.arange(0., maxtime.total_seconds()+1e-5, 60., dtype=np.float64)
    dimensions = {'lon': np.linspace(0, 25000, xdim, dtype=np.float32),
                  'lat': np.linspace(0, 25000, ydim, dtype=np.float32),
                  'time': time}
    data = {'U': np.ones((xdim, ydim, 1), dtype=np.float32) * u_g * np.exp(-gamma_g * time) + (u_0 - u_g) * np.exp(-gamma * time) * np.cos(f * time),
            'V': np.ones((xdim, ydim, 1), dtype=np.float32) * -(u_0 - u_g) * np.exp(-gamma * time) * np.sin(f * time)}
    return FieldSet.from_data(data, dimensions, mesh='flat', transpose=True)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('method, rtol', [
    ('EE', 1e-2),
    ('RK4', 1e-5),
    ('RK45', 1e-5)])
def test_decaying_eddy(fieldset_decaying, mode, method, rtol, npart=1):
    fieldset = fieldset_decaying
    lon = np.linspace(12000, 21000, npart)
    lat = np.linspace(12500, 12500, npart)
    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=lon, lat=lat)
    endtime = delta(hours=6).total_seconds()
    pset.execute(kernel[method], dt=delta(minutes=3), endtime=endtime)
    exp_lon = [truth_decaying(x, y, endtime)[0] for x, y, in zip(lon, lat)]
    exp_lat = [truth_decaying(x, y, endtime)[1] for x, y, in zip(lon, lat)]
    assert np.allclose(np.array([p.lon for p in pset]), exp_lon, rtol=rtol)
    assert np.allclose(np.array([p.lat for p in pset]), exp_lat, rtol=rtol)
