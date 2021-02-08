from parcels import (FieldSet, Field, NestedField, ScipyParticle, JITParticle, Geographic,
                     AdvectionRK4, AdvectionRK4_3D, Variable, ErrorCode)
from parcels.particleset_vectorized import ParticleSet
import numpy as np
import pytest
from math import cos, pi
from datetime import timedelta as delta


ptype = {'scipy': ScipyParticle, 'jit': JITParticle}


def pclass(mode):
    class SampleParticle(ptype[mode]):
        u = Variable('u', dtype=np.float32)
        v = Variable('v', dtype=np.float32)
        p = Variable('p', dtype=np.float32)
    return SampleParticle


def k_sample_uv():
    def SampleUV(particle, fieldset, time):
        particle.u = fieldset.U[time, particle.depth, particle.lat, particle.lon]
        particle.v = fieldset.V[time, particle.depth, particle.lat, particle.lon]
    return SampleUV


@pytest.fixture(name="k_sample_uv")
def k_sample_uv_fixture():
    return k_sample_uv()


def k_sample_p():
    def SampleP(particle, fieldset, time):
        particle.p = fieldset.P[time, particle.depth, particle.lat, particle.lon]
    return SampleP


@pytest.fixture(name="k_sample_p")
def k_sample_P_fixture():
    return k_sample_p()


def fieldset(xdim=200, ydim=100):
    """ Standard fieldset spanning the earth's coordinates with U and V
        equivalent to longitude and latitude in deg.
    """
    lon = np.linspace(-180, 180, xdim, dtype=np.float32)
    lat = np.linspace(-90, 90, ydim, dtype=np.float32)
    U, V = np.meshgrid(lat, lon)
    data = {'U': U, 'V': V}
    dimensions = {'lon': lon, 'lat': lat}

    return FieldSet.from_data(data, dimensions, mesh='flat', transpose=True)


@pytest.fixture(name="fieldset")
def fieldset_fixture(xdim=200, ydim=100):
    return fieldset(xdim=xdim, ydim=ydim)


def fieldset_geometric(xdim=200, ydim=100):
    """ Standard earth fieldset with U and V equivalent to lon/lat in m. """
    lon = np.linspace(-180, 180, xdim, dtype=np.float32)
    lat = np.linspace(-90, 90, ydim, dtype=np.float32)
    U, V = np.meshgrid(lat, lon)
    U *= 1000. * 1.852 * 60.
    V *= 1000. * 1.852 * 60.
    data = {'U': U, 'V': V}
    dimensions = {'lon': lon, 'lat': lat}
    fieldset = FieldSet.from_data(data, dimensions, transpose=True)
    fieldset.U.units = Geographic()
    fieldset.V.units = Geographic()
    return fieldset


@pytest.fixture(name="fieldset_geometric")
def fieldset_geometric_fixture(xdim=200, ydim=100):
    return fieldset_geometric(xdim=xdim, ydim=ydim)


def fieldset_geometric_polar(xdim=200, ydim=100):
    """ Standard earth fieldset with U and V equivalent to lon/lat in m
        and the inversion of the pole correction applied to U.
    """
    lon = np.linspace(-180, 180, xdim, dtype=np.float32)
    lat = np.linspace(-90, 90, ydim, dtype=np.float32)
    U, V = np.meshgrid(lat, lon)
    # Apply inverse of pole correction to U
    for i, y in enumerate(lat):
        U[:, i] *= cos(y * pi / 180)
    U *= 1000. * 1.852 * 60.
    V *= 1000. * 1.852 * 60.
    data = {'U': U, 'V': V}
    dimensions = {'lon': lon, 'lat': lat}
    return FieldSet.from_data(data, dimensions, mesh='spherical', transpose=True)


@pytest.fixture(name="fieldset_geometric_polar")
def fieldset_geometric_polar_fixture(xdim=200, ydim=100):
    return fieldset_geometric_polar(xdim=xdim, ydim=ydim)


def test_fieldset_sample(fieldset, xdim=120, ydim=80):
    """ Sample the fieldset using indexing notation. """
    lon = np.linspace(-170, 170, xdim, dtype=np.float32)
    lat = np.linspace(-80, 80, ydim, dtype=np.float32)
    v_s = np.array([fieldset.V[0, 0., 70., x] for x in lon])
    u_s = np.array([fieldset.U[0, 0., y, -45.] for y in lat])
    assert np.allclose(v_s, lon, rtol=1e-7)
    assert np.allclose(u_s, lat, rtol=1e-7)


def test_fieldset_sample_eval(fieldset, xdim=60, ydim=60):
    """ Sample the fieldset using the explicit eval function. """
    lon = np.linspace(-170, 170, xdim, dtype=np.float32)
    lat = np.linspace(-80, 80, ydim, dtype=np.float32)
    v_s = np.array([fieldset.V.eval(0, 0., 70., x) for x in lon])
    u_s = np.array([fieldset.U.eval(0, 0., y, 0.) for y in lat])
    assert np.allclose(v_s, lon, rtol=1e-7)
    assert np.allclose(u_s, lat, rtol=1e-7)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_fieldset_polar_with_halo(fieldset_geometric_polar, mode):
    fieldset_geometric_polar.add_periodic_halo(zonal=5)
    pset = ParticleSet(fieldset_geometric_polar, pclass=pclass(mode), lon=0, lat=0)
    pset.execute(runtime=1)
    assert(pset[0].lon == 0.)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_variable_init_from_field(mode, npart=9):
    dims = (2, 2)
    dimensions = {'lon': np.linspace(0., 1., dims[0], dtype=np.float32),
                  'lat': np.linspace(0., 1., dims[1], dtype=np.float32)}
    data = {'U': np.zeros(dims, dtype=np.float32),
            'V': np.zeros(dims, dtype=np.float32),
            'P': np.zeros(dims, dtype=np.float32)}
    data['P'][0, 0] = 1.
    fieldset = FieldSet.from_data(data, dimensions, mesh='flat', transpose=True)
    xv, yv = np.meshgrid(np.linspace(0, 1, int(np.sqrt(npart))), np.linspace(0, 1, int(np.sqrt(npart))))

    class VarParticle(pclass(mode)):
        a = Variable('a', dtype=np.float32, initial=fieldset.P)

    pset = ParticleSet(fieldset, pclass=VarParticle,
                       lon=xv.flatten(), lat=yv.flatten(), time=0)
    assert np.all([abs(p.a - fieldset.P[p.time, p.depth, p.lat, p.lon]) < 1e-6 for p in pset])


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pset_from_field(mode, xdim=10, ydim=20, npart=10000):
    np.random.seed(123456)
    dimensions = {'lon': np.linspace(0., 1., xdim, dtype=np.float32),
                  'lat': np.linspace(0., 1., ydim, dtype=np.float32)}
    startfield = np.ones((xdim, ydim), dtype=np.float32)
    for x in range(xdim):
        startfield[x, :] = x
    data = {'U': np.zeros((xdim, ydim), dtype=np.float32),
            'V': np.zeros((xdim, ydim), dtype=np.float32),
            'start': startfield}
    fieldset = FieldSet.from_data(data, dimensions, mesh='flat', transpose=True)

    pset = ParticleSet.from_field(fieldset, size=npart, pclass=pclass(mode),
                                  start_field=fieldset.start)
    densfield = Field(name='densfield', data=np.zeros((xdim+1, ydim+1), dtype=np.float32),
                      lon=np.linspace(-1./(xdim*2), 1.+1./(xdim*2), xdim+1, dtype=np.float32),
                      lat=np.linspace(-1./(ydim*2), 1.+1./(ydim*2), ydim+1, dtype=np.float32), transpose=True)
    pdens = pset.density(field=densfield, relative=True)[:-1, :-1]
    assert np.allclose(np.transpose(pdens), startfield/np.sum(startfield), atol=1e-2)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_nearest_neighbour_interpolation2D(mode, k_sample_p, npart=81):
    dims = (2, 2)
    dimensions = {'lon': np.linspace(0., 1., dims[0], dtype=np.float32),
                  'lat': np.linspace(0., 1., dims[1], dtype=np.float32)}
    data = {'U': np.zeros(dims, dtype=np.float32),
            'V': np.zeros(dims, dtype=np.float32),
            'P': np.zeros(dims, dtype=np.float32)}
    data['P'][0, 1] = 1.
    fieldset = FieldSet.from_data(data, dimensions, mesh='flat', transpose=True)
    fieldset.P.interp_method = 'nearest'
    xv, yv = np.meshgrid(np.linspace(0., 1.0, int(np.sqrt(npart))), np.linspace(0., 1.0, int(np.sqrt(npart))))
    pset = ParticleSet(fieldset, pclass=pclass(mode),
                       lon=xv.flatten(), lat=yv.flatten())
    pset.execute(k_sample_p, endtime=1, dt=1)
    assert np.allclose(np.array([p.p for p in pset if p.lon < 0.5 and p.lat > 0.5]), 1.0, rtol=1e-5)
    assert np.allclose(np.array([p.p for p in pset if p.lon > 0.5 or p.lat < 0.5]), 0.0, rtol=1e-5)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_nearest_neighbour_interpolation3D(mode, k_sample_p, npart=81):
    dims = (2, 2, 2)
    dimensions = {'lon': np.linspace(0., 1., dims[0], dtype=np.float32),
                  'lat': np.linspace(0., 1., dims[1], dtype=np.float32),
                  'depth': np.linspace(0., 1., dims[2], dtype=np.float32)}
    data = {'U': np.zeros(dims, dtype=np.float32),
            'V': np.zeros(dims, dtype=np.float32),
            'P': np.zeros(dims, dtype=np.float32)}
    data['P'][0, 1, 1] = 1.
    fieldset = FieldSet.from_data(data, dimensions, mesh='flat', transpose=True)
    fieldset.P.interp_method = 'nearest'
    xv, yv = np.meshgrid(np.linspace(0, 1.0, int(np.sqrt(npart))), np.linspace(0, 1.0, int(np.sqrt(npart))))
    # combine a pset at 0m with pset at 1m, as meshgrid does not do 3D
    pset = ParticleSet(fieldset, pclass=pclass(mode), lon=xv.flatten(), lat=yv.flatten(), depth=np.zeros(npart))
    pset2 = ParticleSet(fieldset, pclass=pclass(mode), lon=xv.flatten(), lat=yv.flatten(), depth=np.ones(npart))
    pset.add(pset2)

    pset.execute(k_sample_p, endtime=1, dt=1)
    assert np.allclose(np.array([p.p for p in pset if p.lon < 0.5 and p.lat > 0.5 and p.depth > 0.5]), 1.0, rtol=1e-5)
    assert np.allclose(np.array([p.p for p in pset if p.lon > 0.5 or p.lat < 0.5 and p.depth < 0.5]), 0.0, rtol=1e-5)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('lat_flip', [False, True])
def test_fieldset_sample_particle(mode, k_sample_uv, lat_flip, npart=120):
    """ Sample the fieldset using an array of particles.

    Note that the low tolerances (1.e-6) are due to the first-order
    interpolation in JIT mode and give an indication of the
    corresponding sampling error.
    """

    lon = np.linspace(-180, 180, 200, dtype=np.float32)
    if lat_flip:
        lat = np.linspace(90, -90, 100, dtype=np.float32)
    else:
        lat = np.linspace(-90, 90, 100, dtype=np.float32)
    U, V = np.meshgrid(lat, lon)
    data = {'U': U, 'V': V}
    dimensions = {'lon': lon, 'lat': lat}

    fieldset = FieldSet.from_data(data, dimensions, mesh='flat', transpose=True)

    lon = np.linspace(-170, 170, npart)
    lat = np.linspace(-80, 80, npart)
    pset = ParticleSet(fieldset, pclass=pclass(mode), lon=lon, lat=np.zeros(npart) + 70.)
    pset.execute(pset.Kernel(k_sample_uv), endtime=1., dt=1.)
    assert np.allclose(np.array([p.v for p in pset]), lon, rtol=1e-6)

    pset = ParticleSet(fieldset, pclass=pclass(mode), lat=lat, lon=np.zeros(npart) - 45.)
    pset.execute(pset.Kernel(k_sample_uv), endtime=1., dt=1.)
    assert np.allclose(np.array([p.u for p in pset]), lat, rtol=1e-6)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_fieldset_sample_geographic(fieldset_geometric, mode, k_sample_uv, npart=120):
    """ Sample a fieldset with conversion to geographic units (degrees). """
    fieldset = fieldset_geometric
    lon = np.linspace(-170, 170, npart)
    lat = np.linspace(-80, 80, npart)

    pset = ParticleSet(fieldset, pclass=pclass(mode), lon=lon, lat=np.zeros(npart) + 70.)
    pset.execute(pset.Kernel(k_sample_uv), endtime=1., dt=1.)
    assert np.allclose(np.array([p.v for p in pset]), lon, rtol=1e-6)

    pset = ParticleSet(fieldset, pclass=pclass(mode), lat=lat, lon=np.zeros(npart) - 45.)
    pset.execute(pset.Kernel(k_sample_uv), endtime=1., dt=1.)
    assert np.allclose(np.array([p.u for p in pset]), lat, rtol=1e-6)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_fieldset_sample_geographic_polar(fieldset_geometric_polar, mode, k_sample_uv, npart=120):
    """ Sample a fieldset with conversion to geographic units and a pole correction. """
    fieldset = fieldset_geometric_polar
    lon = np.linspace(-170, 170, npart)
    lat = np.linspace(-80, 80, npart)

    pset = ParticleSet(fieldset, pclass=pclass(mode), lon=lon, lat=np.zeros(npart) + 70.)
    pset.execute(pset.Kernel(k_sample_uv), endtime=1., dt=1.)
    assert np.allclose(np.array([p.v for p in pset]), lon, rtol=1e-6)

    pset = ParticleSet(fieldset, pclass=pclass(mode), lat=lat, lon=np.zeros(npart) - 45.)
    pset.execute(pset.Kernel(k_sample_uv), endtime=1., dt=1.)
    # Note: 1.e-2 is a very low rtol, so there seems to be a rather
    # large sampling error for the JIT correction.
    assert np.allclose(np.array([p.u for p in pset]), lat, rtol=1e-2)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_meridionalflow_spherical(mode, xdim=100, ydim=200):
    """ Create uniform NORTHWARD flow on spherical earth and advect particles

    As flow is so simple, it can be directly compared to analytical solution
    """

    maxvel = 1.
    dimensions = {'lon': np.linspace(-180, 180, xdim, dtype=np.float32),
                  'lat': np.linspace(-90, 90, ydim, dtype=np.float32)}
    data = {'U': np.zeros([xdim, ydim]),
            'V': maxvel * np.ones([xdim, ydim])}

    fieldset = FieldSet.from_data(data, dimensions, mesh='spherical', transpose=True)

    lonstart = [0, 45]
    latstart = [0, 45]
    runtime = delta(hours=24)
    pset = ParticleSet(fieldset, pclass=pclass(mode), lon=lonstart, lat=latstart)
    pset.execute(pset.Kernel(AdvectionRK4), runtime=runtime, dt=delta(hours=1))

    assert(pset[0].lat - (latstart[0] + runtime.total_seconds() * maxvel / 1852 / 60) < 1e-4)
    assert(pset[0].lon - lonstart[0] < 1e-4)
    assert(pset[1].lat - (latstart[1] + runtime.total_seconds() * maxvel / 1852 / 60) < 1e-4)
    assert(pset[1].lon - lonstart[1] < 1e-4)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_zonalflow_spherical(mode, k_sample_p, xdim=100, ydim=200):
    """ Create uniform EASTWARD flow on spherical earth and advect particles

    As flow is so simple, it can be directly compared to analytical solution
    Note that in this case the cosine conversion is needed
    """
    maxvel = 1.
    p_fld = 10
    dimensions = {'lon': np.linspace(-180, 180, xdim, dtype=np.float32),
                  'lat': np.linspace(-90, 90, ydim, dtype=np.float32)}
    data = {'U': maxvel * np.ones([xdim, ydim]),
            'V': np.zeros([xdim, ydim]),
            'P': p_fld * np.ones([xdim, ydim])}

    fieldset = FieldSet.from_data(data, dimensions, mesh='spherical', transpose=True)

    lonstart = [0, 45]
    latstart = [0, 45]
    runtime = delta(hours=24)
    pset = ParticleSet(fieldset, pclass=pclass(mode), lon=lonstart, lat=latstart)
    pset.execute(pset.Kernel(AdvectionRK4) + k_sample_p,
                 runtime=runtime, dt=delta(hours=1))

    assert(pset[0].lat - latstart[0] < 1e-4)
    assert(pset[0].lon - (lonstart[0] + runtime.total_seconds() * maxvel / 1852 / 60
                          / cos(latstart[0] * pi / 180)) < 1e-4)
    assert(abs(pset[0].p - p_fld) < 1e-4)
    assert(pset[1].lat - latstart[1] < 1e-4)
    assert(pset[1].lon - (lonstart[1] + runtime.total_seconds() * maxvel / 1852 / 60
                          / cos(latstart[1] * pi / 180)) < 1e-4)
    assert(abs(pset[1].p - p_fld) < 1e-4)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_random_field(mode, k_sample_p, xdim=20, ydim=20, npart=100):
    """Sampling test that tests for overshoots by sampling a field of
    random numbers between 0 and 1.
    """
    np.random.seed(123456)
    dimensions = {'lon': np.linspace(0., 1., xdim, dtype=np.float32),
                  'lat': np.linspace(0., 1., ydim, dtype=np.float32)}
    data = {'U': np.zeros((xdim, ydim), dtype=np.float32),
            'V': np.zeros((xdim, ydim), dtype=np.float32),
            'P': np.random.uniform(0, 1., size=(xdim, ydim)),
            'start': np.ones((xdim, ydim), dtype=np.float32)}

    fieldset = FieldSet.from_data(data, dimensions, mesh='flat', transpose=True)
    pset = ParticleSet.from_field(fieldset, size=npart, pclass=pclass(mode),
                                  start_field=fieldset.start)
    pset.execute(k_sample_p, endtime=1., dt=1.0)
    sampled = np.array([p.p for p in pset])
    assert((sampled >= 0.).all())


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('allow_time_extrapolation', [True, False])
def test_sampling_out_of_bounds_time(mode, allow_time_extrapolation, k_sample_p,
                                     xdim=10, ydim=10, tdim=10):
    dimensions = {'lon': np.linspace(0., 1., xdim, dtype=np.float32),
                  'lat': np.linspace(0., 1., ydim, dtype=np.float32),
                  'time': np.linspace(0., 1., tdim, dtype=np.float64)}
    data = {'U': np.zeros((xdim, ydim, tdim), dtype=np.float32),
            'V': np.zeros((xdim, ydim, tdim), dtype=np.float32),
            'P': np.ones((xdim, ydim, 1), dtype=np.float32) * dimensions['time']}

    fieldset = FieldSet.from_data(data, dimensions, mesh='flat',
                                  allow_time_extrapolation=allow_time_extrapolation, transpose=True)
    pset = ParticleSet(fieldset, pclass=pclass(mode), lon=[0.5], lat=[0.5], time=-1.0)
    if allow_time_extrapolation:
        pset.execute(k_sample_p, endtime=-0.9, dt=0.1)
        assert np.allclose(np.array([p.p for p in pset]), 0.0, rtol=1e-5)
    else:
        with pytest.raises(RuntimeError):
            pset.execute(k_sample_p, endtime=-0.9, dt=0.1)

    pset = ParticleSet(fieldset, pclass=pclass(mode), lon=[0.5], lat=[0.5], time=0)
    pset.execute(k_sample_p, runtime=0.1, dt=0.1)
    assert np.allclose(np.array([p.p for p in pset]), 0.0, rtol=1e-5)

    pset = ParticleSet(fieldset, pclass=pclass(mode), lon=[0.5], lat=[0.5], time=0.5)
    pset.execute(k_sample_p, runtime=0.1, dt=0.1)
    assert np.allclose(np.array([p.p for p in pset]), 0.5, rtol=1e-5)

    pset = ParticleSet(fieldset, pclass=pclass(mode), lon=[0.5], lat=[0.5], time=1.0)
    pset.execute(k_sample_p, runtime=0.1, dt=0.1)
    assert np.allclose(np.array([p.p for p in pset]), 1.0, rtol=1e-5)

    pset = ParticleSet(fieldset, pclass=pclass(mode), lon=[0.5], lat=[0.5], time=2.0)
    if allow_time_extrapolation:
        pset.execute(k_sample_p, runtime=0.1, dt=0.1)
        assert np.allclose(np.array([p.p for p in pset]), 1.0, rtol=1e-5)
    else:
        with pytest.raises(RuntimeError):
            pset.execute(k_sample_p, runtime=0.1, dt=0.1)


@pytest.mark.parametrize('mode', ['jit', 'scipy'])
@pytest.mark.parametrize('npart', [1, 10])
@pytest.mark.parametrize('chs', [False, 'auto', (10, 10)])
def test_sampling_multigrids_non_vectorfield_from_file(mode, npart, tmpdir, chs, filename='test_subsets'):
    xdim, ydim = 100, 200
    filepath = tmpdir.join(filename)
    U = Field('U', np.zeros((ydim, xdim), dtype=np.float32),
              lon=np.linspace(0., 1., xdim, dtype=np.float32),
              lat=np.linspace(0., 1., ydim, dtype=np.float32))
    V = Field('V', np.zeros((ydim, xdim), dtype=np.float32),
              lon=np.linspace(0., 1., xdim, dtype=np.float32),
              lat=np.linspace(0., 1., ydim, dtype=np.float32))
    B = Field('B', np.ones((3*ydim, 4*xdim), dtype=np.float32),
              lon=np.linspace(0., 1., 4*xdim, dtype=np.float32),
              lat=np.linspace(0., 1., 3*ydim, dtype=np.float32))
    fieldset = FieldSet(U, V)
    fieldset.add_field(B, 'B')
    fieldset.write(filepath)
    fieldset = None

    ufiles = [filepath+'U.nc', ] * 4
    vfiles = [filepath+'V.nc', ] * 4
    bfiles = [filepath+'B.nc', ] * 4
    timestamps = np.arange(0, 4, 1) * 86400.0
    timestamps = np.expand_dims(timestamps, 1)
    files = {'U': ufiles, 'V': vfiles, 'B': bfiles}
    variables = {'U': 'vozocrtx', 'V': 'vomecrty', 'B': 'B'}
    dimensions = {'lon': 'nav_lon', 'lat': 'nav_lat'}
    fieldset = FieldSet.from_netcdf(files, variables, dimensions, timestamps=timestamps, allow_time_extrapolation=True,
                                    field_chunksize=chs)

    fieldset.add_constant('sample_depth', 2.5)
    assert fieldset.U.grid is fieldset.V.grid
    assert fieldset.U.grid is not fieldset.B.grid

    class TestParticle(ptype[mode]):
        sample_var = Variable('sample_var', initial=0.)

    pset = ParticleSet.from_line(fieldset, pclass=TestParticle, start=[0.3, 0.3], finish=[0.7, 0.7], size=npart)

    def test_sample(particle, fieldset, time):
        particle.sample_var += fieldset.B[time, fieldset.sample_depth, particle.lat, particle.lon]

    kernels = pset.Kernel(AdvectionRK4) + pset.Kernel(test_sample)
    pset.execute(kernels, runtime=10, dt=1)
    assert np.allclose(np.array([p.sample_var for p in pset]), 10.0)
    if mode == 'jit':
        assert np.all(np.array([len(p.xi) == fieldset.gridset.size for p in pset]))
        assert np.all(np.array([[p.xi[i] >= 0 for i in range(0, len(p.xi))] for p in pset]))
        #assert np.all(pset.xi[:, fieldset.B.igrid] < xdim * 4)
        #assert np.all(pset.xi[:, 0] < xdim)
        #assert pset.yi.shape[0] == len(pset.lon)
        #assert pset.yi.shape[1] == fieldset.gridset.size
        #assert np.all(pset.yi >= 0)
        #assert np.all(pset.yi[:, fieldset.B.igrid] < ydim * 3)
        #assert np.all(pset.yi[:, 0] < ydim)


@pytest.mark.parametrize('mode', ['jit', 'scipy'])
@pytest.mark.parametrize('npart', [1, 10])
def test_sampling_multigrids_non_vectorfield(mode, npart):
    xdim, ydim = 100, 200
    U = Field('U', np.zeros((ydim, xdim), dtype=np.float32),
              lon=np.linspace(0., 1., xdim, dtype=np.float32),
              lat=np.linspace(0., 1., ydim, dtype=np.float32))
    V = Field('V', np.zeros((ydim, xdim), dtype=np.float32),
              lon=np.linspace(0., 1., xdim, dtype=np.float32),
              lat=np.linspace(0., 1., ydim, dtype=np.float32))
    B = Field('B', np.ones((3*ydim, 4*xdim), dtype=np.float32),
              lon=np.linspace(0., 1., 4*xdim, dtype=np.float32),
              lat=np.linspace(0., 1., 3*ydim, dtype=np.float32))
    fieldset = FieldSet(U, V)
    fieldset.add_field(B, 'B')
    fieldset.add_constant('sample_depth', 2.5)
    assert fieldset.U.grid is fieldset.V.grid
    assert fieldset.U.grid is not fieldset.B.grid

    class TestParticle(ptype[mode]):
        sample_var = Variable('sample_var', initial=0.)

    pset = ParticleSet.from_line(fieldset, pclass=TestParticle, start=[0.3, 0.3], finish=[0.7, 0.7], size=npart)

    def test_sample(particle, fieldset, time):
        particle.sample_var += fieldset.B[time, fieldset.sample_depth, particle.lat, particle.lon]

    kernels = pset.Kernel(AdvectionRK4) + pset.Kernel(test_sample)
    pset.execute(kernels, runtime=10, dt=1)
    assert np.allclose(np.array([p.sample_var for p in pset]), 10.0)
    if mode == 'jit':
        assert np.all(np.array([len(p.xi) == fieldset.gridset.size for p in pset]))
        assert np.all(np.array([[p.xi[i] >= 0 for i in range(0, len(p.xi))] for p in pset]))
        #assert np.all(pset.xi[:, fieldset.B.igrid] < xdim * 4)
        #assert np.all(pset.xi[:, 0] < xdim)
        #assert pset.yi.shape[0] == len(pset.lon)
        #assert pset.yi.shape[1] == fieldset.gridset.size
        #assert np.all(pset.yi >= 0)
        #assert np.all(pset.yi[:, fieldset.B.igrid] < ydim * 3)
        #assert np.all(pset.yi[:, 0] < ydim)


@pytest.mark.parametrize('mode', ['jit', 'scipy'])
@pytest.mark.parametrize('ugridfactor', [1, 10])
def test_sampling_multiple_grid_sizes(mode, ugridfactor):
    xdim, ydim = 10, 20
    U = Field('U', np.zeros((ydim*ugridfactor, xdim*ugridfactor), dtype=np.float32),
              lon=np.linspace(0., 1., xdim*ugridfactor, dtype=np.float32),
              lat=np.linspace(0., 1., ydim*ugridfactor, dtype=np.float32))
    V = Field('V', np.zeros((ydim, xdim), dtype=np.float32),
              lon=np.linspace(0., 1., xdim, dtype=np.float32),
              lat=np.linspace(0., 1., ydim, dtype=np.float32))
    fieldset = FieldSet(U, V)
    pset = ParticleSet(fieldset, pclass=pclass(mode), lon=[0.8], lat=[0.9])

    if ugridfactor > 1:
        assert fieldset.U.grid is not fieldset.V.grid
    else:
        assert fieldset.U.grid is fieldset.V.grid
    pset.execute(AdvectionRK4, runtime=10, dt=1)
    assert np.isclose(pset[0].lon, 0.8)
    assert np.all((pset[0].xi >= 0) & (pset[0].xi < xdim*ugridfactor))


def test_multiple_grid_addlater_error():
    xdim, ydim = 10, 20
    U = Field('U', np.zeros((ydim, xdim), dtype=np.float32),
              lon=np.linspace(0., 1., xdim, dtype=np.float32),
              lat=np.linspace(0., 1., ydim, dtype=np.float32))
    V = Field('V', np.zeros((ydim, xdim), dtype=np.float32),
              lon=np.linspace(0., 1., xdim, dtype=np.float32),
              lat=np.linspace(0., 1., ydim, dtype=np.float32))
    fieldset = FieldSet(U, V)

    pset = ParticleSet(fieldset, pclass=pclass('jit'), lon=[0.8], lat=[0.9])

    P = Field('P', np.zeros((ydim*10, xdim*10), dtype=np.float32),
              lon=np.linspace(0., 1., xdim*10, dtype=np.float32),
              lat=np.linspace(0., 1., ydim*10, dtype=np.float32))
    fieldset.add_field(P)

    fail = False
    try:
        pset.execute(AdvectionRK4, runtime=10, dt=1)
    except:
        fail = True
    assert fail


@pytest.mark.parametrize('mode', ['jit', 'scipy'])
def test_sampling_multiple_grid_sizes(mode):
    """Sampling test that tests for FieldSet with different grid sizes

    While this currently works fine in Scipy mode, it fails in JIT mode with
    an out_of_bounds_error because there is only one (xi, yi, zi) for each particle
    A solution would be to define xi, yi, zi for each field separately
    """
    xdim = 10
    ydim = 20
    gf = 10  # factor by which the resolution of U is higher than of V
    U = Field('U', np.zeros((ydim*gf, xdim*gf), dtype=np.float32),
              lon=np.linspace(0., 1., xdim*gf, dtype=np.float32),
              lat=np.linspace(0., 1., ydim*gf, dtype=np.float32))
    V = Field('V', np.zeros((ydim, xdim), dtype=np.float32),
              lon=np.linspace(0., 1., xdim, dtype=np.float32),
              lat=np.linspace(0., 1., ydim, dtype=np.float32))
    fieldset = FieldSet(U, V)
    pset = ParticleSet(fieldset, pclass=pclass(mode), lon=[0.8], lat=[0.9])

    pset.execute(AdvectionRK4, runtime=10, dt=1)
    assert np.isclose(pset[0].lon, 0.8)


@pytest.mark.parametrize('mode', ['jit', 'scipy'])
@pytest.mark.parametrize('with_W', [True, False])
@pytest.mark.parametrize('mesh', ['flat', 'spherical'])
def test_summedfields(mode, with_W, k_sample_p, mesh):
    xdim = 10
    ydim = 20
    zdim = 4
    gf = 10  # factor by which the resolution of grid1 is higher than of grid2
    U1 = Field('U', 0.2*np.ones((zdim*gf, ydim*gf, xdim*gf), dtype=np.float32),
               lon=np.linspace(0., 1., xdim*gf, dtype=np.float32),
               lat=np.linspace(0., 1., ydim*gf, dtype=np.float32),
               depth=np.linspace(0., 20., zdim*gf, dtype=np.float32),
               mesh=mesh)
    U2 = Field('U', 0.1*np.ones((zdim, ydim, xdim), dtype=np.float32),
               lon=np.linspace(0., 1., xdim, dtype=np.float32),
               lat=np.linspace(0., 1., ydim, dtype=np.float32),
               depth=np.linspace(0., 20., zdim, dtype=np.float32),
               mesh=mesh)
    V1 = Field('V', np.zeros((zdim*gf, ydim*gf, xdim*gf), dtype=np.float32), grid=U1.grid, fieldtype='V')
    V2 = Field('V', np.zeros((zdim, ydim, xdim), dtype=np.float32), grid=U2.grid, fieldtype='V')
    fieldsetS = FieldSet(U1+U2, V1+V2)

    conv = 1852*60 if mesh == 'spherical' else 1.
    assert np.allclose(fieldsetS.U[0, 0, 0, 0]*conv, 0.3)

    P1 = Field('P', 30*np.ones((zdim*gf, ydim*gf, xdim*gf), dtype=np.float32), grid=U1.grid)
    P2 = Field('P', 20*np.ones((zdim, ydim, xdim), dtype=np.float32), grid=U2.grid)
    P3 = Field('P', 10*np.ones((zdim, ydim, xdim), dtype=np.float32), grid=U2.grid)
    P4 = Field('P', 0*np.ones((zdim, ydim, xdim), dtype=np.float32), grid=U2.grid)
    fieldsetS.add_field((P1+P4)+(P2+P3), name='P')
    assert np.allclose(fieldsetS.P[0, 0, 0, 0], 60)

    if with_W:
        W1 = Field('W', 2*np.ones((zdim * gf, ydim * gf, xdim * gf), dtype=np.float32), grid=U1.grid)
        W2 = Field('W', np.ones((zdim, ydim, xdim), dtype=np.float32), grid=U2.grid)
        fieldsetS.add_field(W1+W2, name='W')
        pset = ParticleSet(fieldsetS, pclass=pclass(mode), lon=[0], lat=[0.9])
        pset.execute(AdvectionRK4_3D+pset.Kernel(k_sample_p), runtime=2, dt=1)
        assert np.isclose(pset[0].depth, 6)
    else:
        pset = ParticleSet(fieldsetS, pclass=pclass(mode), lon=[0], lat=[0.9])
        pset.execute(AdvectionRK4+pset.Kernel(k_sample_p), runtime=2, dt=1)
    assert np.isclose(pset[0].p, 60)
    assert np.isclose(pset[0].lon*conv, 0.6, atol=1e-3)
    assert np.isclose(pset[0].lat, 0.9)
    assert np.allclose(fieldsetS.UV[0][0, 0, 0, 0], [.2/conv, 0])


@pytest.mark.parametrize('mode', ['jit', 'scipy'])
def test_nestedfields(mode, k_sample_p):
    xdim = 10
    ydim = 20

    U1 = Field('U1', 0.1*np.ones((ydim, xdim), dtype=np.float32),
               lon=np.linspace(0., 1., xdim, dtype=np.float32),
               lat=np.linspace(0., 1., ydim, dtype=np.float32))
    V1 = Field('V1', 0.2*np.ones((ydim, xdim), dtype=np.float32),
               lon=np.linspace(0., 1., xdim, dtype=np.float32),
               lat=np.linspace(0., 1., ydim, dtype=np.float32))
    U2 = Field('U2', 0.3*np.ones((ydim, xdim), dtype=np.float32),
               lon=np.linspace(0., 2., xdim, dtype=np.float32),
               lat=np.linspace(0., 2., ydim, dtype=np.float32))
    V2 = Field('V2', 0.4*np.ones((ydim, xdim), dtype=np.float32),
               lon=np.linspace(0., 2., xdim, dtype=np.float32),
               lat=np.linspace(0., 2., ydim, dtype=np.float32))
    U = NestedField('U', [U1, U2])
    V = NestedField('V', [V1, V2])
    fieldset = FieldSet(U, V)

    P1 = Field('P1', 0.1*np.ones((ydim, xdim), dtype=np.float32),
               lon=np.linspace(0., 1., xdim, dtype=np.float32),
               lat=np.linspace(0., 1., ydim, dtype=np.float32))
    P2 = Field('P2', 0.2*np.ones((ydim, xdim), dtype=np.float32),
               lon=np.linspace(0., 2., xdim, dtype=np.float32),
               lat=np.linspace(0., 2., ydim, dtype=np.float32))
    P = NestedField('P', [P1, P2])
    fieldset.add_field(P)

    def Recover(particle, fieldset, time):
        particle.lon = -1
        particle.lat = -1
        particle.p = 999
        particle.time = particle.time + particle.dt

    pset = ParticleSet(fieldset, pclass=pclass(mode), lon=[0], lat=[.3])
    pset.execute(AdvectionRK4+pset.Kernel(k_sample_p), runtime=1, dt=1)
    assert np.isclose(pset[0].lat, .5)
    assert np.isclose(pset[0].p, .1)
    pset = ParticleSet(fieldset, pclass=pclass(mode), lon=[0], lat=[1.3])
    pset.execute(AdvectionRK4+pset.Kernel(k_sample_p), runtime=1, dt=1)
    assert np.isclose(pset[0].lat, 1.7)
    assert np.isclose(pset[0].p, .2)
    pset = ParticleSet(fieldset, pclass=pclass(mode), lon=[0], lat=[2.3])
    pset.execute(AdvectionRK4+pset.Kernel(k_sample_p), runtime=1, dt=1, recovery={ErrorCode.ErrorOutOfBounds: Recover})
    assert np.isclose(pset[0].lat, -1)
    assert np.isclose(pset[0].p, 999)
    assert np.allclose(fieldset.UV[0][0, 0, 0, 0], [.1, .2])
