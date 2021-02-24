from parcels import (FieldSet, Field, ParticleSet, ScipyParticle, JITParticle, Variable, AdvectionRK4, AdvectionRK4_3D, ErrorCode, UnitConverter)
from parcels import RectilinearZGrid, RectilinearSGrid, CurvilinearZGrid
import numpy as np
import xarray as xr
import math
import pytest
from os import path
from datetime import timedelta as delta

ptype = {'scipy': ScipyParticle, 'jit': JITParticle}


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_multi_structured_grids(mode):

    def temp_func(lon, lat):
        return 20 + lat/1000. + 2 * np.sin(lon*2*np.pi/5000.)

    a = 10000
    b = 10000

    # Grid 0
    xdim_g0 = 201
    ydim_g0 = 201
    # Coordinates of the test fieldset (on A-grid in deg)
    lon_g0 = np.linspace(0, a, xdim_g0, dtype=np.float32)
    lat_g0 = np.linspace(0, b, ydim_g0, dtype=np.float32)
    time_g0 = np.linspace(0., 1000., 2, dtype=np.float64)
    grid_0 = RectilinearZGrid(lon_g0, lat_g0, time=time_g0)

    # Grid 1
    xdim_g1 = 51
    ydim_g1 = 51
    # Coordinates of the test fieldset (on A-grid in deg)
    lon_g1 = np.linspace(0, a, xdim_g1, dtype=np.float32)
    lat_g1 = np.linspace(0, b, ydim_g1, dtype=np.float32)
    time_g1 = np.linspace(0., 1000., 2, dtype=np.float64)
    grid_1 = RectilinearZGrid(lon_g1, lat_g1, time=time_g1)

    u_data = np.ones((lon_g0.size, lat_g0.size, time_g0.size), dtype=np.float32)
    u_data = 2*u_data
    u_field = Field('U', u_data, grid=grid_0, transpose=True)

    temp0_data = np.empty((lon_g0.size, lat_g0.size, time_g0.size), dtype=np.float32)
    for i in range(lon_g0.size):
        for j in range(lat_g0.size):
            temp0_data[i, j, :] = temp_func(lon_g0[i], lat_g0[j])
    temp0_field = Field('temp0', temp0_data, grid=grid_0, transpose=True)

    v_data = np.zeros((lon_g1.size, lat_g1.size, time_g1.size), dtype=np.float32)
    v_field = Field('V', v_data, grid=grid_1, transpose=True)

    temp1_data = np.empty((lon_g1.size, lat_g1.size, time_g1.size), dtype=np.float32)
    for i in range(lon_g1.size):
        for j in range(lat_g1.size):
            temp1_data[i, j, :] = temp_func(lon_g1[i], lat_g1[j])
    temp1_field = Field('temp1', temp1_data, grid=grid_1, transpose=True)

    other_fields = {}
    other_fields['temp0'] = temp0_field
    other_fields['temp1'] = temp1_field

    field_set = FieldSet(u_field, v_field, fields=other_fields)

    def sampleTemp(particle, fieldset, time):
        # Note that fieldset.temp is interpolated at time=time+dt.
        # Indeed, sampleTemp is called at time=time, but the result is written
        # at time=time+dt, after the Kernel update
        particle.temp0 = fieldset.temp0[time+particle.dt, particle.depth, particle.lat, particle.lon]
        particle.temp1 = fieldset.temp1[time+particle.dt, particle.depth, particle.lat, particle.lon]

    class MyParticle(ptype[mode]):
        temp0 = Variable('temp0', dtype=np.float32, initial=20.)
        temp1 = Variable('temp1', dtype=np.float32, initial=20.)

    pset = ParticleSet.from_list(field_set, MyParticle, lon=[3001], lat=[5001], repeatdt=1)

    pset.execute(AdvectionRK4 + pset.Kernel(sampleTemp), runtime=3, dt=1)

    # check if particle xi and yi are different for the two grids
    assert np.all([pset.xi[i, 0] != pset.xi[i, 1] for i in range(3)])
    assert np.all([pset.yi[i, 0] != pset.yi[i, 1] for i in range(3)])

    # advect without updating temperature to test particle deletion
    pset.remove_indices(np.array([1]))
    pset.execute(AdvectionRK4, runtime=1, dt=1)

    assert np.allclose(pset.temp0[0], pset.temp1[0], atol=1e-3)


@pytest.mark.xfail(reason="Grid cannot be computed using a time vector which is neither float nor int", strict=True)
def test_time_format_in_grid():
    lon = np.linspace(0, 1, 2, dtype=np.float32)
    lat = np.linspace(0, 1, 2, dtype=np.float32)
    time = np.array([np.datetime64('2000-01-01')]*2)
    RectilinearZGrid(lon, lat, time=time)


def test_avoid_repeated_grids():

    lon_g0 = np.linspace(0, 1000, 11, dtype=np.float32)
    lat_g0 = np.linspace(0, 1000, 11, dtype=np.float32)
    time_g0 = np.linspace(0, 1000, 2, dtype=np.float64)
    grid_0 = RectilinearZGrid(lon_g0, lat_g0, time=time_g0)

    lon_g1 = np.linspace(0, 1000, 21, dtype=np.float32)
    lat_g1 = np.linspace(0, 1000, 21, dtype=np.float32)
    time_g1 = np.linspace(0, 1000, 2, dtype=np.float64)
    grid_1 = RectilinearZGrid(lon_g1, lat_g1, time=time_g1)

    u_data = np.zeros((lon_g0.size, lat_g0.size, time_g0.size), dtype=np.float32)
    u_field = Field('U', u_data, grid=grid_0, transpose=True)

    v_data = np.zeros((lon_g1.size, lat_g1.size, time_g1.size), dtype=np.float32)
    v_field = Field('V', v_data, grid=grid_1, transpose=True)

    temp0_field = Field('temp', u_data, lon=lon_g0, lat=lat_g0, time=time_g0, transpose=True)

    other_fields = {}
    other_fields['temp'] = temp0_field

    field_set = FieldSet(u_field, v_field, fields=other_fields)
    assert field_set.gridset.size == 2
    assert field_set.U.grid is field_set.temp.grid
    assert field_set.V.grid is not field_set.U.grid


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_multigrids_pointer(mode):
    lon_g0 = np.linspace(0, 1e4, 21, dtype=np.float32)
    lat_g0 = np.linspace(0, 1000, 2, dtype=np.float32)
    depth_g0 = np.zeros((5, lat_g0.size, lon_g0.size), dtype=np.float32)

    def bath_func(lon):
        return lon / 1000. + 10
    bath = bath_func(lon_g0)

    zdim = depth_g0.shape[0]
    for i in range(lon_g0.size):
        for k in range(zdim):
            depth_g0[k, :, i] = bath[i] * k / (zdim-1)

    grid_0 = RectilinearSGrid(lon_g0, lat_g0, depth=depth_g0)
    grid_1 = RectilinearSGrid(lon_g0, lat_g0, depth=depth_g0)

    u_data = np.zeros((zdim, lat_g0.size, lon_g0.size), dtype=np.float32)
    v_data = np.zeros((zdim, lat_g0.size, lon_g0.size), dtype=np.float32)
    w_data = np.zeros((zdim, lat_g0.size, lon_g0.size), dtype=np.float32)

    u_field = Field('U', u_data, grid=grid_0)
    v_field = Field('V', v_data, grid=grid_0)
    w_field = Field('W', w_data, grid=grid_1)

    field_set = FieldSet(u_field, v_field, fields={'W': w_field})
    assert(u_field.grid == v_field.grid)
    assert(u_field.grid == w_field.grid)  # w_field.grid is now supposed to be grid_1

    pset = ParticleSet.from_list(field_set, ptype[mode], lon=[0], lat=[0], depth=[1])

    for i in range(10):
        pset.execute(AdvectionRK4_3D, runtime=1000, dt=500)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('z4d', ['True', 'False'])
def test_rectilinear_s_grid_sampling(mode, z4d):
    lon_g0 = np.linspace(-3e4, 3e4, 61, dtype=np.float32)
    lat_g0 = np.linspace(0, 1000, 2, dtype=np.float32)
    time_g0 = np.linspace(0, 1000, 2, dtype=np.float64)
    if z4d:
        depth_g0 = np.zeros((time_g0.size, 5, lat_g0.size, lon_g0.size), dtype=np.float32)
    else:
        depth_g0 = np.zeros((5, lat_g0.size, lon_g0.size), dtype=np.float32)

    def bath_func(lon):
        bath = (lon <= -2e4) * 20.
        bath += (lon > -2e4) * (lon < 2e4) * (110. + 90 * np.sin(lon/2e4 * np.pi/2.))
        bath += (lon >= 2e4) * 200.
        return bath
    bath = bath_func(lon_g0)

    zdim = depth_g0.shape[-3]
    for i in range(depth_g0.shape[-1]):
        for k in range(zdim):
            if z4d:
                depth_g0[:, k, :, i] = bath[i] * k / (zdim-1)
            else:
                depth_g0[k, :, i] = bath[i] * k / (zdim-1)

    grid = RectilinearSGrid(lon_g0, lat_g0, depth=depth_g0, time=time_g0)

    u_data = np.zeros((grid.tdim, grid.zdim, grid.ydim, grid.xdim), dtype=np.float32)
    v_data = np.zeros((grid.tdim, grid.zdim, grid.ydim, grid.xdim), dtype=np.float32)
    temp_data = np.zeros((grid.tdim, grid.zdim, grid.ydim, grid.xdim), dtype=np.float32)
    for k in range(1, zdim):
        temp_data[:, k, :, :] = k / (zdim-1.)
    u_field = Field('U', u_data, grid=grid)
    v_field = Field('V', v_data, grid=grid)
    temp_field = Field('temp', temp_data, grid=grid)

    other_fields = {}
    other_fields['temp'] = temp_field
    field_set = FieldSet(u_field, v_field, fields=other_fields)

    def sampleTemp(particle, fieldset, time):
        particle.temp = fieldset.temp[time, particle.depth, particle.lat, particle.lon]

    class MyParticle(ptype[mode]):
        temp = Variable('temp', dtype=np.float32, initial=20.)

    lon = 400
    lat = 0
    ratio = .3
    pset = ParticleSet.from_list(field_set, MyParticle, lon=[lon], lat=[lat], depth=[bath_func(lon)*ratio])

    pset.execute(pset.Kernel(sampleTemp), runtime=0, dt=0)
    assert np.allclose(pset.temp[0], ratio, atol=1e-4)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_rectilinear_s_grids_advect1(mode):
    # Constant water transport towards the east. check that the particle stays at the same relative depth (z/bath)
    lon_g0 = np.linspace(0, 1e4, 21, dtype=np.float32)
    lat_g0 = np.linspace(0, 1000, 2, dtype=np.float32)
    depth_g0 = np.zeros((lon_g0.size, lat_g0.size, 5), dtype=np.float32)

    def bath_func(lon):
        return lon / 1000. + 10
    bath = bath_func(lon_g0)

    for i in range(depth_g0.shape[0]):
        for k in range(depth_g0.shape[2]):
            depth_g0[i, :, k] = bath[i] * k / (depth_g0.shape[2]-1)
    depth_g0 = depth_g0.transpose()  # we don't change it on purpose, to check if the transpose op if fixed in jit

    grid = RectilinearSGrid(lon_g0, lat_g0, depth=depth_g0)

    zdim = depth_g0.shape[0]
    u_data = np.zeros((zdim, lat_g0.size, lon_g0.size), dtype=np.float32)
    v_data = np.zeros((zdim, lat_g0.size, lon_g0.size), dtype=np.float32)
    w_data = np.zeros((zdim, lat_g0.size, lon_g0.size), dtype=np.float32)
    for i in range(lon_g0.size):
        u_data[:, :, i] = 1 * 10 / bath[i]
        for k in range(zdim):
            w_data[k, :, i] = u_data[k, :, i] * depth_g0[k, :, i] / bath[i] * 1e-3

    u_field = Field('U', u_data, grid=grid)
    v_field = Field('V', v_data, grid=grid)
    w_field = Field('W', w_data, grid=grid)

    field_set = FieldSet(u_field, v_field, fields={'W': w_field})

    lon = np.zeros((11))
    lat = np.zeros((11))
    ratio = [min(i/10., .99) for i in range(11)]
    depth = bath_func(lon)*ratio
    pset = ParticleSet.from_list(field_set, ptype[mode], lon=lon, lat=lat, depth=depth)

    pset.execute(AdvectionRK4_3D, runtime=10000, dt=500)
    assert np.allclose(pset.depth/bath_func(pset.lon), ratio)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_rectilinear_s_grids_advect2(mode):
    # Move particle towards the east, check relative depth evolution
    lon_g0 = np.linspace(0, 1e4, 21, dtype=np.float32)
    lat_g0 = np.linspace(0, 1000, 2, dtype=np.float32)
    depth_g0 = np.zeros((5, lat_g0.size, lon_g0.size), dtype=np.float32)

    def bath_func(lon):
        return lon / 1000. + 10
    bath = bath_func(lon_g0)

    zdim = depth_g0.shape[0]
    for i in range(lon_g0.size):
        for k in range(zdim):
            depth_g0[k, :, i] = bath[i] * k / (zdim-1)

    grid = RectilinearSGrid(lon_g0, lat_g0, depth=depth_g0)

    u_data = np.zeros((zdim, lat_g0.size, lon_g0.size), dtype=np.float32)
    v_data = np.zeros((zdim, lat_g0.size, lon_g0.size), dtype=np.float32)
    rel_depth_data = np.zeros((zdim, lat_g0.size, lon_g0.size), dtype=np.float32)
    for k in range(1, zdim):
        rel_depth_data[k, :, :] = k / (zdim-1.)

    u_field = Field('U', u_data, grid=grid)
    v_field = Field('V', v_data, grid=grid)
    rel_depth_field = Field('relDepth', rel_depth_data, grid=grid)
    field_set = FieldSet(u_field, v_field, fields={'relDepth': rel_depth_field})

    class MyParticle(ptype[mode]):
        relDepth = Variable('relDepth', dtype=np.float32, initial=20.)

    def moveEast(particle, fieldset, time):
        particle.lon += 5 * particle.dt
        particle.relDepth = fieldset.relDepth[time, particle.depth, particle.lat, particle.lon]

    depth = .9
    pset = ParticleSet.from_list(field_set, MyParticle, lon=[0], lat=[0], depth=[depth])

    kernel = pset.Kernel(moveEast)
    for _ in range(10):
        pset.execute(kernel, runtime=100, dt=50)
        assert np.allclose(pset.relDepth[0], depth/bath_func(pset.lon[0]))


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_curvilinear_grids(mode):

    x = np.linspace(0, 1e3, 7, dtype=np.float32)
    y = np.linspace(0, 1e3, 5, dtype=np.float32)
    (xx, yy) = np.meshgrid(x, y)

    r = np.sqrt(xx*xx+yy*yy)
    theta = np.arctan2(yy, xx)
    theta = theta + np.pi/6.

    lon = r * np.cos(theta)
    lat = r * np.sin(theta)
    time = np.array([0, 86400], dtype=np.float64)
    grid = CurvilinearZGrid(lon, lat, time=time)

    u_data = np.ones((2, y.size, x.size), dtype=np.float32)
    v_data = np.zeros((2, y.size, x.size), dtype=np.float32)
    u_data[0, :, :] = lon[:, :] + lat[:, :]
    u_field = Field('U', u_data, grid=grid, transpose=False)
    v_field = Field('V', v_data, grid=grid, transpose=False)
    field_set = FieldSet(u_field, v_field)

    def sampleSpeed(particle, fieldset, time):
        u = fieldset.U[time, particle.depth, particle.lat, particle.lon]
        v = fieldset.V[time, particle.depth, particle.lat, particle.lon]
        particle.speed = math.sqrt(u*u+v*v)

    class MyParticle(ptype[mode]):
        speed = Variable('speed', dtype=np.float32, initial=0.)

    pset = ParticleSet.from_list(field_set, MyParticle, lon=[400, -200], lat=[600, 600])
    pset.execute(pset.Kernel(sampleSpeed), runtime=0, dt=0)
    assert(np.allclose(pset.speed[0], 1000))


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_nemo_grid(mode):
    data_path = path.join(path.dirname(__file__), 'test_data/')

    filenames = {'U': {'lon': data_path + 'mask_nemo_cross_180lon.nc',
                       'lat': data_path + 'mask_nemo_cross_180lon.nc',
                       'data': data_path + 'Uu_eastward_nemo_cross_180lon.nc'},
                 'V': {'lon': data_path + 'mask_nemo_cross_180lon.nc',
                       'lat': data_path + 'mask_nemo_cross_180lon.nc',
                       'data': data_path + 'Vv_eastward_nemo_cross_180lon.nc'}}
    variables = {'U': 'U', 'V': 'V'}
    dimensions = {'lon': 'glamf', 'lat': 'gphif'}
    field_set = FieldSet.from_nemo(filenames, variables, dimensions)

    # test ParticleSet.from_field on curvilinear grids
    ParticleSet.from_field(field_set, ptype[mode], start_field=field_set.U, size=5)

    def sampleVel(particle, fieldset, time):
        (particle.zonal, particle.meridional) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]

    class MyParticle(ptype[mode]):
        zonal = Variable('zonal', dtype=np.float32, initial=0.)
        meridional = Variable('meridional', dtype=np.float32, initial=0.)

    lonp = 175.5
    latp = 81.5
    pset = ParticleSet.from_list(field_set, MyParticle, lon=[lonp], lat=[latp])
    pset.execute(pset.Kernel(sampleVel), runtime=0, dt=0)
    u = field_set.U.units.to_source(pset.zonal[0], lonp, latp, 0)
    v = field_set.V.units.to_source(pset.meridional[0], lonp, latp, 0)
    assert abs(u - 1) < 1e-4
    assert abs(v) < 1e-4


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_advect_nemo(mode):
    data_path = path.join(path.dirname(__file__), 'test_data/')

    filenames = {'U': {'lon': data_path + 'mask_nemo_cross_180lon.nc',
                       'lat': data_path + 'mask_nemo_cross_180lon.nc',
                       'data': data_path + 'Uu_eastward_nemo_cross_180lon.nc'},
                 'V': {'lon': data_path + 'mask_nemo_cross_180lon.nc',
                       'lat': data_path + 'mask_nemo_cross_180lon.nc',
                       'data': data_path + 'Vv_eastward_nemo_cross_180lon.nc'}}
    variables = {'U': 'U', 'V': 'V'}
    dimensions = {'lon': 'glamf', 'lat': 'gphif'}
    field_set = FieldSet.from_nemo(filenames, variables, dimensions)

    lonp = 175.5
    latp = 81.5
    pset = ParticleSet.from_list(field_set, ptype[mode], lon=[lonp], lat=[latp])
    pset.execute(AdvectionRK4, runtime=delta(days=2), dt=delta(hours=6))
    assert abs(pset.lat[0] - latp) < 1e-3


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('time', [True, False])
def test_cgrid_uniform_2dvel(mode, time):
    lon = np.array([[0, 2], [.4, 1.5]])
    lat = np.array([[0, -.5], [.8, .5]])
    U = np.array([[-99, -99], [4.4721359549995793e-01, 1.3416407864998738e+00]])
    V = np.array([[-99, 1.2126781251816650e+00], [-99, 1.2278812270298409e+00]])

    if time:
        U = np.stack((U, U))
        V = np.stack((V, V))
        dimensions = {'lat': lat, 'lon': lon, 'time': np.array([0, 10])}
    else:
        dimensions = {'lat': lat, 'lon': lon}
    data = {'U': np.array(U, dtype=np.float32), 'V': np.array(V, dtype=np.float32)}
    fieldset = FieldSet.from_data(data, dimensions, mesh='flat')
    fieldset.U.interp_method = 'cgrid_velocity'
    fieldset.V.interp_method = 'cgrid_velocity'

    def sampleVel(particle, fieldset, time):
        (particle.zonal, particle.meridional) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]

    class MyParticle(ptype[mode]):
        zonal = Variable('zonal', dtype=np.float32, initial=0.)
        meridional = Variable('meridional', dtype=np.float32, initial=0.)

    pset = ParticleSet.from_list(fieldset, MyParticle, lon=.7, lat=.3)
    pset.execute(pset.Kernel(sampleVel), runtime=0, dt=0)
    assert abs(pset.zonal[0] - 1) < 1e-6
    assert abs(pset.meridional[0] - 1) < 1e-6


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('vert_mode', ['zlev', 'slev1', 'slev2'])
@pytest.mark.parametrize('time', [True, False])
def test_cgrid_uniform_3dvel(mode, vert_mode, time):

    lon = np.array([[0, 2], [.4, 1.5]])
    lat = np.array([[0, -.5], [.8, .5]])

    u0 = 4.4721359549995793e-01
    u1 = 1.3416407864998738e+00
    v0 = 1.2126781251816650e+00
    v1 = 1.2278812270298409e+00
    w0 = 1
    w1 = 1

    if vert_mode == 'zlev':
        depth = np.array([0, 1])
    elif vert_mode == 'slev1':
        depth = np.array([[[0, 0], [0, 0]], [[1, 1], [1, 1]]])
    elif vert_mode == 'slev2':
        depth = np.array([[[-1, -.6], [-1.1257142857142859, -.9]],
                          [[1, 1.5], [0.50857142857142845, .8]]])
        w0 = 1.0483007922296661e+00
        w1 = 1.3098951476312375e+00

    U = np.array([[[-99, -99], [u0, u1]],
                  [[-99, -99], [-99, -99]]])
    V = np.array([[[-99, v0], [-99, v1]],
                  [[-99, -99], [-99, -99]]])
    W = np.array([[[-99, -99], [-99, w0]],
                  [[-99, -99], [-99, w1]]])

    if time:
        U = np.stack((U, U))
        V = np.stack((V, V))
        W = np.stack((W, W))
        dimensions = {'lat': lat, 'lon': lon, 'depth': depth, 'time': np.array([0, 10])}
    else:
        dimensions = {'lat': lat, 'lon': lon, 'depth': depth}
    data = {'U': np.array(U, dtype=np.float32),
            'V': np.array(V, dtype=np.float32),
            'W': np.array(W, dtype=np.float32)}
    fieldset = FieldSet.from_data(data, dimensions, mesh='flat')
    fieldset.U.interp_method = 'cgrid_velocity'
    fieldset.V.interp_method = 'cgrid_velocity'
    fieldset.W.interp_method = 'cgrid_velocity'

    def sampleVel(particle, fieldset, time):
        (particle.zonal, particle.meridional, particle.vertical) = fieldset.UVW[time, particle.depth, particle.lat, particle.lon]

    class MyParticle(ptype[mode]):
        zonal = Variable('zonal', dtype=np.float32, initial=0.)
        meridional = Variable('meridional', dtype=np.float32, initial=0.)
        vertical = Variable('vertical', dtype=np.float32, initial=0.)

    pset = ParticleSet.from_list(fieldset, MyParticle, lon=.7, lat=.3, depth=.2)
    pset.execute(pset.Kernel(sampleVel), runtime=0, dt=0)
    assert abs(pset.zonal[0] - 1) < 1e-6
    assert abs(pset.meridional[0] - 1) < 1e-6
    assert abs(pset.vertical[0] - 1) < 1e-6


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('vert_mode', ['zlev', 'slev1'])
@pytest.mark.parametrize('time', [True, False])
def test_cgrid_uniform_3dvel_spherical(mode, vert_mode, time):
    data_path = path.join(path.dirname(__file__), 'test_data/')
    dim_file = xr.open_dataset(data_path + 'mask_nemo_cross_180lon.nc')
    u_file = xr.open_dataset(data_path + 'Uu_eastward_nemo_cross_180lon.nc')
    v_file = xr.open_dataset(data_path + 'Vv_eastward_nemo_cross_180lon.nc')
    j = 4
    i = 11
    lon = np.array(dim_file.glamf[0, j:j+2, i:i+2])
    lat = np.array(dim_file.gphif[0, j:j+2, i:i+2])
    U = np.array(u_file.U[0, j:j+2, i:i+2])
    V = np.array(v_file.V[0, j:j+2, i:i+2])
    trash = np.zeros((2, 2))
    U = np.stack((U, trash))
    V = np.stack((V, trash))
    w0 = 1
    w1 = 1
    W = np.array([[[-99, -99], [-99, w0]],
                  [[-99, -99], [-99, w1]]])

    if vert_mode == 'zlev':
        depth = np.array([0, 1])
    elif vert_mode == 'slev1':
        depth = np.array([[[0, 0], [0, 0]], [[1, 1], [1, 1]]])

    if time:
        U = np.stack((U, U))
        V = np.stack((V, V))
        W = np.stack((W, W))
        dimensions = {'lat': lat, 'lon': lon, 'depth': depth, 'time': np.array([0, 10])}
    else:
        dimensions = {'lat': lat, 'lon': lon, 'depth': depth}
    data = {'U': np.array(U, dtype=np.float32),
            'V': np.array(V, dtype=np.float32),
            'W': np.array(W, dtype=np.float32)}
    fieldset = FieldSet.from_data(data, dimensions, mesh='spherical')
    fieldset.U.interp_method = 'cgrid_velocity'
    fieldset.V.interp_method = 'cgrid_velocity'
    fieldset.W.interp_method = 'cgrid_velocity'

    def sampleVel(particle, fieldset, time):
        (particle.zonal, particle.meridional, particle.vertical) = fieldset.UVW[time, particle.depth, particle.lat, particle.lon]

    class MyParticle(ptype[mode]):
        zonal = Variable('zonal', dtype=np.float32, initial=0.)
        meridional = Variable('meridional', dtype=np.float32, initial=0.)
        vertical = Variable('vertical', dtype=np.float32, initial=0.)

    lonp = 179.8
    latp = 81.35
    pset = ParticleSet.from_list(fieldset, MyParticle, lon=lonp, lat=latp, depth=.2)
    pset.execute(pset.Kernel(sampleVel), runtime=0, dt=0)
    pset.zonal[0] = fieldset.U.units.to_source(pset.zonal[0], lonp, latp, 0)
    pset.meridional[0] = fieldset.V.units.to_source(pset.meridional[0], lonp, latp, 0)
    assert abs(pset.zonal[0] - 1) < 1e-3
    assert abs(pset.meridional[0]) < 1e-3
    assert abs(pset.vertical[0] - 1) < 1e-3


@pytest.mark.parametrize('mode', ['scipy'])  # TODO also implement JIT
@pytest.mark.parametrize('interp_method', ['cgrid_velocity', 'cgrid_velocity_incompressible'])
def test_doublegyre_incompressible(mode, interp_method):
    """Implemented following Froyland and Padberg (2009), 10.1016/j.physd.2009.03.002"""
    A = 0.25
    delta = 0.25
    omega = 2 * np.pi

    a, b = 2, 1  # domain size
    lon = np.linspace(0, a, 51, dtype=np.float32)
    lat = np.linspace(0, b, 51, dtype=np.float32)
    times = np.arange(0, 3.1, 0.1)
    dx, dy = lon[2]-lon[1], lat[2]-lat[1]

    U = np.zeros((times.size, lat.size, lon.size), dtype=np.float32)
    V = np.zeros((times.size, lat.size, lon.size), dtype=np.float32)

    for i in range(lon.size):
        for j in range(lat.size):
            x1 = lon[i]-dx/2
            x2 = lat[j]-dy/2
            for t in range(len(times)):
                time = times[t]
                f = delta * np.sin(omega * time) * x1**2 + (1-2 * delta * np.sin(omega * time)) * x1
                U[t, j, i] = -np.pi * A * np.sin(np.pi * f) * np.cos(np.pi * x2)
                V[t, j, i] = np.pi * A * np.cos(np.pi * f) * np.sin(np.pi * x2) * (2 * delta * np.sin(omega * time) * x1 + 1 - 2 * delta * np.sin(omega * time))

    data = {'U': U, 'V': V}
    dimensions = {'lon': lon, 'lat': lat, 'time': times}
    allow_time_extrapolation = True if len(times) == 1 else False
    fieldset = FieldSet.from_data(data, dimensions, mesh='flat', allow_time_extrapolation=allow_time_extrapolation)
    fieldset.U.interp_method = interp_method
    fieldset.V.interp_method = interp_method

    # X, Y = np.meshgrid(np.arange(0.15, 1.85, 0.1), np.arange(0.15, 0.85, 0.1))
    X, Y = 0.15, 0.4
    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=X, lat=Y)
    pset.execute(AdvectionRK4, dt=0.01, runtime=3)
    print(pset)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('vert_discretisation', ['zlevel', 'slevel', 'slevel2'])
@pytest.mark.parametrize('deferred_load', [True, False])
def test_popgrid(mode, vert_discretisation, deferred_load):
    mesh = path.join(path.join(path.dirname(__file__), 'test_data'), 'POPtestdata_time.nc')
    if vert_discretisation == 'zlevel':
        w_dep = 'w_dep'
    elif vert_discretisation == 'slevel':
        w_dep = 'w_deps'   # same as zlevel, but defined as slevel
    elif vert_discretisation == 'slevel2':
        w_dep = 'w_deps2'  # contains shaved cells

    filenames = mesh
    variables = {'U': 'U',
                 'V': 'V',
                 'W': 'W',
                 'T': 'T'}
    dimensions = {'lon': 'lon', 'lat': 'lat', 'depth': w_dep, 'time': 'time'}

    field_set = FieldSet.from_pop(filenames, variables, dimensions, mesh='flat', deferred_load=deferred_load)

    def sampleVel(particle, fieldset, time):
        (particle.zonal, particle.meridional, particle.vert) = fieldset.UVW[time, particle.depth, particle.lat, particle.lon]
        particle.tracer = fieldset.T[time, particle.depth, particle.lat, particle.lon]

    def OutBoundsError(particle, fieldset, time):
        particle.out_of_bounds = 1
        particle.depth -= 3

    class MyParticle(ptype[mode]):
        zonal = Variable('zonal', dtype=np.float32, initial=0.)
        meridional = Variable('meridional', dtype=np.float32, initial=0.)
        vert = Variable('vert', dtype=np.float32, initial=0.)
        tracer = Variable('tracer', dtype=np.float32, initial=0.)
        out_of_bounds = Variable('out_of_bounds', dtype=np.float32, initial=0.)

    pset = ParticleSet.from_list(field_set, MyParticle, lon=[3, 5, 1], lat=[3, 5, 1], depth=[3, 7, 11])
    pset.execute(pset.Kernel(sampleVel), runtime=1, dt=1,
                 recovery={ErrorCode.ErrorOutOfBounds: OutBoundsError})
    if vert_discretisation == 'slevel2':
        assert np.isclose(pset.vert[0], 0.)
        assert np.isclose(pset.zonal[0], 0.)
        assert np.isclose(pset.tracer[0], 99.)
        assert np.isclose(pset.vert[1], -0.0066666666)
        assert np.isclose(pset.zonal[1], .015)
        assert np.isclose(pset.tracer[1], 1.)
        assert pset.out_of_bounds[0] == 0
        assert pset.out_of_bounds[1] == 0
        assert pset.out_of_bounds[2] == 1
    else:
        assert np.allclose(pset.zonal, 0.015)
        assert np.allclose(pset.meridional, 0.01)
        assert np.allclose(pset.vert, -0.01)
        assert np.allclose(pset.tracer, 1)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('gridindexingtype', ['mitgcm', 'nemo'])
def test_cgrid_indexing(mode, gridindexingtype):
    xdim, ydim = 151, 201
    a = b = 20000  # domain size
    lon = np.linspace(-a / 2, a / 2, xdim, dtype=np.float32)
    lat = np.linspace(-b / 2, b / 2, ydim, dtype=np.float32)
    dx, dy = lon[2] - lon[1], lat[2] - lat[1]
    omega = 2 * np.pi / delta(days=1).total_seconds()

    index_signs = {'nemo': -1, 'mitgcm': 1}
    isign = index_signs[gridindexingtype]

    def calc_r_phi(ln, lt):
        return np.sqrt(ln ** 2 + lt ** 2), np.arctan2(ln, lt)

    def calculate_UVR(lat, lon, dx, dy, omega):
        U = np.zeros((lat.size, lon.size), dtype=np.float32)
        V = np.zeros((lat.size, lon.size), dtype=np.float32)
        R = np.zeros((lat.size, lon.size), dtype=np.float32)
        for i in range(lon.size):
            for j in range(lat.size):
                r, phi = calc_r_phi(lon[i], lat[j])
                R[j, i] = r
                r, phi = calc_r_phi(lon[i] + isign * dx / 2, lat[j])
                V[j, i] = -omega * r * np.sin(phi)
                r, phi = calc_r_phi(lon[i], lat[j] + isign * dy / 2)
                U[j, i] = omega * r * np.cos(phi)
        return U, V, R

    U, V, R = calculate_UVR(lat, lon, dx, dy, omega)
    data = {'U': U, 'V': V, 'R': R}
    dimensions = {'lon': lon, 'lat': lat}
    fieldset = FieldSet.from_data(data, dimensions, mesh='flat', gridindexingtype=gridindexingtype)
    fieldset.U.interp_method = 'cgrid_velocity'
    fieldset.V.interp_method = 'cgrid_velocity'

    def UpdateR(particle, fieldset, time):
        particle.radius = fieldset.R[time, particle.depth, particle.lat, particle.lon]

    class MyParticle(ptype[mode]):
        radius = Variable('radius', dtype=np.float32, initial=0.)
        radius_start = Variable('radius_start', dtype=np.float32, initial=fieldset.R)

    pset = ParticleSet(fieldset, pclass=MyParticle, lon=0, lat=4e3, time=0)

    pset.execute(pset.Kernel(UpdateR) + AdvectionRK4,
                 runtime=delta(hours=14), dt=delta(minutes=5))
    assert np.allclose(pset.radius, pset.radius_start, atol=10)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('gridindexingtype', ['mitgcm', 'nemo'])
@pytest.mark.parametrize('withtime', [False, True])
def test_cgrid_indexing_3D(mode, gridindexingtype, withtime):
    xdim = zdim = 201
    ydim = 2
    a = c = 20000  # domain size
    b = 2
    lon = np.linspace(-a / 2, a / 2, xdim, dtype=np.float32)
    lat = np.linspace(-b / 2, b / 2, ydim, dtype=np.float32)
    depth = np.linspace(-c / 2, c / 2, zdim, dtype=np.float32)
    dx, dz = lon[1] - lon[0], depth[1] - depth[0]
    omega = 2 * np.pi / delta(days=1).total_seconds()
    if withtime:
        time = np.linspace(0, 24*60*60, 10)
        dimensions = {"lon": lon, "lat": lat, "depth": depth, "time": time}
        dsize = (time.size, depth.size, lat.size, lon.size)
    else:
        dimensions = {"lon": lon, "lat": lat, "depth": depth}
        dsize = (depth.size, lat.size, lon.size)

    hindex_signs = {'nemo': -1, 'mitgcm': 1}
    hsign = hindex_signs[gridindexingtype]

    def calc_r_phi(ln, dp):
        # r = np.sqrt(ln ** 2 + dp ** 2)
        # phi = np.arcsin(dp/r) if r > 0 else 0
        return np.sqrt(ln ** 2 + dp ** 2), np.arctan2(ln, dp)

    def calculate_UVWR(lat, lon, depth, dx, dz, omega):
        U = np.zeros(dsize, dtype=np.float32)
        V = np.zeros(dsize, dtype=np.float32)
        W = np.zeros(dsize, dtype=np.float32)
        R = np.zeros(dsize, dtype=np.float32)

        for i in range(lon.size):
            for j in range(lat.size):
                for k in range(depth.size):
                    r, phi = calc_r_phi(lon[i], depth[k])
                    if withtime:
                        R[:, k, j, i] = r
                    else:
                        R[k, j, i] = r
                    r, phi = calc_r_phi(lon[i] + hsign * dx / 2, depth[k])
                    if withtime:
                        W[:, k, j, i] = -omega * r * np.sin(phi)
                    else:
                        W[k, j, i] = -omega * r * np.sin(phi)
                    r, phi = calc_r_phi(lon[i], depth[k] + dz / 2)
                    if withtime:
                        U[:, k, j, i] = omega * r * np.cos(phi)
                    else:
                        U[k, j, i] = omega * r * np.cos(phi)
        return U, V, W, R

    U, V, W, R = calculate_UVWR(lat, lon, depth, dx, dz, omega)
    data = {"U": U, "V": V, "W": W, "R": R}
    fieldset = FieldSet.from_data(data, dimensions, mesh="flat", gridindexingtype=gridindexingtype)
    fieldset.U.interp_method = "cgrid_velocity"
    fieldset.V.interp_method = "cgrid_velocity"
    fieldset.W.interp_method = "cgrid_velocity"

    def UpdateR(particle, fieldset, time):
        particle.radius = fieldset.R[time, particle.depth, particle.lat, particle.lon]

    class MyParticle(ptype[mode]):
        radius = Variable('radius', dtype=np.float32, initial=0.)
        radius_start = Variable('radius_start', dtype=np.float32, initial=fieldset.R)

    pset = ParticleSet(fieldset, pclass=MyParticle, depth=4e3, lon=0, lat=0, time=0)

    pset.execute(pset.Kernel(UpdateR) + AdvectionRK4_3D,
                 runtime=delta(hours=14), dt=delta(minutes=5))
    assert np.allclose(pset.radius, pset.radius_start, atol=10)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('gridindexingtype', ['pop', 'mom5'])
@pytest.mark.parametrize('withtime', [False, True])
def test_bgrid_indexing_3D(mode, gridindexingtype, withtime):
    xdim = zdim = 201
    ydim = 2
    a = c = 20000  # domain size
    b = 2
    lon = np.linspace(-a / 2, a / 2, xdim, dtype=np.float32)
    lat = np.linspace(-b / 2, b / 2, ydim, dtype=np.float32)
    depth = np.linspace(-c / 2, c / 2, zdim, dtype=np.float32)
    dx, dz = lon[1] - lon[0], depth[1] - depth[0]
    omega = 2 * np.pi / delta(days=1).total_seconds()
    if withtime:
        time = np.linspace(0, 24*60*60, 10)
        dimensions = {"lon": lon, "lat": lat, "depth": depth, "time": time}
        dsize = (time.size, depth.size, lat.size, lon.size)
    else:
        dimensions = {"lon": lon, "lat": lat, "depth": depth}
        dsize = (depth.size, lat.size, lon.size)

    vindex_signs = {'pop': 1, 'mom5': -1}
    vsign = vindex_signs[gridindexingtype]

    def calc_r_phi(ln, dp):
        return np.sqrt(ln ** 2 + dp ** 2), np.arctan2(ln, dp)

    def calculate_UVWR(lat, lon, depth, dx, dz, omega):
        U = np.zeros(dsize, dtype=np.float32)
        V = np.zeros(dsize, dtype=np.float32)
        W = np.zeros(dsize, dtype=np.float32)
        R = np.zeros(dsize, dtype=np.float32)

        for i in range(lon.size):
            for j in range(lat.size):
                for k in range(depth.size):
                    r, phi = calc_r_phi(lon[i], depth[k])
                    if withtime:
                        R[:, k, j, i] = r
                    else:
                        R[k, j, i] = r
                    r, phi = calc_r_phi(lon[i] - dx / 2, depth[k])
                    if withtime:
                        W[:, k, j, i] = -omega * r * np.sin(phi)
                    else:
                        W[k, j, i] = -omega * r * np.sin(phi)
                    # Since Parcels loads as dimensions only the depth of W-points
                    # and lon/lat of UV-points, W-points are similarly interpolated
                    # in MOM5 and POP. Indexing is shifted for UV-points.
                    r, phi = calc_r_phi(lon[i], depth[k] + vsign * dz / 2)
                    if withtime:
                        U[:, k, j, i] = omega * r * np.cos(phi)
                    else:
                        U[k, j, i] = omega * r * np.cos(phi)
        return U, V, W, R

    U, V, W, R = calculate_UVWR(lat, lon, depth, dx, dz, omega)
    data = {"U": U, "V": V, "W": W, "R": R}
    fieldset = FieldSet.from_data(data, dimensions, mesh="flat", gridindexingtype=gridindexingtype)
    fieldset.U.interp_method = "bgrid_velocity"
    fieldset.V.interp_method = "bgrid_velocity"
    fieldset.W.interp_method = "bgrid_w_velocity"

    def UpdateR(particle, fieldset, time):
        particle.radius = fieldset.R[time, particle.depth, particle.lat, particle.lon]

    class MyParticle(ptype[mode]):
        radius = Variable('radius', dtype=np.float32, initial=0.)
        radius_start = Variable('radius_start', dtype=np.float32, initial=fieldset.R)

    pset = ParticleSet(fieldset, pclass=MyParticle, depth=-9.995e3, lon=0, lat=0, time=0)

    pset.execute(pset.Kernel(UpdateR) + AdvectionRK4_3D,
                 runtime=delta(hours=14), dt=delta(minutes=5))
    assert np.allclose(pset.radius, pset.radius_start, atol=10)


@pytest.mark.parametrize('gridindexingtype', ['pop', 'mom5'])
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('extrapolation', [True, False])
def test_bgrid_interpolation(gridindexingtype, mode, extrapolation):
    xi, yi = 3, 2
    if extrapolation:
        zi = 0 if gridindexingtype == 'mom5' else -1
    else:
        zi = 2
    if gridindexingtype == 'mom5':
        ufile = path.join(path.join(path.dirname(__file__), 'test_data'), 'access-om2-01_u.nc')
        vfile = path.join(path.join(path.dirname(__file__), 'test_data'), 'access-om2-01_v.nc')
        wfile = path.join(path.join(path.dirname(__file__), 'test_data'), 'access-om2-01_wt.nc')

        filenames = {"U": {"lon": ufile, "lat": ufile, "depth": wfile, "data": ufile},
                     "V": {"lon": ufile, "lat": ufile, "depth": wfile, "data": vfile},
                     "W": {"lon": ufile, "lat": ufile, "depth": wfile, "data": wfile}}

        variables = {"U": "u", "V": "v", "W": "wt"}

        dimensions = {"U": {"lon": "xu_ocean", "lat": "yu_ocean", "depth": "sw_ocean", "time": "time"},
                      "V": {"lon": "xu_ocean", "lat": "yu_ocean", "depth": "sw_ocean", "time": "time"},
                      "W": {"lon": "xu_ocean", "lat": "yu_ocean", "depth": "sw_ocean", "time": "time"}}

        fieldset = FieldSet.from_mom5(filenames, variables, dimensions)
        ds_u = xr.open_dataset(ufile)
        ds_v = xr.open_dataset(vfile)
        ds_w = xr.open_dataset(wfile)
        u = ds_u.u.isel(time=0, st_ocean=zi, yu_ocean=yi, xu_ocean=xi)
        v = ds_v.v.isel(time=0, st_ocean=zi, yu_ocean=yi, xu_ocean=xi)
        w = ds_w.wt.isel(time=0, sw_ocean=zi, yt_ocean=yi, xt_ocean=xi)

    elif gridindexingtype == 'pop':
        datafname = path.join(path.join(path.dirname(__file__), 'test_data'), 'popdata.nc')
        coordfname = path.join(path.join(path.dirname(__file__), 'test_data'), 'popcoordinates.nc')
        filenames = {"U": {"lon": coordfname, "lat": coordfname, "depth": coordfname, "data": datafname},
                     "V": {"lon": coordfname, "lat": coordfname, "depth": coordfname, "data": datafname},
                     "W": {"lon": coordfname, "lat": coordfname, "depth": coordfname, "data": datafname}}

        variables = {'U': 'UVEL', 'V': 'VVEL', 'W': 'WVEL'}
        dimensions = {'lon': 'U_LON_2D', 'lat': 'U_LAT_2D', 'depth': 'w_dep'}

        fieldset = FieldSet.from_pop(filenames, variables, dimensions)
        dsc = xr.open_dataset(coordfname)
        dsd = xr.open_dataset(datafname)
        u = dsd.UVEL.isel(k=zi, j=yi, i=xi)
        v = dsd.VVEL.isel(k=zi, j=yi, i=xi)
        w = dsd.WVEL.isel(k=zi, j=yi, i=xi)

    fieldset.U.units = UnitConverter()
    fieldset.V.units = UnitConverter()

    def VelocityInterpolator(particle, fieldset, time):
        particle.Uvel = fieldset.U[time, particle.depth, particle.lat, particle.lon]
        particle.Vvel = fieldset.V[time, particle.depth, particle.lat, particle.lon]
        particle.Wvel = fieldset.W[time, particle.depth, particle.lat, particle.lon]

    class myParticle(ptype[mode]):
        Uvel = Variable("Uvel", dtype=np.float32, initial=0.0)
        Vvel = Variable("Vvel", dtype=np.float32, initial=0.0)
        Wvel = Variable("Wvel", dtype=np.float32, initial=0.0)

    for pointtype in ["U", "V", "W"]:
        if gridindexingtype == "pop":
            if pointtype in ["U", "V"]:
                lons = dsc.U_LON_2D[yi, xi].values
                lats = dsc.U_LAT_2D[yi, xi].values
                deps = dsc.depth_t[zi]
            elif pointtype == "W":
                lons = dsc.T_LON_2D[yi, xi].values
                lats = dsc.T_LAT_2D[yi, xi].values
                deps = dsc.w_dep[zi]
            if extrapolation:
                deps = 5499.
        elif gridindexingtype == "mom5":
            if pointtype in ["U", "V"]:
                lons = u.xu_ocean.data.reshape(1)
                lats = u.yu_ocean.data.reshape(1)
                deps = u.st_ocean.data.reshape(1)
            elif pointtype == "W":
                lons = w.xt_ocean.data.reshape(1)
                lats = w.yt_ocean.data.reshape(1)
                deps = w.sw_ocean.data.reshape(1)
            if extrapolation:
                deps = 0

        pset = ParticleSet.from_list(fieldset=fieldset, pclass=myParticle, lon=lons, lat=lats, depth=deps)
        pset.execute(VelocityInterpolator, dt=0)

        convfactor = 0.01 if gridindexingtype == "pop" else 1.
        if pointtype in ["U", "V"]:
            assert np.allclose(pset.Uvel[0], u*convfactor)
            assert np.allclose(pset.Vvel[0], v*convfactor)
        elif pointtype == "W":
            if extrapolation:
                assert np.allclose(pset.Wvel[0], 0, atol=1e-9)
            else:
                assert np.allclose(pset.Wvel[0], -w*convfactor)
