import os

import cftime
import numpy as np
import pytest
import xarray as xr
from zarr.storage import MemoryStore

from parcels import (
    AdvectionRK4,
    Field,
    FieldSet,
    JITParticle,
    ParticleSet,
    ScipyParticle,
    Variable,
)
from parcels.particlefile import _set_calendar
from parcels.tools.converters import _get_cftime_calendars, _get_cftime_datetimes

ptype = {'scipy': ScipyParticle, 'jit': JITParticle}


def fieldset(xdim=40, ydim=100):
    U = np.zeros((ydim, xdim), dtype=np.float32)
    V = np.zeros((ydim, xdim), dtype=np.float32)
    lon = np.linspace(0, 1, xdim, dtype=np.float32)
    lat = np.linspace(-60, 60, ydim, dtype=np.float32)
    depth = np.zeros(1, dtype=np.float32)
    data = {'U': np.array(U, dtype=np.float32), 'V': np.array(V, dtype=np.float32)}
    dimensions = {'lat': lat, 'lon': lon, 'depth': depth}
    return FieldSet.from_data(data, dimensions)


@pytest.fixture(name="fieldset")
def fieldset_ficture(xdim=40, ydim=100):
    return fieldset(xdim=xdim, ydim=ydim)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_metadata(fieldset, mode, tmpdir):
    filepath = tmpdir.join("pfile_metadata.zarr")
    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=0, lat=0)

    def DoNothing(particle, fieldset, time):
        pass

    pset.execute(DoNothing, runtime=1, output_file=pset.ParticleFile(filepath))

    ds = xr.open_zarr(filepath)
    assert ds.attrs['parcels_kernels'].lower() == f'{mode}ParticleDoNothing'.lower()


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pfile_array_write_zarr_memorystore(fieldset, mode, npart=10):
    """Check that writing to a Zarr MemoryStore works."""
    zarr_store = MemoryStore()
    pset = ParticleSet(fieldset, pclass=ptype[mode],
                       lon=np.linspace(0, 1, npart),
                       lat=0.5*np.ones(npart), time=0)
    pfile = pset.ParticleFile(zarr_store)
    pfile.write(pset, 0)

    ds = xr.open_zarr(zarr_store)
    assert ds.sizes["trajectory"] == npart
    ds.close()


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pfile_array_remove_particles(fieldset, mode, tmpdir, npart=10):
    filepath = tmpdir.join("pfile_array_remove_particles.zarr")
    pset = ParticleSet(fieldset, pclass=ptype[mode],
                       lon=np.linspace(0, 1, npart),
                       lat=0.5*np.ones(npart), time=0)
    pfile = pset.ParticleFile(filepath)
    pfile.write(pset, 0)
    pset.remove_indices(3)
    for p in pset:
        p.time = 1
    pfile.write(pset, 1)

    ds = xr.open_zarr(filepath)
    timearr = ds['time'][:]
    assert (np.isnat(timearr[3, 1])) and (np.isfinite(timearr[3, 0]))
    ds.close()


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pfile_set_towrite_False(fieldset, mode, tmpdir, npart=10):
    filepath = tmpdir.join("pfile_set_towrite_False.zarr")
    pset = ParticleSet(fieldset, pclass=ptype[mode],
                       lon=np.linspace(0, 1, npart),
                       lat=0.5*np.ones(npart))
    pset.set_variable_write_status('depth', False)
    pset.set_variable_write_status('lat', False)
    pfile = pset.ParticleFile(filepath, outputdt=1)

    def Update_lon(particle, fieldset, time):
        particle_dlon += 0.1  # noqa

    pset.execute(Update_lon, runtime=10, output_file=pfile)

    ds = xr.open_zarr(filepath)
    assert 'time' in ds
    assert 'z' not in ds
    assert 'lat' not in ds
    ds.close()

    # For pytest purposes, we need to reset to original status
    pset.set_variable_write_status('depth', True)
    pset.set_variable_write_status('lat', True)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('chunks_obs', [1, None])
def test_pfile_array_remove_all_particles(fieldset, mode, chunks_obs, tmpdir, npart=10):

    filepath = tmpdir.join("pfile_array_remove_particles.zarr")
    pset = ParticleSet(fieldset, pclass=ptype[mode],
                       lon=np.linspace(0, 1, npart),
                       lat=0.5*np.ones(npart), time=0)
    chunks = (npart, chunks_obs) if chunks_obs else None
    pfile = pset.ParticleFile(filepath, chunks=chunks)
    pfile.write(pset, 0)
    for _ in range(npart):
        pset.remove_indices(-1)
    pfile.write(pset, 1)
    pfile.write(pset, 2)

    ds = xr.open_zarr(filepath)
    assert np.allclose(ds['time'][:, 0], np.timedelta64(0, 's'), atol=np.timedelta64(1, 'ms'))
    if chunks_obs is not None:
        assert ds['time'][:].shape == chunks
    else:
        assert ds['time'][:].shape[0] == npart
        assert np.all(np.isnan(ds['time'][:, 1:]))
    ds.close()


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_variable_write_double(fieldset, mode, tmpdir):
    filepath = tmpdir.join("pfile_variable_write_double.zarr")

    def Update_lon(particle, fieldset, time):
        particle_dlon += 0.1  # noqa

    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=[0], lat=[0], lonlatdepth_dtype=np.float64)
    ofile = pset.ParticleFile(name=filepath, outputdt=0.00001)
    pset.execute(pset.Kernel(Update_lon), endtime=0.001, dt=0.00001, output_file=ofile)

    ds = xr.open_zarr(filepath)
    lons = ds['lon'][:]
    assert (isinstance(lons.values[0, 0], np.float64))
    ds.close()


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_write_dtypes_pfile(fieldset, mode, tmpdir):
    filepath = tmpdir.join("pfile_dtypes.zarr")

    dtypes = [np.float32, np.float64, np.int32, np.uint32, np.int64, np.uint64]
    if mode == 'scipy':
        dtypes.extend([np.bool_, np.int8, np.uint8, np.int16, np.uint16])

    extra_vars = [Variable(f'v_{d.__name__}', dtype=d, initial=0.) for d in dtypes]
    MyParticle = ptype[mode].add_variables(extra_vars)

    pset = ParticleSet(fieldset, pclass=MyParticle, lon=0, lat=0, time=0)
    pfile = pset.ParticleFile(name=filepath, outputdt=1)
    pfile.write(pset, 0)

    ds = xr.open_zarr(filepath, mask_and_scale=False)  # Note masking issue at https://stackoverflow.com/questions/68460507/xarray-loading-int-data-as-float
    for d in dtypes:
        assert ds[f'v_{d.__name__}'].dtype == d


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('npart', [1, 2, 5])
def test_variable_written_once(fieldset, mode, tmpdir, npart):
    filepath = tmpdir.join("pfile_once_written_variables.zarr")

    def Update_v(particle, fieldset, time):
        particle.v_once += 1.
        particle.age += particle.dt

    MyParticle = ptype[mode].add_variables([
        Variable('v_once', dtype=np.float64, initial=0., to_write='once'),
        Variable('age', dtype=np.float32, initial=0.)])
    lon = np.linspace(0, 1, npart)
    lat = np.linspace(1, 0, npart)
    time = np.arange(0, npart/10., 0.1, dtype=np.float64)
    pset = ParticleSet(fieldset, pclass=MyParticle, lon=lon, lat=lat, time=time, v_once=time)
    ofile = pset.ParticleFile(name=filepath, outputdt=0.1)
    pset.execute(pset.Kernel(Update_v), endtime=1, dt=0.1, output_file=ofile)

    assert np.allclose(pset.v_once - time - pset.age*10, 1, atol=1e-5)
    ds = xr.open_zarr(filepath)
    vfile = np.ma.filled(ds['v_once'][:], np.nan)
    assert (vfile.shape == (npart, ))
    ds.close()


@pytest.mark.parametrize('type', ['repeatdt', 'timearr'])
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('repeatdt', range(1, 3))
@pytest.mark.parametrize('dt', [-1, 1])
@pytest.mark.parametrize('maxvar', [2, 4, 10])
def test_pset_repeated_release_delayed_adding_deleting(type, fieldset, mode, repeatdt, tmpdir, dt, maxvar, runtime=10):
    fieldset.maxvar = maxvar
    pset = None

    MyParticle = ptype[mode].add_variables([
        Variable('sample_var', initial=0.),
        Variable('v_once', dtype=np.float64, initial=0., to_write='once')])

    if type == 'repeatdt':
        pset = ParticleSet(fieldset, lon=[0], lat=[0], pclass=MyParticle, repeatdt=repeatdt)
    elif type == 'timearr':
        pset = ParticleSet(fieldset, lon=np.zeros(runtime), lat=np.zeros(runtime), pclass=MyParticle, time=list(range(runtime)))
    outfilepath = tmpdir.join("pfile_repeated_release.zarr")
    pfile = pset.ParticleFile(outfilepath, outputdt=abs(dt), chunks=(1, 1))

    def IncrLon(particle, fieldset, time):
        particle.sample_var += 1.
        if particle.sample_var > fieldset.maxvar:
            particle.delete()
    for _ in range(runtime):
        pset.execute(IncrLon, dt=dt, runtime=1., output_file=pfile)

    ds = xr.open_zarr(outfilepath)
    samplevar = ds['sample_var'][:]
    if type == 'repeatdt':
        assert samplevar.shape == (runtime // repeatdt, min(maxvar+1, runtime))
        assert np.allclose(pset.sample_var, np.arange(maxvar, -1, -repeatdt))
    elif type == 'timearr':
        assert samplevar.shape == (runtime, min(maxvar + 1, runtime))
    # test whether samplevar[:, k] = k
    for k in range(samplevar.shape[1]):
        assert np.allclose([p for p in samplevar[:, k] if np.isfinite(p)], k+1)
    filesize = os.path.getsize(str(outfilepath))
    assert filesize < 1024 * 65  # test that chunking leads to filesize less than 65KB
    ds.close()


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('repeatdt', [1, 2])
@pytest.mark.parametrize('nump', [1, 10])
def test_pfile_chunks_repeatedrelease(fieldset, mode, repeatdt, nump, tmpdir):
    runtime = 8
    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=np.zeros((nump, 1)),
                       lat=np.zeros((nump, 1)), repeatdt=repeatdt)
    outfilepath = tmpdir.join("pfile_chunks_repeatedrelease.zarr")
    chunks = (20, 10)
    pfile = pset.ParticleFile(outfilepath, outputdt=1, chunks=chunks)

    def DoNothing(particle, fieldset, time):
        pass

    pset.execute(DoNothing, dt=1, runtime=runtime, output_file=pfile)
    ds = xr.open_zarr(outfilepath)
    assert ds['time'].shape == (int(nump*runtime/repeatdt), chunks[1])


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_write_timebackward(fieldset, mode, tmpdir):
    outfilepath = tmpdir.join("pfile_write_timebackward.zarr")

    def Update_lon(particle, fieldset, time):
        particle_dlon -= 0.1 * particle.dt  # noqa

    pset = ParticleSet(fieldset, pclass=ptype[mode],
                       lat=np.linspace(0, 1, 3), lon=[0, 0, 0], time=[1, 2, 3])
    pfile = pset.ParticleFile(name=outfilepath, outputdt=1.)
    pset.execute(pset.Kernel(Update_lon), runtime=4, dt=-1.,
                 output_file=pfile)
    ds = xr.open_zarr(outfilepath)
    trajs = ds['trajectory'][:]
    assert trajs.values.dtype == 'int64'
    assert np.all(np.diff(trajs.values) < 0)  # all particles written in order of release
    ds.close()


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_write_xiyi(fieldset, mode, tmpdir):
    outfilepath = tmpdir.join("pfile_xiyi.zarr")
    fieldset.U.data[:] = 1  # set a non-zero zonal velocity
    fieldset.add_field(Field(name='P', data=np.zeros((3, 20)), lon=np.linspace(0, 1, 20), lat=[-2, 0, 2]))
    dt = 3600

    XiYiParticle = ptype[mode].add_variables([
        Variable('pxi0', dtype=np.int32, initial=0.),
        Variable('pxi1', dtype=np.int32, initial=0.),
        Variable('pyi', dtype=np.int32, initial=0.)])

    def Get_XiYi(particle, fieldset, time):
        """Kernel to sample the grid indices of the particle.
        Note that this sampling should be done _before_ the advection kernel
        and that the first outputted value is zero.
        Be careful when using multiple grids, as the index may be different for the grids.
        """
        particle.pxi0 = particle.xi[0]
        particle.pxi1 = particle.xi[1]
        particle.pyi = particle.yi[0]

    def SampleP(particle, fieldset, time):
        if time > 5*3600:
            _ = fieldset.P[particle]  # To trigger sampling of the P field

    pset = ParticleSet(fieldset, pclass=XiYiParticle, lon=[0, 0.2], lat=[0.2, 1], lonlatdepth_dtype=np.float64)
    pfile = pset.ParticleFile(name=outfilepath, outputdt=dt)
    pset.execute([SampleP, Get_XiYi, AdvectionRK4], endtime=10*dt, dt=dt, output_file=pfile)

    ds = xr.open_zarr(outfilepath)
    pxi0 = ds['pxi0'][:].values.astype(np.int32)
    pxi1 = ds['pxi1'][:].values.astype(np.int32)
    lons = ds['lon'][:].values
    pyi = ds['pyi'][:].values.astype(np.int32)
    lats = ds['lat'][:].values

    for p in range(pyi.shape[0]):
        assert (pxi0[p, 0] == 0) and (pxi0[p, -1] == pset[p].pxi0)  # check that particle has moved
        assert np.all(pxi1[p, :6] == 0)  # check that particle has not been sampled on grid 1 until time 6
        assert np.all(pxi1[p, 6:] > 0)  # check that particle has not been sampled on grid 1 after time 6
        for xi, lon in zip(pxi0[p, 1:], lons[p, 1:]):
            assert fieldset.U.grid.lon[xi] <= lon < fieldset.U.grid.lon[xi+1]
        for xi, lon in zip(pxi1[p, 6:], lons[p, 6:]):
            assert fieldset.P.grid.lon[xi] <= lon < fieldset.P.grid.lon[xi+1]
        for yi, lat in zip(pyi[p, 1:], lats[p, 1:]):
            assert fieldset.U.grid.lat[yi] <= lat < fieldset.U.grid.lat[yi+1]
    ds.close()


def test_set_calendar():
    for _calendar_name, cf_datetime in zip(_get_cftime_calendars(), _get_cftime_datetimes()):
        date = getattr(cftime, cf_datetime)(1990, 1, 1)
        assert _set_calendar(date.calendar) == date.calendar
    assert _set_calendar('np_datetime64') == 'standard'


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_reset_dt(fieldset, mode, tmpdir):
    # Assert that p.dt gets reset when a write_time is not a multiple of dt
    # for p.dt=0.02 to reach outputdt=0.05 and endtime=0.1, the steps should be [0.2, 0.2, 0.1, 0.2, 0.2, 0.1], resulting in 6 kernel executions
    filepath = tmpdir.join("pfile_reset_dt.zarr")

    def Update_lon(particle, fieldset, time):
        particle_dlon += 0.1  # noqa

    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=[0], lat=[0], lonlatdepth_dtype=np.float64)
    ofile = pset.ParticleFile(name=filepath, outputdt=0.05)
    pset.execute(pset.Kernel(Update_lon), endtime=0.12, dt=0.02, output_file=ofile)

    assert np.allclose(pset.lon, .6)
