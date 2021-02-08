from os import path
from parcels import (FieldSet, ScipyParticle, JITParticle, ErrorCode, KernelError,OutOfBoundsError, AdvectionRK4)
from parcels.particleset_vectorized import ParticleSet
import numpy as np
import pytest
from parcels.tools import idgen
from parcels.tools import logger
from os import getpid

try:
    from mpi4py import MPI
except:
    MPI = None


ptype = {'scipy': ScipyParticle, 'jit': JITParticle}


def DoNothing(particle, fieldset, time):
    return ErrorCode.Success


def fieldset(xdim=20, ydim=20):
    """ Standard unit mesh fieldset """
    lon = np.linspace(0., 1., xdim, dtype=np.float32)
    lat = np.linspace(0., 1., ydim, dtype=np.float32)
    U, V = np.meshgrid(lat, lon)
    data = {'U': np.array(U, dtype=np.float32), 'V': np.array(V, dtype=np.float32)}
    dimensions = {'lat': lat, 'lon': lon}
    return FieldSet.from_data(data, dimensions, mesh='flat')


@pytest.fixture(name="fieldset")
def fieldset_fixture(xdim=20, ydim=20):
    return fieldset(xdim=xdim, ydim=ydim)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('start, end, substeps, dt', [
    (0., 10., 1, 1.),
    (0., 10., 4, 1.),
    (0., 10., 1, 3.),
    (2., 16., 5, 3.),
    (20., 10., 4, -1.),
    (20., -10., 7, -2.),
])
def test_execution_endtime(fieldset, mode, start, end, substeps, dt, npart=10):
    pset = ParticleSet(fieldset, pclass=ptype[mode], time=start,
                       lon=np.linspace(0, 1, npart),
                       lat=np.linspace(1, 0, npart))
    pset.execute(DoNothing, endtime=end, dt=dt)
    assert np.allclose(np.array([p.time for p in pset]), end)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('start, end, substeps, dt', [
    (0., 10., 1, 1.),
    (0., 10., 4, 1.),
    (0., 10., 1, 3.),
    (2., 16., 5, 3.),
    (20., 10., 4, -1.),
    (20., -10., 7, -2.),
])
def test_execution_runtime(fieldset, mode, start, end, substeps, dt, npart=10):
    pset = ParticleSet(fieldset, pclass=ptype[mode], time=start,
                       lon=np.linspace(0, 1, npart),
                       lat=np.linspace(1, 0, npart))
    t_step = abs(end - start) / substeps
    for _ in range(substeps):
        pset.execute(DoNothing, runtime=t_step, dt=dt)
    assert np.allclose(np.array([p.time for p in pset]), end)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('time', [0., 1])
@pytest.mark.parametrize('dt', [0., 1])
def test_pset_execute_dt_0(fieldset, mode, time, dt, npart=2):
    def SetLat(particle, fieldset, time):
        particle.lat = .6
    lon = np.linspace(0, 1, npart)
    lat = np.linspace(1, 0, npart)

    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=lon, lat=lat)
    pset.execute(SetLat, endtime=time, dt=dt)
    assert np.allclose([p.lon for p in pset], lon)
    assert np.allclose([p.lat for p in pset], [.6])
    assert np.allclose([p.time for p in pset], min([time, dt]))

    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=lon, lat=lat)
    pset.execute(SetLat, runtime=time, dt=dt)
    assert np.allclose([p.lon for p in pset], lon)
    assert np.allclose([p.lat for p in pset], [.6])
    assert np.allclose([p.time for p in pset], min([time, dt]))


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_execution_fail_timed(fieldset, mode, npart=10):
    def TimedFail(particle, fieldset, time):
        if particle.time >= 10.:
            return ErrorCode.Error
        else:
            return ErrorCode.Success

    pset = ParticleSet(fieldset, pclass=ptype[mode],
                       lon=np.linspace(0, 1, npart),
                       lat=np.linspace(1, 0, npart))
    error_thrown = False
    try:
        pset.execute(TimedFail, endtime=20., dt=2.)
    except KernelError:
        error_thrown = True
    assert error_thrown
    assert len(pset) == npart
    assert np.allclose(np.array([p.time for p in pset]), 10.)


@pytest.mark.parametrize('mode', ['scipy'])
def test_execution_fail_python_exception(fieldset, mode, npart=10):
    def PythonFail(particle, fieldset, time):
        if particle.time >= 10.:
            raise RuntimeError("Enough is enough!")
        else:
            return ErrorCode.Success

    pset = ParticleSet(fieldset, pclass=ptype[mode],
                       lon=np.linspace(0, 1, npart),
                       lat=np.linspace(1, 0, npart))
    error_thrown = False
    try:
        pset.execute(PythonFail, endtime=20., dt=2.)
    except KernelError:
        error_thrown = True
    assert error_thrown
    assert len(pset) == npart
    assert np.allclose(np.array([p.time for p in pset]), 10.)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_execution_fail_out_of_bounds(fieldset, mode, npart=10):
    def MoveRight(particle, fieldset, time):
        fieldset.U[time, particle.depth, particle.lat, particle.lon + 0.1]
        particle.lon += 0.1

    pset = ParticleSet(fieldset, pclass=ptype[mode],
                       lon=np.linspace(0, 1, npart),
                       lat=np.linspace(1, 0, npart))
    error_thrown = False
    try:
        pset.execute(MoveRight, endtime=10., dt=1.)
    except OutOfBoundsError:
        error_thrown = True
    assert error_thrown
    assert len(pset) == npart
    assert (np.array([p.lon - 1. for p in pset]) > -1.e12).all()


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_execution_recover_out_of_bounds(fieldset, mode, npart=2):
    def MoveRight(particle, fieldset, time):
        fieldset.U[time, particle.depth, particle.lat, particle.lon + 0.1]
        particle.lon += 0.1

    def MoveLeft(particle, fieldset, time):
        particle.lon -= 1.

    lon = np.linspace(0.05, 0.95, npart)
    lat = np.linspace(1, 0, npart)
    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=lon, lat=lat)
    pset.execute(MoveRight, endtime=10., dt=1.,
                 recovery={ErrorCode.ErrorOutOfBounds: MoveLeft})
    assert len(pset) == npart
    assert np.allclose([p.lon for p in pset], lon, rtol=1e-5)
    assert np.allclose([p.lat for p in pset], lat, rtol=1e-5)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_execution_delete_out_of_bounds(fieldset, mode, npart=10):
    def MoveRight(particle, fieldset, time):
        fieldset.U[time, particle.depth, particle.lat, particle.lon + 0.1]
        particle.lon += 0.1

    def DeleteMe(particle, fieldset, time):
        particle.delete()

    lon = np.linspace(0.05, 0.95, npart)
    lat = np.linspace(1, 0, npart)
    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=lon, lat=lat)
    pset.execute(MoveRight, endtime=10., dt=1.,
                 recovery={ErrorCode.ErrorOutOfBounds: DeleteMe})
    assert len(pset) == 0


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_kernel_add_no_new_variables(fieldset, mode):
    def MoveEast(particle, fieldset, time):
        particle.lon += 0.1

    def MoveNorth(particle, fieldset, time):
        particle.lat += 0.1

    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=[0.5], lat=[0.5])
    pset.execute(pset.Kernel(MoveEast) + pset.Kernel(MoveNorth),
                 endtime=1., dt=1.)
    assert np.allclose([p.lon for p in pset], 0.6, rtol=1e-5)
    assert np.allclose([p.lat for p in pset], 0.6, rtol=1e-5)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_multi_kernel_duplicate_varnames(fieldset, mode):
    # Testing for merging of two Kernels with the same variable declared
    # Should throw a warning, but go ahead regardless
    def MoveEast(particle, fieldset, time):
        add_lon = 0.1
        particle.lon += add_lon

    def MoveWest(particle, fieldset, time):
        add_lon = -0.3
        particle.lon += add_lon

    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=[0.5], lat=[0.5])
    pset.execute(pset.Kernel(MoveEast) + pset.Kernel(MoveWest),
                 endtime=1., dt=1.)
    assert np.allclose([p.lon for p in pset], 0.3, rtol=1e-5)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_multi_kernel_reuse_varnames(fieldset, mode):
    # Testing for merging of two Kernels with the same variable declared
    # Should throw a warning, but go ahead regardless
    def MoveEast1(particle, fieldset, time):
        add_lon = 0.2
        particle.lon += add_lon

    def MoveEast2(particle, fieldset, time):
        particle.lon += add_lon  # noga - no flake8 testing of this line

    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=[0.5], lat=[0.5])
    pset.execute(pset.Kernel(MoveEast1) + pset.Kernel(MoveEast2),
                 endtime=1., dt=1.)
    assert np.allclose([p.lon for p in pset], [0.9], rtol=1e-5)  # should be 0.5 + 0.2 + 0.2 = 0.9


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_update_kernel_in_script(fieldset, mode):
    # Testing what happens when kernels are updated during runtime of a script
    # Should throw a warning, but go ahead regardless
    def MoveEast(particle, fieldset, time):
        add_lon = 0.1
        particle.lon += add_lon

    def MoveWest(particle, fieldset, time):
        add_lon = -0.3
        particle.lon += add_lon

    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=[0.5], lat=[0.5])
    pset.execute(pset.Kernel(MoveEast), endtime=1., dt=1.)
    pset.execute(pset.Kernel(MoveWest), endtime=2., dt=1.)
    assert np.allclose([p.lon for p in pset], 0.3, rtol=1e-5)  # should be 0.5 + 0.1 - 0.3 = 0.3


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_errorcode_repeat(fieldset, mode):
    def simpleKernel(particle, fieldset, time):
        if particle.lon > .1 and time < 1.:
            # if particle.lon is not re-setted before kernel repetition, it will break here
            return ErrorCode.Error
        particle.lon += 0.1
        if particle.dt > 1.49:
            # dt is used to leave the repetition loop (dt is the only variable not re-setted)
            return ErrorCode.Success
        particle.dt += .1
        return ErrorCode.Repeat

    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=[0.], lat=[0.])
    pset.execute(pset.Kernel(simpleKernel), endtime=3., dt=1.)


@pytest.mark.parametrize('delete_cfiles', [True, False])
def test_execution_keep_cfiles_and_nocompilation_warnings(fieldset, delete_cfiles):
    pset = ParticleSet(fieldset, pclass=JITParticle, lon=[0.], lat=[0.])
    pset.execute(pset.Kernel(AdvectionRK4, delete_cfiles=delete_cfiles), endtime=1., dt=1.)
    cfile = pset.kernel.src_file
    logfile = pset.kernel.log_file
    del pset.kernel
    if delete_cfiles:
        assert not path.exists(cfile)
    else:
        assert path.exists(cfile)
        with open(logfile) as f:
            assert 'warning' not in f.read(), 'Compilation WARNING in log file'




def run_test_execution_runtime(fset, mode, start, end, substeps, dt, npart=10):
    pset = ParticleSet(fset, pclass=ptype[mode], time=start,
                       lon=np.linspace(0, 1, npart),
                       lat=np.linspace(1, 0, npart))
    t_step = abs(end - start) / substeps
    for _ in range(substeps):
        pset.execute(DoNothing, runtime=t_step, dt=dt)
    if MPI is None:
        assert np.allclose(np.array([p.time for p in pset]), end)
        logger.info(
            "Completed run (mode={}; start={}; end={}; substep={}; dt={}) on process (pid={})".format(mode, start, end, substeps, dt, getpid()))
    else:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        logger.info("Completed run (mode={}; start={}; end={}; substep={}; dt={}) on processor {} (pid={})".format(mode, start, end, substeps, dt, mpi_rank, getpid()))


if __name__ == '__main__':
    fset = fieldset()
    run_test_execution_runtime(fset, 'jit', 0., 10., 1, 1.)
    run_test_execution_runtime(fset, 'jit', 0., 10., 4, 1.)
    run_test_execution_runtime(fset, 'jit', 0., 10., 1, 3.)
    run_test_execution_runtime(fset, 'jit', 2., 16., 5, 3.)
    run_test_execution_runtime(fset, 'jit', 20., 10., 4, -1.)
    run_test_execution_runtime(fset, 'jit', 20., -10., 7, -2.)
    run_test_execution_runtime(fset, 'scipy', 0., 10., 1, 1.)
    run_test_execution_runtime(fset, 'scipy', 0., 10., 4, 1.)
    run_test_execution_runtime(fset, 'scipy', 0., 10., 1, 3.)
    run_test_execution_runtime(fset, 'scipy', 2., 16., 5, 3.)
    run_test_execution_runtime(fset, 'scipy', 20., 10., 4, -1.)
    run_test_execution_runtime(fset, 'scipy', 20., -10., 7, -2.)
    # idgen.close()
