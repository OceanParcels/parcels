from parcels import (FieldSet, Field, ScipyParticle, JITParticle,
                     Variable, ErrorCode, CurvilinearZGrid, AdvectionRK4)
from parcels.particleset_node import ParticleSet
from parcels.particlefile_node import ParticleFile
from parcels.kernel_node import Kernel
from parcels.nodes.Node import Node, NodeJIT
from parcels.tools import idgen
from parcels.tools import get_cache_dir
from os import path
# from parcels.tools import logger
import numpy as np
import pytest


try:
    from mpi4py import MPI
except:
    MPI = None

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
def fieldset_fixture(xdim=40, ydim=100):
    return fieldset(xdim=xdim, ydim=ydim)


#@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('mode', ['jit'])
def test_pset_repeated_release(fieldset, mode, npart=10):
    time = np.arange(0, npart, 1)  # release 1 particle every second
    pset = ParticleSet(fieldset, lon=np.zeros(npart), lat=np.zeros(npart),
                       pclass=ptype[mode], time=time)
    assert np.allclose([n.data.time for n in pset.data], time)

    def IncrLon(particle, fieldset, time):
        particle.lon += 1.
    pset.execute(IncrLon, dt=1., runtime=npart)
    assert np.allclose([n.data.lon for n in pset.data], np.arange(npart, 0, -1))


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pset_dt0(fieldset, mode, npart=10):
    pset = ParticleSet(fieldset, lon=np.zeros(npart), lat=np.zeros(npart),
                       pclass=ptype[mode])

    def IncrLon(particle, fieldset, time):
        particle.lon += 1
    pset.execute(IncrLon, dt=0., runtime=npart)
    assert np.allclose([n.data.lon for n in pset.data], 1.)
    assert np.allclose([n.data.time for n in pset.data], 0.)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pset_custom_ptype(fieldset, mode, npart=100):
    class TestParticle(ptype[mode]):
        user_vars = {'p': np.float32, 'n': np.int32}

        def __init__(self, *args, **kwargs):
            super(TestParticle, self).__init__(*args, **kwargs)
            self.p = 0.33
            self.n = 2

    pset = ParticleSet(fieldset, pclass=TestParticle,
                       lon=np.linspace(0, 1, npart, dtype=np.float32),
                       lat=np.linspace(1, 0, npart, dtype=np.float32))
    assert(pset.size == 100)
    assert np.allclose([n.data.p - 0.33 for n in pset.data], np.zeros(npart), rtol=1e-12)
    assert np.allclose([n.data.n - 2 for n in pset.data], np.zeros(npart), rtol=1e-12)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pset_add_explicit(fieldset, mode, npart=100):
    nclass = Node
    if mode == 'jit':
        nclass = NodeJIT
    lon = np.linspace(0, 1, npart, dtype=np.float64)
    lat = np.linspace(1, 0, npart, dtype=np.float64)
    pset = ParticleSet(fieldset=fieldset, pclass=ptype[mode], lon=[], lat=[], lonlatdepth_dtype=np.float64)
    index_mapping = {}
    for i in range(0, npart):
        index = idgen.total_length
        id = idgen.getID(lon[i], lat[i], 0., 0.)
        index_mapping[i] = id
        pdata = ptype[mode](lon[i], lat[i], pid=id, fieldset=fieldset, index=index)
        ndata = nclass(id=id, data=pdata)
        pset.add(ndata)
    assert(pset.size == 100)
    # ==== of course this is not working as the order in pset.data and lon is not the same ==== #
    # assert np.allclose([n.data.lon for n in pset.data], lon, rtol=1e-12)
    assert np.allclose([pset.get_by_id(index_mapping[i]).data.lon for i in index_mapping.keys()], lon, rtol=1e-12)
    assert np.allclose([pset.get_by_id(index_mapping[i]).data.lat for i in index_mapping.keys()], lat, rtol=1e-12)
    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_comm.Barrier()


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pset_node_execute(fieldset, mode, tmpdir, npart=100):
    nclass = Node
    if mode == 'jit':
        nclass = NodeJIT
    lon = np.linspace(0, 1, npart, dtype=np.float64)
    lat = np.linspace(1, 0, npart, dtype=np.float64)
    outfilepath = tmpdir.join("pfile_node_execute_test.nc")
    pset = ParticleSet(fieldset=fieldset, lon=[], lat=[], pclass=ptype[mode], lonlatdepth_dtype=np.float64)
    pfile = pset.ParticleFile(name=outfilepath, outputdt=1.)
    index_mapping = {}
    for i in range(npart):
        index = idgen.getID(lon[i], lat[i], 0., 0.)
        index_mapping[i] = index
        pdata = ptype[mode](lon[i], lat[i], pid=index, fieldset=fieldset)
        ndata = nclass(id=index, data=pdata)
        pset.add(ndata)
    pset.execute(AdvectionRK4, runtime=0., dt=10., output_file=pfile)
    pfile.close()
    assert(pset.size == 100)
    # ==== of course this is not working as the order in pset.data and lon is not the same ==== #
    # assert np.allclose([n.data.lon for n in pset.data], lon, rtol=1e-12)
    assert np.allclose([pset.get_by_id(index_mapping[i]).data.lon for i in index_mapping.keys()], lon, rtol=1e-12)
    assert np.allclose([pset.get_by_id(index_mapping[i]).data.lat for i in index_mapping.keys()], lat, rtol=1e-12)


def run_test_pset_add_explicit(fset, mode, npart=100):
    nclass = Node
    if mode == 'jit':
        nclass = NodeJIT

    lon = np.linspace(0, 1, npart, dtype=np.float64)
    lat = np.linspace(1, 0, npart, dtype=np.float64)
    outfilepath = path.join(get_cache_dir(), "add_explicit", "pfile_node_add_explicit.nc")
    pset = ParticleSet(fieldset=fset, pclass=ptype[mode], lon=lon, lat=lat, lonlatdepth_dtype=np.float64)
    pfile = pset.ParticleFile(name=outfilepath, outputdt=1.)

    index_mapping = {}
    for i in range(0, npart):
        index = idgen.total_length
        id = idgen.getID(lon[i], lat[i], None, None)
        index_mapping[i] = id
        pdata = ptype[mode](lon[i], lat[i], pid=id, fieldset=fset, index=index)
        ndata = nclass(id=id, data=pdata)
        pset.add(ndata)
    # assert(pset.size >= npart)
    # print("# particles: {}".format(pset.size))
    # logger.info("# particles: {}".format(pset.size))
    for tstep in range(3):
        pfile.write(pset, tstep)
    pfile.close()
    # logger.info("# particles: {}".format(pset.size))
    assert (pset.size > 0)
    assert (pset.size <= 2* npart)
    # ==== of course this is not working as the order in pset.data and lon is not the same ==== #
    # assert np.allclose([n.data.lon for n in pset.data], lon, rtol=1e-12)
    # ==== makes no sence in MPI ==== #
    if MPI is None:
        assert np.allclose([pset.get_by_id(index_mapping[i]).data.lon for i in index_mapping.keys()], lon, rtol=1e-12)
        assert np.allclose([pset.get_by_id(index_mapping[i]).data.lat for i in index_mapping.keys()], lat, rtol=1e-12)


def run_test_pset_node_execute(fset, mode, npart=10000):
    nclass = Node
    if mode == 'jit':
        nclass = NodeJIT
    lon = np.linspace(0, 1, npart, dtype=np.float64)
    lat = np.linspace(1, 0, npart, dtype=np.float64)
    outfilepath = path.join(get_cache_dir(), "pfile_node_execute_test.nc")
    pset = ParticleSet(fieldset=fset, lon=lon, lat=lat, pclass=ptype[mode], lonlatdepth_dtype=np.float64)
    pfile = pset.ParticleFile(name=outfilepath, outputdt=360.0)
    for i in range(npart):
        index = idgen.getID(lon[i], lat[i], 0., 0.)
        pdata = ptype[mode](lon[i], lat[i], pid=index, fieldset=fset)
        ndata = nclass(id=index, data=pdata)
        pset.add(ndata)
    pset.execute(AdvectionRK4, runtime=0., dt=(360.0 * 24.0 * 10.0), output_file=pfile)
    pfile.close()
    # logger.info("# particles: {}".format(pset.size))
    assert (pset.size > 0)
    assert (pset.size <= 2* npart)


if __name__ == '__main__':
    fset = fieldset()
    run_test_pset_add_explicit(fset, 'jit')
    run_test_pset_add_explicit(fset, 'scipy')
    run_test_pset_node_execute(fset, 'jit')
    run_test_pset_node_execute(fset, 'scipy')
    # idgen.close()
    # del idgen
