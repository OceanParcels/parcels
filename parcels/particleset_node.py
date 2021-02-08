import time as time_module
from datetime import date
from datetime import datetime as dtime
from datetime import timedelta as delta

import os
import numpy as np
from scipy.spatial import distance
import itertools
import xarray as xr
import progressbar
# import math  # noga
# import random  # noga

from parcels.nodes.LinkedList import *
from parcels.nodes.Node import Node, NodeJIT
from parcels.tools import idgen

from parcels.tools import get_cache_dir, get_package_dir
# from parcels.tools import cleanup_remove_files, cleanup_unload_lib, get_cache_dir, get_package_dir
# from parcels.wrapping.code_compiler import GNUCompiler
# from parcels.wrapping.code_compiler import GNUCompiler_MS
from parcels import ScipyParticle, JITParticle
from parcels.particlefile_node import ParticleFile
# from parcels import Grid, Field, GridSet, FieldSet
from parcels.grid import GridCode
from parcels.field import NestedField
from parcels.field import SummedField
from parcels.kernelbase import BaseKernel
from parcels.kernel_node import Kernel
from parcels import ErrorCode
from parcels.kernels.advection import AdvectionRK4
from parcels.tools.loggers import logger


try:
    from mpi4py import MPI
except:
    MPI = None
if MPI:
    try:
        from sklearn.cluster import KMeans
    except:
        raise EnvironmentError('sklearn needs to be available if MPI is installed. '
                               'See http://oceanparcels.org/#parallel_install for more information')

__all__ = ['ParticleSet', 'RepeatParameters']


class RepeatParameters(object):
    _n_pts = 0
    _lon = []
    _lat = []
    _depth = []
    _maxID = None
    _pclass = ScipyParticle
    _partitions = None
    kwargs = None

    def __init__(self, pclass=JITParticle, lon=None, lat=None, depth=None, partitions=None, pid_orig=None, **kwargs):
        if lon is None:
            lon = []
        self._lon = lon
        if lat is None:
            lat = []
        self._lat = lat
        if depth is None:
            depth = []
        self._depth = depth
        self._maxID = pid_orig # pid - pclass.lastID
        # assert type(self._lon)==type(self._lat)==type(self._depth)
        if isinstance(self._lon, list):
            self._n_pts = len(self._lon)
        elif isinstance(self._lon, np.ndarray):
            self._n_pts = self._lon.shape[0]
        self._pclass = pclass
        self._partitions = partitions
        self.kwargs = kwargs

    @property
    def num_pts(self):
        return self._n_pts

    @property
    def lon(self):
        return self._lon

    def get_longitude(self, index):
        return self._lon[index]

    @property
    def lat(self):
        return self._lat

    def get_latitude(self, index):
        return self._lat[index]

    @property
    def depth(self):
        return self._depth

    def get_depth_value(self, index):
        if len(self._depth) == 0:
            return 0
        return self._depth[index]

    @property
    def maxID(self):
        return self._maxID

    def get_particle_id(self, index):
        if self._maxID is None:
            return None
        return self._maxID+index

    @property
    def pclass(self):
        return self._pclass

    @property
    def partitions(self):
        return self._partitions


class ParticleSet(object):
    _nodes = None
    _pclass = ScipyParticle
    _nclass = Node
    _kclass = BaseKernel
    _ptype = None
    _fieldset = None
    _kernel = None
    _pu_centers = None
    _lonlatdepth_dtype = None

    @staticmethod
    def _convert_to_array_(var):
        # Convert lists and single integers/floats to one-dimensional numpy arrays
        if isinstance(var, np.ndarray):
            return var.flatten()
        elif isinstance(var, (int, float, np.float32, np.int32)):
            return np.array([var])
        else:
            return np.array(var)

    @staticmethod
    def lonlatdepth_dtype_from_field_interp_method(field):
        if type(field) in [SummedField, NestedField]:
            for f in field:
                if f.interp_method == 'cgrid_velocity':
                    return np.float64
        else:
            if field.interp_method == 'cgrid_velocity':
                return np.float64
        return np.float32

    def __init__(self, fieldset=None, pclass=JITParticle, lon=None, lat=None, depth=None, time=None,
                 repeatdt=None, lonlatdepth_dtype=None, pid_orig=None, **kwargs):

        self._fieldset = fieldset
        if self._fieldset is not None:
            self._fieldset.check_complete()

        if lonlatdepth_dtype is not None:
            self._lonlatdepth_dtype = lonlatdepth_dtype
        else:
            self._lonlatdepth_dtype = self.lonlatdepth_dtype_from_field_interp_method(self._fieldset.U)
        assert self._lonlatdepth_dtype in [np.float32, np.float64], \
            'lon lat depth precision should be set to either np.float32 or np.float64'
        JITParticle.set_lonlatdepth_dtype(self._lonlatdepth_dtype)
        # pid = None if pid_orig is None else pid_orig if isinstance(pid_orig, list) or isinstance(pid_orig, np.ndarray) else pid_orig + pclass.lastID
        pid = None if pid_orig is None else pid_orig if isinstance(pid_orig, list) or isinstance(pid_orig, np.ndarray) else pid_orig + idgen.total_length

        self._pclass = pclass
        self._kclass = Kernel
        self._kernel = None
        self._ptype = self._pclass.getPType()
        self._pu_centers = None  # can be given by parameter
        if self._ptype.uses_jit:
            self._nclass = NodeJIT
        else:
            self._nclass = Node
        self._nodes = RealList(dtype=self._nclass)

        # ---- init common parameters to ParticleSets ---- #
        lon = np.empty(shape=0) if lon is None else self._convert_to_array_(lon)
        lat = np.empty(shape=0) if lat is None else self._convert_to_array_(lat)
        # ==== pid is determined from the ID generator itself, not user-generated data, so to guarantee ID uniqueness ==== #
        # if pid_orig is None:
        #     pid_orig = np.arange(lon.size)
        # pid = pid_orig + pclass.lastID

        if depth is None:
            mindepth, _ = self.fieldset.gridset.dimrange('depth')
            depth = np.ones(lon.size, dtype=self._lonlatdepth_dtype) * mindepth
        else:
            depth = self._convert_to_array_(depth)
        assert lon.size == lat.size and lon.size == depth.size, (
            'lon, lat, depth don''t all have the same lenghts')

        time = self._convert_to_array_(time)
        time = np.repeat(time, lon.size) if time.size == 1 else time
        if time.size > 0 and type(time[0]) in [dtime, date]:
            time = np.array([np.datetime64(t) for t in time])
        self.time_origin = fieldset.time_origin
        if time.size > 0 and isinstance(time[0], np.timedelta64) and not self.time_origin:
            raise NotImplementedError('If fieldset.time_origin is not a date, time of a particle must be a double')
        time = np.array([self.time_origin.reltime(t) if isinstance(t, np.datetime64) else t for t in time])
        assert lon.size == time.size, (
            'time and positions (lon, lat, depth) don''t have the same lengths.')

        # ---- init MPI functionality          ---- #
        _partitions = kwargs.pop('partitions', None)
        if _partitions is not None and _partitions is not False:
            _partitions = self._convert_to_array_(_partitions)

        offset = np.max(pid) if (pid is not None) and (len(pid) > 0) else -1
        if MPI:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()
            mpi_size = mpi_comm.Get_size()

            if lon.size < mpi_size and mpi_size > 1:
                raise RuntimeError('Cannot initialise with fewer particles than MPI processors')

            if mpi_size > 1:
                if _partitions is not False:
                    if _partitions is None or self._pu_centers is None:
                        _partitions = None
                        _pu_centers = None
                        if mpi_rank == 0:
                            coords = np.vstack((lon, lat)).transpose()
                            kmeans = KMeans(n_clusters=mpi_size, random_state=0).fit(coords)
                            _partitions = kmeans.labels_
                            _pu_centers = kmeans.cluster_centers_
                        #mpi_comm.Barrier()
                        _partitions = mpi_comm.bcast(_partitions, root=0)
                        _pu_centers = mpi_comm.bcast(_pu_centers, root=0)
                        self._pu_centers = _pu_centers
                    # elif np.max(_partitions) >= mpi_size or self._pu_centers.shape[0] >= mpi_size:
                    elif np.max(_partitions >= mpi_rank) or self._pu_centers.shape[0] >= mpi_size:
                        raise RuntimeError('Particle partitions must vary between 0 and the number of mpi procs')
                    lon = lon[_partitions == mpi_rank]
                    lat = lat[_partitions == mpi_rank]
                    time = time[_partitions == mpi_rank]
                    depth = depth[_partitions == mpi_rank]
                    if pid is not None and (isinstance(pid, list) or isinstance(pid, np.ndarray)):
                        pid = pid[_partitions == mpi_rank]
                    for kwvar in kwargs:
                        kwargs[kwvar] = kwargs[kwvar][_partitions == mpi_rank]
                offset = mpi_comm.allreduce(offset, op=MPI.MAX)
        pclass.setLastID(offset+1)

        # ---- particle data parameter length assertions ---- #
        for kwvar in kwargs:
            kwargs[kwvar] = self._convert_to_array_(kwargs[kwvar])
            assert lon.size == kwargs[kwvar].size, (
                '%s and positions (lon, lat, depth) don''t have the same lengths.' % kwargs[kwvar])

        self.repeatdt = repeatdt.total_seconds() if isinstance(repeatdt, delta) else repeatdt
        rdata_available = True
        rdata_available &= (lon is not None) and (isinstance(lon, list) or isinstance(lon, np.ndarray))
        rdata_available &= (lat is not None) and (isinstance(lat, list) or isinstance(lat, np.ndarray))
        rdata_available &= (depth is not None) and (isinstance(depth, list) or isinstance(depth, np.ndarray))
        rdata_available &= (time is not None) and (isinstance(time, list) or isinstance(time, np.ndarray))
        if self.repeatdt and rdata_available:
            self.repeat_starttime = self._fieldset.gridset.dimrange('full_time')[0] if time is None else time[0]
            self.rparam = RepeatParameters(self._pclass, lon, lat, depth, None,
                                           None if pid is None else (pid - pclass.lastID), **kwargs)

        # fill / initialize / populate the list
        if lon is not None and lat is not None:
            for i in range(lon.size):
                pdata_id = None
                index = -1
                if pid is not None and (isinstance(pid, list) or isinstance(pid, np.ndarray)):
                    index = pid[i]
                    pdata_id = pid[i]
                else:
                    index = idgen.total_length
                    pdata_id = idgen.nextID(lon[i], lat[i], depth[i], time[i])
                pdata = self._pclass(lon[i], lat[i], pid=pdata_id, fieldset=self._fieldset, depth=depth[i], time=time[i], index=index)
                # Set other Variables if provided
                for kwvar in kwargs:
                    if not hasattr(pdata, kwvar):
                        raise RuntimeError('Particle class does not have Variable %s' % kwvar)
                    setattr(pdata, kwvar, kwargs[kwvar][i])
                ndata = self._nclass(id=pdata_id, data=pdata)
                self._nodes.add(ndata)

    @classmethod
    def from_list(cls, fieldset, pclass, lon, lat, depth=None, time=None, repeatdt=None, lonlatdepth_dtype=None, **kwargs):
        """Initialise the ParticleSet from lists of lon and lat

        :param fieldset: :mod:`parcels.fieldset.FieldSet` object from which to sample velocity
        :param pclass: mod:`parcels.particle.JITParticle` or :mod:`parcels.particle.ScipyParticle`
                 object that defines custom particle
        :param lon: List of initial longitude values for particles
        :param lat: List of initial latitude values for particles
        :param depth: Optional list of initial depth values for particles. Default is 0m
        :param time: Optional list of start time values for particles. Default is fieldset.U.time[0]
        :param repeatdt: Optional interval (in seconds) on which to repeat the release of the ParticleSet
        :param lonlatdepth_dtype: Floating precision for lon, lat, depth particle coordinates.
               It is either np.float32 or np.float64. Default is np.float32 if fieldset.U.interp_method is 'linear'
               and np.float64 if the interpolation method is 'cgrid_velocity'
        Other Variables can be initialised using further arguments (e.g. v=... for a Variable named 'v')
       """
        return cls(fieldset=fieldset, pclass=pclass, lon=lon, lat=lat, depth=depth, time=time, repeatdt=repeatdt, lonlatdepth_dtype=lonlatdepth_dtype, **kwargs)

    @classmethod
    def from_line(cls, fieldset, pclass, start, finish, size, depth=None, time=None, repeatdt=None, lonlatdepth_dtype=None):
        """Initialise the ParticleSet from start/finish coordinates with equidistant spacing
        Note that this method uses simple numpy.linspace calls and does not take into account
        great circles, so may not be a exact on a globe

        :param fieldset: :mod:`parcels.fieldset.FieldSet` object from which to sample velocity
        :param pclass: mod:`parcels.particle.JITParticle` or :mod:`parcels.particle.ScipyParticle`
                 object that defines custom particle
        :param start: Starting point for initialisation of particles on a straight line.
        :param finish: End point for initialisation of particles on a straight line.
        :param size: Initial size of particle set
        :param depth: Optional list of initial depth values for particles. Default is 0m
        :param time: Optional start time value for particles. Default is fieldset.U.time[0]
        :param repeatdt: Optional interval (in seconds) on which to repeat the release of the ParticleSet
        :param lonlatdepth_dtype: Floating precision for lon, lat, depth particle coordinates.
               It is either np.float32 or np.float64. Default is np.float32 if fieldset.U.interp_method is 'linear'
               and np.float64 if the interpolation method is 'cgrid_velocity'
        """

        lon = np.linspace(start[0], finish[0], size)
        lat = np.linspace(start[1], finish[1], size)
        if type(depth) in [int, float]:
            depth = [depth] * size
        return cls(fieldset=fieldset, pclass=pclass, lon=lon, lat=lat, depth=depth, time=time, repeatdt=repeatdt, lonlatdepth_dtype=lonlatdepth_dtype)

    @classmethod
    def from_field(cls, fieldset, pclass, start_field, size, mode='monte_carlo', depth=None, time=None, repeatdt=None, lonlatdepth_dtype=None):
        """Initialise the ParticleSet randomly drawn according to distribution from a field

        :param fieldset: :mod:`parcels.fieldset.FieldSet` object from which to sample velocity
        :param pclass: mod:`parcels.particle.JITParticle` or :mod:`parcels.particle.ScipyParticle`
                 object that defines custom particle
        :param start_field: Field for initialising particles stochastically (horizontally)  according to the presented density field.
        :param size: Initial size of particle set
        :param mode: Type of random sampling. Currently only 'monte_carlo' is implemented
        :param depth: Optional list of initial depth values for particles. Default is 0m
        :param time: Optional start time value for particles. Default is fieldset.U.time[0]
        :param repeatdt: Optional interval (in seconds) on which to repeat the release of the ParticleSet
        :param lonlatdepth_dtype: Floating precision for lon, lat, depth particle coordinates.
               It is either np.float32 or np.float64. Default is np.float32 if fieldset.U.interp_method is 'linear'
               and np.float64 if the interpolation method is 'cgrid_velocity'
        """

        if mode == 'monte_carlo':
            data = start_field.data if isinstance(start_field.data, np.ndarray) else np.array(start_field.data)
            if start_field.interp_method == 'cgrid_tracer':
                p_interior = np.squeeze(data[0, 1:, 1:])
            else:  # if A-grid
                d = data
                p_interior = (d[0, :-1, :-1] + d[0, 1:, :-1] + d[0, :-1, 1:] + d[0, 1:, 1:])/4.
                p_interior = np.where(d[0, :-1, :-1] == 0, 0, p_interior)
                p_interior = np.where(d[0, 1:, :-1] == 0, 0, p_interior)
                p_interior = np.where(d[0, 1:, 1:] == 0, 0, p_interior)
                p_interior = np.where(d[0, :-1, 1:] == 0, 0, p_interior)
            p = np.reshape(p_interior, (1, p_interior.size))
            inds = np.random.choice(p_interior.size, size, replace=True, p=p[0] / np.sum(p))
            xsi = np.random.uniform(size=len(inds))
            eta = np.random.uniform(size=len(inds))
            j, i = np.unravel_index(inds, p_interior.shape)
            grid = start_field.grid
            if grid.gtype in [GridCode.RectilinearZGrid, GridCode.RectilinearSGrid]:
                lon = grid.lon[i] + xsi * (grid.lon[i + 1] - grid.lon[i])
                lat = grid.lat[j] + eta * (grid.lat[j + 1] - grid.lat[j])
            else:
                lons = np.array([grid.lon[j, i], grid.lon[j, i+1], grid.lon[j+1, i+1], grid.lon[j+1, i]])
                if grid.mesh == 'spherical':
                    lons[1:] = np.where(lons[1:] - lons[0] > 180, lons[1:]-360, lons[1:])
                    lons[1:] = np.where(-lons[1:] + lons[0] > 180, lons[1:]+360, lons[1:])
                lon = (1-xsi)*(1-eta) * lons[0] +\
                    xsi*(1-eta) * lons[1] +\
                    xsi*eta * lons[2] +\
                    (1-xsi)*eta * lons[3]
                lat = (1-xsi)*(1-eta) * grid.lat[j, i] +\
                    xsi*(1-eta) * grid.lat[j, i+1] +\
                    xsi*eta * grid.lat[j+1, i+1] +\
                    (1-xsi)*eta * grid.lat[j+1, i]
        else:
            raise NotImplementedError('Mode %s not implemented. Please use "monte carlo" algorithm instead.' % mode)

        return cls(fieldset=fieldset, pclass=pclass, lon=lon, lat=lat, depth=depth, time=time, lonlatdepth_dtype=lonlatdepth_dtype, repeatdt=repeatdt)

    @classmethod
    def from_particlefile(cls, fieldset, pclass, filename, restart=True, repeatdt=None, lonlatdepth_dtype=None):
        """Initialise the ParticleSet from a netcdf ParticleFile.
        This creates a new ParticleSet based on the last locations and time of all particles
        in the netcdf ParticleFile. Particle IDs are preserved if restart=True

        :param fieldset: :mod:`parcels.fieldset.FieldSet` object from which to sample velocity
        :param pclass: mod:`parcels.particle.JITParticle` or :mod:`parcels.particle.ScipyParticle`
                 object that defines custom particle
        :param filename: Name of the particlefile from which to read initial conditions
        :param restart: Boolean to signal if pset is used for a restart (default is True).
               In that case, Particle IDs are preserved.
        :param repeatdt: Optional interval (in seconds) on which to repeat the release of the ParticleSet
        :param lonlatdepth_dtype: Floating precision for lon, lat, depth particle coordinates.
               It is either np.float32 or np.float64. Default is np.float32 if fieldset.U.interp_method is 'linear'
               and np.float64 if the interpolation method is 'cgrid_velocity'
        """

        pfile = xr.open_dataset(str(filename), decode_cf=True)

        lon = np.ma.filled(pfile.variables['lon'][:, -1], np.nan)
        lat = np.ma.filled(pfile.variables['lat'][:, -1], np.nan)
        depth = np.ma.filled(pfile.variables['z'][:, -1], np.nan)
        time = np.ma.filled(pfile.variables['time'][:, -1], np.nan)
        pid = np.ma.filled(pfile.variables['trajectory'][:, -1], np.nan)
        if isinstance(time[0], np.timedelta64):
            time = np.array([t/np.timedelta64(1, 's') for t in time])

        inds = np.where(np.isfinite(lon))[0]
        lon = lon[inds]
        lat = lat[inds]
        depth = depth[inds]
        time = time[inds]
        pid = pid[inds] if restart else None

        return cls(fieldset=fieldset, pclass=pclass, lon=lon, lat=lat, depth=depth, time=time,
                   pid_orig=pid, lonlatdepth_dtype=lonlatdepth_dtype, repeatdt=repeatdt)

    def cptr(self, index):
        if self._ptype.uses_jit:
            node = self._nodes[index]
            return node.data.get_cptr()
        else:
            return None

    def empty(self):
        return self.size <= 0

    def begin(self):
        """
        Returns the begin of the linked particle list (like C++ STL begin() function)
        :return: begin Node (Node whose prev element is None); returns None if ParticleSet is empty
        """
        if not self.empty():
            node = self._nodes[0]
            while node.prev is not None:
                node = node.prev
            return node
        return None

    def end(self):
        """
        Returns the end of the linked partile list. UNLIKE in C++ STL, it returns the last element (valid element),
        not the element past the last element (invalid element). (see http://www.cplusplus.com/reference/list/list/end/)
        :return: end Node (Node whose next element is None); returns None if ParticleSet is empty
        """
        if not self.empty():
            node = self._nodes[self.size - 1]
            while node.next is not None:
                node = node.next
            return node
        return None

    # def set_kernel_class(self, kclass):
    #     self._kclass = kclass

    @property
    def lonlatdepth_dtype(self):
        return self._lonlatdepth_dtype

    @property
    def ptype(self):
        return self._ptype

    @property
    def kernel_class(self):
        return self._kclass

    @kernel_class.setter
    def kernel_class(self, value):
        self._kclass = value

    @property
    def size(self):
        return len(self._nodes)

    @property
    def data(self):
        return self._nodes

    @property
    def particles(self):
        return self._nodes

    @property
    def particle_data(self):
        return self._nodes

    @property
    def fieldset(self):
        return self._fieldset

    def __len__(self):
        return len(self._nodes)

    def __repr__(self):
        result = "\n"
        node = self._nodes[0]
        while node.prev is not None:
            node = node.prev
        while node.next is not None:
            result += str(node) + "\n"
            node = node.next
        result += str(node) + "\n"
        return result
        # return "\n".join([str(p) for p in self])

    def get_index(self, ndata):
        return self._nodes.index(ndata)

    def get(self, index):
        return self.get_by_index(index)

    def get_by_index(self, index):
        return self.__getitem__(index)

    def get_by_id(self, id):
        """
        divide-and-conquer search of SORTED list - needed because the node list internally
        can only be scanned for (a) its list index (non-coherent) or (b) a node itself, but not for a specific
        Node property alone. That is why using the 'bisect' module alone won't work.
        :param id: search Node ID
        :return: Node attached to ID - if node not in list: return None
        """
        lower = 0
        upper = len(self._nodes) - 1
        pos = lower + int((upper - lower) / 2.0)
        current_node = self._nodes[pos]
        _found = False
        _search_done = False
        while current_node.id != id and not _search_done:
            prev_upper = upper
            prev_lower = lower
            if id < current_node.id:
                lower = lower
                upper = pos - 1
                pos = lower + int((upper - lower) / 2.0)
            else:
                lower = pos
                upper = upper
                pos = lower + int((upper - lower) / 2.0) + 1
            if (prev_upper == upper and prev_lower == lower):
                _search_done = True
            current_node = self._nodes[pos]
        if current_node.id == id:
            _found = True
        if _found:
            return current_node
        else:
            return None

    def get_particle(self, index):
        return self.get(index).data

    # def retrieve_item(self, key):
    #    return self.get(key)

    def __getitem__(self, key):
        if key >= 0 and key < len(self._nodes):
            return self._nodes[key]
        return None

    def __setitem__(self, key, value):
        """
        Sets the 'data' portion of the Node list. Replacing the Node itself is error-prone,
        but it is possible to replace the data container (i.e. the particle data) or a specific
        Node.
        :param key: index (int; np.int32) or Node
        :param value: particle data of the particle class
        :return: modified Node
        """
        try:
            assert (isinstance(value, self._pclass))
        except AssertionError:
            print("setting value not of type '{}'".format(str(self._pclass)))
            exit()
        if isinstance(key, int) or isinstance(key, np.int32):
            search_node = self._nodes[key]
            search_node.set_data(value)
        elif isinstance(key, self._nclass):
            assert (key in self._nodes)
            key.set_data(value)

    def __iadd__(self, pdata):
        self.add(pdata)
        return self

    def add(self, pdata):
        if pdata is None:
            return
        if isinstance(pdata, (list, np.ndarray, ParticleSet)):
            self.add_entities(pdata)
        else:
            self.add_entity(pdata)

    def add_entities(self, data_array):
        if len(data_array) <= 0:
            return
        if isinstance(data_array, list) or isinstance(data_array, tuple):
            for item in data_array:
                self.add_entity(item)
        elif isinstance(data_array, ParticleSet):
            for i in range(len(data_array)):
                # self.add_entity(data_array[i])
                self.add_entity(data_array.pop(i))
        elif isinstance(data_array, np.ndarray):
            if data_array.dtype == self._ptype:
                for i in itertools.islice(itertools.count(), 0, data_array.shape[0]):
                    ndata = self._nclass(id=data_array[i].id, data=data_array[i])
                    self._nodes.add(ndata)
            else:
                # expect this to be a nD (2 <= n <= 5) array with [lon, lat, [depth, [time, [dt]]]]
                pu_data = None
                if MPI and MPI.COMM_WORLD.Get_size() > 1:
                    mpi_comm = MPI.COMM_WORLD
                    mpi_rank = mpi_comm.Get_rank()
                    spdata = data_array[:, 0:2]
                    dists = distance.cdist(spdata, self._pu_centers)
                    min_pu = np.argmax(dists, axis=1)
                    pu_data = data_array[ min_pu == mpi_rank ]
                else:
                    pu_data = data_array
                for i in itertools.islice(itertools.count(), 0, pu_data.shape[0]):
                    pdata = self._pclass(lon=pu_data[i, 0], lat=pu_data[i, 1], pid=np.iinfo(np.uint64).max, fieldset=self._fieldset)
                    if pu_data.shape[1]>2:
                        pdata.depth = pu_data[i, 2]
                    if pu_data.shape[1]>3:
                        pdata.time = pu_data[i, 3]
                    if pu_data.shape[1]>4:
                        pdata.dt = pu_data[i, 4]
                    self.add_entity(pdata, pu_checked=True)
        else:
            return

    def add_entity(self, pdata, pu_checked=False):
        """
        Adds the new data in the list - position is auto-determined (because of sorted-list nature)
        :param pdata: new Node or pdata
        :return: index of inserted node
        """
        # Comment: by current workflow, pset modification is only done on the front node, thus
        # the distance determination and assigment is also done on the front node
        _add_to_pu = True
        # if MPI:
        if MPI and MPI.COMM_WORLD.Get_size() > 1 and not pu_checked:
            if self._pu_centers is not None and isinstance(self._pu_centers, np.ndarray):
                mpi_comm = MPI.COMM_WORLD
                mpi_rank = mpi_comm.Get_rank()
                mpi_size = mpi_comm.Get_size()
                min_dist = np.finfo(self._lonlatdepth_dtype).max
                min_pu = 0
                if mpi_size > 1 and mpi_rank == 0:
                    ppos = pdata
                    if isinstance(pdata, self._nclass):
                        ppos = pdata.data
                    spdata = np.array([ppos.lat, ppos.lon], dtype=self._lonlatdepth_dtype)
                    n_clusters = self._pu_centers.shape[0]
                    for i in range(n_clusters):
                        diff = self._pu_centers[i, :] - spdata
                        dist = np.dot(diff, diff)
                        if dist < min_dist:
                            min_dist = dist
                            min_pu = i
                # NOW: move the related center by: (center-spdata) * 1/(cluster_size+1)
                min_pu = mpi_comm.bcast(min_pu, root=0)
                if mpi_rank == min_pu:
                    _add_to_pu = True
                else:
                    _add_to_pu = False
                # ==== old non-numpy code ==== #
                # ppos = pdata
                # if isinstance(pdata, self._nclass):
                #     ppos = pdata.data
                # spdata = np.array([ppos.lat, ppos.lon], dtype=self._lonlatdepth_dtype)
                # n_clusters = self._pu_centers.shape[0]
                # for i in range(n_clusters):
                #     diff = self._pu_centers[i, :] - spdata
                #     dist = np.dot(diff, diff)
                #     if dist < min_dist:
                #         min_dist = dist
                #         min_pu = i
                # if mpi_rank == min_pu:
                #     _add_to_pu = True
                # else:
                #     _add_to_pu = False
                # # TODO: still need to propagate the new information
        if _add_to_pu:
            index = -1
            pid = np.iinfo(np.uint64).max
            if isinstance(pdata, self._nclass):
                self._nodes.add(pdata)
                index = self._nodes.bisect_right(pdata)
            else:
                if pdata.id == pid:
                    index = idgen.total_length
                    pid = idgen.nextID(pdata.lon, pdata.lat, pdata.depth, pdata.time)
                    pdata.id = pid
                    pdata.index = index
                else:
                    pid = pdata.id
                    index = pdata.index
                node = NodeJIT(id=pid, data=pdata)
                self._nodes.add(node)
                index = self._nodes.bisect_right(node)
            if index >= 0:
                # return self._nodes[index]
                return index
        return None

    def __isub__(self, ndata):
        self.remove(ndata)
        return self

    def remove(self, ndata):
        """
        Removes a specific Node from the list. The Node can either be given directly or determined via it's index
        or it's data package (i.e. particle data). When using the index, note though that Nodes are shifting
        (non-coherent indices), so the reliable method is to provide the Node to-be-removed directly
        (similar to an iterator in C++).
        :param ndata: Node object, Particle object or int index to the Node to-be-removed
        """
        if ndata is None:
            pass
        elif isinstance(ndata, list) or isinstance(ndata, np.ndarray):
            self.remove_entities(ndata)  # remove multiple instances
        elif isinstance(ndata, self._nclass):
            self.remove_entity(ndata)
        else:
            pass

    def remove_entity(self, ndata):
        if isinstance(ndata, int) or isinstance(ndata, np.int32):
            del self._nodes[ndata]
            # search_node = self._nodes[ndata]
            # self._nodes.remove(search_node)
        elif isinstance(ndata, self._nclass):
            try:
                # self._nodes.remove(ndata)
                nindex = self.get_index(ndata)
                del self._nodes[nindex]
            except ValueError:
                logger("Node {} not found.".format(ndata))
        elif isinstance(ndata, self._pclass):
            try:
                node = self.get_by_id(ndata.id)
                nindex = self.get_index(node)
                # self._nodes.remove(node)
                del self._nodes[nindex]
            except ValueError:
                logger("Particle data {} not found.".format(ndata))

    def remove_entities(self, ndata_array):
        rm_list = ndata_array
        if len(ndata_array) <= 0:
            return
        if isinstance(rm_list[0], int) or isinstance(rm_list[0], np.int32) or isinstance(rm_list[0], np.int64):
            rm_list = []
            for index in ndata_array:
                rm_list.append(self.get_by_index(index))
        for ndata in rm_list:
            self.remove_entity(ndata)

    def merge(self, key1, key2):
        # TODO
        pass

    def split(self, key):
        """
        splits a node, returning the result 2 new nodes
        :param key: index (int; np.int32), Node
        :return: 'node1, node2' or 'index1, index2'
        """
        # TODO

    def pop(self, idx=-1, deepcopy_elem=False):
        try:
            return self._nodes.pop(idx, deepcopy_elem)
        except IndexError:
            return None

    def insert(self, node_or_pdata):
        """
        Inserts new data in the list - position is auto-determined (semantically equal to 'add')
        :param node_or_pdata: new Node or pdata
        :return: index of inserted node
        """
        return self.add(node_or_pdata)

    # ==== high-level functions to execute operations (Add, Delete, Merge, Split) requested by the ==== #
    # ==== internal :variables Particle.state of each Node.                                        ==== #

    def get_deleted_item_indices(self):
        indices = [i for i, n in enumerate(self._nodes) if n.data.state == ErrorCode.Delete]
        return indices

    def remove_deleted_items_by_indices(self, indices):
        if len(indices) > 0:
            indices.sort(reverse=True)
            for index in indices:
                del self._nodes[index]

    def remove_deleted_items(self):
        node = self.begin()
        while node is not None:
            next_node = node.next
            if node.data.state == ErrorCode.Delete:
                self._nodes.remove(node)
            node = next_node

    def execute(self, pyfunc=AdvectionRK4, endtime=None, runtime=None, dt=1.,
                moviedt=None, recovery=None, output_file=None, movie_background_field=None,
                verbose_progress=None, postIterationCallbacks=None, callbackdt=None):
        """Execute a given kernel function over the particle set for
        multiple timesteps. Optionally also provide sub-timestepping
        for particle output.

        :param pyfunc: Kernel function to execute. This can be the name of a
                       defined Python function or a :class:`parcels.kernel.Kernel` object.
                       Kernels can be concatenated using the + operator
        :param endtime: End time for the timestepping loop.
                        It is either a datetime object or a positive double.
        :param runtime: Length of the timestepping loop. Use instead of endtime.
                        It is either a timedelta object or a positive double. [DURATION]
        :param dt: Timestep interval to be passed to the kernel.
                   It is either a timedelta object or a double.
                   Use a negative value for a backward-in-time simulation.
        :param recovery: Dictionary with additional `:mod:parcels.tools.error`
                         recovery kernels to allow custom recovery behaviour in case of
                         kernel errors.
        :param output_file: :mod:`parcels.particlefile.ParticleFile` object for particle output
        :param verbose_progress: Boolean for providing a progress bar for the kernel execution loop.
        :param postIterationCallbacks: (Optional) Array of functions that are to be called after each iteration (post-process, non-Kernel)
        :param callbackdt: (Optional, in conjecture with 'postIterationCallbacks) timestep inverval to (latestly) interrupt the running kernel and invoke post-iteration callbacks from 'postIterationCallbacks'
        """

        # check if pyfunc has changed since last compile. If so, recompile
        if self._kernel is None or (self._kernel.pyfunc is not pyfunc and self._kernel is not pyfunc):
            # Generate and store Kernel
            if isinstance(pyfunc, self._kclass):
                self._kernel = pyfunc
            else:
                self._kernel = self.Kernel(pyfunc)
            # Prepare JIT kernel execution
            if self._ptype.uses_jit:
                self._kernel.remove_lib()
                cppargs = ['-DDOUBLE_COORD_VARIABLES'] if self.lonlatdepth_dtype == np.float64 else None
                # self._kernel.compile(compiler=GNUCompiler(cppargs=cppargs))
                # self._kernel.compile(compiler=GNUCompiler(cppargs=cppargs, incdirs=[os.path.join(get_package_dir(), 'include'), os.path.join(get_package_dir(), 'nodes'), "."], libdirs=[".", get_cache_dir()], libs=["node"]))
                self._kernel.compile(compiler=GNUCompiler_MS(cppargs=cppargs, incdirs=[os.path.join(get_package_dir(), 'include'), os.path.join(get_package_dir(), 'nodes'), "."], tmp_dir=get_cache_dir()))
                self._kernel.load_lib()

        # Convert all time variables to seconds
        if isinstance(endtime, delta):
            raise RuntimeError('endtime must be either a datetime or a double')
        if isinstance(endtime, dtime):
            endtime = np.datetime64(endtime)

        if isinstance(endtime, np.datetime64):
            if self.time_origin.calendar is None:
                raise NotImplementedError('If fieldset.time_origin is not a date, execution endtime must be a double')
            endtime = self.time_origin.reltime(endtime)

        if isinstance(runtime, delta):
            runtime = runtime.total_seconds()
        if isinstance(dt, delta):
            dt = dt.total_seconds()
        outputdt = output_file.outputdt if output_file else np.infty
        if isinstance(outputdt, delta):
            outputdt = outputdt.total_seconds()

        if isinstance(moviedt, delta):
            moviedt = moviedt.total_seconds()
        if isinstance(callbackdt, delta):
            callbackdt = callbackdt.total_seconds()

        assert runtime is None or runtime >= 0, 'runtime must be positive'
        assert outputdt is None or outputdt >= 0, 'outputdt must be positive'
        assert moviedt is None or moviedt >= 0, 'moviedt must be positive'

        # ==== Set particle.time defaults based on sign of dt, if not set at ParticleSet construction => moved below (l. xyz)
        # piter = 0
        # while piter < len(self._nodes):
        #     pdata = self._nodes[piter].data
        # #node = self.begin()
        # #while node is not None:
        # #    pdata = node.data
        #     if np.isnan(pdata.time):
        #         mintime, maxtime = self._fieldset.gridset.dimrange('time_full')
        #         pdata.time = mintime if dt >= 0 else maxtime
        # #    node.set_data(pdata)
        #     self._nodes[piter].set_data(pdata)
        #     piter += 1

        # Derive _starttime and endtime from arguments or fieldset defaults
        if runtime is not None and endtime is not None:
            raise RuntimeError('Only one of (endtime, runtime) can be specified')

        mintime, maxtime = self._fieldset.gridset.dimrange('time_full')
        _starttime = min([n.data.time for n in self._nodes if not np.isnan(n.data.time)] + [mintime, ]) if dt >= 0 else max([n.data.time for n in self._nodes if not np.isnan(n.data.time)] + [maxtime, ])
        if self.repeatdt is not None and self.repeat_starttime is None:
            self.repeat_starttime = _starttime
        if runtime is not None:
            endtime = _starttime + runtime * np.sign(dt)
        elif endtime is None:
            endtime = maxtime if dt >= 0 else mintime

        # print("Fieldset min-max: {} to {}".format(mintime, maxtime))
        # print("starttime={} to endtime={} (runtime={})".format(_starttime, endtime, runtime))

        execute_once = False
        if abs(endtime-_starttime) < 1e-5 or dt == 0 or runtime == 0:
            dt = 0
            runtime = 0
            endtime = _starttime

            logger.warning_once("dt or runtime are zero, or endtime is equal to Particle.time. "
                                "The kernels will be executed once, without incrementing time")
            execute_once = True

        # ==== Initialise particle timestepping
        # for p in self:
        #     p.dt = dt
        piter = 0
        while piter < len(self._nodes):
            pdata = self._nodes[piter].data
            pdata.dt = dt
            if np.isnan(pdata.time):
                pdata.time = _starttime
            self._nodes[piter].set_data(pdata)
            piter += 1

        # First write output_file, because particles could have been added
        if output_file is not None:
            output_file.write(self, _starttime)

        if moviedt:
            self.show(field=movie_background_field, show_time=_starttime, animation=True)
        if moviedt is None:
            moviedt = np.infty
        if callbackdt is None:
            interupt_dts = [np.infty, moviedt, outputdt]
            if self.repeatdt is not None:
                interupt_dts.append(self.repeatdt)
            callbackdt = np.min(np.array(interupt_dts))

        time = _starttime
        if self.repeatdt and self.rparam is not None:
            next_prelease = self.repeat_starttime + (abs(time - self.repeat_starttime) // self.repeatdt + 1) * self.repeatdt * np.sign(dt)
        else:
            next_prelease = np.infty if dt > 0 else - np.infty
        next_output = time + outputdt if dt > 0 else time - outputdt

        next_movie = time + moviedt if dt > 0 else time - moviedt
        next_callback = time + callbackdt if dt > 0 else time - callbackdt

        next_input = self._fieldset.computeTimeChunk(time, np.sign(dt))

        tol = 1e-12

        if verbose_progress is None:
            walltime_start = time_module.time()
        if verbose_progress:
            pbar = self._create_progressbar_(_starttime, endtime)

        while (time < endtime and dt > 0) or (time > endtime and dt < 0) or dt == 0:

            if verbose_progress is None and time_module.time() - walltime_start > 10:
                # Showing progressbar if runtime > 10 seconds
                if output_file:
                    logger.info('Temporary output files are stored in %s.' % output_file.tempwritedir_base)
                    logger.info('You can use "parcels_convert_npydir_to_netcdf %s" to convert these '
                                'to a NetCDF file during the run.' % output_file.tempwritedir_base)
                pbar = self._create_progressbar_(_starttime, endtime)
                verbose_progress = True

            if dt > 0:
                time = min(next_prelease, next_input, next_output, next_movie, next_callback, endtime)
            else:
                time = max(next_prelease, next_input, next_output, next_movie, next_callback, endtime)
            self._kernel.execute(self, endtime=time, dt=dt, recovery=recovery, output_file=output_file, execute_once=execute_once)
            if abs(time-next_prelease) < tol:
                if self.rparam.get_particle_id(0) is None:
                    pdata = np.array(self.rparam.lon, dtype=self._lonlatdepth_dtype).reshape((self.rparam.num_pts, 1))
                    pdata = np.concatenate((pdata, np.array(self.rparam.lat, dtype=self.lonlatdepth_dtype).reshape((self.rparam.num_pts, 1))), axis=1)
                    if len(self.rparam.depth) > 0:
                        pdata = np.concatenate((pdata, np.array(self.rparam.depth, dtype=self.lonlatdepth_dtype).reshape((self.rparam.num_pts, 1))), axis=1)
                    else:
                        pdata = np.concatenate((pdata, np.zeros(pdata.shape[0], dtype=self._lonlatdepth_dtype).reshape((self.rparam.num_pts, 1))), axis=1)
                    pdata = np.concatenate((pdata, (np.ones(pdata.shape[0], dtype=np.float64) * time).reshape((self.rparam.num_pts, 1))), axis=1)
                    pdata = np.concatenate((pdata, (np.ones(pdata.shape[0], dtype=np.float64) * dt).reshape((self.rparam.num_pts, 1))), axis=1)
                    self.add(pdata)
                else:
                    add_iter = 0
                    while add_iter < self.rparam.num_pts:
                        gen_id = self.rparam.get_particle_id(add_iter)
                        lon = self.rparam.get_longitude(add_iter)
                        lat = self.rparam.get_latitude(add_iter)
                        pdepth = self.rparam.get_depth_value(add_iter)
                        ptime = time
                        # pindex = idgen.total_length
                        pid = idgen.nextID(lon, lat, pdepth, ptime) if gen_id is None else gen_id
                        # pid = np.iinfo(np.uint64).max if gen_id is None else gen_id
                        # pdata = self._pclass(lon=lon, lat=lat, pid=pid, fieldset=self._fieldset, depth=pdepth, time=ptime, index=pindex)
                        pdata = self._pclass(lon=lon, lat=lat, pid=pid, fieldset=self._fieldset, depth=pdepth, time=ptime)
                        pdata.dt = dt
                        self.add(self._nclass(id=pid, data=pdata))
                        add_iter += 1
                next_prelease += self.repeatdt * np.sign(dt)
            if abs(time-next_output) < tol:
                if output_file is not None:
                    output_file.write(self, time)
                next_output += outputdt * np.sign(dt)

            if abs(time-next_movie) < tol:
                self.show(field=movie_background_field, show_time=time, animation=True)
                next_movie += moviedt * np.sign(dt)
            # ==== insert post-process here to also allow for memory clean-up via external func ==== #
            if abs(time-next_callback) < tol:
                if postIterationCallbacks is not None:
                    for extFunc in postIterationCallbacks:
                        extFunc()
                next_callback += callbackdt * np.sign(dt)

            if time != endtime:
                next_input = self._fieldset.computeTimeChunk(time, dt)
            if dt == 0:
                break

            if verbose_progress:
                pbar.update(abs(time - _starttime))

        if output_file is not None:
            output_file.write(self, time)

        if verbose_progress:
            pbar.finish()

    def Kernel(self, pyfunc, c_include="", delete_cfiles=True):
        """Wrapper method to convert a `pyfunc` into a :class:`parcels.kernel.Kernel` object
        based on `fieldset` and `ptype` of the ParticleSet
        :param delete_cfiles: Boolean whether to delete the C-files after compilation in JIT mode (default is True)
        """
        return self._kclass(self._fieldset, self._ptype, pyfunc=pyfunc, c_include=c_include, delete_cfiles=delete_cfiles)

    def ParticleFile(self, *args, **kwargs):
        """Wrapper method to initialise a :class:`parcels.particlefile.ParticleFile`
        object from the ParticleSet"""
        return ParticleFile(*args, particleset=self, **kwargs)

    def _create_progressbar_(self, starttime, endtime):
        pbar = None
        try:
            pbar = progressbar.ProgressBar(max_value=abs(endtime - starttime)).start()
        except:  # for old versions of progressbar
            try:
                pbar = progressbar.ProgressBar(maxvalue=abs(endtime - starttime)).start()
            except:  # for even older OR newer versions
                pbar = progressbar.ProgressBar(maxval=abs(endtime - starttime)).start()
        return pbar

    def show(self, with_particles=True, show_time=None, field=None, domain=None, projection=None,
             land=True, vmin=None, vmax=None, savefile=None, animation=False, **kwargs):
        """Method to 'show' a Parcels ParticleSet

        :param with_particles: Boolean whether to show particles
        :param show_time: Time at which to show the ParticleSet
        :param field: Field to plot under particles (either None, a Field object, or 'vector')
        :param domain: dictionary (with keys 'N', 'S', 'E', 'W') defining domain to show
        :param projection: type of cartopy projection to use (default PlateCarree)
        :param land: Boolean whether to show land. This is ignored for flat meshes
        :param vmin: minimum colour scale (only in single-plot mode)
        :param vmax: maximum colour scale (only in single-plot mode)
        :param savefile: Name of a file to save the plot to
        :param animation: Boolean whether result is a single plot, or an animation
        """
        from parcels.plotting import plotparticles
        plotparticles(particles=self, with_particles=with_particles, show_time=show_time, field=field, domain=domain,
                      projection=projection, land=land, vmin=vmin, vmax=vmax, savefile=savefile, animation=animation, **kwargs)

    def density(self, field=None, particle_val=None, relative=False, area_scale=False):
        """Method to calculate the density of particles in a ParticleSet from their locations,
        through a 2D histogram.

        :param field: Optional :mod:`parcels.field.Field` object to calculate the histogram
                      on. Default is `fieldset.U`
        :param particle_val: Optional numpy-array of values to weigh each particle with,
                             or string name of particle variable to use weigh particles with.
                             Default is None, resulting in a value of 1 for each particle
        :param relative: Boolean to control whether the density is scaled by the total
                         weight of all particles. Default is False
        :param area_scale: Boolean to control whether the density is scaled by the area
                           (in m^2) of each grid cell. Default is False
        """

        field = field if field else self.fieldset.U
        if isinstance(particle_val, str):
            particle_val = [getattr(p, particle_val) for p in self.particles]
        else:
            particle_val = particle_val if particle_val else np.ones(len(self.particles))
        density = np.zeros((field.grid.lat.size, field.grid.lon.size), dtype=np.float32)

        for pi, p in enumerate(self.particles):
            try:  # breaks if either p.xi, p.yi, p.zi, p.ti do not exist (in scipy) or field not in fieldset
                if p.ti[field.igrid] < 0:  # xi, yi, zi, ti, not initialised
                    raise('error')
                xi = p.xi[field.igrid]
                yi = p.yi[field.igrid]
            except:
                _, _, _, xi, yi, _ = field.search_indices(p.lon, p.lat, p.depth, 0, 0, search2D=True)
            density[yi, xi] += particle_val[pi]

        if relative:
            density /= np.sum(particle_val)

        if area_scale:
            density /= field.cell_areas()

        return density
