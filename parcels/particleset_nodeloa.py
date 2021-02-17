import collections
import time as time_module
from datetime import date
from datetime import datetime
from datetime import timedelta as delta

import numpy as np
import xarray as xr
import progressbar

from parcels.compiler import GNUCompiler
from parcels.field import NestedField
from parcels.field import SummedField
from parcels.grid import GridCode
from parcels.kernel import Kernel
from parcels.kernels.advection import AdvectionRK4
from parcels.particle import JITParticle
from parcels.particlefile import ParticleFile
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

__all__ = ['ParticleSet']


class ParticleSet(object):
    """Container class for storing particle and executing kernel over them.

    Please note that this currently only supports fixed size particle sets.

    :param fieldset: :mod:`parcels.fieldset.FieldSet` object from which to sample velocity
    :param pclass: Optional :mod:`parcels.particle.JITParticle` or
                 :mod:`parcels.particle.ScipyParticle` object that defines custom particle
    :param lon: List of initial longitude values for particles
    :param lat: List of initial latitude values for particles
    :param depth: Optional list of initial depth values for particles. Default is 0m
    :param time: Optional list of initial time values for particles. Default is fieldset.U.grid.time[0]
    :param repeatdt: Optional interval (in seconds) on which to repeat the release of the ParticleSet
    :param lonlatdepth_dtype: Floating precision for lon, lat, depth particle coordinates.
           It is either np.float32 or np.float64. Default is np.float32 if fieldset.U.interp_method is 'linear'
           and np.float64 if the interpolation method is 'cgrid_velocity'
    :param partitions: List of cores on which to distribute the particles for MPI runs. Default: None, in which case particles
           are distributed automatically on the processors
    Other Variables can be initialised using further arguments (e.g. v=... for a Variable named 'v')
    """

    def __init__(self, fieldset, pclass=JITParticle, lon=None, lat=None, depth=None, time=None, repeatdt=None, lonlatdepth_dtype=None, pid_orig=None, **kwargs):
        self.fieldset = fieldset
        self.fieldset.check_complete()
        partitions = kwargs.pop('partitions', None)

        def convert_to_array(var):
            # Convert lists and single integers/floats to one-dimensional numpy arrays
            if isinstance(var, np.ndarray):
                return var.flatten()
            elif isinstance(var, (int, float, np.float32, np.int32)):
                return np.array([var])
            else:
                return np.array(var)

        lon = np.empty(shape=0) if lon is None else convert_to_array(lon)
        lat = np.empty(shape=0) if lat is None else convert_to_array(lat)
        if pid_orig is None:
            pid_orig = np.arange(lon.size)
        pid = pid_orig + pclass.lastID

        if depth is None:
            mindepth, _ = self.fieldset.gridset.dimrange('depth')
            depth = np.ones(lon.size) * mindepth
        else:
            depth = convert_to_array(depth)
        assert lon.size == lat.size and lon.size == depth.size, (
            'lon, lat, depth don''t all have the same lenghts')

        time = convert_to_array(time)
        time = np.repeat(time, lon.size) if time.size == 1 else time
        if time.size > 0 and type(time[0]) in [datetime, date]:
            time = np.array([np.datetime64(t) for t in time])
        self.time_origin = fieldset.time_origin
        if time.size > 0 and isinstance(time[0], np.timedelta64) and not self.time_origin:
            raise NotImplementedError('If fieldset.time_origin is not a date, time of a particle must be a double')
        time = np.array([self.time_origin.reltime(t) if isinstance(t, np.datetime64) else t for t in time])
        assert lon.size == time.size, (
            'time and positions (lon, lat, depth) don''t have the same lengths.')

        if partitions is not None and partitions is not False:
            partitions = convert_to_array(partitions)

        for kwvar in kwargs:
            kwargs[kwvar] = convert_to_array(kwargs[kwvar])
            assert lon.size == kwargs[kwvar].size, (
                '%s and positions (lon, lat, depth) don''t have the same lengths.' % kwargs[kwvar])

        offset = np.max(pid) if len(pid) > 0 else -1
        if MPI:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()
            mpi_size = mpi_comm.Get_size()

            if lon.size < mpi_size and mpi_size > 1:
                raise RuntimeError('Cannot initialise with fewer particles than MPI processors')

            if mpi_size > 1:
                if partitions is not False:
                    if partitions is None:
                        if mpi_rank == 0:
                            coords = np.vstack((lon, lat)).transpose()
                            kmeans = KMeans(n_clusters=mpi_size, random_state=0).fit(coords)
                            partitions = kmeans.labels_
                        else:
                            partitions = None
                        partitions = mpi_comm.bcast(partitions, root=0)
                    elif np.max(partitions >= mpi_rank):
                        raise RuntimeError('Particle partitions must vary between 0 and the number of mpi procs')
                    lon = lon[partitions == mpi_rank]
                    lat = lat[partitions == mpi_rank]
                    time = time[partitions == mpi_rank]
                    depth = depth[partitions == mpi_rank]
                    pid = pid[partitions == mpi_rank]
                    for kwvar in kwargs:
                        kwargs[kwvar] = kwargs[kwvar][partitions == mpi_rank]
                offset = MPI.COMM_WORLD.allreduce(offset, op=MPI.MAX)

        self.repeatdt = repeatdt.total_seconds() if isinstance(repeatdt, delta) else repeatdt
        if self.repeatdt:
            if self.repeatdt <= 0:
                raise('Repeatdt should be > 0')
            if time[0] and not np.allclose(time, time[0]):
                raise ('All Particle.time should be the same when repeatdt is not None')
            self.repeat_starttime = time[0]
            self.repeatlon = lon
            self.repeatlat = lat
            if not hasattr(self, 'repeatpid'):
                self.repeatpid = pid - pclass.lastID
            self.repeatdepth = depth
            self.repeatpclass = pclass
            self.partitions = partitions
            self.repeatkwargs = kwargs
        pclass.setLastID(offset+1)

        if lonlatdepth_dtype is None:
            self.lonlatdepth_dtype = self.lonlatdepth_dtype_from_field_interp_method(fieldset.U)
        else:
            self.lonlatdepth_dtype = lonlatdepth_dtype
        assert self.lonlatdepth_dtype in [np.float32, np.float64], \
            'lon lat depth precision should be set to either np.float32 or np.float64'
        JITParticle.set_lonlatdepth_dtype(self.lonlatdepth_dtype)

        # == this should be a list not a dict! == #
        self._pid_mapping_bounds = []  # plist bracket_index -> (min_id, max_id, size_bracket)
        # ======================================= #
        self.nlist_limit = 4096
        # self.particles = np.empty(lon.size, dtype=pclass)
        assert lon.shape[0] == lat.shape[0], ('Length of lon and lat do not match.')
        # self._plist = []
        self._plist = collections.deque([])
        self._bracket_no = 0
        start_index = 0
        end_index = 0
        while end_index < lon.shape[0]:
            end_index = min(lon.shape[0], start_index + self.nlist_limit)
            self._plist.append(np.empty(end_index-start_index, dtype=pclass))
            self._pid_mapping_bounds.append((np.iinfo(np.int32).max, np.iinfo(np.int32).min, end_index-start_index))
            # bracket_index = len(self._plist)-1
            start_index = end_index
            # end_index = min(lon.shape[0], end_index+self.nlist_limit)
        # self._particle_data = None
        self._plist_c = None
        self._pclass = pclass
        self._ptype = self.pclass.getPType()
        self.kernel = None

        # ==== ANNOTATION ==== #
        # the particles themselves (self.particles, dtype: pclass) are already separate memory fields from
        # their c-pointers (self._particle_data, dtype: PType.dtype [ie. padded memory block]). Thus, it makes little
        # sense to require them being static over the whole ParticleSet lifetime. In other words: there should be a mechanism
        # to update the particles, reallocate the c-pointer field, and then just value-copy the particle data into the c-memory.
        # nparticles = self.allocate_cptrs()
        if self._ptype.uses_jit:
            # self._plist_c = []
            self._plist_c = collections.deque([])
            for bracket_index in range(len(self._plist)):
                self._plist_c.append(np.empty(self._pid_mapping_bounds[bracket_index][2], dtype=self.ptype.dtype))

        if lon is not None and lat is not None:
            # == Initialise from arrays of lon/lat coordinates == #
            # assert self.particles.size == lon.size and self.particles.size == lat.size, ('Size of ParticleSet does not match length of lon and lat.')

            nbrackets = 0
            for i in range(lon.size):
                bracket_index = int(float(i)/self.nlist_limit)
                slot_index = int(i % self.nlist_limit)
                # self._plist[bracket_index][slot_index] = pclass(lon[i], lat[i], pid[i], fieldset=fieldset, depth=depth[i], cptr=self.cptr(bracket_index, slot_index), time=time[i])
                self.current_p_node()[slot_index] = pclass(lon[i], lat[i], pid[i], fieldset=fieldset, depth=depth[i], cptr=self.current_c_node()(slot_index), time=time[i])
                bracket_info = self._pid_mapping_bounds[bracket_index]
                self._pid_mapping_bounds[bracket_index] = (min(pid[i], bracket_info[0]), max(pid[i], bracket_info[1]), bracket_info[2])
                # self.particles[i] = pclass(lon[i], lat[i], pid[i], fieldset=fieldset, depth=depth[i], cptr=cptr(i), time=time[i])
                # == Set other Variables if provided == #
                for kwvar in kwargs:
                    # if not hasattr(self._plist[bracket_index][slot_index], kwvar):
                    if not hasattr(self.current_p_node()[slot_index], kwvar):
                        raise RuntimeError('Particle class does not have Variable %s' % kwvar)
                    # setattr(self._plist[bracket_index][slot_index], kwvar, kwargs[kwvar][i])
                    setattr(self.current_p_node()[slot_index], kwvar, kwargs[kwvar][i])
        #             if not hasattr(self.particles[i], kwvar):
        #                 raise RuntimeError('Particle class does not have Variable %s' % kwvar)
        #             setattr(self.particles[i], kwvar, kwargs[kwvar][i])
                if slot_index >= (self.nlist_limit-1):
                    self.next()
            self.reset_current_node()
        else:
            raise ValueError("Latitude and longitude required for generating ParticleSet")

    def next(self):
        self._bracket_no += 1
        self._plist.rotate(-1)
        if self._plist_c is not None and self._ptype.uses_jit:
            self._plist_c.rotate(-1)

    def prev(self):
        self._bracket_no -= 1
        self._plist.rotate(1)
        if self._plist_c is not None and self._ptype.uses_jit:
            self._plist_c.rotate(1)

    def reset_current_node(self):
        self._plist.rotate(self._bracket_no)
        if self._plist_c is not None and self._ptype.uses_jit:
            self._plist_c.rotate(self._bracket_no)
        self._bracket_no = 0

    @property
    def current_p_node(self):
        return self._plist[0]

    @current_p_node.setter
    def current_p_node(self, value):
        self._plist[0] = value

    @property
    def current_c_node(self):
        if self._plist_c is not None and self._ptype.uses_jit:
            return self._plist_c[0]
        return None

    @current_c_node.setter
    def current_c_node(self, value):
        if self._plist_c is not None and self._ptype.uses_jit:
            self._plist_c[0] = value

    @property
    def particles(self):
        if len(self._plist) == 0:
            return None
        result = self._plist[0]
        if len(self._plist) > 1:
            # for bracket_index in range(0, len(self._plist)):
            #     result = np.concatenate((result, self._plist[bracket_index]), axis=0)
            for bracket in self._plist:
                result = np.concatenate((result, bracket), axis=0)
        return result
        # return self._plist

    @property
    def particles_a(self):
        return self._plist

    @property
    def pid_mapping_bounds(self):
        return self._pid_mapping_bounds

    @property
    def particles_c(self):
        return self._plist_c

    @property
    def ptype(self):
        return self._ptype

    @property
    def pclass(self):
        return self._pclass

    def cptr(self, bracket_index, slot_index):
        """
        THIS IS EXTREMELY SLOW
        """
        # if self._particle_data is None:
        if self._plist_c is None:
            return None
        if self.ptype.uses_jit:
            return self._plist_c[bracket_index][slot_index]
        else:
            return None

    # def allocate_cptrs(self):
    #     nparticles = 0
    #     for sublist in self.plist:
    #         nparticles += sublist.shape[0]
    #     # == Allocate underlying data for C-allocated particles == #
    #     if self.ptype.uses_jit:
    #         self._particle_data = np.empty(nparticles, dtype=self.ptype.dtype)
    #     return nparticles

    def get_list_array_index(self, pdata):
        """Searches for the list-array indices for a given particle.

        :param pdata: :mod:`int` global index or :mod:`parcels.particle._Particle` (or its subclass) object to search for
        :return :mod:`tuple` in format (<list index>, <np.ndarray[dtype=Particle] index>), to index
        """
        if isinstance(pdata, int):
            # == parameter is the global particle index == #
            bracket_index = 0
            start_index = 0
            end_index = self._pid_mapping_bounds[bracket_index][2]
            while pdata >= end_index:
                start_index = end_index
                end_index = start_index+self._pid_mapping_bounds[bracket_index][2]
                bracket_index += 1
            slot_index = pdata-start_index
            return (bracket_index, slot_index)
        elif isinstance(pdata, self.pclass):
            # == parameter is a particle itself == #
            bracket_index = 0
            bracket_info = self.pid_mapping_bounds[bracket_index]
            while bracket_index < len(self._plist) and ((pdata.id < bracket_info[0]) or (pdata.id > bracket_info[1])):
                bracket_index += 1
                bracket_info = self.pid_mapping_bounds[bracket_index]
                self.next()
            if bracket_index >= len(self._plist):
                logger.warn_once("ParticleSet.get_cptr_index() - requested particle ({}) not found.".format(pdata))
                self.reset_current_node()
                return -1
            slot_index = 0
            while (slot_index < self.current_p_node().shape[0]) and (self.current_p_node()[slot_index].id != pdata.id):
                slot_index += 1
            if slot_index >= self.current_p_node().shape[0]:
                logger.warn_once("ParticleSet.get_cptr_index() - requested particle ({}) not found.".format(pdata))
                self.reset_current_node()
                return -1
            self.reset_current_node()
            return (bracket_index, slot_index)

    def get_list_array_index_by_PID(self, pdata):
        # == parameter is the particle ID == #
        located_particle = [(p, bracket_index, slot_index) for bracket_index, sublist in enumerate(self._plist) for slot_index, p in enumerate(sublist) if p.id == pdata]
        if len(located_particle) < 1:
            logger.warn_once("ParticleSet.get_cptr_index() - requested particle ID ({}) not existent.".format(pdata))
            return -1
        if len(located_particle) > 1:
            logger.warn_once("ParticleSet.get_cptr_index() - requested particle ID ({}) ambiguous.".format(pdata))
            return -1
        located_particle = located_particle[0]
        # nparticle = 0
        # for i in range(located_particle[1]):
        #     nparticle += self.pid_mapping_bounds[i][2]
        # return nparticle+located_particle[2]
        return (located_particle[1], located_particle[2])

    def covert_static_array_to_list_array(self, property_array):
        if not isinstance(property_array, np.ndarray):
            property_array = np.array(property_array)
        nparticles = 0
        for bracket_index in range(len(self._pid_mapping_bounds)):
            nparticles += self._pid_mapping_bounds[bracket_index][2]
        assert property_array.shape[0] == nparticles, ("Attaching the property array to the list of particle prohibited because num. of entries do not match.")
        results = []
        bracket_index = 0
        start_index = 0
        end_index = self._pid_mapping_bounds[bracket_index][2]
        while start_index < property_array.shape[0]:
            results.append(np.array(property_array[start_index:min(end_index, property_array.shape[0])]))
            start_index = end_index
            end_index = start_index+self._pid_mapping_bounds[bracket_index][2]
            bracket_index += 1
        return results

    @staticmethod
    def convert_list_of_arrays_to_static_list(pset):
        return [p
                for sublist in pset
                if sublist is not None
                for p in sublist]

    def convert_pset_to_static_list(self):
        return [p
                for sublist in self.particles_a
                if sublist is not None
                for p in sublist]

    def convert_pset_to_static_array(self):
        return np.array(self.convert_pset_to_static_list(), dtype=self._pclass)

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

    @property
    def size(self):
        nparticles = 0
        for bracket_index in range(len(self._pid_mapping_bounds)):
            nparticles += self._pid_mapping_bounds[bracket_index][2]
        return nparticles
        # return self.particles.size

    def __repr__(self):
        return "\n".join([str(p) for sublist in self._plist for p in sublist])

    def __len__(self):
        # return len(self._plist)
        return self.size

    def __getitem__(self, key):
        bracket_index = 0
        start_index = 0
        end_index = self._pid_mapping_bounds[bracket_index][2]
        # while start_index<key:
        # while end_index >= key:
        while key >= end_index:
            start_index = end_index
            end_index = start_index+self._pid_mapping_bounds[bracket_index][2]
            bracket_index += 1
            self.next()
        slot_index = key - start_index
        value = self.current_p_node[slot_index]
        self.reset_current_node()
        return value
    #     return self._plist[bracket_index][slot_index]
    #     return self.particles[key]

    def __setitem__(self, key, value):
        assert isinstance(key, tuple) and len(key) == 2, ("Error: trying to set/address data item without (bracket_index, slot_index) key.")
        # self._plist[key[0]][key[1]] = value
        for i in range(0, key[0]):
            self.next()
        self.current_p_node[key[1]] = value
        self.reset_current_node()
    # def __setitem__(self, key, value):
    #     self.particles[key] = value

    def __iadd__(self, particles):
        self.add(particles)
        return self

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

    def add(self, particles):
        """Method to add particles to the ParticleSet"""
        if isinstance(particles, ParticleSet):
            self._plist += particles.particles_a
            self._pid_mapping_bounds += particles.pid_mapping_bounds

            if self.ptype.uses_jit:
                for sublist in particles.particles_c:
                    self._plist_c.append(sublist)
        else:
            raise NotImplementedError('Only ParticleSets can be added to a ParticleSet')

        # if isinstance(particles, ParticleSet):
        #     particles = particles.particles
        # else:
        #     raise NotImplementedError('Only ParticleSets can be added to a ParticleSet')
        # self.particles = np.append(self.particles, particles)
        # if self.ptype.uses_jit:
        #     particles_data = [p._cptr for p in particles]
        #     self._particle_data = np.append(self._particle_data, particles_data)
        #     # Update C-pointer on particles
        #     for p, pdata in zip(self.particles, self._particle_data):
        #         p._cptr = pdata

    def _merge_brackets_(self):
        lw_bound_nlist = int(self.nlist_limit / 2)
        nmerges = 1
        while nmerges > 0:
            nmerges = 0
            trg_bracket = None
            trg_bracket_c = None
            trg_bracket_index = -1
            src_bracket = None
            src_bracket_c = None
            src_bracket_index = -1
            for bracket_index in range(len(self._plist)):
                if self._pid_mapping_bounds[bracket_index][2] < lw_bound_nlist:
                    if trg_bracket is None:
                        trg_bracket = self._plist[bracket_index]
                        if self._plist_c is not None and self.ptype.uses_jit:
                            trg_bracket_c = self._plist_c[bracket_index]
                        trg_bracket_index = bracket_index
                    elif src_bracket is None:
                        src_bracket = self._plist[bracket_index]
                        if self._plist_c is not None and self.ptype.uses_jit:
                            src_bracket_c = self._plist_c[bracket_index]
                        src_bracket_index = bracket_index
                if src_bracket is not None and trg_bracket is not None:
                    break
            if src_bracket is not None and trg_bracket is not None:
                for i in range(0, trg_bracket_index):
                    self.next()
                # self._plist[trg_bracket_index] = np.append(trg_bracket, src_bracket)
                self.current_p_node = np.append(trg_bracket, src_bracket)
                # local_ids = np.array([p.id for p in self._plist[trg_bracket_index]])
                local_ids = np.array([p.id for p in self.current_p_node])
                self.pid_mapping_bounds[trg_bracket_index] = (np.min(local_ids), np.max(local_ids), local_ids.shape[0])
                if self.ptype.uses_jit:
                    # self._plist_c[trg_bracket_index] = np.append(self._plist_c[trg_bracket_index], self._plist_c[src_bracket_index])
                    self.current_c_node = np.append(trg_bracket_c, src_bracket_c)
                    # for p, pdata in zip(self._plist[trg_bracket_index], self._plist_c[trg_bracket_index]):
                    for p, pdata in zip(self.current_p_node, self.current_c_node):
                        p._cptr = pdata
                self._plist.pop(src_bracket_index)
                # self._plist.remove(self._plist[src_bracket_index])
                # del self._plist[src_bracket_index]
                if self.ptype.uses_jit:
                    self._plist_c.pop(src_bracket_index)
                    # self._plist_c.remove(self._plist_c[src_bracket_index])
                    # del self._plist_c[src_bracket_index]
                # self._pid_mapping_bounds.remove(self._pid_mapping_bounds[src_bracket_index])
                del self._pid_mapping_bounds[src_bracket_index]
                nmerges += 1

    def pop(self, indices=-1):
        """
        Method to remove particles from the ParticleSet, based on their `indices`.
        Returns numpy.ndarray of removed particles.
        """
        lindices = indices
        if isinstance(indices, np.ndarray) or not isinstance(indices, list):
            ncounter = 0
            for bracket in self._pid_mapping_bounds:
                ncounter += bracket[2]
            if isinstance(indices, np.ndarray):
                indices = indices.tolist()
            if not isinstance(indices, list):
                indices = [indices, ]
            lindices = []
            for i in range(len(self._plist)):
                lindices.append(None)
            for gindex in indices:
                while gindex < 0:
                    gindex = ncounter+gindex
                (bracket_index, slot_index) = self.get_list_array_index(gindex)
                if lindices[bracket_index] is None:
                    lindices[bracket_index] = [slot_index, ]
                else:
                    lindices[bracket_index].append(slot_index)
        assert len(lindices) == len(self._plist), ("ParticleSet.remove() - particle set length ({}) does not match index list length ({})".format(
            len(self._plist), len(lindices)))
        result = None
        for bracket_index in range(len(lindices)):
            local_indices = lindices[bracket_index]
            if local_indices is None:
                continue
            if isinstance(local_indices, collections.Iterable):
                local_indices = np.array(local_indices)
            if result is None:
                result = self._remove_(bracket_index, local_indices)
            else:
                result = np.concatenate((result, self._remove_(bracket_index, local_indices)), axis=0)
        self._merge_brackets_()
        if result is None:
            return result
        result = result.tolist()
        if len(result) == 1:
            return result[0]
        return result

    def remove(self, indices):
        """Method to remove particles from the ParticleSet, based on their `indices`"""
        lindices = indices
        if isinstance(indices, np.ndarray) or not isinstance(indices, list):
            ncounter = 0
            for bracket in self._pid_mapping_bounds:
                ncounter += bracket[2]
            if isinstance(indices, np.ndarray):
                indices = indices.tolist()
            if not isinstance(indices, list):
                indices = [indices, ]
            lindices = []
            for i in range(len(self._plist)):
                lindices.append(None)
            for gindex in indices:
                while gindex < 0:
                    gindex = ncounter+gindex
                (bracket_index, slot_index) = self.get_list_array_index(gindex)
                if lindices[bracket_index] is None:
                    lindices[bracket_index] = [slot_index, ]
                else:
                    lindices[bracket_index].append(slot_index)
        assert len(lindices) == len(self._plist), ("ParticleSet.remove() - particle set length ({}) does not match index list length ({})".format(
            len(self._plist), len(lindices)))

        for bracket_index in range(len(lindices)):
            local_indices = lindices[bracket_index]
            if local_indices is None:
                continue
            if isinstance(local_indices, collections.Iterable):
                local_indices = np.array(local_indices)
            self._remove_(bracket_index, local_indices)
        self._merge_brackets_()

        # if isinstance(indices, collections.Iterable):
        #     particles = [self.particles[i] for i in indices]
        # else:
        #     particles = self.particles[indices]
        # self.particles = np.delete(self.particles, indices)
        # if self.ptype.uses_jit:
        #     self._particle_data = np.delete(self._particle_data, indices)
        #     # == Update C-pointer on particles == #
        #     for p, pdata in zip(self.particles, self._particle_data):
        #         p._cptr = pdata
        # return particles

    def remove_local_particles(self, bracket_index, local_indices):
        return self._remove_(bracket_index, local_indices)

    def _remove_(self, bracket_index, local_indices):
        # if isinstance(indices, collections.Iterable):
        #     particles = [self.particles[i] for i in indices]
        # else:
        #     particles = self.particles[indices]
        if isinstance(local_indices, collections.Iterable):
            local_indices = np.array(local_indices)
        for i in range(0, bracket_index):
            self.next()
        # return_p = self._plist[bracket_index][local_indices]
        return_p = self.current_p_node[local_indices]
        # self._plist[bracket_index] = np.delete(self._plist[bracket_index], local_indices)
        self.current_p_node = np.delete(self.current_p_node, local_indices)
        if self._ptype.uses_jit:
            # self._plist_c[bracket_index] = np.delete(self._plist_c[bracket_index], local_indices)
            self.current_c_node = np.delete(self.current_c_node, local_indices)
        # if self._plist[bracket_index] is not None and self._plist[bracket_index].shape[0] > 0:
        if self.current_p_node is not None and self.current_p_node.shape[0] > 0:
            # local_ids = np.array([p.id for p in self._plist[bracket_index]])
            local_ids = np.array([p.id for p in self.current_p_node])
            self.pid_mapping_bounds[bracket_index] = (np.min(local_ids), np.max(local_ids), local_ids.shape[0])
            if self._ptype.uses_jit:
                # == Update C-pointer on particles == #
                # -- OLD -- #
                # for p, pdata in zip(self._plist[bracket_index], self._plist_c[bracket_index]):
                #    p._cptr = pdata
                # -- END -- #
                # for slot_index in range(len(self._plist[bracket_index])):
                #     self._plist[bracket_index][slot_index].update_cptr(self._plist_c[bracket_index][slot_index])
                for slot_index in range(len(self.current_p_node)):
                    self.current_p_node[slot_index].update_cptr(self.current_p_node[slot_index])
        else:
            if len(self._plist) > 1:
                # self._plist.remove(self._plist[bracket_index])
                self._plist.remove(self.current_p_node)
                if self._ptype.uses_jit:
                    # self._plist_c.remove(self._plist_c[bracket_index])
                    self._plist_c.remove(self.current_c_node)
                self._pid_mapping_bounds.remove(self._pid_mapping_bounds[bracket_index])
            else:
                self.pid_mapping_bounds[bracket_index] = (np.iinfo(np.int32).max, np.iinfo(np.int32).min, 0)
        # return particles
        return return_p

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
                        It is either a timedelta object or a positive double.
        :param dt: Timestep interval to be passed to the kernel.
                   It is either a timedelta object or a double.
                   Use a negative value for a backward-in-time simulation.
        :param moviedt:  Interval for inner sub-timestepping (leap), which dictates
                         the update frequency of animation.
                         It is either a timedelta object or a positive double.
                         None value means no animation.
        :param output_file: :mod:`parcels.particlefile.ParticleFile` object for particle output
        :param recovery: Dictionary with additional `:mod:parcels.tools.error`
                         recovery kernels to allow custom recovery behaviour in case of
                         kernel errors.
        :param movie_background_field: field plotted as background in the movie if moviedt is set.
                                       'vector' shows the velocity as a vector field.
        :param verbose_progress: Boolean for providing a progress bar for the kernel execution loop.
        :param postIterationCallbacks: (Optional) Array of functions that are to be called after each iteration (post-process, non-Kernel)
        :param callbackdt: (Optional, in conjecture with 'postIterationCallbacks) timestep inverval to (latestly) interrupt the running kernel and invoke post-iteration callbacks from 'postIterationCallbacks'
        """

        # check if pyfunc has changed since last compile. If so, recompile
        if self.kernel is None or (self.kernel.pyfunc is not pyfunc and self.kernel is not pyfunc):
            # Generate and store Kernel
            if isinstance(pyfunc, Kernel):
                self.kernel = pyfunc
            else:
                self.kernel = self.Kernel(pyfunc)
            # Prepare JIT kernel execution
            if self._ptype.uses_jit:
                self.kernel.remove_lib()
                cppargs = ['-DDOUBLE_COORD_VARIABLES'] if self.lonlatdepth_dtype == np.float64 else None
                self.kernel.compile(compiler=GNUCompiler(cppargs=cppargs))
                self.kernel.load_lib()

        # Convert all time variables to seconds
        if isinstance(endtime, delta):
            raise RuntimeError('endtime must be either a datetime or a double')
        if isinstance(endtime, datetime):
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

        # Set particle.time defaults based on sign of dt, if not set at ParticleSet construction
        for sublist in self._plist:
            for p in sublist:
                if np.isnan(p.time):
                    mintime, maxtime = self.fieldset.gridset.dimrange('time_full')
                    p.time = mintime if dt >= 0 else maxtime

        # Derive _starttime and endtime from arguments or fieldset defaults
        if runtime is not None and endtime is not None:
            raise RuntimeError('Only one of (endtime, runtime) can be specified')
        # _starttime = min([p.time for p in self]) if dt >= 0 else max([p.time for p in self])
        _starttime = min([p.time for sublist in self._plist for p in sublist]) if dt >= 0 else max([p.time for sublist in self._plist for p in sublist])
        if self.repeatdt is not None and self.repeat_starttime is None:
            self.repeat_starttime = _starttime
        if runtime is not None:
            endtime = _starttime + runtime * np.sign(dt)
        elif endtime is None:
            mintime, maxtime = self.fieldset.gridset.dimrange('time_full')
            endtime = maxtime if dt >= 0 else mintime

        execute_once = False
        if abs(endtime-_starttime) < 1e-5 or dt == 0 or runtime == 0:
            dt = 0
            runtime = 0
            endtime = _starttime
            logger.warning_once("dt or runtime are zero, or endtime is equal to Particle.time. "
                                "The kernels will be executed once, without incrementing time")
            execute_once = True

        # Initialise particle timestepping
        for sublist in self._plist:
            for p in sublist:
                p.dt = dt

        # First write output_file, because particles could have been added
        if output_file:
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
        if self.repeatdt:
            next_prelease = self.repeat_starttime + (abs(time - self.repeat_starttime) // self.repeatdt + 1) * self.repeatdt * np.sign(dt)
        else:
            next_prelease = np.infty if dt > 0 else - np.infty
        next_output = time + outputdt if dt > 0 else time - outputdt
        next_movie = time + moviedt if dt > 0 else time - moviedt
        next_callback = time + callbackdt if dt > 0 else time - callbackdt
        next_input = self.fieldset.computeTimeChunk(time, np.sign(dt))

        tol = 1e-12
        if verbose_progress is None:
            walltime_start = time_module.time()
        if verbose_progress:
            pbar = self._create_progressbar_(_starttime, endtime)
        while (time < endtime and dt > 0) or (time > endtime and dt < 0) or dt == 0:
            pbar = None
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
            self.kernel.execute(self, endtime=time, dt=dt, recovery=recovery, output_file=output_file, execute_once=execute_once)
            logger.info("Computed time = {}".format(time))
            if abs(time-next_prelease) < tol:
                pset_new = ParticleSet(fieldset=self.fieldset, time=time, lon=self.repeatlon,
                                       lat=self.repeatlat, depth=self.repeatdepth,
                                       pclass=self.repeatpclass, lonlatdepth_dtype=self.lonlatdepth_dtype,
                                       partitions=False, pid_orig=self.repeatpid, **self.repeatkwargs)
                for sublist in pset_new.particles_a:
                    for p in sublist:
                        p.dt = dt
                self.add(pset_new)
                next_prelease += self.repeatdt * np.sign(dt)
            if abs(time-next_output) < tol:
                if output_file:
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
                next_input = self.fieldset.computeTimeChunk(time, dt)
            if dt == 0:
                break
            if verbose_progress:
                pbar.update(abs(time - _starttime))

        if output_file:
            output_file.write(self, time)
        if verbose_progress:
            pbar.finish()

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
            particle_val = [np.array([getattr(p, particle_val) for p in sublist]) for sublist in self._plist]
        else:
            particle_val = particle_val if particle_val else np.ones(self.size)
            particle_val = self.covert_static_array_to_list_array(particle_val)
        density = np.zeros((field.grid.lat.size, field.grid.lon.size), dtype=np.float32)
# =================================================================================================================== #
        for bracket_index, subfield in enumerate(self._plist):
            for slot_index, p in enumerate(subfield):
                try:  # breaks if either p.xi, p.yi, p.zi, p.ti do not exist (in scipy) or field not in fieldset
                    if p.ti[field.igrid] < 0:  # xi, yi, zi, ti, not initialised
                        raise('error')
                    xi = p.xi[field.igrid]
                    yi = p.yi[field.igrid]
                except:
                    _, _, _, xi, yi, _ = field.search_indices(p.lon, p.lat, p.depth, 0, 0, search2D=True)
                density[yi, xi] += particle_val[bracket_index][slot_index]

        # for pi, p in enumerate(self.particles):
        #     try:  # breaks if either p.xi, p.yi, p.zi, p.ti do not exist (in scipy) or field not in fieldset
        #         if p.ti[field.igrid] < 0:  # xi, yi, zi, ti, not initialised
        #             raise('error')
        #         xi = p.xi[field.igrid]
        #         yi = p.yi[field.igrid]
        #     except:
        #         _, _, _, xi, yi, _ = field.search_indices(p.lon, p.lat, p.depth, 0, 0, search2D=True)
        #     density[yi, xi] += particle_val[pi]

        if relative:
            psum = 0
            for bracket_index, subfield in enumerate(particle_val):
                psum += np.sum(particle_val[bracket_index])
            density /= psum
            # density /= np.sum(particle_val)

        if area_scale:
            density /= field.cell_areas()

        return density

    def Kernel(self, pyfunc, c_include="", delete_cfiles=True):
        """Wrapper method to convert a `pyfunc` into a :class:`parcels.kernel.Kernel` object
        based on `fieldset` and `ptype` of the ParticleSet
        :param delete_cfiles: Boolean whether to delete the C-files after compilation in JIT mode (default is True)
        """
        return Kernel(self.fieldset, self.ptype, pyfunc=pyfunc, c_include=c_include,
                      delete_cfiles=delete_cfiles)

    def ParticleFile(self, *args, **kwargs):
        """Wrapper method to initialise a :class:`parcels.particlefile.ParticleFile`
        object from the ParticleSet"""
        return ParticleFile(*args, particleset=self, **kwargs)
