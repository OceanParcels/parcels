import os
import sys
import warnings
from copy import copy
from datetime import date, datetime, timedelta

import cftime
import numpy as np
import xarray as xr
from scipy.spatial import KDTree
from tqdm import tqdm

from parcels._compat import MPI
from parcels.application_kernels.advection import AdvectionRK4
from parcels.compilation.codecompiler import GNUCompiler
from parcels.field import NestedField
from parcels.grid import CurvilinearGrid, GridType
from parcels.interaction.interactionkernel import InteractionKernel
from parcels.interaction.neighborsearch import (
    BruteFlatNeighborSearch,
    BruteSphericalNeighborSearch,
    HashSphericalNeighborSearch,
    KDTreeFlatNeighborSearch,
)
from parcels.kernel import Kernel
from parcels.particle import JITParticle, Variable
from parcels.particledata import ParticleData, ParticleDataIterator
from parcels.particlefile import ParticleFile
from parcels.tools._helpers import deprecated, deprecated_made_private
from parcels.tools.converters import _get_cftime_calendars, convert_to_flat_array
from parcels.tools.global_statics import get_package_dir
from parcels.tools.loggers import logger
from parcels.tools.statuscodes import StatusCode
from parcels.tools.warnings import FileWarning

__all__ = ["ParticleSet"]


def _convert_to_reltime(time):
    """Check to determine if the value of the time parameter needs to be converted to a relative value (relative to the time_origin)."""
    if isinstance(time, np.datetime64) or (hasattr(time, "calendar") and time.calendar in _get_cftime_calendars()):
        return True
    return False


class ParticleSet:
    """Class for storing particle and executing kernel over them.

    Please note that this currently only supports fixed size particle sets, meaning that the particle set only
    holds the particles defined on construction. Individual particles can neither be added nor deleted individually,
    and individual particles can only be deleted as a set procedurally (i.e. by 'particle.delete()'-call during
    kernel execution).

    Parameters
    ----------
    fieldset :
        mod:`parcels.fieldset.FieldSet` object from which to sample velocity.
    pclass : parcels.particle.JITParticle or parcels.particle.ScipyParticle
        Optional :mod:`parcels.particle.JITParticle` or
        :mod:`parcels.particle.ScipyParticle` object that defines custom particle
    lon :
        List of initial longitude values for particles
    lat :
        List of initial latitude values for particles
    depth :
        Optional list of initial depth values for particles. Default is 0m
    time :
        Optional list of initial time values for particles. Default is fieldset.U.grid.time[0]
    repeatdt : datetime.timedelta or float, optional
        Optional interval on which to repeat the release of the ParticleSet. Either timedelta object, or float in seconds.
    lonlatdepth_dtype :
        Floating precision for lon, lat, depth particle coordinates.
        It is either np.float32 or np.float64. Default is np.float32 if fieldset.U.interp_method is 'linear'
        and np.float64 if the interpolation method is 'cgrid_velocity'
    pid_orig :
        Optional list of (offsets for) the particle IDs
    partition_function :
        Function to use for partitioning particles over processors. Default is to use kMeans
    periodic_domain_zonal :
        Zonal domain size, used to apply zonally periodic boundaries for particle-particle
        interaction. If None, no zonally periodic boundaries are applied

        Other Variables can be initialised using further arguments (e.g. v=... for a Variable named 'v')
    """

    def __init__(
        self,
        fieldset,
        pclass=JITParticle,
        lon=None,
        lat=None,
        depth=None,
        time=None,
        repeatdt=None,
        lonlatdepth_dtype=None,
        pid_orig=None,
        interaction_distance=None,
        periodic_domain_zonal=None,
        **kwargs,
    ):
        self.particledata = None
        self._repeat_starttime = None
        self._repeatlon = None
        self._repeatlat = None
        self._repeatdepth = None
        self._repeatpclass = None
        self._repeatkwargs = None
        self._kernel = None
        self._interaction_kernel = None

        self.fieldset = fieldset
        self.fieldset._check_complete()
        self.time_origin = fieldset.time_origin

        # ==== first: create a new subclass of the pclass that includes the required variables ==== #
        # ==== see dynamic-instantiation trick here: https://www.python-course.eu/python3_classes_and_type.php ==== #
        class_name = pclass.__name__
        array_class = None
        if class_name not in dir():

            def ArrayClass_init(self, *args, **kwargs):
                fieldset = kwargs.get("fieldset", None)
                ngrids = kwargs.get("ngrids", None)
                if type(self).ngrids.initial < 0:
                    numgrids = ngrids
                    if numgrids is None and fieldset is not None:
                        numgrids = fieldset.gridset.size
                    assert numgrids is not None, "Neither fieldsets nor number of grids are specified - exiting."
                    type(self).ngrids.initial = numgrids
                self.ngrids = type(self).ngrids.initial
                if self.ngrids >= 0:
                    for index in ["xi", "yi", "zi", "ti"]:
                        if index != "ti":
                            setattr(self, index, np.zeros(self.ngrids, dtype=np.int32))
                        else:
                            setattr(self, index, -1 * np.ones(self.ngrids, dtype=np.int32))
                super(type(self), self).__init__(*args, **kwargs)

            array_class_vdict = {
                "ngrids": Variable("ngrids", dtype=np.int32, to_write=False, initial=-1),
                "xi": Variable("xi", dtype=np.int32, to_write=False),
                "yi": Variable("yi", dtype=np.int32, to_write=False),
                "zi": Variable("zi", dtype=np.int32, to_write=False),
                "ti": Variable("ti", dtype=np.int32, to_write=False, initial=-1),
                "__init__": ArrayClass_init,
            }
            array_class = type(class_name, (pclass,), array_class_vdict)
        else:
            array_class = locals()[class_name]
        # ==== dynamic re-classing completed ==== #
        _pclass = array_class

        lon = np.empty(shape=0) if lon is None else convert_to_flat_array(lon)
        lat = np.empty(shape=0) if lat is None else convert_to_flat_array(lat)

        if isinstance(pid_orig, (type(None), bool)):
            pid_orig = np.arange(lon.size)

        if depth is None:
            mindepth = self.fieldset.gridset.dimrange("depth")[0]
            depth = np.ones(lon.size) * mindepth
        else:
            depth = convert_to_flat_array(depth)
        assert lon.size == lat.size and lon.size == depth.size, "lon, lat, depth don't all have the same lenghts"

        time = convert_to_flat_array(time)
        time = np.repeat(time, lon.size) if time.size == 1 else time

        if time.size > 0 and type(time[0]) in [datetime, date]:
            time = np.array([np.datetime64(t) for t in time])
        if time.size > 0 and isinstance(time[0], np.timedelta64) and not self.time_origin:
            raise NotImplementedError("If fieldset.time_origin is not a date, time of a particle must be a double")
        time = np.array([self.time_origin.reltime(t) if _convert_to_reltime(t) else t for t in time])
        assert lon.size == time.size, "time and positions (lon, lat, depth) do not have the same lengths."

        if lonlatdepth_dtype is None:
            lonlatdepth_dtype = self.lonlatdepth_dtype_from_field_interp_method(fieldset.U)
        assert lonlatdepth_dtype in [
            np.float32,
            np.float64,
        ], "lon lat depth precision should be set to either np.float32 or np.float64"

        for kwvar in kwargs:
            if kwvar not in ["partition_function"]:
                kwargs[kwvar] = convert_to_flat_array(kwargs[kwvar])
                assert (
                    lon.size == kwargs[kwvar].size
                ), f"{kwvar} and positions (lon, lat, depth) don't have the same lengths."

        self.repeatdt = repeatdt.total_seconds() if isinstance(repeatdt, timedelta) else repeatdt
        if self.repeatdt:
            if self.repeatdt <= 0:
                raise "Repeatdt should be > 0"
            if time[0] and not np.allclose(time, time[0]):
                raise "All Particle.time should be the same when repeatdt is not None"
            self._repeatpclass = pclass
            self._repeatkwargs = kwargs
            self._repeatkwargs.pop("partition_function", None)

        ngrids = fieldset.gridset.size

        # Variables used for interaction kernels.
        inter_dist_horiz = None
        inter_dist_vert = None
        # The _dirty_neighbor attribute keeps track of whether
        # the neighbor search structure needs to be rebuilt.
        # If indices change (for example adding/deleting a particle)
        # The NS structure needs to be rebuilt and _dirty_neighbor should be
        # set to true. Since the NS structure isn't immediately initialized,
        # it is set to True here.
        self._dirty_neighbor = True

        self.particledata = ParticleData(
            _pclass,
            lon=lon,
            lat=lat,
            depth=depth,
            time=time,
            lonlatdepth_dtype=lonlatdepth_dtype,
            pid_orig=pid_orig,
            ngrid=ngrids,
            **kwargs,
        )

        # Initialize neighbor search data structure (used for interaction).
        if interaction_distance is not None:
            meshes = [g.mesh for g in fieldset.gridset.grids]
            # Assert all grids have the same mesh type
            assert np.all(np.array(meshes) == meshes[0])
            mesh_type = meshes[0]
            if mesh_type == "spherical":
                if len(self) < 1000:
                    interaction_class = BruteSphericalNeighborSearch
                else:
                    interaction_class = HashSphericalNeighborSearch
            elif mesh_type == "flat":
                if len(self) < 1000:
                    interaction_class = BruteFlatNeighborSearch
                else:
                    interaction_class = KDTreeFlatNeighborSearch
            else:
                assert False, "Interaction is only possible on 'flat' and 'spherical' meshes"
            try:
                if len(interaction_distance) == 2:
                    inter_dist_vert, inter_dist_horiz = interaction_distance
                else:
                    inter_dist_vert = interaction_distance[0]
                    inter_dist_horiz = interaction_distance[0]
            except TypeError:
                inter_dist_vert = interaction_distance
                inter_dist_horiz = interaction_distance
            self._neighbor_tree = interaction_class(
                inter_dist_vert=inter_dist_vert,
                inter_dist_horiz=inter_dist_horiz,
                periodic_domain_zonal=periodic_domain_zonal,
            )
        # End of neighbor search data structure initialization.

        if self.repeatdt:
            if len(time) > 0 and time[0] is None:
                self._repeat_starttime = time[0]
            else:
                if self.particledata.data["time"][0] and not np.allclose(
                    self.particledata.data["time"], self.particledata.data["time"][0]
                ):
                    raise ValueError("All Particle.time should be the same when repeatdt is not None")
                self._repeat_starttime = copy(self.particledata.data["time"][0])
            self._repeatlon = copy(self.particledata.data["lon"])
            self._repeatlat = copy(self.particledata.data["lat"])
            self._repeatdepth = copy(self.particledata.data["depth"])
            for kwvar in kwargs:
                if kwvar not in ["partition_function"]:
                    self._repeatkwargs[kwvar] = copy(self.particledata.data[kwvar])

        if self.repeatdt:
            if MPI and self.particledata.pu_indicators is not None:
                mpi_comm = MPI.COMM_WORLD
                mpi_rank = mpi_comm.Get_rank()
                self._repeatpid = pid_orig[self.particledata.pu_indicators == mpi_rank]

        self._kernel = None

    @property
    @deprecated_made_private  # TODO: Remove 6 months after v3.1.0
    def repeat_starttime(self):
        return self._repeat_starttime

    @property
    @deprecated_made_private  # TODO: Remove 6 months after v3.1.0
    def repeatlon(self):
        return self._repeatlon

    @property
    @deprecated_made_private  # TODO: Remove 6 months after v3.1.0
    def repeatlat(self):
        return self._repeatlat

    @property
    @deprecated_made_private  # TODO: Remove 6 months after v3.1.0
    def repeatdepth(self):
        return self._repeatdepth

    @property
    @deprecated_made_private  # TODO: Remove 6 months after v3.1.0
    def repeatpclass(self):
        return self._repeatpclass

    @property
    @deprecated_made_private  # TODO: Remove 6 months after v3.1.0
    def repeatkwargs(self):
        return self._repeatkwargs

    @property
    @deprecated_made_private  # TODO: Remove 6 months after v3.1.0
    def kernel(self):
        return self._kernel

    @property
    @deprecated_made_private  # TODO: Remove 6 months after v3.1.0
    def interaction_kernel(self):
        return self._interaction_kernel

    @property
    @deprecated_made_private  # TODO: Remove 6 months after v3.1.0
    def repeatpid(self):
        return self._repeatpid

    def __del__(self):
        if self.particledata is not None and isinstance(self.particledata, ParticleData):
            del self.particledata
        self.particledata = None

    @deprecated(
        "Use iter(pset) instead, or just use the object in an iterator context (e.g. for p in pset: ...)."
    )  # TODO: Remove 6 months after v3.1.0 (or 9 months; doesn't contribute to code debt)
    def iterator(self):
        return iter(self)

    def __iter__(self):
        return iter(self.particledata)

    def __getattr__(self, name):
        """
        Access a single property of all particles.

        Parameters
        ----------
        name : str
            Name of the property
        """
        for v in self.particledata.ptype.variables:
            if v.name == name:
                return getattr(self.particledata, name)
        if name in self.__dict__ and name[0] != "_":
            return self.__dict__[name]
        else:
            return False

    def __getitem__(self, index):
        """Get a single particle by index."""
        return self.particledata.get_single_by_index(index)

    @staticmethod
    def lonlatdepth_dtype_from_field_interp_method(field):
        if isinstance(field, NestedField):
            for f in field:
                if f.interp_method == "cgrid_velocity":
                    return np.float64
        else:
            if field.interp_method == "cgrid_velocity":
                return np.float64
        return np.float32

    def cstruct(self):
        cstruct = self.particledata.cstruct()
        return cstruct

    @property
    def ctypes_struct(self):
        return self.cstruct()

    @property
    def size(self):
        # ==== to change at some point - len and size are different things ==== #
        return len(self.particledata)

    def __repr__(self):
        return "\n".join([str(p) for p in self])

    def __len__(self):
        return len(self.particledata)

    def __sizeof__(self):
        return sys.getsizeof(self.particledata)

    def add(self, particles):
        """Add particles to the ParticleSet. Note that this is an
        incremental add, the particles will be added to the ParticleSet
        on which this function is called.

        Parameters
        ----------
        particles :
            Another ParticleSet containing particles to add to this one.

        Returns
        -------
        type
            The current ParticleSet

        """
        if isinstance(particles, type(self)):
            particles = particles.particledata
        self.particledata += particles
        # Adding particles invalidates the neighbor search structure.
        self._dirty_neighbor = True
        return self

    def __iadd__(self, particles):
        """Add particles to the ParticleSet.

        Note that this is an incremental add, the particles will be added to the ParticleSet
        on which this function is called.

        Parameters
        ----------
        particles :
            Another ParticleSet containing particles to add to this one.

        Returns
        -------
        type
            The current ParticleSet
        """
        self.add(particles)
        return self

    def remove_indices(self, indices):
        """Method to remove particles from the ParticleSet, based on their `indices`."""
        # Removing particles invalidates the neighbor search structure.
        self._dirty_neighbor = True
        if type(indices) in [int, np.int32, np.intp]:
            self.particledata.remove_single_by_index(indices)
        else:
            self.particledata.remove_multi_by_indices(indices)

    def remove_booleanvector(self, indices):
        """Method to remove particles from the ParticleSet, based on an array of booleans."""
        # Removing particles invalidates the neighbor search structure.
        self._dirty_neighbor = True
        self.remove_indices(np.where(indices)[0])

    @deprecated_made_private  # TODO: Remove 6 months after v3.1.0
    def active_particles_mask(self, *args, **kwargs):
        return self._active_particles_mask(*args, **kwargs)

    def _active_particles_mask(self, time, dt):
        active_indices = (time - self.particledata.data["time"]) / dt >= 0
        non_err_indices = np.isin(self.particledata.data["state"], [StatusCode.Success, StatusCode.Evaluate])
        active_indices = np.logical_and(active_indices, non_err_indices)
        self._active_particle_idx = np.where(active_indices)[0]
        return active_indices

    @deprecated_made_private  # TODO: Remove 6 months after v3.1.0
    def compute_neighbor_tree(self, *args, **kwargs):
        return self._compute_neighbor_tree(*args, **kwargs)

    def _compute_neighbor_tree(self, time, dt):
        active_mask = self._active_particles_mask(time, dt)

        self._values = np.vstack(
            (
                self.particledata.data["depth"],
                self.particledata.data["lat"],
                self.particledata.data["lon"],
            )
        )
        if self._dirty_neighbor:
            self._neighbor_tree.rebuild(self._values, active_mask=active_mask)
            self._dirty_neighbor = False
        else:
            self._neighbor_tree.update_values(self._values, new_active_mask=active_mask)

    @deprecated_made_private  # TODO: Remove 6 months after v3.1.0
    def neighbors_by_index(self, *args, **kwargs):
        return self._neighbors_by_index(*args, **kwargs)

    def _neighbors_by_index(self, particle_idx):
        neighbor_idx, distances = self._neighbor_tree.find_neighbors_by_idx(particle_idx)
        neighbor_idx = self._active_particle_idx[neighbor_idx]
        mask = neighbor_idx != particle_idx
        neighbor_idx = neighbor_idx[mask]
        if "horiz_dist" in self.particledata._ptype.variables:
            self.particledata.data["vert_dist"][neighbor_idx] = distances[0, mask]
            self.particledata.data["horiz_dist"][neighbor_idx] = distances[1, mask]
        return ParticleDataIterator(self.particledata, subset=neighbor_idx)

    @deprecated_made_private  # TODO: Remove 6 months after v3.1.0
    def neighbors_by_coor(self, *args, **kwargs):
        return self._neighbors_by_coor(*args, **kwargs)

    def _neighbors_by_coor(self, coor):
        neighbor_idx = self._neighbor_tree.find_neighbors_by_coor(coor)
        neighbor_ids = self.particledata.data["id"][neighbor_idx]
        return neighbor_ids

    # TODO: This method is only tested in tutorial notebook. Add unit test?
    def populate_indices(self):
        """Pre-populate guesses of particle xi/yi indices using a kdtree.

        This is only intended for curvilinear grids, where the initial index search
        may be quite expensive.
        """
        for i, grid in enumerate(self.fieldset.gridset.grids):
            if not isinstance(grid, CurvilinearGrid):
                continue

            tree_data = np.stack((grid.lon.flat, grid.lat.flat), axis=-1)
            IN = np.all(~np.isnan(tree_data), axis=1)
            tree = KDTree(tree_data[IN, :])
            # stack all the particle positions for a single query
            pts = np.stack((self.particledata.data["lon"], self.particledata.data["lat"]), axis=-1)
            # query datatype needs to match tree datatype
            _, idx_nan = tree.query(pts.astype(tree_data.dtype))

            idx = np.where(IN)[0][idx_nan]
            yi, xi = np.unravel_index(idx, grid.lon.shape)

            self.particledata.data["xi"][:, i] = xi
            self.particledata.data["yi"][:, i] = yi

    @classmethod
    def from_list(
        cls, fieldset, pclass, lon, lat, depth=None, time=None, repeatdt=None, lonlatdepth_dtype=None, **kwargs
    ):
        """Initialise the ParticleSet from lists of lon and lat.

        Parameters
        ----------
        fieldset :
            mod:`parcels.fieldset.FieldSet` object from which to sample velocity
        pclass : parcels.particle.JITParticle or parcels.particle.ScipyParticle
            Particle class. May be a particle class as defined in parcels, or a subclass defining a custom particle.
        lon :
            List of initial longitude values for particles
        lat :
            List of initial latitude values for particles
        depth :
            Optional list of initial depth values for particles. Default is 0m
        time :
            Optional list of start time values for particles. Default is fieldset.U.time[0]
        repeatdt :
            Optional interval (in seconds) on which to repeat the release of the ParticleSet (Default value = None)
        lonlatdepth_dtype :
            Floating precision for lon, lat, depth particle coordinates.
            It is either np.float32 or np.float64. Default is np.float32 if fieldset.U.interp_method is 'linear'
            and np.float64 if the interpolation method is 'cgrid_velocity'
            Other Variables can be initialised using further arguments (e.g. v=... for a Variable named 'v')
        **kwargs :
            Keyword arguments passed to the particleset constructor.
        """
        return cls(
            fieldset=fieldset,
            pclass=pclass,
            lon=lon,
            lat=lat,
            depth=depth,
            time=time,
            repeatdt=repeatdt,
            lonlatdepth_dtype=lonlatdepth_dtype,
            **kwargs,
        )

    @classmethod
    def from_line(
        cls,
        fieldset,
        pclass,
        start,
        finish,
        size,
        depth=None,
        time=None,
        repeatdt=None,
        lonlatdepth_dtype=None,
        **kwargs,
    ):
        """Create a particleset in the shape of a line (according to a cartesian grid).

        Initialise the ParticleSet from start/finish coordinates with equidistant spacing
        Note that this method uses simple numpy.linspace calls and does not take into account
        great circles, so may not be a exact on a globe

        Parameters
        ----------
        fieldset :
            mod:`parcels.fieldset.FieldSet` object from which to sample velocity
        pclass : parcels.particle.JITParticle or parcels.particle.ScipyParticle
            Particle class. May be a particle class as defined in parcels, or a subclass defining a custom particle.
        start :
            Start point (longitude, latitude) for initialisation of particles on a straight line.
        finish :
            End point (longitude, latitude) for initialisation of particles on a straight line.
        size :
            Initial size of particle set
        depth :
            Optional list of initial depth values for particles. Default is 0m
        time :
            Optional start time value for particles. Default is fieldset.U.time[0]
        repeatdt :
            Optional interval (in seconds) on which to repeat the release of the ParticleSet (Default value = None)
        lonlatdepth_dtype :
            Floating precision for lon, lat, depth particle coordinates.
            It is either np.float32 or np.float64. Default is np.float32 if fieldset.U.interp_method is 'linear'
            and np.float64 if the interpolation method is 'cgrid_velocity'
        """
        lon = np.linspace(start[0], finish[0], size)
        lat = np.linspace(start[1], finish[1], size)
        if type(depth) in [int, float]:
            depth = [depth] * size
        return cls(
            fieldset=fieldset,
            pclass=pclass,
            lon=lon,
            lat=lat,
            depth=depth,
            time=time,
            repeatdt=repeatdt,
            lonlatdepth_dtype=lonlatdepth_dtype,
            **kwargs,
        )

    @classmethod
    @deprecated_made_private  # TODO: Remove 6 months after v3.1.0
    def monte_carlo_sample(self, *args, **kwargs):
        return self._monte_carlo_sample(*args, **kwargs)

    @classmethod
    def _monte_carlo_sample(cls, start_field, size, mode="monte_carlo"):
        """Converts a starting field into a monte-carlo sample of lons and lats.

        Parameters
        ----------
        start_field : parcels.field.Field
            mod:`parcels.fieldset.Field` object for initialising particles stochastically (horizontally)  according to the presented density field.
        size :

        mode :
             (Default value = 'monte_carlo')

        Returns
        -------
        list of float
            A list of longitude values.
        list of float
            A list of latitude values.
        """
        if mode == "monte_carlo":
            data = start_field.data if isinstance(start_field.data, np.ndarray) else np.array(start_field.data)
            if start_field.interp_method == "cgrid_tracer":
                p_interior = np.squeeze(data[0, 1:, 1:])
            else:  # if A-grid
                d = data
                p_interior = (d[0, :-1, :-1] + d[0, 1:, :-1] + d[0, :-1, 1:] + d[0, 1:, 1:]) / 4.0
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
            lon, lat = ([], [])
            if grid._gtype in [GridType.RectilinearZGrid, GridType.RectilinearSGrid]:
                lon = grid.lon[i] + xsi * (grid.lon[i + 1] - grid.lon[i])
                lat = grid.lat[j] + eta * (grid.lat[j + 1] - grid.lat[j])
            else:
                lons = np.array([grid.lon[j, i], grid.lon[j, i + 1], grid.lon[j + 1, i + 1], grid.lon[j + 1, i]])
                if grid.mesh == "spherical":
                    lons[1:] = np.where(lons[1:] - lons[0] > 180, lons[1:] - 360, lons[1:])
                    lons[1:] = np.where(-lons[1:] + lons[0] > 180, lons[1:] + 360, lons[1:])
                lon = (
                    (1 - xsi) * (1 - eta) * lons[0]
                    + xsi * (1 - eta) * lons[1]
                    + xsi * eta * lons[2]
                    + (1 - xsi) * eta * lons[3]
                )
                lat = (
                    (1 - xsi) * (1 - eta) * grid.lat[j, i]
                    + xsi * (1 - eta) * grid.lat[j, i + 1]
                    + xsi * eta * grid.lat[j + 1, i + 1]
                    + (1 - xsi) * eta * grid.lat[j + 1, i]
                )
            return list(lon), list(lat)
        else:
            raise NotImplementedError(f'Mode {mode} not implemented. Please use "monte carlo" algorithm instead.')

    @classmethod
    def from_field(
        cls,
        fieldset,
        pclass,
        start_field,
        size,
        mode="monte_carlo",
        depth=None,
        time=None,
        repeatdt=None,
        lonlatdepth_dtype=None,
    ):
        """Initialise the ParticleSet randomly drawn according to distribution from a field.

        Parameters
        ----------
        fieldset : parcels.fieldset.FieldSet
            mod:`parcels.fieldset.FieldSet` object from which to sample velocity
        pclass : parcels.particle.JITParticle or parcels.particle.ScipyParticle
            Particle class. May be a particle class as defined in parcels, or a subclass defining a custom particle.
        start_field : parcels.field.Field
            Field for initialising particles stochastically (horizontally)  according to the presented density field.
        size :
            Initial size of particle set
        mode :
            Type of random sampling. Currently only 'monte_carlo' is implemented (Default value = 'monte_carlo')
        depth :
            Optional list of initial depth values for particles. Default is 0m
        time :
            Optional start time value for particles. Default is fieldset.U.time[0]
        repeatdt :
            Optional interval (in seconds) on which to repeat the release of the ParticleSet (Default value = None)
        lonlatdepth_dtype :
            Floating precision for lon, lat, depth particle coordinates.
            It is either np.float32 or np.float64. Default is np.float32 if fieldset.U.interp_method is 'linear'
            and np.float64 if the interpolation method is 'cgrid_velocity'
        """
        lon, lat = cls._monte_carlo_sample(start_field, size, mode)

        return cls(
            fieldset=fieldset,
            pclass=pclass,
            lon=lon,
            lat=lat,
            depth=depth,
            time=time,
            lonlatdepth_dtype=lonlatdepth_dtype,
            repeatdt=repeatdt,
        )

    @classmethod
    def from_particlefile(
        cls, fieldset, pclass, filename, restart=True, restarttime=None, repeatdt=None, lonlatdepth_dtype=None, **kwargs
    ):
        """Initialise the ParticleSet from a zarr ParticleFile.
        This creates a new ParticleSet based on locations of all particles written
        in a zarr ParticleFile at a certain time. Particle IDs are preserved if restart=True

        Parameters
        ----------
        fieldset : parcels.fieldset.FieldSet
            mod:`parcels.fieldset.FieldSet` object from which to sample velocity
        pclass : parcels.particle.JITParticle or parcels.particle.ScipyParticle
            Particle class. May be a particle class as defined in parcels, or a subclass defining a custom particle.
        filename : str
            Name of the particlefile from which to read initial conditions
        restart : bool
            BSignal if pset is used for a restart (default is True).
            In that case, Particle IDs are preserved.
        restarttime :
            time at which the Particles will be restarted. Default is the last time written.
            Alternatively, restarttime could be a time value (including np.datetime64) or
            a callable function such as np.nanmin. The last is useful when running with dt < 0.
        repeatdt : datetime.timedelta or float, optional
            Optional interval on which to repeat the release of the ParticleSet. Either timedelta object, or float in seconds.
        lonlatdepth_dtype :
            Floating precision for lon, lat, depth particle coordinates.
            It is either np.float32 or np.float64. Default is np.float32 if fieldset.U.interp_method is 'linear'
            and np.float64 if the interpolation method is 'cgrid_velocity'
        **kwargs :
            Keyword arguments passed to the particleset constructor.
        """
        if repeatdt is not None:
            warnings.warn(
                f"Note that the `repeatdt` argument is not retained from {filename}, and that "
                "setting a new repeatdt will start particles from the _new_ particle "
                "locations.",
                FileWarning,
                stacklevel=2,
            )

        pfile = xr.open_zarr(str(filename))
        pfile_vars = [v for v in pfile.data_vars]

        vars = {}
        to_write = {}
        for v in pclass.getPType().variables:
            if v.name in pfile_vars:
                vars[v.name] = np.ma.filled(pfile.variables[v.name], np.nan)
            elif (
                v.name
                not in [
                    "xi",
                    "yi",
                    "zi",
                    "ti",
                    "dt",
                    "depth",
                    "id",
                    "obs_written",
                    "state",
                    "lon_nextloop",
                    "lat_nextloop",
                    "depth_nextloop",
                    "time_nextloop",
                ]
                and v.to_write
            ):
                raise RuntimeError(f"Variable {v.name} is in pclass but not in the particlefile")
            to_write[v.name] = v.to_write
        vars["depth"] = np.ma.filled(pfile.variables["z"], np.nan)
        vars["id"] = np.ma.filled(pfile.variables["trajectory"], np.nan)

        for v in ["lon", "lat", "depth", "time"]:
            to_write[v] = True

        if isinstance(vars["time"][0, 0], np.timedelta64):
            vars["time"] = np.array([t / np.timedelta64(1, "s") for t in vars["time"]])

        if restarttime is None:
            restarttime = np.nanmax(vars["time"])
        elif callable(restarttime):
            restarttime = restarttime(vars["time"])
        else:
            restarttime = restarttime

        inds = np.where(vars["time"] == restarttime)
        for v in vars:
            if to_write[v] is True:
                vars[v] = vars[v][inds]
            elif to_write[v] == "once":
                vars[v] = vars[v][inds[0]]
            if v not in ["lon", "lat", "depth", "time", "id"]:
                kwargs[v] = vars[v]

        if restart:
            pclass.setLastID(0)  # reset to zero offset
        else:
            vars["id"] = None

        return cls(
            fieldset=fieldset,
            pclass=pclass,
            lon=vars["lon"],
            lat=vars["lat"],
            depth=vars["depth"],
            time=vars["time"],
            pid_orig=vars["id"],
            lonlatdepth_dtype=lonlatdepth_dtype,
            repeatdt=repeatdt,
            **kwargs,
        )

    def Kernel(self, pyfunc, c_include="", delete_cfiles=True):
        """Wrapper method to convert a `pyfunc` into a :class:`parcels.kernel.Kernel` object.

        Conversion is based on `fieldset` and `ptype` of the ParticleSet.

        Parameters
        ----------
        pyfunc : function or list of functions
            Python function to convert into kernel. If a list of functions is provided,
            the functions will be converted to kernels and combined into a single kernel.
        delete_cfiles : bool
            Whether to delete the C-files after compilation in JIT mode (default is True)
        pyfunc :

        c_include :
             (Default value = "")
        """
        if isinstance(pyfunc, list):
            return Kernel.from_list(
                self.fieldset,
                self.particledata.ptype,
                pyfunc,
                c_include=c_include,
                delete_cfiles=delete_cfiles,
            )
        return Kernel(
            self.fieldset,
            self.particledata.ptype,
            pyfunc=pyfunc,
            c_include=c_include,
            delete_cfiles=delete_cfiles,
        )

    def InteractionKernel(self, pyfunc_inter, delete_cfiles=True):
        if pyfunc_inter is None:
            return None
        return InteractionKernel(
            self.fieldset, self.particledata.ptype, pyfunc=pyfunc_inter, delete_cfiles=delete_cfiles
        )

    def ParticleFile(self, *args, **kwargs):
        """Wrapper method to initialise a :class:`parcels.particlefile.ParticleFile` object from the ParticleSet."""
        return ParticleFile(*args, particleset=self, **kwargs)

    def data_indices(self, variable_name, compare_values, invert=False):
        """Get the indices of all particles where the value of `variable_name` equals (one of) `compare_values`.

        Parameters
        ----------
        variable_name : str
            Name of the variable to check.
        compare_values :
            Value or list of values to compare to.
        invert :
            Whether to invert the selection. I.e., when True,
            return all indices that do not equal (one of)
            `compare_values`. (Default value = False)

        Returns
        -------
        np.ndarray
            Numpy array of indices that satisfy the test.

        """
        compare_values = (
            np.array([compare_values]) if type(compare_values) not in [list, dict, np.ndarray] else compare_values
        )
        return np.where(np.isin(self.particledata.data[variable_name], compare_values, invert=invert))[0]

    @property
    @deprecated_made_private  # TODO: Remove 6 months after v3.1.0
    def error_particles(self):
        return self._error_particles

    @property
    def _error_particles(self):
        """Get an iterator over all particles that are in an error state.

        Returns
        -------
        iterator
            ParticleDataIterator over error particles.
        """
        error_indices = self.data_indices("state", [StatusCode.Success, StatusCode.Evaluate], invert=True)
        return ParticleDataIterator(self.particledata, subset=error_indices)

    @property
    @deprecated_made_private  # TODO: Remove 6 months after v3.1.0
    def num_error_particles(self):
        return self._num_error_particles

    @property
    def _num_error_particles(self):
        """Get the number of particles that are in an error state.

        Returns
        -------
        int
            Number of error particles.
        """
        return np.sum(np.isin(self.particledata.data["state"], [StatusCode.Success, StatusCode.Evaluate], invert=True))

    def set_variable_write_status(self, var, write_status):
        """Method to set the write status of a Variable.

        Parameters
        ----------
        var :
            Name of the variable (string)
        write_status :
            Write status of the variable (True, False or 'once')
        """
        self.particledata.set_variable_write_status(var, write_status)

    def execute(
        self,
        pyfunc=AdvectionRK4,
        pyfunc_inter=None,
        endtime=None,
        runtime=None,
        dt=1.0,
        output_file=None,
        verbose_progress=True,
        postIterationCallbacks=None,
        callbackdt=None,
        delete_cfiles=True,
    ):
        """Execute a given kernel function over the particle set for multiple timesteps.

        Optionally also provide sub-timestepping
        for particle output.

        Parameters
        ----------
        pyfunc :
            Kernel function to execute. This can be the name of a
            defined Python function or a :class:`parcels.kernel.Kernel` object.
            Kernels can be concatenated using the + operator (Default value = AdvectionRK4)
        endtime :
            End time for the timestepping loop.
            It is either a datetime object or a positive double. (Default value = None)
        runtime :
            Length of the timestepping loop. Use instead of endtime.
            It is either a timedelta object or a positive double. (Default value = None)
        dt :
            Timestep interval (in seconds) to be passed to the kernel.
            It is either a timedelta object or a double.
            Use a negative value for a backward-in-time simulation. (Default value = 1 second)
        output_file :
            mod:`parcels.particlefile.ParticleFile` object for particle output (Default value = None)
        verbose_progress : bool
            Boolean for providing a progress bar for the kernel execution loop. (Default value = True)
        postIterationCallbacks :
            Optional, array of functions that are to be called after each iteration (post-process, non-Kernel) (Default value = None)
        callbackdt :
            Optional, in conjecture with 'postIterationCallbacks', timestep interval to (latest) interrupt the running kernel and invoke post-iteration callbacks from 'postIterationCallbacks' (Default value = None)
        pyfunc_inter :
            (Default value = None)
        delete_cfiles : bool
            Whether to delete the C-files after compilation in JIT mode (default is True)

        Notes
        -----
        ``ParticleSet.execute()`` acts as the main entrypoint for simulations, and provides the simulation time-loop. This method encapsulates the logic controlling the switching between kernel execution (where control in handed to C in JIT mode), output file writing, reading in fields for new timesteps, adding new particles to the simulation domain, stopping the simulation, and executing custom functions (``postIterationCallbacks`` provided by the user).
        """
        # check if particleset is empty. If so, return immediately
        if len(self) == 0:
            return

        # check if pyfunc has changed since last compile. If so, recompile
        if self._kernel is None or (self._kernel.pyfunc is not pyfunc and self._kernel is not pyfunc):
            # Generate and store Kernel
            if isinstance(pyfunc, Kernel):
                self._kernel = pyfunc
            else:
                self._kernel = self.Kernel(pyfunc, delete_cfiles=delete_cfiles)
            # Prepare JIT kernel execution
            if self.particledata.ptype.uses_jit:
                self._kernel.remove_lib()
                cppargs = ["-DDOUBLE_COORD_VARIABLES"] if self.particledata.lonlatdepth_dtype else None
                self._kernel.compile(
                    compiler=GNUCompiler(cppargs=cppargs, incdirs=[os.path.join(get_package_dir(), "include"), "."])
                )
                self._kernel.load_lib()
        if output_file:
            output_file.add_metadata("parcels_kernels", self._kernel.name)

        # Set up the interaction kernel(s) if not set and given.
        if self._interaction_kernel is None and pyfunc_inter is not None:
            if isinstance(pyfunc_inter, InteractionKernel):
                self._interaction_kernel = pyfunc_inter
            else:
                self._interaction_kernel = self.InteractionKernel(pyfunc_inter, delete_cfiles=delete_cfiles)

        # Convert all time variables to seconds
        if isinstance(endtime, timedelta):
            raise TypeError("endtime must be either a datetime or a double")
        if isinstance(endtime, datetime):
            endtime = np.datetime64(endtime)
        elif isinstance(endtime, cftime.datetime):
            endtime = self.time_origin.reltime(endtime)
        if isinstance(endtime, np.datetime64):
            if self.time_origin.calendar is None:
                raise NotImplementedError("If fieldset.time_origin is not a date, execution endtime must be a double")
            endtime = self.time_origin.reltime(endtime)
        if isinstance(runtime, timedelta):
            runtime = runtime.total_seconds()
        if isinstance(dt, timedelta):
            dt = dt.total_seconds()
        if abs(dt) <= 1e-6:
            raise ValueError("Time step dt is too small")
        if (dt * 1e6) % 1 != 0:
            raise ValueError("Output interval should not have finer precision than 1e-6 s")
        outputdt = output_file.outputdt if output_file else np.inf
        if isinstance(outputdt, timedelta):
            outputdt = outputdt.total_seconds()
        if isinstance(callbackdt, timedelta):
            callbackdt = callbackdt.total_seconds()

        assert runtime is None or runtime >= 0, "runtime must be positive"
        assert outputdt is None or outputdt >= 0, "outputdt must be positive"

        if runtime is not None and endtime is not None:
            raise RuntimeError("Only one of (endtime, runtime) can be specified")

        mintime, maxtime = self.fieldset.gridset.dimrange("time_full")

        default_release_time = mintime if dt >= 0 else maxtime
        if np.any(np.isnan(self.particledata.data["time"])):
            self.particledata.data["time"][np.isnan(self.particledata.data["time"])] = default_release_time
            self.particledata.data["time_nextloop"][np.isnan(self.particledata.data["time_nextloop"])] = (
                default_release_time
            )
        min_rt = np.min(self.particledata.data["time_nextloop"])
        max_rt = np.max(self.particledata.data["time_nextloop"])

        # Derive starttime and endtime from arguments or fieldset defaults
        starttime = min_rt if dt >= 0 else max_rt
        if self.repeatdt is not None and self._repeat_starttime is None:
            self._repeat_starttime = starttime
        if runtime is not None:
            endtime = starttime + runtime * np.sign(dt)
        elif endtime is None:
            mintime, maxtime = self.fieldset.gridset.dimrange("time_full")
            endtime = maxtime if dt >= 0 else mintime

        if (abs(endtime - starttime) < 1e-5 or runtime == 0) and dt == 0:
            raise RuntimeError(
                "dt and runtime are zero, or endtime is equal to Particle.time. "
                "ParticleSet.execute() will not do anything."
            )

        self.particledata._data["dt"][:] = dt

        if callbackdt is None:
            interupt_dts = [np.inf, outputdt]
            if self.repeatdt is not None:
                interupt_dts.append(self.repeatdt)
            callbackdt = np.min(np.array(interupt_dts))

        # Set up pbar
        if output_file:
            logger.info(f"Output files are stored in {output_file.fname}.")

        if verbose_progress:
            pbar = tqdm(total=abs(endtime - starttime), file=sys.stdout)

        # Set up variables for first iteration
        if self.repeatdt:
            next_prelease = self._repeat_starttime + (
                abs(starttime - self._repeat_starttime) // self.repeatdt + 1
            ) * self.repeatdt * np.sign(dt)
        else:
            next_prelease = np.inf if dt > 0 else -np.inf
        if output_file:
            next_output = starttime + dt
        else:
            next_output = np.inf * np.sign(dt)
        next_callback = starttime + callbackdt * np.sign(dt)

        tol = 1e-12
        time = starttime

        while (time < endtime and dt > 0) or (time > endtime and dt < 0):
            time_at_startofloop = time

            next_input = self.fieldset.computeTimeChunk(time, dt)

            # Define next_time (the timestamp when the execution needs to be handed back to python)
            if dt > 0:
                next_time = min(next_prelease, next_input, next_output, next_callback, endtime)
            else:
                next_time = max(next_prelease, next_input, next_output, next_callback, endtime)

            # If we don't perform interaction, only execute the normal kernel efficiently.
            if self._interaction_kernel is None:
                res = self._kernel.execute(self, endtime=next_time, dt=dt)
                if res == StatusCode.StopAllExecution:
                    return StatusCode.StopAllExecution
            # Interaction: interleave the interaction and non-interaction kernel for each time step.
            # E.g. Normal -> Inter -> Normal -> Inter if endtime-time == 2*dt
            else:
                cur_time = time
                while (cur_time < next_time and dt > 0) or (cur_time > next_time and dt < 0):
                    if dt > 0:
                        cur_end_time = min(cur_time + dt, next_time)
                    else:
                        cur_end_time = max(cur_time + dt, next_time)
                    self._kernel.execute(self, endtime=cur_end_time, dt=dt)
                    self._interaction_kernel.execute(self, endtime=cur_end_time, dt=dt)
                    cur_time += dt
            # End of interaction specific code
            time = next_time

            if abs(time - next_output) < tol:
                for fld in self.fieldset.get_fields():
                    if hasattr(fld, "to_write") and fld.to_write:
                        if fld.grid.tdim > 1:
                            raise RuntimeError(
                                "Field writing during execution only works for Fields with one snapshot in time"
                            )
                        fldfilename = str(output_file.fname).replace(".zarr", "_%.4d" % fld.to_write)
                        fld.write(fldfilename)
                        fld.to_write += 1

            if abs(time - next_output) < tol:
                if output_file:
                    if output_file.analytical:  # output analytical solution at later time
                        output_file.write_latest_locations(self, time)
                    else:
                        output_file.write(self, time_at_startofloop)
                if np.isfinite(outputdt):
                    next_output += outputdt * np.sign(dt)

            # ==== insert post-process here to also allow for memory clean-up via external func ==== #
            if abs(time - next_callback) < tol:
                if postIterationCallbacks is not None:
                    for extFunc in postIterationCallbacks:
                        extFunc()
                next_callback += callbackdt * np.sign(dt)

            if abs(time - next_prelease) < tol:
                pset_new = self.__class__(
                    fieldset=self.fieldset,
                    time=time,
                    lon=self._repeatlon,
                    lat=self._repeatlat,
                    depth=self._repeatdepth,
                    pclass=self._repeatpclass,
                    lonlatdepth_dtype=self.particledata.lonlatdepth_dtype,
                    partition_function=False,
                    pid_orig=self._repeatpid,
                    **self._repeatkwargs,
                )
                for p in pset_new:
                    p.dt = dt
                self.add(pset_new)
                next_prelease += self.repeatdt * np.sign(dt)

            if time != endtime:
                next_input = self.fieldset.computeTimeChunk(time, dt)
            if verbose_progress:
                pbar.update(abs(time - time_at_startofloop))

        if verbose_progress:
            pbar.close()
