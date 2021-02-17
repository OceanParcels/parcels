import functools
from ctypes import c_double
from ctypes import c_float
from ctypes import c_int
from ctypes import c_void_p
from ctypes import cast
from ctypes import POINTER
from ctypes import pointer
from ctypes import Structure
from enum import IntEnum

import numpy as np

from parcels.tools.converters import TimeConverter
from parcels.tools.loggers import logger

__all__ = ['GridCode', 'RectilinearZGrid', 'RectilinearSGrid', 'CurvilinearZGrid', 'CurvilinearSGrid', 'CGrid', 'Grid']


class GridCode(IntEnum):
    RectilinearZGrid = 0
    RectilinearSGrid = 1
    CurvilinearZGrid = 2
    CurvilinearSGrid = 3


class CGrid(Structure):
    _fields_ = [('gtype', c_int),
                ('grid', c_void_p)]


class Grid(object):
    """Grid class that defines a (spatial and temporal) grid on which Fields are defined

    """

    def __init__(self, lon, lat, time, time_origin, mesh):
        self.lon = lon  # should be a C-contiguous array of floats
        if not self.lon.flags['C_CONTIGUOUS']:
            self.lon = np.array(self.lon, order='C')
        self.lat = lat  # should be a C-contiguous array of floats
        if not self.lat.flags['C_CONTIGUOUS']:
            self.lat = np.array(self.lat, order='C')
        self.time = np.zeros(1, dtype=np.float64) if time is None else time # should be a C-contiguous array of float64
        if not self.time.flags['C_CONTIGUOUS']:
            self.time = np.array(self.time, order='C')
        if not self.lon.dtype == np.float32:
            logger.warning_once("Casting lon data to np.float32")
            self.lon = self.lon.astype(np.float32)
        if not self.lat.dtype == np.float32:
            logger.warning_once("Casting lat data to np.float32")
            self.lat = self.lat.astype(np.float32)
        if not self.time.dtype == np.float64:
            assert isinstance(self.time[0], (np.integer, np.floating, float, int)), 'Time vector must be an array of int or floats'
            logger.warning_once("Casting time data to np.float64")
            self.time = self.time.astype(np.float64)
        self.time_full = self.time  # needed for deferred_loaded Fields
        self.time_origin = TimeConverter() if time_origin is None else time_origin
        assert isinstance(self.time_origin, TimeConverter), 'time_origin needs to be a TimeConverter object'
        self.mesh = mesh
        self.cstruct = None
        self.cell_edge_sizes = {}
        self.zonal_periodic = False
        self.zonal_halo = 0
        self.meridional_halo = 0
        self.lat_flipped = False
        self.defer_load = False
        self.lonlat_minmax = np.array([np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)], dtype=np.float32)
        self.periods = 0
        self.load_chunk = []    # should be a C-contiguous array of ints
        self.chunk_info = None
        self.master_chunksize = None
        self._add_last_periodic_data_timestep = False
        self.depth_field = None

    @staticmethod
    def create_grid(lon, lat, depth, time, time_origin, mesh, **kwargs):
        if len(lon.shape) == 1:
            if depth is None or len(depth.shape) == 1:
                return RectilinearZGrid(lon, lat, depth, time, time_origin=time_origin, mesh=mesh, **kwargs)
            else:
                return RectilinearSGrid(lon, lat, depth, time, time_origin=time_origin, mesh=mesh, **kwargs)
        else:
            if depth is None or len(depth.shape) == 1:
                return CurvilinearZGrid(lon, lat, depth, time, time_origin=time_origin, mesh=mesh, **kwargs)
            else:
                return CurvilinearSGrid(lon, lat, depth, time, time_origin=time_origin, mesh=mesh, **kwargs)

    @property
    def ctypes_struct(self):
        # This is unnecessary for the moment, but it could be useful when going will fully unstructured grids
        self.cgrid = cast(pointer(self.child_ctypes_struct), c_void_p)
        cstruct = CGrid(self.gtype, self.cgrid.value)
        return cstruct

    @property
    def child_ctypes_struct(self):
        """Returns a ctypes struct object containing all relevant
        pointers and sizes for this grid."""

        class CStructuredGrid(Structure):
            # z4d is only to have same cstruct as RectilinearSGrid
            _fields_ = [('xdim', c_int), ('ydim', c_int), ('zdim', c_int),
                        ('tdim', c_int), ('z4d', c_int),
                        ('mesh_spherical', c_int), ('zonal_periodic', c_int),
                        ('chunk_info', POINTER(c_int)),
                        ('load_chunk', POINTER(c_int)),
                        ('tfull_min', c_double), ('tfull_max', c_double), ('periods', POINTER(c_int)),
                        ('lonlat_minmax', POINTER(c_float)),
                        ('lon', POINTER(c_float)), ('lat', POINTER(c_float)),
                        ('depth', POINTER(c_float)), ('time', POINTER(c_double))
                        ]

        # Create and populate the c-struct object
        if not self.cstruct:  # Not to point to the same grid various times if grid in various fields
            if not isinstance(self.periods, c_int):
                self.periods = c_int()
                self.periods.value = 0
            self.cstruct = CStructuredGrid(self.xdim, self.ydim, self.zdim,
                                           self.tdim, self.z4d,
                                           self.mesh == 'spherical', self.zonal_periodic,
                                           (c_int * len(self.chunk_info))(*self.chunk_info),
                                           self.load_chunk.ctypes.data_as(POINTER(c_int)),
                                           self.time_full[0], self.time_full[-1], pointer(self.periods),
                                           self.lonlat_minmax.ctypes.data_as(POINTER(c_float)),
                                           self.lon.ctypes.data_as(POINTER(c_float)),
                                           self.lat.ctypes.data_as(POINTER(c_float)),
                                           self.depth.ctypes.data_as(POINTER(c_float)),
                                           self.time.ctypes.data_as(POINTER(c_double)))
        return self.cstruct

    def lon_grid_to_target(self):
        if self.lon_remapping:
            self.lon = self.lon_remapping.to_target(self.lon)

    def lon_grid_to_source(self):
        if self.lon_remapping:
            self.lon = self.lon_remapping.to_source(self.lon)

    def lon_particle_to_target(self, lon):
        if self.lon_remapping:
            return self.lon_remapping.particle_to_target(lon)
        return lon

    def advancetime(self, grid_new):
        assert isinstance(grid_new.time_origin, type(self.time_origin)), 'time_origin of new and old grids must be either both None or both a date'
        if self.time_origin:
            grid_new.time = grid_new.time + self.time_origin.reltime(grid_new.time_origin)
        if len(grid_new.time) != 1:
            raise RuntimeError('New FieldSet needs to have only one snapshot')
        if grid_new.time > self.time[-1]:  # forward in time, so appending at end
            self.time = np.concatenate((self.time[1:], grid_new.time))
            return 1
        elif grid_new.time < self.time[0]:  # backward in time, so prepending at start
            self.time = np.concatenate((grid_new.time, self.time[:-1]))
            return -1
        else:
            raise RuntimeError("Time of field_new in Field.advancetime() overlaps with times in old Field")

    def check_zonal_periodic(self):
        if self.zonal_periodic or self.mesh == 'flat':
            return
        dx = (self.lon[1:] - self.lon[:-1]) if len(self.lon.shape) == 1 else self.lon[0, 1:] - self.lon[0, :-1]
        dx = np.where(dx < -180, dx+360, dx)
        dx = np.where(dx > 180, dx-360, dx)
        self.zonal_periodic = sum(dx) > 359.9

    def add_Sdepth_periodic_halo(self, zonal, meridional, halosize):
        if zonal:
            if len(self.depth.shape) == 3:
                self.depth = np.concatenate((self.depth[:, :, -halosize:], self.depth,
                                             self.depth[:, :, 0:halosize]), axis=len(self.depth.shape) - 1)
                assert self.depth.shape[2] == self.xdim, "Third dim must be x."
            else:
                self.depth = np.concatenate((self.depth[:, :, :, -halosize:], self.depth,
                                             self.depth[:, :, :, 0:halosize]), axis=len(self.depth.shape) - 1)
                assert self.depth.shape[3] == self.xdim, "Fourth dim must be x."
        if meridional:
            if len(self.depth.shape) == 3:
                self.depth = np.concatenate((self.depth[:, -halosize:, :], self.depth,
                                             self.depth[:, 0:halosize, :]), axis=len(self.depth.shape) - 2)
                assert self.depth.shape[1] == self.ydim, "Second dim must be y."
            else:
                self.depth = np.concatenate((self.depth[:, :, -halosize:, :], self.depth,
                                             self.depth[:, :, 0:halosize, :]), axis=len(self.depth.shape) - 2)
                assert self.depth.shape[2] == self.ydim, "Third dim must be y."

    def computeTimeChunk(self, f, time, signdt):
        nextTime_loc = np.infty if signdt >= 0 else -np.infty
        periods = self.periods.value if isinstance(self.periods, c_int) else self.periods
        prev_time_indices = self.time
        if self.update_status == 'not_updated':
            if self.ti >= 0:
                if (time - periods*(self.time_full[-1]-self.time_full[0]) < self.time[0] or time - periods*(self.time_full[-1]-self.time_full[0]) > self.time[2]):
                    self.ti = -1  # reset
                elif signdt >= 0 and (time - periods*(self.time_full[-1]-self.time_full[0]) < self.time_full[0] or time - periods*(self.time_full[-1]-self.time_full[0]) >= self.time_full[-1]):
                    self.ti = -1  # reset
                elif signdt < 0 and (time - periods*(self.time_full[-1]-self.time_full[0]) <= self.time_full[0] or time - periods*(self.time_full[-1]-self.time_full[0]) > self.time_full[-1]):
                    self.ti = -1  # reset
                elif signdt >= 0 and time - periods*(self.time_full[-1]-self.time_full[0]) >= self.time[1] and self.ti < len(self.time_full)-3:
                    self.ti += 1
                    self.time = self.time_full[self.ti:self.ti+3]
                    self.update_status = 'updated'
                elif signdt == -1 and time - periods*(self.time_full[-1]-self.time_full[0]) < self.time[1] and self.ti > 0:
                    self.ti -= 1
                    self.time = self.time_full[self.ti:self.ti+3]
                    self.update_status = 'updated'
            if self.ti == -1:
                self.time = self.time_full
                self.ti, _ = f.time_index(time)
                periods = self.periods.value if isinstance(self.periods, c_int) else self.periods
                if signdt == -1 and self.ti == 0 and (time - periods*(self.time_full[-1]-self.time_full[0])) == self.time[0] and f.time_periodic:
                    self.ti = len(self.time)-2
                    periods -= 1
                if signdt == -1 and self.ti > 0:
                    self.ti -= 1
                if self.ti >= len(self.time_full) - 2:
                    self.ti = len(self.time_full) - 3

                self.time = self.time_full[self.ti:self.ti+3]
                self.tdim = 3
                if prev_time_indices is None or len(prev_time_indices) != 3 or len(prev_time_indices) != len(self.time):
                    self.update_status = 'first_updated'
                elif functools.reduce(lambda i, j: i and j, map(lambda m, k: m == k, self.time, prev_time_indices), True) and len(prev_time_indices) == len(self.time):
                    self.update_status = 'not_updated'
                elif functools.reduce(lambda i, j: i and j, map(lambda m, k: m == k, self.time[:2], prev_time_indices[:2]), True) and len(prev_time_indices) == len(self.time):
                    self.update_status = 'updated'
                else:
                    self.update_status = 'first_updated'
            if signdt >= 0 and (self.ti < len(self.time_full)-3 or not f.allow_time_extrapolation):
                nextTime_loc = self.time[2] + periods*(self.time_full[-1]-self.time_full[0])
            elif signdt == -1 and (self.ti > 0 or not f.allow_time_extrapolation):
                nextTime_loc = self.time[0] + periods*(self.time_full[-1]-self.time_full[0])
        return nextTime_loc


class RectilinearGrid(Grid):
    """Rectilinear Grid
       Private base class for RectilinearZGrid and RectilinearSGrid

    """

    def __init__(self, lon, lat, time, time_origin, mesh):
        assert(isinstance(lon, np.ndarray) and len(lon.shape) == 1), 'lon is not a numpy vector'
        assert(isinstance(lat, np.ndarray) and len(lat.shape) == 1), 'lat is not a numpy vector'
        assert (isinstance(time, np.ndarray) or not time), 'time is not a numpy array'
        if isinstance(time, np.ndarray):
            assert(len(time.shape) == 1), 'time is not a vector'

        super(RectilinearGrid, self).__init__(lon, lat, time, time_origin, mesh)
        self.xdim = self.lon.size
        self.ydim = self.lat.size
        self.tdim = self.time.size
        if self.lat[-1] < self.lat[0]:
            self.lat = np.flip(self.lat, axis=0)
            self.lat_flipped = True
            logger.warning_once("Flipping lat data from North-South to South-North")

    def add_periodic_halo(self, zonal, meridional, halosize=5):
        """Add a 'halo' to the Grid, through extending the Grid (and lon/lat)
        similarly to the halo created for the Fields

        :param zonal: Create a halo in zonal direction (boolean)
        :param meridional: Create a halo in meridional direction (boolean)
        :param halosize: size of the halo (in grid points). Default is 5 grid points
        """
        if zonal:
            lonshift = (self.lon[-1] - 2 * self.lon[0] + self.lon[1])
            if not np.allclose(self.lon[1]-self.lon[0], self.lon[-1]-self.lon[-2]):
                logger.warning_once("The zonal halo is located at the east and west of current grid, with a dx = lon[1]-lon[0] between the last nodes of the original grid and the first ones of the halo. In your grid, lon[1]-lon[0] != lon[-1]-lon[-2]. Is the halo computed as you expect?")
            self.lon = np.concatenate((self.lon[-halosize:] - lonshift,
                                      self.lon, self.lon[0:halosize] + lonshift))
            self.xdim = self.lon.size
            self.zonal_periodic = True
            self.zonal_halo = halosize
        if meridional:
            if not np.allclose(self.lat[1]-self.lat[0], self.lat[-1]-self.lat[-2]):
                logger.warning_once("The meridional halo is located at the north and south of current grid, with a dy = lat[1]-lat[0] between the last nodes of the original grid and the first ones of the halo. In your grid, lat[1]-lat[0] != lat[-1]-lat[-2]. Is the halo computed as you expect?")
            latshift = (self.lat[-1] - 2 * self.lat[0] + self.lat[1])
            self.lat = np.concatenate((self.lat[-halosize:] - latshift,
                                      self.lat, self.lat[0:halosize] + latshift))
            self.ydim = self.lat.size
            self.meridional_halo = halosize
        self.lonlat_minmax = np.array([np.nanmin(self.lon), np.nanmax(self.lon), np.nanmin(self.lat), np.nanmax(self.lat)], dtype=np.float32)
        if isinstance(self, RectilinearSGrid):
            self.add_Sdepth_periodic_halo(zonal, meridional, halosize)


class RectilinearZGrid(RectilinearGrid):
    """Rectilinear Z Grid

    :param lon: Vector containing the longitude coordinates of the grid
    :param lat: Vector containing the latitude coordinates of the grid
    :param depth: Vector containing the vertical coordinates of the grid, which are z-coordinates.
           The depth of the different layers is thus constant.
    :param time: Vector containing the time coordinates of the grid
    :param time_origin: Time origin (TimeConverter object) of the time axis
    :param mesh: String indicating the type of mesh coordinates and
           units used during velocity interpolation:

           1. spherical (default): Lat and lon in degree, with a
              correction for zonal velocity U near the poles.
           2. flat: No conversion, lat/lon are assumed to be in m.
    """

    def __init__(self, lon, lat, depth=None, time=None, time_origin=None, mesh='flat'):
        super(RectilinearZGrid, self).__init__(lon, lat, time, time_origin, mesh)
        if isinstance(depth, np.ndarray):
            assert(len(depth.shape) == 1), 'depth is not a vector'

        self.gtype = GridCode.RectilinearZGrid
        self.depth = np.zeros(1, dtype=np.float32) if depth is None else depth  # should be a C-contiguous array of floats
        if not self.depth.flags['C_CONTIGUOUS']:
            self.depth = np.array(self.depth, order='C')
        self.zdim = self.depth.size
        self.z4d = -1  # only used in RectilinearSGrid
        if not self.depth.dtype == np.float32:
            logger.warning_once("Casting depth data to np.float32")
            self.depth = self.depth.astype(np.float32)


class RectilinearSGrid(RectilinearGrid):
    """Rectilinear S Grid. Same horizontal discretisation as a rectilinear z grid,
       but with s vertical coordinates

    :param lon: Vector containing the longitude coordinates of the grid
    :param lat: Vector containing the latitude coordinates of the grid
    :param depth: 4D (time-evolving) or 3D (time-independent) array containing the vertical coordinates of the grid,
           which are s-coordinates.
           s-coordinates can be terrain-following (sigma) or iso-density (rho) layers,
           or any generalised vertical discretisation.
           The depth of each node depends then on the horizontal position (lon, lat),
           the number of the layer and the time is depth is a 4D array.
           depth array is either a 4D array[xdim][ydim][zdim][tdim] or a 3D array[xdim][ydim[zdim].
    :param time: Vector containing the time coordinates of the grid
    :param time_origin: Time origin (TimeConverter object) of the time axis
    :param mesh: String indicating the type of mesh coordinates and
           units used during velocity interpolation:

           1. spherical (default): Lat and lon in degree, with a
              correction for zonal velocity U near the poles.
           2. flat: No conversion, lat/lon are assumed to be in m.
    """

    def __init__(self, lon, lat, depth, time=None, time_origin=None, mesh='flat'):
        super(RectilinearSGrid, self).__init__(lon, lat, time, time_origin, mesh)
        assert(isinstance(depth, np.ndarray) and len(depth.shape) in [3, 4]), 'depth is not a 3D or 4D numpy array'

        self.gtype = GridCode.RectilinearSGrid
        self.depth = depth  # should be a C-contiguous array of floats
        if not self.depth.flags['C_CONTIGUOUS']:
            self.depth = np.array(self.depth, order='C')
        self.zdim = self.depth.shape[-3]
        self.z4d = len(self.depth.shape) == 4
        if self.z4d:
            # self.depth.shape[0] is 0 for S grids loaded from netcdf file
            assert self.tdim == self.depth.shape[0] or self.depth.shape[0] == 0, 'depth dimension has the wrong format. It should be [tdim, zdim, ydim, xdim]'
            assert self.xdim == self.depth.shape[-1] or self.depth.shape[-1] == 0, 'depth dimension has the wrong format. It should be [tdim, zdim, ydim, xdim]'
            assert self.ydim == self.depth.shape[-2] or self.depth.shape[-2] == 0, 'depth dimension has the wrong format. It should be [tdim, zdim, ydim, xdim]'
        else:
            assert self.xdim == self.depth.shape[-1], 'depth dimension has the wrong format. It should be [zdim, ydim, xdim]'
            assert self.ydim == self.depth.shape[-2], 'depth dimension has the wrong format. It should be [zdim, ydim, xdim]'
        if not self.depth.dtype == np.float32:
            logger.warning_once("Casting depth data to np.float32")
            self.depth = self.depth.astype(np.float32)
        if self.lat_flipped:
            self.depth = np.flip(self.depth, axis=-2)


class CurvilinearGrid(Grid):

    def __init__(self, lon, lat, time=None, time_origin=None, mesh='flat'):
        assert(isinstance(lon, np.ndarray) and len(lon.squeeze().shape) == 2), 'lon is not a 2D numpy array'
        assert(isinstance(lat, np.ndarray) and len(lat.squeeze().shape) == 2), 'lat is not a 2D numpy array'
        assert (isinstance(time, np.ndarray) or not time), 'time is not a numpy array'
        if isinstance(time, np.ndarray):
            assert(len(time.shape) == 1), 'time is not a vector'

        lon = lon.squeeze()
        lat = lat.squeeze()
        super(CurvilinearGrid, self).__init__(lon, lat, time, time_origin, mesh)
        self.xdim = self.lon.shape[1]
        self.ydim = self.lon.shape[0]
        self.tdim = self.time.size

    def add_periodic_halo(self, zonal, meridional, halosize=5):
        """Add a 'halo' to the Grid, through extending the Grid (and lon/lat)
        similarly to the halo created for the Fields

        :param zonal: Create a halo in zonal direction (boolean)
        :param meridional: Create a halo in meridional direction (boolean)
        :param halosize: size of the halo (in grid points). Default is 5 grid points
        """
        if zonal:
            lonshift = self.lon[:, -1] - 2 * self.lon[:, 0] + self.lon[:, 1]
            if not np.allclose(self.lon[:, 1]-self.lon[:, 0], self.lon[:, -1]-self.lon[:, -2]):
                logger.warning_once("The zonal halo is located at the east and west of current grid, with a dx = lon[:,1]-lon[:,0] between the last nodes of the original grid and the first ones of the halo. In your grid, lon[:,1]-lon[:,0] != lon[:,-1]-lon[:,-2]. Is the halo computed as you expect?")
            self.lon = np.concatenate((self.lon[:, -halosize:] - lonshift[:, np.newaxis],
                                       self.lon, self.lon[:, 0:halosize] + lonshift[:, np.newaxis]),
                                      axis=len(self.lon.shape)-1)
            self.lat = np.concatenate((self.lat[:, -halosize:],
                                       self.lat, self.lat[:, 0:halosize]),
                                      axis=len(self.lat.shape)-1)
            self.xdim = self.lon.shape[1]
            self.ydim = self.lat.shape[0]
            self.zonal_periodic = True
            self.zonal_halo = halosize
        if meridional:
            if not np.allclose(self.lat[1, :]-self.lat[0, :], self.lat[-1, :]-self.lat[-2, :]):
                logger.warning_once("The meridional halo is located at the north and south of current grid, with a dy = lat[1,:]-lat[0,:] between the last nodes of the original grid and the first ones of the halo. In your grid, lat[1,:]-lat[0,:] != lat[-1,:]-lat[-2,:]. Is the halo computed as you expect?")
            latshift = self.lat[-1, :] - 2 * self.lat[0, :] + self.lat[1, :]
            self.lat = np.concatenate((self.lat[-halosize:, :] - latshift[np.newaxis, :],
                                       self.lat, self.lat[0:halosize, :] + latshift[np.newaxis, :]),
                                      axis=len(self.lat.shape)-2)
            self.lon = np.concatenate((self.lon[-halosize:, :],
                                       self.lon, self.lon[0:halosize, :]),
                                      axis=len(self.lon.shape)-2)
            self.xdim = self.lon.shape[1]
            self.ydim = self.lat.shape[0]
            self.meridional_halo = halosize
        if isinstance(self, CurvilinearSGrid):
            self.add_Sdepth_periodic_halo(zonal, meridional, halosize)


class CurvilinearZGrid(CurvilinearGrid):
    """Curvilinear Z Grid.

    :param lon: 2D array containing the longitude coordinates of the grid
    :param lat: 2D array containing the latitude coordinates of the grid
    :param depth: Vector containing the vertical coordinates of the grid, which are z-coordinates.
           The depth of the different layers is thus constant.
    :param time: Vector containing the time coordinates of the grid
    :param time_origin: Time origin (TimeConverter object) of the time axis
    :param mesh: String indicating the type of mesh coordinates and
           units used during velocity interpolation:

           1. spherical (default): Lat and lon in degree, with a
              correction for zonal velocity U near the poles.
           2. flat: No conversion, lat/lon are assumed to be in m.
    """

    def __init__(self, lon, lat, depth=None, time=None, time_origin=None, mesh='flat'):
        super(CurvilinearZGrid, self).__init__(lon, lat, time, time_origin, mesh)
        if isinstance(depth, np.ndarray):
            assert(len(depth.shape) == 1), 'depth is not a vector'

        self.gtype = GridCode.CurvilinearZGrid
        self.depth = np.zeros(1, dtype=np.float32) if depth is None else depth  # should be a C-contiguous array of floats
        if not self.depth.flags['C_CONTIGUOUS']:
            self.depth = np.array(self.depth, order='C')
        self.zdim = self.depth.size
        self.z4d = -1  # only for SGrid
        if not self.depth.dtype == np.float32:
            logger.warning_once("Casting depth data to np.float32")
            self.depth = self.depth.astype(np.float32)


class CurvilinearSGrid(CurvilinearGrid):
    """Curvilinear S Grid.

    :param lon: 2D array containing the longitude coordinates of the grid
    :param lat: 2D array containing the latitude coordinates of the grid
    :param depth: 4D (time-evolving) or 3D (time-independent) array containing the vertical coordinates of the grid,
           which are s-coordinates.
           s-coordinates can be terrain-following (sigma) or iso-density (rho) layers,
           or any generalised vertical discretisation.
           The depth of each node depends then on the horizontal position (lon, lat),
           the number of the layer and the time is depth is a 4D array.
           depth array is either a 4D array[xdim][ydim][zdim][tdim] or a 3D array[xdim][ydim[zdim].
    :param time: Vector containing the time coordinates of the grid
    :param time_origin: Time origin (TimeConverter object) of the time axis
    :param mesh: String indicating the type of mesh coordinates and
           units used during velocity interpolation:

           1. spherical (default): Lat and lon in degree, with a
              correction for zonal velocity U near the poles.
           2. flat: No conversion, lat/lon are assumed to be in m.
    """

    def __init__(self, lon, lat, depth, time=None, time_origin=None, mesh='flat'):
        super(CurvilinearSGrid, self).__init__(lon, lat, time, time_origin, mesh)
        assert(isinstance(depth, np.ndarray) and len(depth.shape) in [3, 4]), 'depth is not a 4D numpy array'

        self.gtype = GridCode.CurvilinearSGrid
        self.depth = depth  # should be a C-contiguous array of floats
        if not self.depth.flags['C_CONTIGUOUS']:
            self.depth = np.array(self.depth, order='C')
        self.zdim = self.depth.shape[-3]
        self.z4d = len(self.depth.shape) == 4
        if self.z4d:
            # self.depth.shape[0] is 0 for S grids loaded from netcdf file
            assert self.tdim == self.depth.shape[0] or self.depth.shape[0] == 0, 'depth dimension has the wrong format. It should be [tdim, zdim, ydim, xdim]'
            assert self.xdim == self.depth.shape[-1] or self.depth.shape[-1] == 0, 'depth dimension has the wrong format. It should be [tdim, zdim, ydim, xdim]'
            assert self.ydim == self.depth.shape[-2] or self.depth.shape[-2] == 0, 'depth dimension has the wrong format. It should be [tdim, zdim, ydim, xdim]'
        else:
            assert self.xdim == self.depth.shape[-1], 'depth dimension has the wrong format. It should be [zdim, ydim, xdim]'
            assert self.ydim == self.depth.shape[-2], 'depth dimension has the wrong format. It should be [zdim, ydim, xdim]'
        if not self.depth.dtype == np.float32:
            logger.warning_once("Casting depth data to np.float32")
            self.depth = self.depth.astype(np.float32)
