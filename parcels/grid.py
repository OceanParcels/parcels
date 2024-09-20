import functools
import warnings
from ctypes import POINTER, Structure, c_double, c_float, c_int, c_void_p, cast, pointer
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from parcels._typing import Mesh, UpdateStatus, assert_valid_mesh
from parcels.tools.converters import TimeConverter
from parcels.tools.warnings import FieldSetWarning

if TYPE_CHECKING:
    from parcels.field import Field

__all__ = [
    "GridType",
    "GridCode",
    "RectilinearZGrid",
    "RectilinearSGrid",
    "CurvilinearZGrid",
    "CurvilinearSGrid",
    "CGrid",
    "Grid",
]


class GridType(IntEnum):
    RectilinearZGrid = 0
    RectilinearSGrid = 1
    CurvilinearZGrid = 2
    CurvilinearSGrid = 3


# GridCode has been renamed to GridType for consistency.
# TODO: Remove alias in Parcels v4
GridCode = GridType


class CGrid(Structure):
    _fields_ = [("gtype", c_int), ("grid", c_void_p)]


class Grid:
    """Grid class that defines a (spatial and temporal) grid on which Fields are defined."""

    def __init__(
        self,
        lon: npt.NDArray,
        lat: npt.NDArray,
        time: npt.NDArray | None,
        time_origin: TimeConverter | None,
        mesh: Mesh,
    ):
        self.xi = None
        self.yi = None
        self.zi = None
        self.ti = -1
        self.lon = lon
        self.update_status: UpdateStatus | None = None
        if not self.lon.flags["C_CONTIGUOUS"]:
            self.lon = np.array(self.lon, order="C")
        self.lat = lat
        if not self.lat.flags["C_CONTIGUOUS"]:
            self.lat = np.array(self.lat, order="C")
        self.time = np.zeros(1, dtype=np.float64) if time is None else time
        if not self.time.flags["C_CONTIGUOUS"]:
            self.time = np.array(self.time, order="C")
        if not self.lon.dtype == np.float32:
            self.lon = self.lon.astype(np.float32)
        if not self.lat.dtype == np.float32:
            self.lat = self.lat.astype(np.float32)
        if not self.time.dtype == np.float64:
            assert isinstance(
                self.time[0], (np.integer, np.floating, float, int)
            ), "Time vector must be an array of int or floats"
            self.time = self.time.astype(np.float64)
        self.time_full = self.time  # needed for deferred_loaded Fields
        self.time_origin = TimeConverter() if time_origin is None else time_origin
        assert isinstance(self.time_origin, TimeConverter), "time_origin needs to be a TimeConverter object"
        assert_valid_mesh(mesh)
        self.mesh = mesh
        self.cstruct = None
        self.cell_edge_sizes: dict[str, npt.NDArray] = {}
        self.zonal_periodic = False
        self.zonal_halo = 0
        self.meridional_halo = 0
        self.lat_flipped = False
        self.defer_load = False
        self.lonlat_minmax = np.array(
            [np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)], dtype=np.float32
        )
        self.periods = 0
        self.load_chunk: npt.NDArray = np.array([])
        self.chunk_info = None
        self.chunksize = None
        self._add_last_periodic_data_timestep = False
        self.depth_field = None

    def __repr__(self):
        with np.printoptions(threshold=5, suppress=True, linewidth=120, formatter={"float": "{: 0.2f}".format}):
            return (
                f"{type(self).__name__}("
                f"lon={self.lon!r}, lat={self.lat!r}, time={self.time!r}, "
                f"time_origin={self.time_origin!r}, mesh={self.mesh!r})"
            )

    @staticmethod
    def create_grid(
        lon: npt.ArrayLike,
        lat: npt.ArrayLike,
        depth,
        time,
        time_origin,
        mesh: Mesh,
        **kwargs,
    ):
        lon = np.array(lon)
        lat = np.array(lat)

        if depth is not None:
            depth = np.array(depth)

        if len(lon.shape) <= 1:
            if depth is None or len(depth.shape) <= 1:
                return RectilinearZGrid(lon, lat, depth, time, time_origin=time_origin, mesh=mesh, **kwargs)
            else:
                return RectilinearSGrid(lon, lat, depth, time, time_origin=time_origin, mesh=mesh, **kwargs)
        else:
            if depth is None or len(depth.shape) <= 1:
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
        pointers and sizes for this grid.
        """

        class CStructuredGrid(Structure):
            # z4d is only to have same cstruct as RectilinearSGrid
            _fields_ = [
                ("xdim", c_int),
                ("ydim", c_int),
                ("zdim", c_int),
                ("tdim", c_int),
                ("z4d", c_int),
                ("mesh_spherical", c_int),
                ("zonal_periodic", c_int),
                ("chunk_info", POINTER(c_int)),
                ("load_chunk", POINTER(c_int)),
                ("tfull_min", c_double),
                ("tfull_max", c_double),
                ("periods", POINTER(c_int)),
                ("lonlat_minmax", POINTER(c_float)),
                ("lon", POINTER(c_float)),
                ("lat", POINTER(c_float)),
                ("depth", POINTER(c_float)),
                ("time", POINTER(c_double)),
            ]

        # Create and populate the c-struct object
        if not self.cstruct:  # Not to point to the same grid various times if grid in various fields
            if not isinstance(self.periods, c_int):
                self.periods = c_int()
                self.periods.value = 0
            self.cstruct = CStructuredGrid(
                self.xdim,
                self.ydim,
                self.zdim,
                self.tdim,
                self.z4d,
                int(self.mesh == "spherical"),
                int(self.zonal_periodic),
                (c_int * len(self.chunk_info))(*self.chunk_info),
                self.load_chunk.ctypes.data_as(POINTER(c_int)),
                self.time_full[0],
                self.time_full[-1],
                pointer(self.periods),
                self.lonlat_minmax.ctypes.data_as(POINTER(c_float)),
                self.lon.ctypes.data_as(POINTER(c_float)),
                self.lat.ctypes.data_as(POINTER(c_float)),
                self.depth.ctypes.data_as(POINTER(c_float)),
                self.time.ctypes.data_as(POINTER(c_double)),
            )
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

    def check_zonal_periodic(self):
        if self.zonal_periodic or self.mesh == "flat" or self.lon.size == 1:
            return
        dx = (self.lon[1:] - self.lon[:-1]) if len(self.lon.shape) == 1 else self.lon[0, 1:] - self.lon[0, :-1]
        dx = np.where(dx < -180, dx + 360, dx)
        dx = np.where(dx > 180, dx - 360, dx)
        self.zonal_periodic = sum(dx) > 359.9

    def add_Sdepth_periodic_halo(self, zonal, meridional, halosize):
        if zonal:
            if len(self.depth.shape) == 3:
                self.depth = np.concatenate(
                    (self.depth[:, :, -halosize:], self.depth, self.depth[:, :, 0:halosize]),
                    axis=len(self.depth.shape) - 1,
                )
                assert self.depth.shape[2] == self.xdim, "Third dim must be x."
            else:
                self.depth = np.concatenate(
                    (self.depth[:, :, :, -halosize:], self.depth, self.depth[:, :, :, 0:halosize]),
                    axis=len(self.depth.shape) - 1,
                )
                assert self.depth.shape[3] == self.xdim, "Fourth dim must be x."
        if meridional:
            if len(self.depth.shape) == 3:
                self.depth = np.concatenate(
                    (self.depth[:, -halosize:, :], self.depth, self.depth[:, 0:halosize, :]),
                    axis=len(self.depth.shape) - 2,
                )
                assert self.depth.shape[1] == self.ydim, "Second dim must be y."
            else:
                self.depth = np.concatenate(
                    (self.depth[:, :, -halosize:, :], self.depth, self.depth[:, :, 0:halosize, :]),
                    axis=len(self.depth.shape) - 2,
                )
                assert self.depth.shape[2] == self.ydim, "Third dim must be y."

    def computeTimeChunk(self, f: "Field", time, signdt):
        nextTime_loc = np.inf if signdt >= 0 else -np.inf
        periods = self.periods.value if isinstance(self.periods, c_int) else self.periods
        prev_time_indices = self.time
        if self.update_status == "not_updated":
            if self.ti >= 0:
                if (
                    time - periods * (self.time_full[-1] - self.time_full[0]) < self.time[0]
                    or time - periods * (self.time_full[-1] - self.time_full[0]) > self.time[1]
                ):
                    self.ti = -1  # reset
                elif signdt >= 0 and (
                    time - periods * (self.time_full[-1] - self.time_full[0]) < self.time_full[0]
                    or time - periods * (self.time_full[-1] - self.time_full[0]) >= self.time_full[-1]
                ):
                    self.ti = -1  # reset
                elif signdt < 0 and (
                    time - periods * (self.time_full[-1] - self.time_full[0]) <= self.time_full[0]
                    or time - periods * (self.time_full[-1] - self.time_full[0]) > self.time_full[-1]
                ):
                    self.ti = -1  # reset
                elif (
                    signdt >= 0
                    and time - periods * (self.time_full[-1] - self.time_full[0]) >= self.time[1]
                    and self.ti < len(self.time_full) - 2
                ):
                    self.ti += 1
                    self.time = self.time_full[self.ti : self.ti + 2]
                    self.update_status = "updated"
                elif (
                    signdt < 0
                    and time - periods * (self.time_full[-1] - self.time_full[0]) <= self.time[0]
                    and self.ti > 0
                ):
                    self.ti -= 1
                    self.time = self.time_full[self.ti : self.ti + 2]
                    self.update_status = "updated"
            if self.ti == -1:
                self.time = self.time_full
                self.ti, _ = f._time_index(time)
                periods = self.periods.value if isinstance(self.periods, c_int) else self.periods
                if (
                    signdt == -1
                    and self.ti == 0
                    and (time - periods * (self.time_full[-1] - self.time_full[0])) == self.time[0]
                    and f.time_periodic
                ):
                    self.ti = len(self.time) - 1
                    periods -= 1
                if signdt == -1 and self.ti > 0 and self.time_full[self.ti] == time:
                    self.ti -= 1
                if self.ti >= len(self.time_full) - 1:
                    self.ti = len(self.time_full) - 2

                self.time = self.time_full[self.ti : self.ti + 2]
                self.tdim = 2
                if prev_time_indices is None or len(prev_time_indices) != 2 or len(prev_time_indices) != len(self.time):
                    self.update_status = "first_updated"
                elif functools.reduce(
                    lambda i, j: i and j, map(lambda m, k: m == k, self.time, prev_time_indices), True
                ) and len(prev_time_indices) == len(self.time):
                    self.update_status = "not_updated"
                elif functools.reduce(
                    lambda i, j: i and j, map(lambda m, k: m == k, self.time[:1], prev_time_indices[:1]), True
                ) and len(prev_time_indices) == len(self.time):
                    self.update_status = "updated"
                else:
                    self.update_status = "first_updated"
            if signdt >= 0 and (self.ti < len(self.time_full) - 2 or not f.allow_time_extrapolation):
                nextTime_loc = self.time[1] + periods * (self.time_full[-1] - self.time_full[0])
            elif signdt < 0 and (self.ti > 0 or not f.allow_time_extrapolation):
                nextTime_loc = self.time[0] + periods * (self.time_full[-1] - self.time_full[0])
        return nextTime_loc

    @property
    def chunk_not_loaded(self):
        return 0

    @property
    def chunk_loading_requested(self):
        return 1

    @property
    def chunk_loaded_touched(self):
        return 2

    @property
    def chunk_deprecated(self):
        return 3

    @property
    def chunk_loaded(self):
        return [2, 3]


class RectilinearGrid(Grid):
    """Rectilinear Grid class

    Private base class for RectilinearZGrid and RectilinearSGrid

    """

    def __init__(self, lon, lat, time, time_origin, mesh: Mesh):
        assert isinstance(lon, np.ndarray) and len(lon.shape) <= 1, "lon is not a numpy vector"
        assert isinstance(lat, np.ndarray) and len(lat.shape) <= 1, "lat is not a numpy vector"
        assert isinstance(time, np.ndarray) or not time, "time is not a numpy array"
        if isinstance(time, np.ndarray):
            assert len(time.shape) == 1, "time is not a vector"

        super().__init__(lon, lat, time, time_origin, mesh)
        self.xdim = self.lon.size
        self.ydim = self.lat.size
        self.tdim = self.time.size
        if self.ydim > 1 and self.lat[-1] < self.lat[0]:
            self.lat = np.flip(self.lat, axis=0)
            self.lat_flipped = True
            warnings.warn(
                "Flipping lat data from North-South to South-North. "
                "Note that this may lead to wrong sign for meridional velocity, so tread very carefully",
                FieldSetWarning,
                stacklevel=2,
            )

    def add_periodic_halo(self, zonal: bool, meridional: bool, halosize: int = 5):
        """Add a 'halo' to the Grid, through extending the Grid (and lon/lat)
        similarly to the halo created for the Fields

        Parameters
        ----------
        zonal : bool
            Create a halo in zonal direction
        meridional : bool
            Create a halo in meridional direction
        halosize : int
            size of the halo (in grid points). Default is 5 grid points
        """
        if zonal:
            lonshift = self.lon[-1] - 2 * self.lon[0] + self.lon[1]
            if not np.allclose(self.lon[1] - self.lon[0], self.lon[-1] - self.lon[-2]):
                warnings.warn(
                    "The zonal halo is located at the east and west of current grid, "
                    "with a dx = lon[1]-lon[0] between the last nodes of the original grid and the first ones of the halo. "
                    "In your grid, lon[1]-lon[0] != lon[-1]-lon[-2]. Is the halo computed as you expect?",
                    FieldSetWarning,
                    stacklevel=2,
                )
            self.lon = np.concatenate((self.lon[-halosize:] - lonshift, self.lon, self.lon[0:halosize] + lonshift))
            self.xdim = self.lon.size
            self.zonal_periodic = True
            self.zonal_halo = halosize
        if meridional:
            if not np.allclose(self.lat[1] - self.lat[0], self.lat[-1] - self.lat[-2]):
                warnings.warn(
                    "The meridional halo is located at the north and south of current grid, "
                    "with a dy = lat[1]-lat[0] between the last nodes of the original grid and the first ones of the halo. "
                    "In your grid, lat[1]-lat[0] != lat[-1]-lat[-2]. Is the halo computed as you expect?",
                    FieldSetWarning,
                    stacklevel=2,
                )
            latshift = self.lat[-1] - 2 * self.lat[0] + self.lat[1]
            self.lat = np.concatenate((self.lat[-halosize:] - latshift, self.lat, self.lat[0:halosize] + latshift))
            self.ydim = self.lat.size
            self.meridional_halo = halosize
        self.lonlat_minmax = np.array(
            [np.nanmin(self.lon), np.nanmax(self.lon), np.nanmin(self.lat), np.nanmax(self.lat)], dtype=np.float32
        )
        if isinstance(self, RectilinearSGrid):
            self.add_Sdepth_periodic_halo(zonal, meridional, halosize)


class RectilinearZGrid(RectilinearGrid):
    """Rectilinear Z Grid.

    Parameters
    ----------
    lon :
        Vector containing the longitude coordinates of the grid
    lat :
        Vector containing the latitude coordinates of the grid
    depth :
        Vector containing the vertical coordinates of the grid, which are z-coordinates.
        The depth of the different layers is thus constant.
    time :
        Vector containing the time coordinates of the grid
    time_origin : parcels.tools.converters.TimeConverter
        Time origin of the time axis
    mesh : str
        String indicating the type of mesh coordinates and
        units used during velocity interpolation:

        1. spherical (default): Lat and lon in degree, with a
           correction for zonal velocity U near the poles.
        2. flat: No conversion, lat/lon are assumed to be in m.
    """

    def __init__(self, lon, lat, depth=None, time=None, time_origin=None, mesh: Mesh = "flat"):
        super().__init__(lon, lat, time, time_origin, mesh)
        if isinstance(depth, np.ndarray):
            assert len(depth.shape) <= 1, "depth is not a vector"

        self.gtype = GridType.RectilinearZGrid
        self.depth = np.zeros(1, dtype=np.float32) if depth is None else depth
        if not self.depth.flags["C_CONTIGUOUS"]:
            self.depth = np.array(self.depth, order="C")
        self.zdim = self.depth.size
        self.z4d = -1  # only used in RectilinearSGrid
        if not self.depth.dtype == np.float32:
            self.depth = self.depth.astype(np.float32)


class RectilinearSGrid(RectilinearGrid):
    """Rectilinear S Grid. Same horizontal discretisation as a rectilinear z grid,
       but with s vertical coordinates

    Parameters
    ----------
    lon :
        Vector containing the longitude coordinates of the grid
    lat :
        Vector containing the latitude coordinates of the grid
    depth :
        4D (time-evolving) or 3D (time-independent) array containing the vertical coordinates of the grid,
        which are s-coordinates.
        s-coordinates can be terrain-following (sigma) or iso-density (rho) layers,
        or any generalised vertical discretisation.
        The depth of each node depends then on the horizontal position (lon, lat),
        the number of the layer and the time is depth is a 4D array.
        depth array is either a 4D array[xdim][ydim][zdim][tdim] or a 3D array[xdim][ydim[zdim].
    time :
        Vector containing the time coordinates of the grid
    time_origin : parcels.tools.converters.TimeConverter
        Time origin of the time axis
    mesh : str
        String indicating the type of mesh coordinates and
        units used during velocity interpolation:

        1. spherical (default): Lat and lon in degree, with a
           correction for zonal velocity U near the poles.
        2. flat: No conversion, lat/lon are assumed to be in m.
    """

    def __init__(
        self,
        lon: npt.NDArray,
        lat: npt.NDArray,
        depth: npt.NDArray,
        time: npt.NDArray | None = None,
        time_origin: TimeConverter | None = None,
        mesh: Mesh = "flat",
    ):
        super().__init__(lon, lat, time, time_origin, mesh)
        assert isinstance(depth, np.ndarray) and len(depth.shape) in [3, 4], "depth is not a 3D or 4D numpy array"

        self.gtype = GridType.RectilinearSGrid
        self.depth = depth
        if not self.depth.flags["C_CONTIGUOUS"]:
            self.depth = np.array(self.depth, order="C")
        self.zdim = self.depth.shape[-3]
        self.z4d = 1 if len(self.depth.shape) == 4 else 0
        if self.z4d:
            # self.depth.shape[0] is 0 for S grids loaded from netcdf file
            assert (
                self.tdim == self.depth.shape[0] or self.depth.shape[0] == 0
            ), "depth dimension has the wrong format. It should be [tdim, zdim, ydim, xdim]"
            assert (
                self.xdim == self.depth.shape[-1] or self.depth.shape[-1] == 0
            ), "depth dimension has the wrong format. It should be [tdim, zdim, ydim, xdim]"
            assert (
                self.ydim == self.depth.shape[-2] or self.depth.shape[-2] == 0
            ), "depth dimension has the wrong format. It should be [tdim, zdim, ydim, xdim]"
        else:
            assert (
                self.xdim == self.depth.shape[-1]
            ), "depth dimension has the wrong format. It should be [zdim, ydim, xdim]"
            assert (
                self.ydim == self.depth.shape[-2]
            ), "depth dimension has the wrong format. It should be [zdim, ydim, xdim]"
        if not self.depth.dtype == np.float32:
            self.depth = self.depth.astype(np.float32)
        if self.lat_flipped:
            self.depth = np.flip(self.depth, axis=-2)


class CurvilinearGrid(Grid):
    def __init__(
        self,
        lon: npt.NDArray,
        lat: npt.NDArray,
        time: npt.NDArray | None = None,
        time_origin: TimeConverter | None = None,
        mesh: Mesh = "flat",
    ):
        assert isinstance(lon, np.ndarray) and len(lon.squeeze().shape) == 2, "lon is not a 2D numpy array"
        assert isinstance(lat, np.ndarray) and len(lat.squeeze().shape) == 2, "lat is not a 2D numpy array"
        assert isinstance(time, np.ndarray) or not time, "time is not a numpy array"
        if isinstance(time, np.ndarray):
            assert len(time.shape) == 1, "time is not a vector"

        lon = lon.squeeze()
        lat = lat.squeeze()
        super().__init__(lon, lat, time, time_origin, mesh)
        self.xdim = self.lon.shape[1]
        self.ydim = self.lon.shape[0]
        self.tdim = self.time.size

    def add_periodic_halo(self, zonal, meridional, halosize=5):
        """Add a 'halo' to the Grid, through extending the Grid (and lon/lat)
        similarly to the halo created for the Fields

        Parameters
        ----------
        zonal : bool
            Create a halo in zonal direction
        meridional : bool
            Create a halo in meridional direction
        halosize : int
            size of the halo (in grid points). Default is 5 grid points
        """
        if zonal:
            lonshift = self.lon[:, -1] - 2 * self.lon[:, 0] + self.lon[:, 1]
            if not np.allclose(self.lon[:, 1] - self.lon[:, 0], self.lon[:, -1] - self.lon[:, -2]):
                warnings.warn(
                    "The zonal halo is located at the east and west of current grid, "
                    "with a dx = lon[1]-lon[0] between the last nodes of the original grid and the first ones of the halo. "
                    "In your grid, lon[1]-lon[0] != lon[-1]-lon[-2]. Is the halo computed as you expect?",
                    FieldSetWarning,
                    stacklevel=2,
                )
            self.lon = np.concatenate(
                (
                    self.lon[:, -halosize:] - lonshift[:, np.newaxis],
                    self.lon,
                    self.lon[:, 0:halosize] + lonshift[:, np.newaxis],
                ),
                axis=len(self.lon.shape) - 1,
            )
            self.lat = np.concatenate(
                (self.lat[:, -halosize:], self.lat, self.lat[:, 0:halosize]), axis=len(self.lat.shape) - 1
            )
            self.xdim = self.lon.shape[1]
            self.ydim = self.lat.shape[0]
            self.zonal_periodic = True
            self.zonal_halo = halosize
        if meridional:
            if not np.allclose(self.lat[1, :] - self.lat[0, :], self.lat[-1, :] - self.lat[-2, :]):
                warnings.warn(
                    "The meridional halo is located at the north and south of current grid, "
                    "with a dy = lat[1]-lat[0] between the last nodes of the original grid and the first ones of the halo. "
                    "In your grid, lat[1]-lat[0] != lat[-1]-lat[-2]. Is the halo computed as you expect?",
                    FieldSetWarning,
                    stacklevel=2,
                )
            latshift = self.lat[-1, :] - 2 * self.lat[0, :] + self.lat[1, :]
            self.lat = np.concatenate(
                (
                    self.lat[-halosize:, :] - latshift[np.newaxis, :],
                    self.lat,
                    self.lat[0:halosize, :] + latshift[np.newaxis, :],
                ),
                axis=len(self.lat.shape) - 2,
            )
            self.lon = np.concatenate(
                (self.lon[-halosize:, :], self.lon, self.lon[0:halosize, :]), axis=len(self.lon.shape) - 2
            )
            self.xdim = self.lon.shape[1]
            self.ydim = self.lat.shape[0]
            self.meridional_halo = halosize
        if isinstance(self, CurvilinearSGrid):
            self.add_Sdepth_periodic_halo(zonal, meridional, halosize)


class CurvilinearZGrid(CurvilinearGrid):
    """Curvilinear Z Grid.

    Parameters
    ----------
    lon :
        2D array containing the longitude coordinates of the grid
    lat :
        2D array containing the latitude coordinates of the grid
    depth :
        Vector containing the vertical coordinates of the grid, which are z-coordinates.
        The depth of the different layers is thus constant.
    time :
        Vector containing the time coordinates of the grid
    time_origin : parcels.tools.converters.TimeConverter
        Time origin of the time axis
    mesh : str
        String indicating the type of mesh coordinates and
        units used during velocity interpolation:

        1. spherical (default): Lat and lon in degree, with a
           correction for zonal velocity U near the poles.
        2. flat: No conversion, lat/lon are assumed to be in m.
    """

    def __init__(
        self,
        lon: npt.NDArray,
        lat: npt.NDArray,
        depth: npt.NDArray | None = None,
        time: npt.NDArray | None = None,
        time_origin: TimeConverter | None = None,
        mesh: Mesh = "flat",
    ):
        super().__init__(lon, lat, time, time_origin, mesh)
        if isinstance(depth, np.ndarray):
            assert len(depth.shape) == 1, "depth is not a vector"

        self.gtype = GridType.CurvilinearZGrid
        self.depth = np.zeros(1, dtype=np.float32) if depth is None else depth
        if not self.depth.flags["C_CONTIGUOUS"]:
            self.depth = np.array(self.depth, order="C")
        self.zdim = self.depth.size
        self.z4d = -1  # only for SGrid
        if not self.depth.dtype == np.float32:
            self.depth = self.depth.astype(np.float32)


class CurvilinearSGrid(CurvilinearGrid):
    """Curvilinear S Grid.

    Parameters
    ----------
    lon :
        2D array containing the longitude coordinates of the grid
    lat :
        2D array containing the latitude coordinates of the grid
    depth :
        4D (time-evolving) or 3D (time-independent) array containing the vertical coordinates of the grid,
        which are s-coordinates.
        s-coordinates can be terrain-following (sigma) or iso-density (rho) layers,
        or any generalised vertical discretisation.
        The depth of each node depends then on the horizontal position (lon, lat),
        the number of the layer and the time is depth is a 4D array.
        depth array is either a 4D array[xdim][ydim][zdim][tdim] or a 3D array[xdim][ydim[zdim].
    time :
        Vector containing the time coordinates of the grid
    time_origin : parcels.tools.converters.TimeConverter
        Time origin of the time axis
    mesh : str
        String indicating the type of mesh coordinates and
        units used during velocity interpolation:

        1. spherical (default): Lat and lon in degree, with a
           correction for zonal velocity U near the poles.
        2. flat: No conversion, lat/lon are assumed to be in m.
    """

    def __init__(
        self,
        lon: npt.NDArray,
        lat: npt.NDArray,
        depth: npt.NDArray,
        time: npt.NDArray | None = None,
        time_origin: TimeConverter | None = None,
        mesh: Mesh = "flat",
    ):
        super().__init__(lon, lat, time, time_origin, mesh)
        assert isinstance(depth, np.ndarray) and len(depth.shape) in [3, 4], "depth is not a 4D numpy array"

        self.gtype = GridType.CurvilinearSGrid
        self.depth = depth  # should be a C-contiguous array of floats
        if not self.depth.flags["C_CONTIGUOUS"]:
            self.depth = np.array(self.depth, order="C")
        self.zdim = self.depth.shape[-3]
        self.z4d = 1 if len(self.depth.shape) == 4 else 0
        if self.z4d:
            # self.depth.shape[0] is 0 for S grids loaded from netcdf file
            assert (
                self.tdim == self.depth.shape[0] or self.depth.shape[0] == 0
            ), "depth dimension has the wrong format. It should be [tdim, zdim, ydim, xdim]"
            assert (
                self.xdim == self.depth.shape[-1] or self.depth.shape[-1] == 0
            ), "depth dimension has the wrong format. It should be [tdim, zdim, ydim, xdim]"
            assert (
                self.ydim == self.depth.shape[-2] or self.depth.shape[-2] == 0
            ), "depth dimension has the wrong format. It should be [tdim, zdim, ydim, xdim]"
        else:
            assert (
                self.xdim == self.depth.shape[-1]
            ), "depth dimension has the wrong format. It should be [zdim, ydim, xdim]"
            assert (
                self.ydim == self.depth.shape[-2]
            ), "depth dimension has the wrong format. It should be [zdim, ydim, xdim]"
        if not self.depth.dtype == np.float32:
            self.depth = self.depth.astype(np.float32)
