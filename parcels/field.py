import collections
import datetime
import math
from ctypes import POINTER, Structure, c_float, c_int, pointer
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Type

import dask.array as da
import numpy as np
import xarray as xr

import parcels.tools.interpolation_utils as i_u
from parcels._typing import GridIndexingType, InterpMethod, Mesh, TimePeriodic, VectorType
from parcels.tools.converters import (
    Geographic,
    GeographicPolar,
    TimeConverter,
    UnitConverter,
    unitconverters_map,
)
from parcels.tools.loggers import logger
from parcels.tools.statuscodes import (
    AllParcelsErrorCodes,
    FieldOutOfBoundError,
    FieldOutOfBoundSurfaceError,
    FieldSamplingError,
    TimeExtrapolationError,
)

from .fieldfilebuffer import (
    DaskFileBuffer,
    DeferredDaskFileBuffer,
    DeferredNetcdfFileBuffer,
    NetcdfFileBuffer,
)
from .grid import CGrid, Grid, GridType

if TYPE_CHECKING:
    from ctypes import _Pointer as PointerType

    from parcels.fieldset import FieldSet

__all__ = ["Field", "VectorField", "NestedField"]


def _isParticle(key):
    if hasattr(key, "obs_written"):
        return True
    else:
        return False


def _deal_with_errors(error, key, vector_type: VectorType):
    if _isParticle(key):
        key.state = AllParcelsErrorCodes[type(error)]
    elif _isParticle(key[-1]):
        key[-1].state = AllParcelsErrorCodes[type(error)]
    else:
        raise RuntimeError(f"{error}. Error could not be handled because particle was not part of the Field Sampling.")

    if vector_type == "3D":
        return (0, 0, 0)
    elif vector_type == "2D":
        return (0, 0)
    else:
        return 0


class Field:
    """Class that encapsulates access to field data.

    Parameters
    ----------
    name : str
        Name of the field
    data : np.ndarray
        2D, 3D or 4D numpy array of field data.

        1. If data shape is [xdim, ydim], [xdim, ydim, zdim], [xdim, ydim, tdim] or [xdim, ydim, zdim, tdim],
           whichever is relevant for the dataset, use the flag transpose=True
        2. If data shape is [ydim, xdim], [zdim, ydim, xdim], [tdim, ydim, xdim] or [tdim, zdim, ydim, xdim],
           use the flag transpose=False
        3. If data has any other shape, you first need to reorder it
    lon : np.ndarray or list
        Longitude coordinates (numpy vector or array) of the field (only if grid is None)
    lat : np.ndarray or list
        Latitude coordinates (numpy vector or array) of the field (only if grid is None)
    depth : np.ndarray or list
        Depth coordinates (numpy vector or array) of the field (only if grid is None)
    time : np.ndarray
        Time coordinates (numpy vector) of the field (only if grid is None)
    mesh : str
        String indicating the type of mesh coordinates and
        units used during velocity interpolation: (only if grid is None)

        1. spherical: Lat and lon in degree, with a
           correction for zonal velocity U near the poles.
        2. flat (default): No conversion, lat/lon are assumed to be in m.
    timestamps : np.ndarray
        A numpy array containing the timestamps for each of the files in filenames, for loading
        from netCDF files only. Default is None if the netCDF dimensions dictionary includes time.
    grid : parcels.grid.Grid
        :class:`parcels.grid.Grid` object containing all the lon, lat depth, time
        mesh and time_origin information. Can be constructed from any of the Grid objects
    fieldtype : str
        Type of Field to be used for UnitConverter (either 'U', 'V', 'Kh_zonal', 'Kh_meridional' or None)
    transpose : bool
        Transpose data to required (lon, lat) layout
    vmin : float
        Minimum allowed value on the field. Data below this value are set to zero
    vmax : float
        Maximum allowed value on the field. Data above this value are set to zero
    cast_data_dtype : str
        Cast Field data to dtype. Supported dtypes are "float32" (np.float32 (default)) and "float64 (np.float64).
        Note that dtype can only be "float32" in JIT mode
    time_origin : parcels.tools.converters.TimeConverter
        Time origin of the time axis (only if grid is None)
    interp_method : str
        Method for interpolation. Options are 'linear' (default), 'nearest',
        'linear_invdist_land_tracer', 'cgrid_velocity', 'cgrid_tracer' and 'bgrid_velocity'
    allow_time_extrapolation : bool
        boolean whether to allow for extrapolation in time
        (i.e. beyond the last available time snapshot)
    time_periodic : bool, float or datetime.timedelta
        To loop periodically over the time component of the Field. It is set to either False or the length of the period (either float in seconds or datetime.timedelta object).
        The last value of the time series can be provided (which is the same as the initial one) or not (Default: False)
        This flag overrides the allow_time_extrapolation and sets it to False
    chunkdims_name_map : str, optional
        Gives a name map to the FieldFileBuffer that declared a mapping between chunksize name, NetCDF dimension and Parcels dimension;
        required only if currently incompatible OCM field is loaded and chunking is used by 'chunksize' (which is the default)
    to_write : bool
        Write the Field in NetCDF format at the same frequency as the ParticleFile outputdt,
        using a filenaming scheme based on the ParticleFile name

    Examples
    --------
    For usage examples see the following tutorials:

    * `Nested Fields <../examples/tutorial_NestedFields.ipynb>`__
    """

    def __init__(
        self,
        name: str | tuple[str, str],
        data,
        lon=None,
        lat=None,
        depth=None,
        time=None,
        grid=None,
        mesh: Mesh = "flat",
        timestamps=None,
        fieldtype=None,
        transpose=False,
        vmin=None,
        vmax=None,
        cast_data_dtype="float32",
        time_origin=None,
        interp_method: InterpMethod = "linear",
        allow_time_extrapolation: bool | None = None,
        time_periodic: TimePeriodic = False,
        gridindexingtype: GridIndexingType = "nemo",
        to_write=False,
        **kwargs,
    ):
        if not isinstance(name, tuple):
            self.name = name
            self.filebuffername = name
        else:
            self.name = name[0]
            self.filebuffername = name[1]
        self.data = data
        if grid:
            if grid.defer_load and isinstance(data, np.ndarray):
                raise ValueError(
                    "Cannot combine Grid from defer_loaded Field with np.ndarray data. please specify lon, lat, depth and time dimensions separately"
                )
            self.grid = grid
        else:
            if (time is not None) and isinstance(time[0], np.datetime64):
                time_origin = TimeConverter(time[0])
                time = np.array([time_origin.reltime(t) for t in time])
            else:
                time_origin = TimeConverter(0)
            self.grid = Grid.create_grid(lon, lat, depth, time, time_origin=time_origin, mesh=mesh)
        self.igrid = -1
        # self.lon, self.lat, self.depth and self.time are not used any more in parcels.
        # self.grid should be used instead.
        # Those variables are still defined for backwards compatibility with users codes.
        self.lon = self.grid.lon
        self.lat = self.grid.lat
        self.depth = self.grid.depth
        self.fieldtype = self.name if fieldtype is None else fieldtype
        self.to_write = to_write
        if self.grid.mesh == "flat" or (self.fieldtype not in unitconverters_map.keys()):
            self.units = UnitConverter()
        elif self.grid.mesh == "spherical":
            self.units = unitconverters_map[self.fieldtype]
        else:
            raise ValueError("Unsupported mesh type. Choose either: 'spherical' or 'flat'")
        self.timestamps = timestamps
        if isinstance(interp_method, dict):
            if self.name in interp_method:
                self.interp_method = interp_method[self.name]
            else:
                raise RuntimeError(f"interp_method is a dictionary but {name} is not in it")
        else:
            self.interp_method = interp_method
        self.gridindexingtype = gridindexingtype
        if self.interp_method in ["bgrid_velocity", "bgrid_w_velocity", "bgrid_tracer"] and self.grid.gtype in [
            GridType.RectilinearSGrid,
            GridType.CurvilinearSGrid,
        ]:
            logger.warning_once(  # type: ignore
                "General s-levels are not supported in B-grid. RectilinearSGrid and CurvilinearSGrid can still be used to deal with shaved cells, but the levels must be horizontal."
            )

        self.fieldset: "FieldSet" | None = None
        if allow_time_extrapolation is None:
            self.allow_time_extrapolation = True if len(self.grid.time) == 1 else False
        else:
            self.allow_time_extrapolation = allow_time_extrapolation

        self.time_periodic = time_periodic
        if self.time_periodic is not False and self.allow_time_extrapolation:
            logger.warning_once(  # type: ignore
                "allow_time_extrapolation and time_periodic cannot be used together.\n \
                                 allow_time_extrapolation is set to False"
            )
            self.allow_time_extrapolation = False
        if self.time_periodic is True:
            raise ValueError(
                "Unsupported time_periodic=True. time_periodic must now be either False or the length of the period (either float in seconds or datetime.timedelta object."
            )
        if self.time_periodic is not False:
            if isinstance(self.time_periodic, datetime.timedelta):
                self.time_periodic = self.time_periodic.total_seconds()
            if not np.isclose(self.grid.time[-1] - self.grid.time[0], self.time_periodic):
                if self.grid.time[-1] - self.grid.time[0] > self.time_periodic:
                    raise ValueError("Time series provided is longer than the time_periodic parameter")
                self.grid._add_last_periodic_data_timestep = True
                self.grid.time = np.append(self.grid.time, self.grid.time[0] + self.time_periodic)
                self.grid.time_full = self.grid.time

        self.vmin = vmin
        self.vmax = vmax
        self.cast_data_dtype = cast_data_dtype
        if self.cast_data_dtype == "float32":
            self.cast_data_dtype = np.float32
        elif self.cast_data_dtype == "float64":
            self.cast_data_dtype = np.float64

        if not self.grid.defer_load:
            self.data = self.reshape(self.data, transpose)

            # Hack around the fact that NaN and ridiculously large values
            # propagate in SciPy's interpolators
            lib = np if isinstance(self.data, np.ndarray) else da
            self.data[lib.isnan(self.data)] = 0.0
            if self.vmin is not None:
                self.data[self.data < self.vmin] = 0.0
            if self.vmax is not None:
                self.data[self.data > self.vmax] = 0.0

            if self.grid._add_last_periodic_data_timestep:
                self.data = lib.concatenate((self.data, self.data[:1, :]), axis=0)

        self._scaling_factor = None

        # Variable names in JIT code
        self.dimensions = kwargs.pop("dimensions", None)
        self.indices = kwargs.pop("indices", None)
        self.dataFiles = kwargs.pop("dataFiles", None)
        if self.grid._add_last_periodic_data_timestep and self.dataFiles is not None:
            self.dataFiles = np.append(self.dataFiles, self.dataFiles[0])
        self._field_fb_class = kwargs.pop("FieldFileBuffer", None)
        self.netcdf_engine = kwargs.pop("netcdf_engine", "netcdf4")
        self.netcdf_decodewarning = kwargs.pop("netcdf_decodewarning", True)
        self.loaded_time_indices: Iterable[int] = []
        self.creation_log: str = kwargs.pop("creation_log", "")
        self.chunksize = kwargs.pop("chunksize", None)
        self.netcdf_chunkdims_name_map = kwargs.pop("chunkdims_name_map", None)
        self.grid.depth_field = kwargs.pop("depth_field", None)

        if self.grid.depth_field == "not_yet_set":
            assert (
                self.grid.z4d
            ), "Providing the depth dimensions from another field data is only available for 4d S grids"

        # data_full_zdim is the vertical dimension of the complete field data, ignoring the indices.
        # (data_full_zdim = grid.zdim if no indices are used, for A- and C-grids and for some B-grids). It is used for the B-grid,
        # since some datasets do not provide the deeper level of data (which is ignored by the interpolation).
        self.data_full_zdim = kwargs.pop("data_full_zdim", None)
        self.data_chunks = []  # type: ignore # the data buffer of the FileBuffer raw loaded data - shall be a list of C-contiguous arrays
        self.c_data_chunks: list["PointerType" | None] = []  # C-pointers to the data_chunks array
        self.nchunks: tuple[int, ...] = ()
        self.chunk_set: bool = False
        self.filebuffers = [None] * 2
        if len(kwargs) > 0:
            raise SyntaxError(f'Field received an unexpected keyword argument "{list(kwargs.keys())[0]}"')

    @classmethod
    def get_dim_filenames(cls, filenames, dim):
        if isinstance(filenames, str) or not isinstance(filenames, collections.abc.Iterable):
            return [filenames]
        elif isinstance(filenames, dict):
            assert dim in filenames.keys(), "filename dimension keys must be lon, lat, depth or data"
            filename = filenames[dim]
            if isinstance(filename, str):
                return [filename]
            else:
                return filename
        else:
            return filenames

    @staticmethod
    def collect_timeslices(
        timestamps, data_filenames, _grid_fb_class, dimensions, indices, netcdf_engine, netcdf_decodewarning=True
    ):
        if timestamps is not None:
            dataFiles = []
            for findex in range(len(data_filenames)):
                stamps_in_file = 1 if isinstance(timestamps[findex], (int, np.datetime64)) else len(timestamps[findex])
                for f in [data_filenames[findex]] * stamps_in_file:
                    dataFiles.append(f)
            timeslices = np.array([stamp for file in timestamps for stamp in file])
            time = timeslices
        else:
            timeslices = []
            dataFiles = []
            for fname in data_filenames:
                with _grid_fb_class(
                    fname, dimensions, indices, netcdf_engine=netcdf_engine, netcdf_decodewarning=netcdf_decodewarning
                ) as filebuffer:
                    ftime = filebuffer.time
                    timeslices.append(ftime)
                    dataFiles.append([fname] * len(ftime))
            time = np.concatenate(timeslices).ravel()
            dataFiles = np.concatenate(dataFiles).ravel()
        if time.size == 1 and time[0] is None:
            time[0] = 0
        time_origin = TimeConverter(time[0])
        time = time_origin.reltime(time)

        if not np.all((time[1:] - time[:-1]) > 0):
            id_not_ordered = np.where(time[1:] < time[:-1])[0][0]
            raise AssertionError(
                f"Please make sure your netCDF files are ordered in time. First pair of non-ordered files: {dataFiles[id_not_ordered]}, {dataFiles[id_not_ordered + 1]}"
            )
        return time, time_origin, timeslices, dataFiles

    @classmethod
    def from_netcdf(
        cls,
        filenames,
        variable,
        dimensions,
        indices=None,
        grid=None,
        mesh: Mesh = "spherical",
        timestamps=None,
        allow_time_extrapolation: bool | None = None,
        time_periodic: TimePeriodic = False,
        deferred_load: bool = True,
        **kwargs,
    ) -> "Field":
        """Create field from netCDF file.

        Parameters
        ----------
        filenames : list of str or dict
            list of filenames to read for the field. filenames can be a list ``[files]`` or
            a dictionary ``{dim:[files]}`` (if lon, lat, depth and/or data not stored in same files as data)
            In the latter case, time values are in filenames[data]
        variable : dict, tuple of str or str
            Dict or tuple mapping field name to variable name in the NetCDF file.
        dimensions : dict
            Dictionary mapping variable names for the relevant dimensions in the NetCDF file
        indices :
            dictionary mapping indices for each dimension to read from file.
            This can be used for reading in only a subregion of the NetCDF file.
            Note that negative indices are not allowed. (Default value = None)
        mesh :
            String indicating the type of mesh coordinates and
            units used during velocity interpolation:

            1. spherical (default): Lat and lon in degree, with a
               correction for zonal velocity U near the poles.
            2. flat: No conversion, lat/lon are assumed to be in m.
        timestamps :
            A numpy array of datetime64 objects containing the timestamps for each of the files in filenames.
            Default is None if dimensions includes time.
        allow_time_extrapolation : bool
            boolean whether to allow for extrapolation in time
            (i.e. beyond the last available time snapshot)
            Default is False if dimensions includes time, else True
        time_periodic : bool, float or datetime.timedelta
            boolean whether to loop periodically over the time component of the FieldSet
            This flag overrides the allow_time_extrapolation and sets it to False (Default value = False)
        deferred_load : bool
            boolean whether to only pre-load data (in deferred mode) or
            fully load them (default: True). It is advised to deferred load the data, since in
            that case Parcels deals with a better memory management during particle set execution.
            deferred_load=False is however sometimes necessary for plotting the fields.
        gridindexingtype : str
            The type of gridindexing. Either 'nemo' (default) or 'mitgcm' are supported.
            See also the Grid indexing documentation on oceanparcels.org
        chunksize :
            size of the chunks in dask loading
        netcdf_decodewarning : bool
            Whether to show a warning id there is a problem decoding the netcdf files.
            Default is True, but in some cases where these warnings are expected, it may be useful to silence them
            by setting netcdf_decodewarning=False.
        grid :
             (Default value = None)
        **kwargs :
            Keyword arguments passed to the :class:`Field` constructor.

        Examples
        --------
        For usage examples see the following tutorial:

        * `Timestamps <../examples/tutorial_timestamps.ipynb>`__

        """
        # Ensure the timestamps array is compatible with the user-provided datafiles.
        if timestamps is not None:
            if isinstance(filenames, list):
                assert len(filenames) == len(
                    timestamps
                ), "Outer dimension of timestamps should correspond to number of files."
            elif isinstance(filenames, dict):
                for k in filenames.keys():
                    if k not in ["lat", "lon", "depth", "time"]:
                        if isinstance(filenames[k], list):
                            assert len(filenames[k]) == len(
                                timestamps
                            ), "Outer dimension of timestamps should correspond to number of files."
                        else:
                            assert (
                                len(timestamps) == 1
                            ), "Outer dimension of timestamps should correspond to number of files."
                        for t in timestamps:
                            assert isinstance(t, (list, np.ndarray)), "timestamps should be a list for each file"

            else:
                raise TypeError(
                    "Filenames type is inconsistent with manual timestamp provision." + "Should be dict or list"
                )

        if isinstance(variable, str):  # for backward compatibility with Parcels < 2.0.0
            variable = (variable, variable)
        elif isinstance(variable, dict):
            assert (
                len(variable) == 1
            ), "Field.from_netcdf() supports only one variable at a time. Use FieldSet.from_netcdf() for multiple variables."
            variable = tuple(variable.items())[0]
        assert (
            len(variable) == 2
        ), "The variable tuple must have length 2. Use FieldSet.from_netcdf() for multiple variables"

        data_filenames = cls.get_dim_filenames(filenames, "data")
        lonlat_filename = cls.get_dim_filenames(filenames, "lon")
        if isinstance(filenames, dict):
            assert len(lonlat_filename) == 1
        if lonlat_filename != cls.get_dim_filenames(filenames, "lat"):
            raise NotImplementedError(
                "longitude and latitude dimensions are currently processed together from one single file"
            )
        lonlat_filename = lonlat_filename[0]
        if "depth" in dimensions:
            depth_filename = cls.get_dim_filenames(filenames, "depth")
            if isinstance(filenames, dict) and len(depth_filename) != 1:
                raise NotImplementedError("Vertically adaptive meshes not implemented for from_netcdf()")
            depth_filename = depth_filename[0]

        netcdf_engine = kwargs.pop("netcdf_engine", "netcdf4")
        netcdf_decodewarning = kwargs.pop("netcdf_decodewarning", True)

        indices = {} if indices is None else indices.copy()
        for ind in indices:
            if len(indices[ind]) == 0:
                raise RuntimeError(f"Indices for {ind} can not be empty")
            assert np.min(indices[ind]) >= 0, (
                "Negative indices are currently not allowed in Parcels. "
                "This is related to the non-increasing dimension it could generate "
                "if the domain goes from lon[-4] to lon[6] for example. "
                "Please raise an issue on https://github.com/OceanParcels/parcels/issues "
                "if you would need such feature implemented."
            )

        interp_method: InterpMethod = kwargs.pop("interp_method", "linear")
        if type(interp_method) is dict:
            if variable[0] in interp_method:
                interp_method = interp_method[variable[0]]
            else:
                raise RuntimeError(f"interp_method is a dictionary but {variable[0]} is not in it")

        _grid_fb_class = NetcdfFileBuffer

        with _grid_fb_class(
            lonlat_filename, dimensions, indices, netcdf_engine, netcdf_decodewarning=netcdf_decodewarning
        ) as filebuffer:
            lon, lat = filebuffer.lonlat
            indices = filebuffer.indices
            # Check if parcels_mesh has been explicitly set in file
            if "parcels_mesh" in filebuffer.dataset.attrs:
                mesh = filebuffer.dataset.attrs["parcels_mesh"]

        if "depth" in dimensions:
            with _grid_fb_class(
                depth_filename,
                dimensions,
                indices,
                netcdf_engine,
                interp_method=interp_method,
                netcdf_decodewarning=netcdf_decodewarning,
            ) as filebuffer:
                filebuffer.name = filebuffer.parse_name(variable[1])
                if dimensions["depth"] == "not_yet_set":
                    depth = filebuffer.depth_dimensions
                    kwargs["depth_field"] = "not_yet_set"
                else:
                    depth = filebuffer.depth
                data_full_zdim = filebuffer.data_full_zdim
        else:
            indices["depth"] = [0]
            depth = np.zeros(1)
            data_full_zdim = 1

        kwargs["data_full_zdim"] = data_full_zdim

        if len(data_filenames) > 1 and "time" not in dimensions and timestamps is None:
            raise RuntimeError("Multiple files given but no time dimension specified")

        if grid is None:
            # Concatenate time variable to determine overall dimension
            # across multiple files
            time, time_origin, timeslices, dataFiles = cls.collect_timeslices(
                timestamps, data_filenames, _grid_fb_class, dimensions, indices, netcdf_engine, netcdf_decodewarning
            )
            grid = Grid.create_grid(lon, lat, depth, time, time_origin=time_origin, mesh=mesh)
            grid.timeslices = timeslices
            kwargs["dataFiles"] = dataFiles
        elif grid is not None and ("dataFiles" not in kwargs or kwargs["dataFiles"] is None):
            # ==== means: the field has a shared grid, but may have different data files, so we need to collect the
            # ==== correct file time series again.
            _, _, _, dataFiles = cls.collect_timeslices(
                timestamps, data_filenames, _grid_fb_class, dimensions, indices, netcdf_engine, netcdf_decodewarning
            )
            kwargs["dataFiles"] = dataFiles

        chunksize: bool | None = kwargs.get("chunksize", None)
        grid.chunksize = chunksize

        if "time" in indices:
            logger.warning_once("time dimension in indices is not necessary anymore. It is then ignored.")  # type: ignore

        if "full_load" in kwargs:  # for backward compatibility with Parcels < v2.0.0
            deferred_load = not kwargs["full_load"]

        if grid.time.size <= 2 or deferred_load is False:
            deferred_load = False

        _field_fb_class: Type[DeferredDaskFileBuffer | DaskFileBuffer | DeferredNetcdfFileBuffer | NetcdfFileBuffer]
        if chunksize not in [False, None]:
            if deferred_load:
                _field_fb_class = DeferredDaskFileBuffer
            else:
                _field_fb_class = DaskFileBuffer
        elif deferred_load:
            _field_fb_class = DeferredNetcdfFileBuffer
        else:
            _field_fb_class = NetcdfFileBuffer
        kwargs["FieldFileBuffer"] = _field_fb_class

        if not deferred_load:
            # Pre-allocate data before reading files into buffer
            data_list = []
            ti = 0
            for tslice, fname in zip(grid.timeslices, data_filenames, strict=True):
                with _field_fb_class(  # type: ignore[operator]
                    fname,
                    dimensions,
                    indices,
                    netcdf_engine,
                    interp_method=interp_method,
                    data_full_zdim=data_full_zdim,
                    chunksize=chunksize,
                    netcdf_decodewarning=netcdf_decodewarning,
                ) as filebuffer:
                    # If Field.from_netcdf is called directly, it may not have a 'data' dimension
                    # In that case, assume that 'name' is the data dimension
                    filebuffer.name = filebuffer.parse_name(variable[1])
                    buffer_data = filebuffer.data
                    if len(buffer_data.shape) == 4:
                        errormessage = (
                            f"Field {filebuffer.name} expecting a data shape of [tdim={grid.tdim}, zdim={grid.zdim}, "
                            f"ydim={grid.ydim - 2 * grid.meridional_halo}, xdim={grid.xdim - 2 * grid.zonal_halo}] "
                            f"but got shape {buffer_data.shape}. Flag transpose=True could help to reorder the data."
                        )
                        assert buffer_data.shape[0] == grid.tdim, errormessage
                        assert buffer_data.shape[2] == grid.ydim - 2 * grid.meridional_halo, errormessage
                        assert buffer_data.shape[3] == grid.xdim - 2 * grid.zonal_halo, errormessage

                    if len(buffer_data.shape) == 2:
                        data_list.append(buffer_data.reshape(sum(((len(tslice), 1), buffer_data.shape), ())))
                    elif len(buffer_data.shape) == 3:
                        if len(filebuffer.indices["depth"]) > 1:
                            data_list.append(buffer_data.reshape(sum(((1,), buffer_data.shape), ())))
                        else:
                            if type(tslice) not in [list, np.ndarray, da.Array, xr.DataArray]:
                                tslice = [tslice]
                            data_list.append(buffer_data.reshape(sum(((len(tslice), 1), buffer_data.shape[1:]), ())))
                    else:
                        data_list.append(buffer_data)
                    if type(tslice) not in [list, np.ndarray, da.Array, xr.DataArray]:
                        tslice = [tslice]
                ti += len(tslice)
            lib = np if isinstance(data_list[0], np.ndarray) else da
            data = lib.concatenate(data_list, axis=0)
        else:
            grid.defer_load = True
            grid.ti = -1
            data = DeferredArray()
            data.compute_shape(grid.xdim, grid.ydim, grid.zdim, grid.tdim, len(grid.timeslices))

        if allow_time_extrapolation is None:
            allow_time_extrapolation = False if "time" in dimensions else True

        kwargs["dimensions"] = dimensions.copy()
        kwargs["indices"] = indices
        kwargs["time_periodic"] = time_periodic
        kwargs["netcdf_engine"] = netcdf_engine
        kwargs["netcdf_decodewarning"] = netcdf_decodewarning

        return cls(
            variable,
            data,
            grid=grid,
            timestamps=timestamps,
            allow_time_extrapolation=allow_time_extrapolation,
            interp_method=interp_method,
            **kwargs,
        )

    @classmethod
    def from_xarray(
        cls,
        da: xr.DataArray,
        name: str,
        dimensions,
        mesh: Mesh = "spherical",
        allow_time_extrapolation: bool | None = None,
        time_periodic: TimePeriodic = False,
        **kwargs,
    ):
        """Create field from xarray Variable.

        Parameters
        ----------
        da : xr.DataArray
            Xarray DataArray
        name : str
            Name of the Field
        dimensions : dict
            Dictionary mapping variable names for the relevant dimensions in the DataArray
        mesh : str
            String indicating the type of mesh coordinates and
            units used during velocity interpolation:

            1. spherical (default): Lat and lon in degree, with a
               correction for zonal velocity U near the poles.
            2. flat: No conversion, lat/lon are assumed to be in m.
        allow_time_extrapolation : bool
            boolean whether to allow for extrapolation in time
            (i.e. beyond the last available time snapshot)
            Default is False if dimensions includes time, else True
        time_periodic : bool, float or datetime.timedelta
            boolean whether to loop periodically over the time component of the FieldSet
            This flag overrides the allow_time_extrapolation and sets it to False (Default value = False)
        **kwargs :
            Keyword arguments passed to the :class:`Field` constructor.
        """
        data = da.data
        interp_method = kwargs.pop("interp_method", "linear")

        time = da[dimensions["time"]].values if "time" in dimensions else np.array([0.0])
        depth = da[dimensions["depth"]].values if "depth" in dimensions else np.array([0])
        lon = da[dimensions["lon"]].values
        lat = da[dimensions["lat"]].values

        time_origin = TimeConverter(time[0])
        time = time_origin.reltime(time)

        grid = Grid.create_grid(lon, lat, depth, time, time_origin=time_origin, mesh=mesh)
        kwargs["time_periodic"] = time_periodic
        return cls(
            name,
            data,
            grid=grid,
            allow_time_extrapolation=allow_time_extrapolation,
            interp_method=interp_method,
            **kwargs,
        )

    def reshape(self, data, transpose=False):
        # Ensure that field data is the right data type
        if not isinstance(data, (np.ndarray, da.core.Array)):
            data = np.array(data)
        if (self.cast_data_dtype == np.float32) and (data.dtype != np.float32):
            data = data.astype(np.float32)
        elif (self.cast_data_dtype == np.float64) and (data.dtype != np.float64):
            data = data.astype(np.float64)
        lib = np if isinstance(data, np.ndarray) else da
        if transpose:
            data = lib.transpose(data)
        if self.grid.lat_flipped:
            data = lib.flip(data, axis=-2)

        if self.grid.xdim == 1 or self.grid.ydim == 1:
            data = lib.squeeze(data)  # First remove all length-1 dimensions in data, so that we can add them below
        if self.grid.xdim == 1 and len(data.shape) < 4:
            if lib == da:
                raise NotImplementedError(
                    "Length-one dimensions with field chunking not implemented, as dask does not have an `expand_dims` method. Use chunksize=None"
                )
            data = lib.expand_dims(data, axis=-1)
        if self.grid.ydim == 1 and len(data.shape) < 4:
            if lib == da:
                raise NotImplementedError(
                    "Length-one dimensions with field chunking not implemented, as dask does not have an `expand_dims` method. Use chunksize=None"
                )
            data = lib.expand_dims(data, axis=-2)
        if self.grid.tdim == 1:
            if len(data.shape) < 4:
                data = data.reshape(sum(((1,), data.shape), ()))
        if self.grid.zdim == 1:
            if len(data.shape) == 4:
                data = data.reshape(sum(((data.shape[0],), data.shape[2:]), ()))
        if len(data.shape) == 4:
            errormessage = (
                f"Field {self.name} expecting a data shape of [tdim, zdim, ydim, xdim]. "
                "Flag transpose=True could help to reorder the data."
            )
            assert data.shape[0] == self.grid.tdim, errormessage
            assert data.shape[2] == self.grid.ydim - 2 * self.grid.meridional_halo, errormessage
            assert data.shape[3] == self.grid.xdim - 2 * self.grid.zonal_halo, errormessage
            if self.gridindexingtype == "pop":
                assert data.shape[1] == self.grid.zdim or data.shape[1] == self.grid.zdim - 1, errormessage
            else:
                assert data.shape[1] == self.grid.zdim, errormessage
        else:
            assert data.shape == (
                self.grid.tdim,
                self.grid.ydim - 2 * self.grid.meridional_halo,
                self.grid.xdim - 2 * self.grid.zonal_halo,
            ), (
                f"Field {self.name} expecting a data shape of [tdim, ydim, xdim]. "
                "Flag transpose=True could help to reorder the data."
            )
        if self.grid.meridional_halo > 0 or self.grid.zonal_halo > 0:
            data = self.add_periodic_halo(
                zonal=self.grid.zonal_halo > 0,
                meridional=self.grid.meridional_halo > 0,
                halosize=max(self.grid.meridional_halo, self.grid.zonal_halo),
                data=data,
            )
        return data

    def set_scaling_factor(self, factor):
        """Scales the field data by some constant factor.

        Parameters
        ----------
        factor :
            scaling factor


        Examples
        --------
        For usage examples see the following tutorial:

        * `Unit converters <../examples/tutorial_unitconverters.ipynb>`__
        """
        if self._scaling_factor:
            raise NotImplementedError(f"Scaling factor for field {self.name} already defined.")
        self._scaling_factor = factor
        if not self.grid.defer_load:
            self.data *= factor

    def set_depth_from_field(self, field):
        """Define the depth dimensions from another (time-varying) field.

        Notes
        -----
        See `this tutorial <../examples/tutorial_timevaryingdepthdimensions.ipynb>`__
        for a detailed explanation on how to set up time-evolving depth dimensions.

        """
        self.grid.depth_field = field
        if self.grid != field.grid:
            field.grid.depth_field = field

    def calc_cell_edge_sizes(self):
        """Method to calculate cell sizes based on numpy.gradient method.

        Currently only works for Rectilinear Grids
        """
        if not self.grid.cell_edge_sizes:
            if self.grid.gtype in (GridType.RectilinearZGrid, GridType.RectilinearSGrid):
                self.grid.cell_edge_sizes["x"] = np.zeros((self.grid.ydim, self.grid.xdim), dtype=np.float32)
                self.grid.cell_edge_sizes["y"] = np.zeros((self.grid.ydim, self.grid.xdim), dtype=np.float32)

                x_conv = GeographicPolar() if self.grid.mesh == "spherical" else UnitConverter()
                y_conv = Geographic() if self.grid.mesh == "spherical" else UnitConverter()
                for y, (lat, dy) in enumerate(zip(self.grid.lat, np.gradient(self.grid.lat), strict=False)):
                    for x, (lon, dx) in enumerate(zip(self.grid.lon, np.gradient(self.grid.lon), strict=False)):
                        self.grid.cell_edge_sizes["x"][y, x] = x_conv.to_source(dx, lon, lat, self.grid.depth[0])
                        self.grid.cell_edge_sizes["y"][y, x] = y_conv.to_source(dy, lon, lat, self.grid.depth[0])
                self.cell_edge_sizes = self.grid.cell_edge_sizes
            else:
                logger.error(
                    (
                        "Field.cell_edge_sizes() not implemented for ",
                        self.grid.gtype,
                        "grids.",
                        "You can provide Field.grid.cell_edge_sizes yourself",
                        "by in e.g. NEMO using the e1u fields etc from the mesh_mask.nc file",
                    )
                )
                exit(-1)

    def cell_areas(self):
        """Method to calculate cell sizes based on cell_edge_sizes.

        Currently only works for Rectilinear Grids
        """
        if not self.grid.cell_edge_sizes:
            self.calc_cell_edge_sizes()
        return self.grid.cell_edge_sizes["x"] * self.grid.cell_edge_sizes["y"]

    def search_indices_vertical_z(self, z):
        grid = self.grid
        z = np.float32(z)
        if grid.depth[-1] > grid.depth[0]:
            if z < grid.depth[0]:
                # Since MOM5 is indexed at cell bottom, allow z at depth[0] - dz where dz = (depth[1] - depth[0])
                if self.gridindexingtype == "mom5" and z > 2 * grid.depth[0] - grid.depth[1]:
                    return (-1, z / grid.depth[0])
                else:
                    raise FieldOutOfBoundSurfaceError(0, 0, z, field=self)
            elif z > grid.depth[-1]:
                raise FieldOutOfBoundError(0, 0, z, field=self)
            depth_indices = grid.depth <= z
            if z >= grid.depth[-1]:
                zi = len(grid.depth) - 2
            else:
                zi = depth_indices.argmin() - 1 if z >= grid.depth[0] else 0
        else:
            if z > grid.depth[0]:
                raise FieldOutOfBoundSurfaceError(0, 0, z, field=self)
            elif z < grid.depth[-1]:
                raise FieldOutOfBoundError(0, 0, z, field=self)
            depth_indices = grid.depth >= z
            if z <= grid.depth[-1]:
                zi = len(grid.depth) - 2
            else:
                zi = depth_indices.argmin() - 1 if z <= grid.depth[0] else 0
        zeta = (z - grid.depth[zi]) / (grid.depth[zi + 1] - grid.depth[zi])
        return (zi, zeta)

    def search_indices_vertical_s(
        self, x: float, y: float, z: float, xi: int, yi: int, xsi: float, eta: float, ti: int, time: float
    ):
        grid = self.grid
        if self.interp_method in ["bgrid_velocity", "bgrid_w_velocity", "bgrid_tracer"]:
            xsi = 1
            eta = 1
        if time < grid.time[ti]:
            ti -= 1
        if grid.z4d:
            if ti == len(grid.time) - 1:
                depth_vector = (
                    (1 - xsi) * (1 - eta) * grid.depth[-1, :, yi, xi]
                    + xsi * (1 - eta) * grid.depth[-1, :, yi, xi + 1]
                    + xsi * eta * grid.depth[-1, :, yi + 1, xi + 1]
                    + (1 - xsi) * eta * grid.depth[-1, :, yi + 1, xi]
                )
            else:
                dv2 = (
                    (1 - xsi) * (1 - eta) * grid.depth[ti : ti + 2, :, yi, xi]
                    + xsi * (1 - eta) * grid.depth[ti : ti + 2, :, yi, xi + 1]
                    + xsi * eta * grid.depth[ti : ti + 2, :, yi + 1, xi + 1]
                    + (1 - xsi) * eta * grid.depth[ti : ti + 2, :, yi + 1, xi]
                )
                tt = (time - grid.time[ti]) / (grid.time[ti + 1] - grid.time[ti])
                assert tt >= 0 and tt <= 1, "Vertical s grid is being wrongly interpolated in time"
                depth_vector = dv2[0, :] * (1 - tt) + dv2[1, :] * tt
        else:
            depth_vector = (
                (1 - xsi) * (1 - eta) * grid.depth[:, yi, xi]
                + xsi * (1 - eta) * grid.depth[:, yi, xi + 1]
                + xsi * eta * grid.depth[:, yi + 1, xi + 1]
                + (1 - xsi) * eta * grid.depth[:, yi + 1, xi]
            )
        z = np.float32(z)  # type: ignore # TODO: remove type ignore once we migrate to float64

        if depth_vector[-1] > depth_vector[0]:
            depth_indices = depth_vector <= z
            if z >= depth_vector[-1]:
                zi = len(depth_vector) - 2
            else:
                zi = depth_indices.argmin() - 1 if z >= depth_vector[0] else 0
            if z < depth_vector[zi]:
                raise FieldOutOfBoundSurfaceError(0, 0, z, field=self)
            elif z > depth_vector[zi + 1]:
                raise FieldOutOfBoundError(x, y, z, field=self)
        else:
            depth_indices = depth_vector >= z
            if z <= depth_vector[-1]:
                zi = len(depth_vector) - 2
            else:
                zi = depth_indices.argmin() - 1 if z <= depth_vector[0] else 0
            if z > depth_vector[zi]:
                raise FieldOutOfBoundSurfaceError(0, 0, z, field=self)
            elif z < depth_vector[zi + 1]:
                raise FieldOutOfBoundError(x, y, z, field=self)
        zeta = (z - depth_vector[zi]) / (depth_vector[zi + 1] - depth_vector[zi])
        return (zi, zeta)

    def reconnect_bnd_indices(self, xi, yi, xdim, ydim, sphere_mesh):
        if xi < 0:
            if sphere_mesh:
                xi = xdim - 2
            else:
                xi = 0
        if xi > xdim - 2:
            if sphere_mesh:
                xi = 0
            else:
                xi = xdim - 2
        if yi < 0:
            yi = 0
        if yi > ydim - 2:
            yi = ydim - 2
            if sphere_mesh:
                xi = xdim - xi
        return xi, yi

    def search_indices_rectilinear(self, x: float, y: float, z: float, ti=-1, time=-1, particle=None, search2D=False):
        grid = self.grid

        if grid.xdim > 1 and (not grid.zonal_periodic):
            if x < grid.lonlat_minmax[0] or x > grid.lonlat_minmax[1]:
                raise FieldOutOfBoundError(x, y, z, field=self)
        if grid.ydim > 1 and (y < grid.lonlat_minmax[2] or y > grid.lonlat_minmax[3]):
            raise FieldOutOfBoundError(x, y, z, field=self)

        if grid.xdim > 1:
            if grid.mesh != "spherical":
                lon_index = grid.lon < x
                if lon_index.all():
                    xi = len(grid.lon) - 2
                else:
                    xi = lon_index.argmin() - 1 if lon_index.any() else 0
                xsi = (x - grid.lon[xi]) / (grid.lon[xi + 1] - grid.lon[xi])
                if xsi < 0:
                    xi -= 1
                    xsi = (x - grid.lon[xi]) / (grid.lon[xi + 1] - grid.lon[xi])
                elif xsi > 1:
                    xi += 1
                    xsi = (x - grid.lon[xi]) / (grid.lon[xi + 1] - grid.lon[xi])
            else:
                lon_fixed = grid.lon.copy()
                indices = lon_fixed >= lon_fixed[0]
                if not indices.all():
                    lon_fixed[indices.argmin() :] += 360
                if x < lon_fixed[0]:
                    lon_fixed -= 360

                lon_index = lon_fixed < x
                if lon_index.all():
                    xi = len(lon_fixed) - 2
                else:
                    xi = lon_index.argmin() - 1 if lon_index.any() else 0
                xsi = (x - lon_fixed[xi]) / (lon_fixed[xi + 1] - lon_fixed[xi])
                if xsi < 0:
                    xi -= 1
                    xsi = (x - lon_fixed[xi]) / (lon_fixed[xi + 1] - lon_fixed[xi])
                elif xsi > 1:
                    xi += 1
                    xsi = (x - lon_fixed[xi]) / (lon_fixed[xi + 1] - lon_fixed[xi])
        else:
            xi, xsi = -1, 0

        if grid.ydim > 1:
            lat_index = grid.lat < y
            if lat_index.all():
                yi = len(grid.lat) - 2
            else:
                yi = lat_index.argmin() - 1 if lat_index.any() else 0

            eta = (y - grid.lat[yi]) / (grid.lat[yi + 1] - grid.lat[yi])
            if eta < 0:
                yi -= 1
                eta = (y - grid.lat[yi]) / (grid.lat[yi + 1] - grid.lat[yi])
            elif eta > 1:
                yi += 1
                eta = (y - grid.lat[yi]) / (grid.lat[yi + 1] - grid.lat[yi])
        else:
            yi, eta = -1, 0

        if grid.zdim > 1 and not search2D:
            if grid.gtype == GridType.RectilinearZGrid:
                # Never passes here, because in this case, we work with scipy
                try:
                    (zi, zeta) = self.search_indices_vertical_z(z)
                except FieldOutOfBoundError:
                    raise FieldOutOfBoundError(x, y, z, field=self)
                except FieldOutOfBoundSurfaceError:
                    raise FieldOutOfBoundSurfaceError(x, y, z, field=self)
            elif grid.gtype == GridType.RectilinearSGrid:
                (zi, zeta) = self.search_indices_vertical_s(x, y, z, xi, yi, xsi, eta, ti, time)
        else:
            zi, zeta = -1, 0

        if not ((0 <= xsi <= 1) and (0 <= eta <= 1) and (0 <= zeta <= 1)):
            raise FieldSamplingError(x, y, z, field=self)

        if particle:
            particle.xi[self.igrid] = xi
            particle.yi[self.igrid] = yi
            particle.zi[self.igrid] = zi

        return (xsi, eta, zeta, xi, yi, zi)

    def search_indices_curvilinear(self, x, y, z, ti=-1, time=-1, particle=None, search2D=False):
        if particle:
            xi = particle.xi[self.igrid]
            yi = particle.yi[self.igrid]
        else:
            xi = int(self.grid.xdim / 2) - 1
            yi = int(self.grid.ydim / 2) - 1
        xsi = eta = -1
        grid = self.grid
        invA = np.array([[1, 0, 0, 0], [-1, 1, 0, 0], [-1, 0, 0, 1], [1, -1, 1, -1]])
        maxIterSearch = 1e6
        it = 0
        tol = 1.0e-10
        if not grid.zonal_periodic:
            if x < grid.lonlat_minmax[0] or x > grid.lonlat_minmax[1]:
                if grid.lon[0, 0] < grid.lon[0, -1]:
                    raise FieldOutOfBoundError(x, y, z, field=self)
                elif x < grid.lon[0, 0] and x > grid.lon[0, -1]:  # This prevents from crashing in [160, -160]
                    raise FieldOutOfBoundError(x, y, z, field=self)
        if y < grid.lonlat_minmax[2] or y > grid.lonlat_minmax[3]:
            raise FieldOutOfBoundError(x, y, z, field=self)

        while xsi < -tol or xsi > 1 + tol or eta < -tol or eta > 1 + tol:
            px = np.array([grid.lon[yi, xi], grid.lon[yi, xi + 1], grid.lon[yi + 1, xi + 1], grid.lon[yi + 1, xi]])
            if grid.mesh == "spherical":
                px[0] = px[0] + 360 if px[0] < x - 225 else px[0]
                px[0] = px[0] - 360 if px[0] > x + 225 else px[0]
                px[1:] = np.where(px[1:] - px[0] > 180, px[1:] - 360, px[1:])
                px[1:] = np.where(-px[1:] + px[0] > 180, px[1:] + 360, px[1:])
            py = np.array([grid.lat[yi, xi], grid.lat[yi, xi + 1], grid.lat[yi + 1, xi + 1], grid.lat[yi + 1, xi]])
            a = np.dot(invA, px)
            b = np.dot(invA, py)

            aa = a[3] * b[2] - a[2] * b[3]
            bb = a[3] * b[0] - a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + x * b[3] - y * a[3]
            cc = a[1] * b[0] - a[0] * b[1] + x * b[1] - y * a[1]
            if abs(aa) < 1e-12:  # Rectilinear cell, or quasi
                eta = -cc / bb
            else:
                det2 = bb * bb - 4 * aa * cc
                if det2 > 0:  # so, if det is nan we keep the xsi, eta from previous iter
                    det = np.sqrt(det2)
                    eta = (-bb + det) / (2 * aa)
            if abs(a[1] + a[3] * eta) < 1e-12:  # this happens when recti cell rotated of 90deg
                xsi = ((y - py[0]) / (py[1] - py[0]) + (y - py[3]) / (py[2] - py[3])) * 0.5
            else:
                xsi = (x - a[0] - a[2] * eta) / (a[1] + a[3] * eta)
            if xsi < 0 and eta < 0 and xi == 0 and yi == 0:
                raise FieldOutOfBoundError(x, y, 0, field=self)
            if xsi > 1 and eta > 1 and xi == grid.xdim - 1 and yi == grid.ydim - 1:
                raise FieldOutOfBoundError(x, y, 0, field=self)
            if xsi < -tol:
                xi -= 1
            elif xsi > 1 + tol:
                xi += 1
            if eta < -tol:
                yi -= 1
            elif eta > 1 + tol:
                yi += 1
            (xi, yi) = self.reconnect_bnd_indices(xi, yi, grid.xdim, grid.ydim, grid.mesh)
            it += 1
            if it > maxIterSearch:
                print("Correct cell not found after %d iterations" % maxIterSearch)
                raise FieldOutOfBoundError(x, y, 0, field=self)
        xsi = max(0.0, xsi)
        eta = max(0.0, eta)
        xsi = min(1.0, xsi)
        eta = min(1.0, eta)

        if grid.zdim > 1 and not search2D:
            if grid.gtype == GridType.CurvilinearZGrid:
                try:
                    (zi, zeta) = self.search_indices_vertical_z(z)
                except FieldOutOfBoundError:
                    raise FieldOutOfBoundError(x, y, z, field=self)
            elif grid.gtype == GridType.CurvilinearSGrid:
                (zi, zeta) = self.search_indices_vertical_s(x, y, z, xi, yi, xsi, eta, ti, time)
        else:
            zi = -1
            zeta = 0

        if not ((0 <= xsi <= 1) and (0 <= eta <= 1) and (0 <= zeta <= 1)):
            raise FieldSamplingError(x, y, z, field=self)

        if particle:
            particle.xi[self.igrid] = xi
            particle.yi[self.igrid] = yi
            particle.zi[self.igrid] = zi

        return (xsi, eta, zeta, xi, yi, zi)

    def search_indices(self, x, y, z, ti=-1, time=-1, particle=None, search2D=False):
        if self.grid.gtype in [GridType.RectilinearSGrid, GridType.RectilinearZGrid]:
            return self.search_indices_rectilinear(x, y, z, ti, time, particle=particle, search2D=search2D)
        else:
            return self.search_indices_curvilinear(x, y, z, ti, time, particle=particle, search2D=search2D)

    def interpolator2D(self, ti, z, y, x, particle=None):
        (xsi, eta, _, xi, yi, _) = self.search_indices(x, y, z, particle=particle)
        if self.interp_method == "nearest":
            xii = xi if xsi <= 0.5 else xi + 1
            yii = yi if eta <= 0.5 else yi + 1
            return self.data[ti, yii, xii]
        elif self.interp_method in ["linear", "bgrid_velocity", "partialslip", "freeslip"]:
            val = (
                (1 - xsi) * (1 - eta) * self.data[ti, yi, xi]
                + xsi * (1 - eta) * self.data[ti, yi, xi + 1]
                + xsi * eta * self.data[ti, yi + 1, xi + 1]
                + (1 - xsi) * eta * self.data[ti, yi + 1, xi]
            )
            return val
        elif self.interp_method == "linear_invdist_land_tracer":
            land = np.isclose(self.data[ti, yi : yi + 2, xi : xi + 2], 0.0)
            nb_land = np.sum(land)
            if nb_land == 4:
                return 0
            elif nb_land > 0:
                val = 0
                w_sum = 0
                for j in range(2):
                    for i in range(2):
                        distance = pow((eta - j), 2) + pow((xsi - i), 2)
                        if np.isclose(distance, 0):
                            if land[j][i] == 1:  # index search led us directly onto land
                                return 0
                            else:
                                return self.data[ti, yi + j, xi + i]
                        elif land[j][i] == 0:
                            val += self.data[ti, yi + j, xi + i] / distance
                            w_sum += 1 / distance
                return val / w_sum
            else:
                val = (
                    (1 - xsi) * (1 - eta) * self.data[ti, yi, xi]
                    + xsi * (1 - eta) * self.data[ti, yi, xi + 1]
                    + xsi * eta * self.data[ti, yi + 1, xi + 1]
                    + (1 - xsi) * eta * self.data[ti, yi + 1, xi]
                )
                return val
        elif self.interp_method in ["cgrid_tracer", "bgrid_tracer"]:
            return self.data[ti, yi + 1, xi + 1]
        elif self.interp_method == "cgrid_velocity":
            raise RuntimeError(
                f"{self.name} is a scalar field. cgrid_velocity interpolation method should be used for vector fields (e.g. FieldSet.UV)"
            )
        else:
            raise RuntimeError(self.interp_method + " is not implemented for 2D grids")

    def interpolator3D(self, ti, z, y, x, time, particle=None):
        (xsi, eta, zeta, xi, yi, zi) = self.search_indices(x, y, z, ti, time, particle=particle)
        if self.interp_method == "nearest":
            xii = xi if xsi <= 0.5 else xi + 1
            yii = yi if eta <= 0.5 else yi + 1
            zii = zi if zeta <= 0.5 else zi + 1
            return self.data[ti, zii, yii, xii]
        elif self.interp_method == "cgrid_velocity":
            # evaluating W velocity in c_grid
            if self.gridindexingtype == "nemo":
                f0 = self.data[ti, zi, yi + 1, xi + 1]
                f1 = self.data[ti, zi + 1, yi + 1, xi + 1]
            elif self.gridindexingtype == "mitgcm":
                f0 = self.data[ti, zi, yi, xi]
                f1 = self.data[ti, zi + 1, yi, xi]
            return (1 - zeta) * f0 + zeta * f1
        elif self.interp_method == "linear_invdist_land_tracer":
            land = np.isclose(self.data[ti, zi : zi + 2, yi : yi + 2, xi : xi + 2], 0.0)
            nb_land = np.sum(land)
            if nb_land == 8:
                return 0
            elif nb_land > 0:
                val = 0
                w_sum = 0
                for k in range(2):
                    for j in range(2):
                        for i in range(2):
                            distance = pow((zeta - k), 2) + pow((eta - j), 2) + pow((xsi - i), 2)
                            if np.isclose(distance, 0):
                                if land[k][j][i] == 1:  # index search led us directly onto land
                                    return 0
                                else:
                                    return self.data[ti, zi + k, yi + j, xi + i]
                            elif land[k][j][i] == 0:
                                val += self.data[ti, zi + k, yi + j, xi + i] / distance
                                w_sum += 1 / distance
                return val / w_sum
            else:
                data = self.data[ti, zi, :, :]
                f0 = (
                    (1 - xsi) * (1 - eta) * data[yi, xi]
                    + xsi * (1 - eta) * data[yi, xi + 1]
                    + xsi * eta * data[yi + 1, xi + 1]
                    + (1 - xsi) * eta * data[yi + 1, xi]
                )
                data = self.data[ti, zi + 1, :, :]
                f1 = (
                    (1 - xsi) * (1 - eta) * data[yi, xi]
                    + xsi * (1 - eta) * data[yi, xi + 1]
                    + xsi * eta * data[yi + 1, xi + 1]
                    + (1 - xsi) * eta * data[yi + 1, xi]
                )
                return (1 - zeta) * f0 + zeta * f1
        elif self.interp_method in ["linear", "bgrid_velocity", "bgrid_w_velocity", "partialslip", "freeslip"]:
            if self.interp_method == "bgrid_velocity":
                if self.gridindexingtype == "mom5":
                    zeta = 1.0
                else:
                    zeta = 0.0
            elif self.interp_method == "bgrid_w_velocity":
                eta = 1.0
                xsi = 1.0
            data = self.data[ti, zi, :, :]
            f0 = (
                (1 - xsi) * (1 - eta) * data[yi, xi]
                + xsi * (1 - eta) * data[yi, xi + 1]
                + xsi * eta * data[yi + 1, xi + 1]
                + (1 - xsi) * eta * data[yi + 1, xi]
            )
            if self.gridindexingtype == "pop" and zi >= self.grid.zdim - 2:
                # Since POP is indexed at cell top, allow linear interpolation of W to zero in lowest cell
                return (1 - zeta) * f0
            data = self.data[ti, zi + 1, :, :]
            f1 = (
                (1 - xsi) * (1 - eta) * data[yi, xi]
                + xsi * (1 - eta) * data[yi, xi + 1]
                + xsi * eta * data[yi + 1, xi + 1]
                + (1 - xsi) * eta * data[yi + 1, xi]
            )
            if self.interp_method == "bgrid_w_velocity" and self.gridindexingtype == "mom5" and zi == -1:
                # Since MOM5 is indexed at cell bottom, allow linear interpolation of W to zero in uppermost cell
                return zeta * f1
            else:
                return (1 - zeta) * f0 + zeta * f1
        elif self.interp_method in ["cgrid_tracer", "bgrid_tracer"]:
            return self.data[ti, zi, yi + 1, xi + 1]
        else:
            raise RuntimeError(self.interp_method + " is not implemented for 3D grids")

    def temporal_interpolate_fullfield(self, ti, time):
        """Calculate the data of a field between two snapshots using linear interpolation.

        Parameters
        ----------
        ti :
            Index in time array associated with time (via :func:`time_index`)
        time :
            Time to interpolate to
        """
        t0 = self.grid.time[ti]
        if time == t0:
            return self.data[ti, :]
        elif ti + 1 >= len(self.grid.time):
            raise TimeExtrapolationError(time, field=self, msg="show_time")
        else:
            t1 = self.grid.time[ti + 1]
            f0 = self.data[ti, :]
            f1 = self.data[ti + 1, :]
            return f0 + (f1 - f0) * ((time - t0) / (t1 - t0))

    def spatial_interpolation(self, ti, z, y, x, time, particle=None):
        """Interpolate horizontal field values using a SciPy interpolator."""
        if self.grid.zdim == 1:
            val = self.interpolator2D(ti, z, y, x, particle=particle)
        else:
            val = self.interpolator3D(ti, z, y, x, time, particle=particle)
        if np.isnan(val):
            # Detect Out-of-bounds sampling and raise exception
            raise FieldOutOfBoundError(x, y, z, field=self)
        else:
            if isinstance(val, da.core.Array):
                val = val.compute()
            return val

    def time_index(self, time):
        """Find the index in the time array associated with a given time.

        Note that we normalize to either the first or the last index
        if the sampled value is outside the time value range.
        """
        if (
            not self.time_periodic
            and not self.allow_time_extrapolation
            and (time < self.grid.time[0] or time > self.grid.time[-1])
        ):
            raise TimeExtrapolationError(time, field=self)
        time_index = self.grid.time <= time
        if self.time_periodic:
            if time_index.all() or np.logical_not(time_index).all():
                periods = int(
                    math.floor((time - self.grid.time_full[0]) / (self.grid.time_full[-1] - self.grid.time_full[0]))
                )
                if isinstance(self.grid.periods, c_int):
                    self.grid.periods.value = periods
                else:
                    self.grid.periods = periods
                time -= periods * (self.grid.time_full[-1] - self.grid.time_full[0])
                time_index = self.grid.time <= time
                ti = time_index.argmin() - 1 if time_index.any() else 0
                return (ti, periods)
            return (time_index.argmin() - 1 if time_index.any() else 0, 0)
        if time_index.all():
            # If given time > last known field time, use
            # the last field frame without interpolation
            return (len(self.grid.time) - 1, 0)
        elif np.logical_not(time_index).all():
            # If given time < any time in the field, use
            # the first field frame without interpolation
            return (0, 0)
        else:
            return (time_index.argmin() - 1 if time_index.any() else 0, 0)

    def _check_velocitysampling(self):
        if self.name in ["U", "V", "W"]:
            logger.warning_once(
                "Sampling of velocities should normally be done using fieldset.UV or fieldset.UVW object; tread carefully"
            )

    def __getitem__(self, key):
        self._check_velocitysampling()
        try:
            if _isParticle(key):
                return self.eval(key.time, key.depth, key.lat, key.lon, key)
            else:
                return self.eval(*key)
        except tuple(AllParcelsErrorCodes.keys()) as error:
            return _deal_with_errors(error, key, vector_type=None)

    def eval(self, time, z, y, x, particle=None, applyConversion=True):
        """Interpolate field values in space and time.

        We interpolate linearly in time and apply implicit unit
        conversion to the result. Note that we defer to
        scipy.interpolate to perform spatial interpolation.
        """
        (ti, periods) = self.time_index(time)
        time -= periods * (self.grid.time_full[-1] - self.grid.time_full[0])
        if ti < self.grid.tdim - 1 and time > self.grid.time[ti]:
            f0 = self.spatial_interpolation(ti, z, y, x, time, particle=particle)
            f1 = self.spatial_interpolation(ti + 1, z, y, x, time, particle=particle)
            t0 = self.grid.time[ti]
            t1 = self.grid.time[ti + 1]
            value = f0 + (f1 - f0) * ((time - t0) / (t1 - t0))
        else:
            # Skip temporal interpolation if time is outside
            # of the defined time range or if we have hit an
            # exact value in the time array.
            value = self.spatial_interpolation(ti, z, y, x, self.grid.time[ti], particle=particle)

        if applyConversion:
            return self.units.to_target(value, x, y, z)
        else:
            return value

    def ccode_eval(self, var, t, z, y, x):
        self._check_velocitysampling()
        ccode_str = f"temporal_interpolation({x}, {y}, {z}, {t}, {self.ccode_name}, &particles->xi[pnum*ngrid], &particles->yi[pnum*ngrid], &particles->zi[pnum*ngrid], &particles->ti[pnum*ngrid], &{var})"
        return ccode_str

    def ccode_convert(self, _, z, y, x):
        return self.units.ccode_to_target(x, y, z)

    def get_block_id(self, block):
        return np.ravel_multi_index(block, self.nchunks)

    def get_block(self, bid):
        return np.unravel_index(bid, self.nchunks[1:])

    def chunk_setup(self):
        if isinstance(self.data, da.core.Array):
            chunks = self.data.chunks
            self.nchunks = self.data.numblocks
            npartitions = 1
            for n in self.nchunks[1:]:
                npartitions *= n
        elif isinstance(self.data, np.ndarray):
            chunks = tuple((t,) for t in self.data.shape)
            self.nchunks = (1,) * len(self.data.shape)
            npartitions = 1
        elif isinstance(self.data, DeferredArray):
            self.nchunks = (1,) * len(self.data.data_shape)
            return
        else:
            return

        self.data_chunks = [None] * npartitions
        self.c_data_chunks = [None] * npartitions
        self.grid.load_chunk = np.zeros(npartitions, dtype=c_int, order="C")
        # self.grid.chunk_info format: number of dimensions (without tdim); number of chunks per dimensions;
        #      chunksizes (the 0th dim sizes for all chunk of dim[0], then so on for next dims
        self.grid.chunk_info = [
            [len(self.nchunks) - 1],
            list(self.nchunks[1:]),
            sum(list(list(ci) for ci in chunks[1:]), []),  # noqa: RUF017 # TODO: Perhaps avoid quadratic list summation here
        ]
        self.grid.chunk_info = sum(self.grid.chunk_info, [])  # noqa: RUF017
        self.chunk_set = True

    def chunk_data(self):
        if not self.chunk_set:
            self.chunk_setup()
        g = self.grid
        if isinstance(self.data, da.core.Array):
            for block_id in range(len(self.grid.load_chunk)):
                if (
                    g.load_chunk[block_id] == g.chunk_loading_requested
                    or g.load_chunk[block_id] in g.chunk_loaded
                    and self.data_chunks[block_id] is None
                ):
                    block = self.get_block(block_id)
                    self.data_chunks[block_id] = np.array(self.data.blocks[(slice(self.grid.tdim),) + block], order="C")
                elif g.load_chunk[block_id] == g.chunk_not_loaded:
                    if isinstance(self.data_chunks, list):
                        self.data_chunks[block_id] = None
                    else:
                        self.data_chunks[block_id, :] = None
                    self.c_data_chunks[block_id] = None
        else:
            if isinstance(self.data_chunks, list):
                self.data_chunks[0] = None
            else:
                self.data_chunks[0, :] = None
            self.c_data_chunks[0] = None
            self.grid.load_chunk[0] = g.chunk_loaded_touched
            self.data_chunks[0] = np.array(self.data, order="C")

    @property
    def ctypes_struct(self):
        """Returns a ctypes struct object containing all relevant pointers and sizes for this field."""

        # Ctypes struct corresponding to the type definition in parcels.h
        class CField(Structure):
            _fields_ = [
                ("xdim", c_int),
                ("ydim", c_int),
                ("zdim", c_int),
                ("tdim", c_int),
                ("igrid", c_int),
                ("allow_time_extrapolation", c_int),
                ("time_periodic", c_int),
                ("gridindexingtype", c_int),
                ("interp_method", c_int),
                ("data_chunks", POINTER(POINTER(POINTER(c_float)))),
                ("grid", POINTER(CGrid)),
            ]

        # Create and populate the c-struct object
        allow_time_extrapolation = 1 if self.allow_time_extrapolation else 0
        time_periodic = 1 if self.time_periodic else 0
        for i in range(len(self.grid.load_chunk)):
            if self.grid.load_chunk[i] == self.grid.chunk_loading_requested:
                raise ValueError(
                    "data_chunks should have been loaded by now if requested. grid.load_chunk[bid] cannot be 1"
                )
            if self.grid.load_chunk[i] in self.grid.chunk_loaded:
                if not self.data_chunks[i].flags["C_CONTIGUOUS"]:
                    self.data_chunks[i] = np.array(self.data_chunks[i], order="C")
                self.c_data_chunks[i] = self.data_chunks[i].ctypes.data_as(POINTER(POINTER(c_float)))
            else:
                self.c_data_chunks[i] = None

        gridindexingtype_mapping = {"nemo": 0, "mitgcm": 1, "mom5": 2, "pop": 3}
        interp_method_mapping = {
            "LINEAR": 0,
            "NEAREST": 1,
            "CGRID_VELOCITY": 2,
            "CGRID_TRACER": 3,
            "BGRID_VELOCITY": 4,
            "BGRID_W_VELOCITY": 5,
            "BGRID_TRACER": 6,
            "LINEAR_INVDIST_LAND_TRACER": 7,
            "PARTIALSLIP": 8,
            "FREESLIP": 9,
        }

        cstruct = CField(
            self.grid.xdim,
            self.grid.ydim,
            self.grid.zdim,
            self.grid.tdim,
            self.igrid,
            allow_time_extrapolation,
            time_periodic,
            gridindexingtype_mapping[self.gridindexingtype],
            interp_method_mapping[self.interp_method.upper()],
            (POINTER(POINTER(c_float)) * len(self.c_data_chunks))(*self.c_data_chunks),
            pointer(self.grid.ctypes_struct),
        )
        return cstruct

    def add_periodic_halo(self, zonal, meridional, halosize=5, data=None):
        """Add a 'halo' to all Fields in a FieldSet.

        Add a 'halo' to all Fields in a FieldSet, through extending the Field (and lon/lat)
        by copying a small portion of the field on one side of the domain to the other.
        Before adding a periodic halo to the Field, it has to be added to the Grid on which the Field depends

        See `this tutorial <../examples/tutorial_periodic_boundaries.ipynb>`__
        for a detailed explanation on how to set up periodic boundaries

        Parameters
        ----------
        zonal : bool
            Create a halo in zonal direction.
        meridional : bool
            Create a halo in meridional direction.
        halosize : int
            Size of the halo (in grid points). Default is 5 grid points
        data :
            if data is not None, the periodic halo will be achieved on data instead of self.data and data will be returned (Default value = None)
        """
        dataNone = not isinstance(data, (np.ndarray, da.core.Array))
        if self.grid.defer_load and dataNone:
            return
        data = self.data if dataNone else data
        lib = np if isinstance(data, np.ndarray) else da
        if zonal:
            if len(data.shape) == 3:
                data = lib.concatenate((data[:, :, -halosize:], data, data[:, :, 0:halosize]), axis=len(data.shape) - 1)
                assert data.shape[2] == self.grid.xdim, "Third dim must be x."
            else:
                data = lib.concatenate(
                    (data[:, :, :, -halosize:], data, data[:, :, :, 0:halosize]), axis=len(data.shape) - 1
                )
                assert data.shape[3] == self.grid.xdim, "Fourth dim must be x."
            self.lon = self.grid.lon
            self.lat = self.grid.lat
        if meridional:
            if len(data.shape) == 3:
                data = lib.concatenate((data[:, -halosize:, :], data, data[:, 0:halosize, :]), axis=len(data.shape) - 2)
                assert data.shape[1] == self.grid.ydim, "Second dim must be y."
            else:
                data = lib.concatenate(
                    (data[:, :, -halosize:, :], data, data[:, :, 0:halosize, :]), axis=len(data.shape) - 2
                )
                assert data.shape[2] == self.grid.ydim, "Third dim must be y."
            self.lat = self.grid.lat
        if dataNone:
            self.data = data
        else:
            return data

    def write(self, filename, varname=None):
        """Write a :class:`Field` to a netcdf file.

        Parameters
        ----------
        filename : str
            Basename of the file (i.e. '{filename}{Field.name}.nc')
        varname : str
            Name of the field, to be appended to the filename. (Default value = None)
        """
        filepath = str(Path(f"{filename}{self.name}.nc"))
        if varname is None:
            varname = self.name
        # Derive name of 'depth' variable for NEMO convention
        vname_depth = f"depth{self.name.lower()}"

        # Create DataArray objects for file I/O
        if self.grid.gtype == GridType.RectilinearZGrid:
            nav_lon = xr.DataArray(
                self.grid.lon + np.zeros((self.grid.ydim, self.grid.xdim), dtype=np.float32),
                coords=[("y", self.grid.lat), ("x", self.grid.lon)],
            )
            nav_lat = xr.DataArray(
                self.grid.lat.reshape(self.grid.ydim, 1) + np.zeros(self.grid.xdim, dtype=np.float32),
                coords=[("y", self.grid.lat), ("x", self.grid.lon)],
            )
        elif self.grid.gtype == GridType.CurvilinearZGrid:
            nav_lon = xr.DataArray(self.grid.lon, coords=[("y", range(self.grid.ydim)), ("x", range(self.grid.xdim))])
            nav_lat = xr.DataArray(self.grid.lat, coords=[("y", range(self.grid.ydim)), ("x", range(self.grid.xdim))])
        else:
            raise NotImplementedError("Field.write only implemented for RectilinearZGrid and CurvilinearZGrid")

        attrs = {"units": "seconds since " + str(self.grid.time_origin)} if self.grid.time_origin.calendar else {}
        time_counter = xr.DataArray(self.grid.time, dims=["time_counter"], attrs=attrs)
        vardata = xr.DataArray(
            self.data.reshape((self.grid.tdim, self.grid.zdim, self.grid.ydim, self.grid.xdim)),
            dims=["time_counter", vname_depth, "y", "x"],
        )
        # Create xarray Dataset and output to netCDF format
        attrs = {"parcels_mesh": self.grid.mesh}
        dset = xr.Dataset(
            {varname: vardata},
            coords={"nav_lon": nav_lon, "nav_lat": nav_lat, "time_counter": time_counter, vname_depth: self.grid.depth},
            attrs=attrs,
        )
        dset.to_netcdf(filepath, unlimited_dims="time_counter")

    def rescale_and_set_minmax(self, data):
        data[np.isnan(data)] = 0
        if self._scaling_factor:
            data *= self._scaling_factor
        if self.vmin is not None:
            data[data < self.vmin] = 0
        if self.vmax is not None:
            data[data > self.vmax] = 0
        return data

    def data_concatenate(self, data, data_to_concat, tindex):
        if data[tindex] is not None:
            if isinstance(data, np.ndarray):
                data[tindex] = None
            elif isinstance(data, list):
                del data[tindex]
        lib = np if isinstance(data, np.ndarray) else da
        if tindex == 0:
            data = lib.concatenate([data_to_concat, data[tindex + 1 :, :]], axis=0)
        elif tindex == 1:
            data = lib.concatenate([data[:tindex, :], data_to_concat], axis=0)
        else:
            raise ValueError("data_concatenate is used for computeTimeChunk, with tindex in [0, 1]")
        return data

    def computeTimeChunk(self, data, tindex):
        g = self.grid
        timestamp = self.timestamps
        if timestamp is not None:
            summedlen = np.cumsum([len(ls) for ls in self.timestamps])
            if g.ti + tindex >= summedlen[-1]:
                ti = g.ti + tindex - summedlen[-1]
            else:
                ti = g.ti + tindex
            timestamp = self.timestamps[np.where(ti < summedlen)[0][0]]

        rechunk_callback_fields = self.chunk_setup if isinstance(tindex, list) else None
        filebuffer = self._field_fb_class(
            self.dataFiles[g.ti + tindex],
            self.dimensions,
            self.indices,
            netcdf_engine=self.netcdf_engine,
            timestamp=timestamp,
            interp_method=self.interp_method,
            data_full_zdim=self.data_full_zdim,
            chunksize=self.chunksize,
            cast_data_dtype=self.cast_data_dtype,
            rechunk_callback_fields=rechunk_callback_fields,
            chunkdims_name_map=self.netcdf_chunkdims_name_map,
            netcdf_decodewarning=self.netcdf_decodewarning,
        )
        filebuffer.__enter__()
        time_data = filebuffer.time
        time_data = g.time_origin.reltime(time_data)
        filebuffer.ti = (time_data <= g.time[tindex]).argmin() - 1
        if self.netcdf_engine != "xarray":
            filebuffer.name = filebuffer.parse_name(self.filebuffername)
        buffer_data = filebuffer.data
        lib = np if isinstance(buffer_data, np.ndarray) else da
        if len(buffer_data.shape) == 2:
            buffer_data = lib.reshape(buffer_data, sum(((1, 1), buffer_data.shape), ()))
        elif len(buffer_data.shape) == 3 and g.zdim > 1:
            buffer_data = lib.reshape(buffer_data, sum(((1,), buffer_data.shape), ()))
        elif len(buffer_data.shape) == 3:
            buffer_data = lib.reshape(
                buffer_data,
                sum(
                    (
                        (
                            buffer_data.shape[0],
                            1,
                        ),
                        buffer_data.shape[1:],
                    ),
                    (),
                ),
            )
        data = self.data_concatenate(data, buffer_data, tindex)
        self.filebuffers[tindex] = filebuffer
        return data


class VectorField:
    """Class VectorField stores 2 or 3 fields which defines together a vector field.
    This enables to interpolate them as one single vector field in the kernels.

    Parameters
    ----------
    name : str
        Name of the vector field
    U : parcels.field.Field
        field defining the zonal component
    V : parcels.field.Field
        field defining the meridional component
    W : parcels.field.Field
        field defining the vertical component (default: None)
    """

    def __init__(self, name: str, U: Field, V: Field, W: Field | None = None):
        self.name = name
        self.U = U
        self.V = V
        self.W = W
        self.vector_type: VectorType = "3D" if W else "2D"
        self.gridindexingtype = U.gridindexingtype
        if self.U.interp_method == "cgrid_velocity":
            assert self.V.interp_method == "cgrid_velocity", "Interpolation methods of U and V are not the same."
            assert self._check_grid_dimensions(U.grid, V.grid), "Dimensions of U and V are not the same."
            if W is not None:
                assert W.interp_method == "cgrid_velocity", "Interpolation methods of U and W are not the same."
                assert self._check_grid_dimensions(U.grid, W.grid), "Dimensions of U and W are not the same."

    @staticmethod
    def _check_grid_dimensions(grid1, grid2):
        return (
            np.allclose(grid1.lon, grid2.lon)
            and np.allclose(grid1.lat, grid2.lat)
            and np.allclose(grid1.depth, grid2.depth)
            and np.allclose(grid1.time_full, grid2.time_full)
        )

    def dist(self, lon1: float, lon2: float, lat1: float, lat2: float, mesh: Mesh, lat: float):
        if mesh == "spherical":
            rad = np.pi / 180.0
            deg2m = 1852 * 60.0
            return np.sqrt(((lon2 - lon1) * deg2m * math.cos(rad * lat)) ** 2 + ((lat2 - lat1) * deg2m) ** 2)
        else:
            return np.sqrt((lon2 - lon1) ** 2 + (lat2 - lat1) ** 2)

    def jacobian(self, xsi: float, eta: float, px: np.ndarray, py: np.ndarray):
        dphidxsi = [eta - 1, 1 - eta, eta, -eta]
        dphideta = [xsi - 1, -xsi, xsi, 1 - xsi]

        dxdxsi = np.dot(px, dphidxsi)
        dxdeta = np.dot(px, dphideta)
        dydxsi = np.dot(py, dphidxsi)
        dydeta = np.dot(py, dphideta)
        jac = dxdxsi * dydeta - dxdeta * dydxsi
        return jac

    def spatial_c_grid_interpolation2D(self, ti, z, y, x, time, particle=None, applyConversion=True):
        grid = self.U.grid
        (xsi, eta, zeta, xi, yi, zi) = self.U.search_indices(x, y, z, ti, time, particle=particle)

        if grid.gtype in [GridType.RectilinearSGrid, GridType.RectilinearZGrid]:
            px = np.array([grid.lon[xi], grid.lon[xi + 1], grid.lon[xi + 1], grid.lon[xi]])
            py = np.array([grid.lat[yi], grid.lat[yi], grid.lat[yi + 1], grid.lat[yi + 1]])
        else:
            px = np.array([grid.lon[yi, xi], grid.lon[yi, xi + 1], grid.lon[yi + 1, xi + 1], grid.lon[yi + 1, xi]])
            py = np.array([grid.lat[yi, xi], grid.lat[yi, xi + 1], grid.lat[yi + 1, xi + 1], grid.lat[yi + 1, xi]])

        if grid.mesh == "spherical":
            px[0] = px[0] + 360 if px[0] < x - 225 else px[0]
            px[0] = px[0] - 360 if px[0] > x + 225 else px[0]
            px[1:] = np.where(px[1:] - px[0] > 180, px[1:] - 360, px[1:])
            px[1:] = np.where(-px[1:] + px[0] > 180, px[1:] + 360, px[1:])
        xx = (1 - xsi) * (1 - eta) * px[0] + xsi * (1 - eta) * px[1] + xsi * eta * px[2] + (1 - xsi) * eta * px[3]
        assert abs(xx - x) < 1e-4
        c1 = self.dist(px[0], px[1], py[0], py[1], grid.mesh, np.dot(i_u.phi2D_lin(xsi, 0.0), py))
        c2 = self.dist(px[1], px[2], py[1], py[2], grid.mesh, np.dot(i_u.phi2D_lin(1.0, eta), py))
        c3 = self.dist(px[2], px[3], py[2], py[3], grid.mesh, np.dot(i_u.phi2D_lin(xsi, 1.0), py))
        c4 = self.dist(px[3], px[0], py[3], py[0], grid.mesh, np.dot(i_u.phi2D_lin(0.0, eta), py))
        if grid.zdim == 1:
            if self.gridindexingtype == "nemo":
                U0 = self.U.data[ti, yi + 1, xi] * c4
                U1 = self.U.data[ti, yi + 1, xi + 1] * c2
                V0 = self.V.data[ti, yi, xi + 1] * c1
                V1 = self.V.data[ti, yi + 1, xi + 1] * c3
            elif self.gridindexingtype == "mitgcm":
                U0 = self.U.data[ti, yi, xi] * c4
                U1 = self.U.data[ti, yi, xi + 1] * c2
                V0 = self.V.data[ti, yi, xi] * c1
                V1 = self.V.data[ti, yi + 1, xi] * c3
        else:
            if self.gridindexingtype == "nemo":
                U0 = self.U.data[ti, zi, yi + 1, xi] * c4
                U1 = self.U.data[ti, zi, yi + 1, xi + 1] * c2
                V0 = self.V.data[ti, zi, yi, xi + 1] * c1
                V1 = self.V.data[ti, zi, yi + 1, xi + 1] * c3
            elif self.gridindexingtype == "mitgcm":
                U0 = self.U.data[ti, zi, yi, xi] * c4
                U1 = self.U.data[ti, zi, yi, xi + 1] * c2
                V0 = self.V.data[ti, zi, yi, xi] * c1
                V1 = self.V.data[ti, zi, yi + 1, xi] * c3
        U = (1 - xsi) * U0 + xsi * U1
        V = (1 - eta) * V0 + eta * V1
        rad = np.pi / 180.0
        deg2m = 1852 * 60.0
        if applyConversion:
            meshJac = (deg2m * deg2m * math.cos(rad * y)) if grid.mesh == "spherical" else 1
        else:
            meshJac = deg2m if grid.mesh == "spherical" else 1

        jac = self.jacobian(xsi, eta, px, py) * meshJac

        u = (
            (-(1 - eta) * U - (1 - xsi) * V) * px[0]
            + ((1 - eta) * U - xsi * V) * px[1]
            + (eta * U + xsi * V) * px[2]
            + (-eta * U + (1 - xsi) * V) * px[3]
        ) / jac
        v = (
            (-(1 - eta) * U - (1 - xsi) * V) * py[0]
            + ((1 - eta) * U - xsi * V) * py[1]
            + (eta * U + xsi * V) * py[2]
            + (-eta * U + (1 - xsi) * V) * py[3]
        ) / jac
        if isinstance(u, da.core.Array):
            u = u.compute()
            v = v.compute()
        return (u, v)

    def spatial_c_grid_interpolation3D_full(self, ti, z, y, x, time, particle=None):
        grid = self.U.grid
        (xsi, eta, zet, xi, yi, zi) = self.U.search_indices(x, y, z, ti, time, particle=particle)

        if grid.gtype in [GridType.RectilinearSGrid, GridType.RectilinearZGrid]:
            px = np.array([grid.lon[xi], grid.lon[xi + 1], grid.lon[xi + 1], grid.lon[xi]])
            py = np.array([grid.lat[yi], grid.lat[yi], grid.lat[yi + 1], grid.lat[yi + 1]])
        else:
            px = np.array([grid.lon[yi, xi], grid.lon[yi, xi + 1], grid.lon[yi + 1, xi + 1], grid.lon[yi + 1, xi]])
            py = np.array([grid.lat[yi, xi], grid.lat[yi, xi + 1], grid.lat[yi + 1, xi + 1], grid.lat[yi + 1, xi]])

        if grid.mesh == "spherical":
            px[0] = px[0] + 360 if px[0] < x - 225 else px[0]
            px[0] = px[0] - 360 if px[0] > x + 225 else px[0]
            px[1:] = np.where(px[1:] - px[0] > 180, px[1:] - 360, px[1:])
            px[1:] = np.where(-px[1:] + px[0] > 180, px[1:] + 360, px[1:])
        xx = (1 - xsi) * (1 - eta) * px[0] + xsi * (1 - eta) * px[1] + xsi * eta * px[2] + (1 - xsi) * eta * px[3]
        assert abs(xx - x) < 1e-4

        px = np.concatenate((px, px))
        py = np.concatenate((py, py))
        if grid.z4d:
            pz = np.array(
                [
                    grid.depth[0, zi, yi, xi],
                    grid.depth[0, zi, yi, xi + 1],
                    grid.depth[0, zi, yi + 1, xi + 1],
                    grid.depth[0, zi, yi + 1, xi],
                    grid.depth[0, zi + 1, yi, xi],
                    grid.depth[0, zi + 1, yi, xi + 1],
                    grid.depth[0, zi + 1, yi + 1, xi + 1],
                    grid.depth[0, zi + 1, yi + 1, xi],
                ]
            )
        else:
            pz = np.array(
                [
                    grid.depth[zi, yi, xi],
                    grid.depth[zi, yi, xi + 1],
                    grid.depth[zi, yi + 1, xi + 1],
                    grid.depth[zi, yi + 1, xi],
                    grid.depth[zi + 1, yi, xi],
                    grid.depth[zi + 1, yi, xi + 1],
                    grid.depth[zi + 1, yi + 1, xi + 1],
                    grid.depth[zi + 1, yi + 1, xi],
                ]
            )

        u0 = self.U.data[ti, zi, yi + 1, xi]
        u1 = self.U.data[ti, zi, yi + 1, xi + 1]
        v0 = self.V.data[ti, zi, yi, xi + 1]
        v1 = self.V.data[ti, zi, yi + 1, xi + 1]
        w0 = self.W.data[ti, zi, yi + 1, xi + 1]
        w1 = self.W.data[ti, zi + 1, yi + 1, xi + 1]

        U0 = u0 * i_u.jacobian3D_lin_face(px, py, pz, 0, eta, zet, "zonal", grid.mesh)
        U1 = u1 * i_u.jacobian3D_lin_face(px, py, pz, 1, eta, zet, "zonal", grid.mesh)
        V0 = v0 * i_u.jacobian3D_lin_face(px, py, pz, xsi, 0, zet, "meridional", grid.mesh)
        V1 = v1 * i_u.jacobian3D_lin_face(px, py, pz, xsi, 1, zet, "meridional", grid.mesh)
        W0 = w0 * i_u.jacobian3D_lin_face(px, py, pz, xsi, eta, 0, "vertical", grid.mesh)
        W1 = w1 * i_u.jacobian3D_lin_face(px, py, pz, xsi, eta, 1, "vertical", grid.mesh)

        # Computing fluxes in half left hexahedron -> flux_u05
        xx = [
            px[0],
            (px[0] + px[1]) / 2,
            (px[2] + px[3]) / 2,
            px[3],
            px[4],
            (px[4] + px[5]) / 2,
            (px[6] + px[7]) / 2,
            px[7],
        ]
        yy = [
            py[0],
            (py[0] + py[1]) / 2,
            (py[2] + py[3]) / 2,
            py[3],
            py[4],
            (py[4] + py[5]) / 2,
            (py[6] + py[7]) / 2,
            py[7],
        ]
        zz = [
            pz[0],
            (pz[0] + pz[1]) / 2,
            (pz[2] + pz[3]) / 2,
            pz[3],
            pz[4],
            (pz[4] + pz[5]) / 2,
            (pz[6] + pz[7]) / 2,
            pz[7],
        ]
        flux_u0 = u0 * i_u.jacobian3D_lin_face(xx, yy, zz, 0, 0.5, 0.5, "zonal", grid.mesh)
        flux_v0_halfx = v0 * i_u.jacobian3D_lin_face(xx, yy, zz, 0.5, 0, 0.5, "meridional", grid.mesh)
        flux_v1_halfx = v1 * i_u.jacobian3D_lin_face(xx, yy, zz, 0.5, 1, 0.5, "meridional", grid.mesh)
        flux_w0_halfx = w0 * i_u.jacobian3D_lin_face(xx, yy, zz, 0.5, 0.5, 0, "vertical", grid.mesh)
        flux_w1_halfx = w1 * i_u.jacobian3D_lin_face(xx, yy, zz, 0.5, 0.5, 1, "vertical", grid.mesh)
        flux_u05 = flux_u0 + flux_v0_halfx - flux_v1_halfx + flux_w0_halfx - flux_w1_halfx

        # Computing fluxes in half front hexahedron -> flux_v05
        xx = [
            px[0],
            px[1],
            (px[1] + px[2]) / 2,
            (px[0] + px[3]) / 2,
            px[4],
            px[5],
            (px[5] + px[6]) / 2,
            (px[4] + px[7]) / 2,
        ]
        yy = [
            py[0],
            py[1],
            (py[1] + py[2]) / 2,
            (py[0] + py[3]) / 2,
            py[4],
            py[5],
            (py[5] + py[6]) / 2,
            (py[4] + py[7]) / 2,
        ]
        zz = [
            pz[0],
            pz[1],
            (pz[1] + pz[2]) / 2,
            (pz[0] + pz[3]) / 2,
            pz[4],
            pz[5],
            (pz[5] + pz[6]) / 2,
            (pz[4] + pz[7]) / 2,
        ]
        flux_u0_halfy = u0 * i_u.jacobian3D_lin_face(xx, yy, zz, 0, 0.5, 0.5, "zonal", grid.mesh)
        flux_u1_halfy = u1 * i_u.jacobian3D_lin_face(xx, yy, zz, 1, 0.5, 0.5, "zonal", grid.mesh)
        flux_v0 = v0 * i_u.jacobian3D_lin_face(xx, yy, zz, 0.5, 0, 0.5, "meridional", grid.mesh)
        flux_w0_halfy = w0 * i_u.jacobian3D_lin_face(xx, yy, zz, 0.5, 0.5, 0, "vertical", grid.mesh)
        flux_w1_halfy = w1 * i_u.jacobian3D_lin_face(xx, yy, zz, 0.5, 0.5, 1, "vertical", grid.mesh)
        flux_v05 = flux_u0_halfy - flux_u1_halfy + flux_v0 + flux_w0_halfy - flux_w1_halfy

        # Computing fluxes in half lower hexahedron -> flux_w05
        xx = [
            px[0],
            px[1],
            px[2],
            px[3],
            (px[0] + px[4]) / 2,
            (px[1] + px[5]) / 2,
            (px[2] + px[6]) / 2,
            (px[3] + px[7]) / 2,
        ]
        yy = [
            py[0],
            py[1],
            py[2],
            py[3],
            (py[0] + py[4]) / 2,
            (py[1] + py[5]) / 2,
            (py[2] + py[6]) / 2,
            (py[3] + py[7]) / 2,
        ]
        zz = [
            pz[0],
            pz[1],
            pz[2],
            pz[3],
            (pz[0] + pz[4]) / 2,
            (pz[1] + pz[5]) / 2,
            (pz[2] + pz[6]) / 2,
            (pz[3] + pz[7]) / 2,
        ]
        flux_u0_halfz = u0 * i_u.jacobian3D_lin_face(xx, yy, zz, 0, 0.5, 0.5, "zonal", grid.mesh)
        flux_u1_halfz = u1 * i_u.jacobian3D_lin_face(xx, yy, zz, 1, 0.5, 0.5, "zonal", grid.mesh)
        flux_v0_halfz = v0 * i_u.jacobian3D_lin_face(xx, yy, zz, 0.5, 0, 0.5, "meridional", grid.mesh)
        flux_v1_halfz = v1 * i_u.jacobian3D_lin_face(xx, yy, zz, 0.5, 1, 0.5, "meridional", grid.mesh)
        flux_w0 = w0 * i_u.jacobian3D_lin_face(xx, yy, zz, 0.5, 0.5, 0, "vertical", grid.mesh)
        flux_w05 = flux_u0_halfz - flux_u1_halfz + flux_v0_halfz - flux_v1_halfz + flux_w0

        surf_u05 = i_u.jacobian3D_lin_face(px, py, pz, 0.5, 0.5, 0.5, "zonal", grid.mesh)
        jac_u05 = i_u.jacobian3D_lin_face(px, py, pz, 0.5, eta, zet, "zonal", grid.mesh)
        U05 = flux_u05 / surf_u05 * jac_u05

        surf_v05 = i_u.jacobian3D_lin_face(px, py, pz, 0.5, 0.5, 0.5, "meridional", grid.mesh)
        jac_v05 = i_u.jacobian3D_lin_face(px, py, pz, xsi, 0.5, zet, "meridional", grid.mesh)
        V05 = flux_v05 / surf_v05 * jac_v05

        surf_w05 = i_u.jacobian3D_lin_face(px, py, pz, 0.5, 0.5, 0.5, "vertical", grid.mesh)
        jac_w05 = i_u.jacobian3D_lin_face(px, py, pz, xsi, eta, 0.5, "vertical", grid.mesh)
        W05 = flux_w05 / surf_w05 * jac_w05

        jac = i_u.jacobian3D_lin(px, py, pz, xsi, eta, zet, grid.mesh)
        dxsidt = i_u.interpolate(i_u.phi1D_quad, [U0, U05, U1], xsi) / jac
        detadt = i_u.interpolate(i_u.phi1D_quad, [V0, V05, V1], eta) / jac
        dzetdt = i_u.interpolate(i_u.phi1D_quad, [W0, W05, W1], zet) / jac

        dphidxsi, dphideta, dphidzet = i_u.dphidxsi3D_lin(xsi, eta, zet)

        u = np.dot(dphidxsi, px) * dxsidt + np.dot(dphideta, px) * detadt + np.dot(dphidzet, px) * dzetdt
        v = np.dot(dphidxsi, py) * dxsidt + np.dot(dphideta, py) * detadt + np.dot(dphidzet, py) * dzetdt
        w = np.dot(dphidxsi, pz) * dxsidt + np.dot(dphideta, pz) * detadt + np.dot(dphidzet, pz) * dzetdt

        if isinstance(u, da.core.Array):
            u = u.compute()
            v = v.compute()
            w = w.compute()
        return (u, v, w)

    def spatial_c_grid_interpolation3D(self, ti, z, y, x, time, particle=None, applyConversion=True):
        """Perform C grid interpolation in 3D. ::

            +---+---+---+
            |   |V1 |   |
            +---+---+---+
            |U0 |   |U1 |
            +---+---+---+
            |   |V0 |   |
            +---+---+---+

        The interpolation is done in the following by
        interpolating linearly U depending on the longitude coordinate and
        interpolating linearly V depending on the latitude coordinate.
        Curvilinear grids are treated properly, since the element is projected to a rectilinear parent element.
        """
        if self.U.grid.gtype in [GridType.RectilinearSGrid, GridType.CurvilinearSGrid]:
            (u, v, w) = self.spatial_c_grid_interpolation3D_full(ti, z, y, x, time, particle=particle)
        else:
            (u, v) = self.spatial_c_grid_interpolation2D(ti, z, y, x, time, particle=particle)
            w = self.W.eval(time, z, y, x, particle=particle, applyConversion=False)
            if applyConversion:
                w = self.W.units.to_target(w, x, y, z)
        return (u, v, w)

    def _is_land2D(self, di, yi, xi):
        if self.U.data.ndim == 3:
            if di < np.shape(self.U.data)[0]:
                return np.isclose(self.U.data[di, yi, xi], 0.0) and np.isclose(self.V.data[di, yi, xi], 0.0)
            else:
                return True
        else:
            if di < self.U.grid.zdim and yi < np.shape(self.U.data)[-2] and xi < np.shape(self.U.data)[-1]:
                return np.isclose(self.U.data[0, di, yi, xi], 0.0) and np.isclose(self.V.data[0, di, yi, xi], 0.0)
            else:
                return True

    def spatial_slip_interpolation(self, ti, z, y, x, time, particle=None, applyConversion=True):
        (xsi, eta, zeta, xi, yi, zi) = self.U.search_indices(x, y, z, ti, time, particle=particle)
        di = ti if self.U.grid.zdim == 1 else zi  # general third dimension

        f_u, f_v, f_w = 1, 1, 1
        if (
            self._is_land2D(di, yi, xi)
            and self._is_land2D(di, yi, xi + 1)
            and self._is_land2D(di + 1, yi, xi)
            and self._is_land2D(di + 1, yi, xi + 1)
            and eta > 0
        ):
            if self.U.interp_method == "partialslip":
                f_u = f_u * (0.5 + 0.5 * eta) / eta
                if self.vector_type == "3D":
                    f_w = f_w * (0.5 + 0.5 * eta) / eta
            elif self.U.interp_method == "freeslip":
                f_u = f_u / eta
                if self.vector_type == "3D":
                    f_w = f_w / eta
        if (
            self._is_land2D(di, yi + 1, xi)
            and self._is_land2D(di, yi + 1, xi + 1)
            and self._is_land2D(di + 1, yi + 1, xi)
            and self._is_land2D(di + 1, yi + 1, xi + 1)
            and eta < 1
        ):
            if self.U.interp_method == "partialslip":
                f_u = f_u * (1 - 0.5 * eta) / (1 - eta)
                if self.vector_type == "3D":
                    f_w = f_w * (1 - 0.5 * eta) / (1 - eta)
            elif self.U.interp_method == "freeslip":
                f_u = f_u / (1 - eta)
                if self.vector_type == "3D":
                    f_w = f_w / (1 - eta)
        if (
            self._is_land2D(di, yi, xi)
            and self._is_land2D(di, yi + 1, xi)
            and self._is_land2D(di + 1, yi, xi)
            and self._is_land2D(di + 1, yi + 1, xi)
            and xsi > 0
        ):
            if self.U.interp_method == "partialslip":
                f_v = f_v * (0.5 + 0.5 * xsi) / xsi
                if self.vector_type == "3D":
                    f_w = f_w * (0.5 + 0.5 * xsi) / xsi
            elif self.U.interp_method == "freeslip":
                f_v = f_v / xsi
                if self.vector_type == "3D":
                    f_w = f_w / xsi
        if (
            self._is_land2D(di, yi, xi + 1)
            and self._is_land2D(di, yi + 1, xi + 1)
            and self._is_land2D(di + 1, yi, xi + 1)
            and self._is_land2D(di + 1, yi + 1, xi + 1)
            and xsi < 1
        ):
            if self.U.interp_method == "partialslip":
                f_v = f_v * (1 - 0.5 * xsi) / (1 - xsi)
                if self.vector_type == "3D":
                    f_w = f_w * (1 - 0.5 * xsi) / (1 - xsi)
            elif self.U.interp_method == "freeslip":
                f_v = f_v / (1 - xsi)
                if self.vector_type == "3D":
                    f_w = f_w / (1 - xsi)
        if self.U.grid.zdim > 1:
            if (
                self._is_land2D(di, yi, xi)
                and self._is_land2D(di, yi, xi + 1)
                and self._is_land2D(di, yi + 1, xi)
                and self._is_land2D(di, yi + 1, xi + 1)
                and zeta > 0
            ):
                if self.U.interp_method == "partialslip":
                    f_u = f_u * (0.5 + 0.5 * zeta) / zeta
                    f_v = f_v * (0.5 + 0.5 * zeta) / zeta
                elif self.U.interp_method == "freeslip":
                    f_u = f_u / zeta
                    f_v = f_v / zeta
            if (
                self._is_land2D(di + 1, yi, xi)
                and self._is_land2D(di + 1, yi, xi + 1)
                and self._is_land2D(di + 1, yi + 1, xi)
                and self._is_land2D(di + 1, yi + 1, xi + 1)
                and zeta < 1
            ):
                if self.U.interp_method == "partialslip":
                    f_u = f_u * (1 - 0.5 * zeta) / (1 - zeta)
                    f_v = f_v * (1 - 0.5 * zeta) / (1 - zeta)
                elif self.U.interp_method == "freeslip":
                    f_u = f_u / (1 - zeta)
                    f_v = f_v / (1 - zeta)

        u = f_u * self.U.eval(time, z, y, x, particle, applyConversion=applyConversion)
        v = f_v * self.V.eval(time, z, y, x, particle, applyConversion=applyConversion)
        if self.vector_type == "3D":
            w = f_w * self.W.eval(time, z, y, x, particle, applyConversion=applyConversion)
            return u, v, w
        else:
            return u, v

    def eval(self, time, z, y, x, particle=None, applyConversion=True):
        if self.U.interp_method not in ["cgrid_velocity", "partialslip", "freeslip"]:
            u = self.U.eval(time, z, y, x, particle=particle, applyConversion=False)
            v = self.V.eval(time, z, y, x, particle=particle, applyConversion=False)
            if applyConversion:
                u = self.U.units.to_target(u, x, y, z)
                v = self.V.units.to_target(v, x, y, z)
            if self.vector_type == "3D":
                w = self.W.eval(time, z, y, x, particle=particle, applyConversion=False)
                if applyConversion:
                    w = self.W.units.to_target(w, x, y, z)
                return (u, v, w)
            else:
                return (u, v)
        else:
            interp = {
                "cgrid_velocity": {
                    "2D": self.spatial_c_grid_interpolation2D,
                    "3D": self.spatial_c_grid_interpolation3D,
                },
                "partialslip": {"2D": self.spatial_slip_interpolation, "3D": self.spatial_slip_interpolation},
                "freeslip": {"2D": self.spatial_slip_interpolation, "3D": self.spatial_slip_interpolation},
            }
            grid = self.U.grid
            (ti, periods) = self.U.time_index(time)
            time -= periods * (grid.time_full[-1] - grid.time_full[0])
            if ti < grid.tdim - 1 and time > grid.time[ti]:
                t0 = grid.time[ti]
                t1 = grid.time[ti + 1]
                if self.vector_type == "3D":
                    (u0, v0, w0) = interp[self.U.interp_method]["3D"](
                        ti, z, y, x, time, particle=particle, applyConversion=applyConversion
                    )
                    (u1, v1, w1) = interp[self.U.interp_method]["3D"](
                        ti + 1, z, y, x, time, particle=particle, applyConversion=applyConversion
                    )
                    w = w0 + (w1 - w0) * ((time - t0) / (t1 - t0))
                else:
                    (u0, v0) = interp[self.U.interp_method]["2D"](
                        ti, z, y, x, time, particle=particle, applyConversion=applyConversion
                    )
                    (u1, v1) = interp[self.U.interp_method]["2D"](
                        ti + 1, z, y, x, time, particle=particle, applyConversion=applyConversion
                    )
                u = u0 + (u1 - u0) * ((time - t0) / (t1 - t0))
                v = v0 + (v1 - v0) * ((time - t0) / (t1 - t0))
                if self.vector_type == "3D":
                    return (u, v, w)
                else:
                    return (u, v)
            else:
                # Skip temporal interpolation if time is outside
                # of the defined time range or if we have hit an
                # exact value in the time array.
                if self.vector_type == "3D":
                    return interp[self.U.interp_method]["3D"](
                        ti, z, y, x, grid.time[ti], particle=particle, applyConversion=applyConversion
                    )
                else:
                    return interp[self.U.interp_method]["2D"](
                        ti, z, y, x, grid.time[ti], particle=particle, applyConversion=applyConversion
                    )

    def __getitem__(self, key):
        try:
            if _isParticle(key):
                return self.eval(key.time, key.depth, key.lat, key.lon, key)
            else:
                return self.eval(*key)
        except tuple(AllParcelsErrorCodes.keys()) as error:
            return _deal_with_errors(error, key, vector_type=self.vector_type)

    def ccode_eval(self, varU, varV, varW, U, V, W, t, z, y, x):
        ccode_str = ""
        if self.vector_type == "3D":
            ccode_str = (
                f"temporal_interpolationUVW({x}, {y}, {z}, {t}, {U.ccode_name}, {V.ccode_name}, {W.ccode_name}, "
                + "&particles->xi[pnum*ngrid], &particles->yi[pnum*ngrid], &particles->zi[pnum*ngrid], &particles->ti[pnum*ngrid],"
                + f"&{varU}, &{varV}, &{varW})"
            )
        else:
            ccode_str = (
                f"temporal_interpolationUV({x}, {y}, {z}, {t}, {U.ccode_name}, {V.ccode_name}, "
                + "&particles->xi[pnum*ngrid], &particles->yi[pnum*ngrid], &particles->zi[pnum*ngrid], &particles->ti[pnum*ngrid],"
                + f" &{varU}, &{varV})"
            )
        return ccode_str


class DeferredArray:
    """Class used for throwing error when Field.data is not read in deferred loading mode."""

    data_shape = ()

    def __init__(self):
        self.data_shape = (1,)

    def compute_shape(self, xdim, ydim, zdim, tdim, tslices):
        if zdim == 1 and tdim == 1:
            self.data_shape = (tslices, 1, ydim, xdim)
        elif zdim > 1 or tdim > 1:
            if zdim > 1:
                self.data_shape = (1, zdim, ydim, xdim)
            else:
                self.data_shape = (max(tdim, tslices), 1, ydim, xdim)
        else:
            self.data_shape = (tdim, zdim, ydim, xdim)
        return self.data_shape

    def __getitem__(self, key):
        raise RuntimeError(
            "Field is in deferred_load mode, so can't be accessed. Use .computeTimeChunk() method to force loading of data"
        )


class NestedField(list):
    """NestedField is a class that allows for interpolation of fields on different grids of potentially varying resolution.

    The NestedField class is a list of Fields where the first Field that contains the particle within the domain is then used for interpolation.
    This induces that the order of the fields in the list matters.
    Each one it its turn, a field is interpolated: if the interpolation succeeds or if an error other
    than `ErrorOutOfBounds` is thrown, the function is stopped. Otherwise, next field is interpolated.
    NestedField returns an `ErrorOutOfBounds` only if last field is as well out of boundaries.
    NestedField is composed of either Fields or VectorFields.

    Parameters
    ----------
    name : str
        Name of the NestedField
    F : list of Field
        List of fields (order matters). F can be a scalar Field, a VectorField, or the zonal component (U) of the VectorField
    V : list of Field
        List of fields defining the meridional component of a VectorField, if F is the zonal component. (default: None)
    W : list of Field
        List of fields defining the vertical component of a VectorField, if F and V are the zonal and meridional components (default: None)


    Examples
    --------
    See `here <../examples/tutorial_NestedFields.ipynb>`__
    for a detailed tutorial

    """

    def __init__(self, name: str, F, V=None, W=None):
        if V is None:
            if isinstance(F[0], VectorField):
                vector_type = F[0].vector_type
            for Fi in F:
                assert isinstance(Fi, Field) or (
                    isinstance(Fi, VectorField) and Fi.vector_type == vector_type
                ), "Components of a NestedField must be Field or VectorField"
                self.append(Fi)
        elif W is None:
            for i, Fi, Vi in zip(range(len(F)), F, V, strict=True):
                assert isinstance(Fi, Field) and isinstance(
                    Vi, Field
                ), "F, and V components of a NestedField must be Field"
                self.append(VectorField(name + "_%d" % i, Fi, Vi))
        else:
            for i, Fi, Vi, Wi in zip(range(len(F)), F, V, W, strict=True):
                assert (
                    isinstance(Fi, Field) and isinstance(Vi, Field) and isinstance(Wi, Field)
                ), "F, V and W components of a NestedField must be Field"
                self.append(VectorField(name + "_%d" % i, Fi, Vi, Wi))
        self.name = name

    def __getitem__(self, key):
        if isinstance(key, int):
            return list.__getitem__(self, key)
        else:
            for iField in range(len(self)):
                try:
                    if _isParticle(key):
                        val = list.__getitem__(self, iField).eval(key.time, key.depth, key.lat, key.lon, particle=None)
                    else:
                        val = list.__getitem__(self, iField).eval(*key)
                    break
                except tuple(AllParcelsErrorCodes.keys()) as error:
                    if iField == len(self) - 1:
                        vector_type = self[iField].vector_type if isinstance(self[iField], VectorField) else None
                        return _deal_with_errors(error, key, vector_type=vector_type)
                    else:
                        pass
            return val
