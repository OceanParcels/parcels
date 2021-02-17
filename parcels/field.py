import collections
import datetime
import math
from ctypes import c_float
from ctypes import c_int
from ctypes import POINTER
from ctypes import pointer
from ctypes import Structure

import psutil
import dask.array as da
from dask import config as da_conf
from dask import utils as da_utils
import numpy as np
import xarray as xr
from pathlib import Path

import parcels.tools.interpolation_utils as i_u
from .grid import CGrid
from .grid import Grid
from .grid import GridCode
from parcels.tools.converters import Geographic
from parcels.tools.converters import GeographicPolar
from parcels.tools.converters import TimeConverter
from parcels.tools.converters import UnitConverter
from parcels.tools.converters import unitconverters_map
from parcels.tools.converters import convert_xarray_time_units
from parcels.tools.error import FieldOutOfBoundError
from parcels.tools.error import FieldOutOfBoundSurfaceError
from parcels.tools.error import FieldSamplingError
from parcels.tools.error import TimeExtrapolationError
from parcels.tools.loggers import logger
from netCDF4 import Dataset as ncDataset


__all__ = ['Field', 'VectorField', 'SummedField', 'NestedField']


class Field(object):
    """Class that encapsulates access to field data.

    :param name: Name of the field
    :param data: 2D, 3D or 4D numpy array of field data.

           1. If data shape is [xdim, ydim], [xdim, ydim, zdim], [xdim, ydim, tdim] or [xdim, ydim, zdim, tdim],
              whichever is relevant for the dataset, use the flag transpose=True
           2. If data shape is [ydim, xdim], [zdim, ydim, xdim], [tdim, ydim, xdim] or [tdim, zdim, ydim, xdim],
              use the flag transpose=False
           3. If data has any other shape, you first need to reorder it
    :param lon: Longitude coordinates (numpy vector or array) of the field (only if grid is None)
    :param lat: Latitude coordinates (numpy vector or array) of the field (only if grid is None)
    :param depth: Depth coordinates (numpy vector or array) of the field (only if grid is None)
    :param time: Time coordinates (numpy vector) of the field (only if grid is None)
    :param mesh: String indicating the type of mesh coordinates and
           units used during velocity interpolation: (only if grid is None)

           1. spherical: Lat and lon in degree, with a
              correction for zonal velocity U near the poles.
           2. flat (default): No conversion, lat/lon are assumed to be in m.
    :param timestamps: A numpy array containing the timestamps for each of the files in filenames, for loading
           from netCDF files only. Default is None if the netCDF dimensions dictionary includes time.
    :param grid: :class:`parcels.grid.Grid` object containing all the lon, lat depth, time
           mesh and time_origin information. Can be constructed from any of the Grid objects
    :param fieldtype: Type of Field to be used for UnitConverter when using SummedFields
           (either 'U', 'V', 'Kh_zonal', 'Kh_meridional' or None)
    :param transpose: Transpose data to required (lon, lat) layout
    :param vmin: Minimum allowed value on the field. Data below this value are set to zero
    :param vmax: Maximum allowed value on the field. Data above this value are set to zero
    :param time_origin: Time origin (TimeConverter object) of the time axis (only if grid is None)
    :param interp_method: Method for interpolation. Either 'linear' or 'nearest'
    :param allow_time_extrapolation: boolean whether to allow for extrapolation in time
           (i.e. beyond the last available time snapshot)
    :param time_periodic: To loop periodically over the time component of the Field. It is set to either False or the length of the period (either float in seconds or datetime.timedelta object).
           The last value of the time series can be provided (which is the same as the initial one) or not (Default: False)
           This flag overrides the allow_time_interpolation and sets it to False
    """

    def __init__(self, name, data, lon=None, lat=None, depth=None, time=None, grid=None, mesh='flat', timestamps=None,
                 fieldtype=None, transpose=False, vmin=None, vmax=None, time_origin=None,
                 interp_method='linear', allow_time_extrapolation=None, time_periodic=False, **kwargs):
        if not isinstance(name, tuple):
            self.name = name
            self.filebuffername = name
        else:
            self.name, self.filebuffername = name
        self.data = data
        time_origin = TimeConverter(0) if time_origin is None else time_origin
        if grid:
            # if grid.defer_load and isinstance(data, np.ndarray):
            #     raise ValueError('Cannot combine Grid from defer_loaded Field with np.ndarray data. please specify lon, lat, depth and time dimensions separately')
            self.grid = grid
        else:
            self.grid = Grid.create_grid(lon, lat, depth, time, time_origin=time_origin, mesh=mesh)
        self.igrid = -1
        # self.lon, self.lat, self.depth and self.time are not used anymore in parcels.
        # self.grid should be used instead.
        # Those variables are still defined for backwards compatibility with users codes.
        self.lon = self.grid.lon
        self.lat = self.grid.lat
        self.depth = self.grid.depth
        self.fieldtype = self.name if fieldtype is None else fieldtype
        if self.grid.mesh == 'flat' or (self.fieldtype not in unitconverters_map.keys()):
            self.units = UnitConverter()
        elif self.grid.mesh == 'spherical':
            self.units = unitconverters_map[self.fieldtype]
        else:
            raise ValueError("Unsupported mesh type. Choose either: 'spherical' or 'flat'")
        self.timestamps = timestamps
        if type(interp_method) is dict:
            if self.name in interp_method:
                self.interp_method = interp_method[self.name]
            else:
                raise RuntimeError('interp_method is a dictionary but %s is not in it' % name)
        else:
            self.interp_method = interp_method
        if self.interp_method in ['bgrid_velocity', 'bgrid_w_velocity', 'bgrid_tracer'] and \
           self.grid.gtype in [GridCode.RectilinearSGrid, GridCode.CurvilinearSGrid]:
            logger.warning_once('General s-levels are not supported in B-grid. RectilinearSGrid and CurvilinearSGrid can still be used to deal with shaved cells, but the levels must be horizontal.')

        self.fieldset = None
        if allow_time_extrapolation is None:
            self.allow_time_extrapolation = True if len(self.grid.time) == 1 else False
        else:
            self.allow_time_extrapolation = allow_time_extrapolation

        self.time_periodic = time_periodic
        if self.time_periodic is not False and self.allow_time_extrapolation:
            logger.warning_once("allow_time_extrapolation and time_periodic cannot be used together.\n \
                                 allow_time_extrapolation is set to False")
            self.allow_time_extrapolation = False
        if self.time_periodic is True:
            raise ValueError("Unsupported time_periodic=True. time_periodic must now be either False or the length of the period (either float in seconds or datetime.timedelta object.")
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

        if not self.grid.defer_load:
            self.data = self.reshape(self.data, transpose)

            # Hack around the fact that NaN and ridiculously large values
            # propagate in SciPy's interpolators
            self.data[np.isnan(self.data)] = 0.
            if self.vmin is not None:
                self.data[self.data < self.vmin] = 0.
            if self.vmax is not None:
                self.data[self.data > self.vmax] = 0.

            if self.grid._add_last_periodic_data_timestep:
                lib = np if isinstance(self.data, np.ndarray) else da
                self.data = lib.concatenate((self.data, self.data[:1, :]), axis=0)

        self._scaling_factor = None

        # Variable names in JIT code
        self.dimensions = kwargs.pop('dimensions', None)
        self.indices = kwargs.pop('indices', None)
        self.dataFiles = kwargs.pop('dataFiles', None)
        if self.grid._add_last_periodic_data_timestep and self.dataFiles is not None:
            self.dataFiles = np.append(self.dataFiles, self.dataFiles[0])
        self.netcdf_engine = kwargs.pop('netcdf_engine', 'netcdf4')
        self.loaded_time_indices = []
        self.creation_log = kwargs.pop('creation_log', '')
        self.field_chunksize = kwargs.pop('field_chunksize', None)
        self.grid.depth_field = kwargs.pop('depth_field', None)

        if self.grid.depth_field == 'not_yet_set':
            assert self.grid.z4d, 'Providing the depth dimensions from another field data is only available for 4d S grids'

        # data_full_zdim is the vertical dimension of the complete field data, ignoring the indices.
        # (data_full_zdim = grid.zdim if no indices are used, for A- and C-grids and for some B-grids). It is used for the B-grid,
        # since some datasets do not provide the deeper level of data (which is ignored by the interpolation).
        self.data_full_zdim = kwargs.pop('data_full_zdim', None)
        self.data_chunks = []   # the data buffer of the FileBuffer raw loaded data - shall be a list of C-contiguous arrays
        self.c_data_chunks = []  # C-pointers to the data_chunks array
        self.nchunks = []
        self.chunk_set = False
        self.filebuffers = [None] * 3
        if len(kwargs) > 0:
            raise SyntaxError('Field received an unexpected keyword argument "%s"' % list(kwargs.keys())[0])

    @classmethod
    def get_dim_filenames(cls, filenames, dim):
        if isinstance(filenames, str) or not isinstance(filenames, collections.abc.Iterable):
            return [filenames]
        elif isinstance(filenames, dict):
            assert dim in filenames.keys(), \
                'filename dimension keys must be lon, lat, depth or data'
            filename = filenames[dim]
            if isinstance(filename, str):
                return [filename]
            else:
                return filename
        else:
            return filenames

    @classmethod
    def from_netcdf(cls, filenames, variable, dimensions, indices=None, grid=None,
                    mesh='spherical', timestamps=None, allow_time_extrapolation=None, time_periodic=False,
                    deferred_load=True, **kwargs):
        """Create field from netCDF file

        :param filenames: list of filenames to read for the field. filenames can be a list [files] or
               a dictionary {dim:[files]} (if lon, lat, depth and/or data not stored in same files as data)
               In the latetr case, time values are in filenames[data]
        :param variable: Tuple mapping field name to variable name in the NetCDF file.
        :param dimensions: Dictionary mapping variable names for the relevant dimensions in the NetCDF file
        :param indices: dictionary mapping indices for each dimension to read from file.
               This can be used for reading in only a subregion of the NetCDF file.
               Note that negative indices are not allowed.
        :param mesh: String indicating the type of mesh coordinates and
               units used during velocity interpolation:

               1. spherical (default): Lat and lon in degree, with a
                  correction for zonal velocity U near the poles.
               2. flat: No conversion, lat/lon are assumed to be in m.
        :param timestamps: A numpy array of datetime64 objects containing the timestamps for each of the files in filenames.
               Default is None if dimensions includes time.
        :param allow_time_extrapolation: boolean whether to allow for extrapolation in time
               (i.e. beyond the last available time snapshot)
               Default is False if dimensions includes time, else True
        :param time_periodic: boolean whether to loop periodically over the time component of the FieldSet
               This flag overrides the allow_time_interpolation and sets it to False
        :param deferred_load: boolean whether to only pre-load data (in deferred mode) or
               fully load them (default: True). It is advised to deferred load the data, since in
               that case Parcels deals with a better memory management during particle set execution.
               deferred_load=False is however sometimes necessary for plotting the fields.
        :param field_chunksize: size of the chunks in dask loading
        """
        # Ensure the timestamps array is compatible with the user-provided datafiles.
        if timestamps is not None:
            if isinstance(filenames, list):
                assert len(filenames) == len(timestamps), 'Outer dimension of timestamps should correspond to number of files.'
            elif isinstance(filenames, dict):
                for k in filenames.keys():
                    if k not in ['lat', 'lon', 'depth', 'time']:
                        assert(len(filenames[k]) == len(timestamps)), 'Outer dimension of timestamps should correspond to number of files.'
            else:
                raise TypeError("Filenames type is inconsistent with manual timestamp provision."
                                + "Should be dict or list")

        if isinstance(variable, str):  # for backward compatibility with Parcels < 2.0.0
            variable = (variable, variable)
        assert len(variable) == 2, 'The variable tuple must have length 2. Use FieldSet.from_netcdf() for multiple variables'

        data_filenames = cls.get_dim_filenames(filenames, 'data')
        lonlat_filename = cls.get_dim_filenames(filenames, 'lon')
        if isinstance(filenames, dict):
            assert len(lonlat_filename) == 1
        if lonlat_filename != cls.get_dim_filenames(filenames, 'lat'):
            raise NotImplementedError('longitude and latitude dimensions are currently processed together from one single file')
        lonlat_filename = lonlat_filename[0]
        if 'depth' in dimensions:
            depth_filename = cls.get_dim_filenames(filenames, 'depth')
            if isinstance(filenames, dict) and len(depth_filename) != 1:
                raise NotImplementedError('Vertically adaptive meshes not implemented for from_netcdf()')
            depth_filename = depth_filename[0]

        netcdf_engine = kwargs.pop('netcdf_engine', 'netcdf4')

        indices = {} if indices is None else indices.copy()
        for ind in indices.values():
            assert np.min(ind) >= 0, \
                ('Negative indices are currently not allowed in Parcels. '
                 + 'This is related to the non-increasing dimension it could generate '
                 + 'if the domain goes from lon[-4] to lon[6] for example. '
                 + 'Please raise an issue on https://github.com/OceanParcels/parcels/issues '
                 + 'if you would need such feature implemented.')

        interp_method = kwargs.pop('interp_method', 'linear')
        if type(interp_method) is dict:
            if variable[0] in interp_method:
                interp_method = interp_method[variable[0]]
            else:
                raise RuntimeError('interp_method is a dictionary but %s is not in it' % variable[0])

        with NetcdfFileBuffer(lonlat_filename, dimensions, indices, netcdf_engine, field_chunksize=False, lock_file=False) as filebuffer:
            lon, lat = filebuffer.read_lonlat
            indices = filebuffer.indices
            # Check if parcels_mesh has been explicitly set in file
            if 'parcels_mesh' in filebuffer.dataset.attrs:
                mesh = filebuffer.dataset.attrs['parcels_mesh']

        if 'depth' in dimensions:
            with NetcdfFileBuffer(depth_filename, dimensions, indices, netcdf_engine, interp_method=interp_method, field_chunksize=False, lock_file=False) as filebuffer:
                filebuffer.name = filebuffer.parse_name(variable[1])
                if dimensions['depth'] == 'not_yet_set':
                    depth = filebuffer.read_depth_dimensions
                    kwargs['depth_field'] = 'not_yet_set'
                else:
                    depth = filebuffer.read_depth
                data_full_zdim = filebuffer.data_full_zdim
        else:
            indices['depth'] = [0]
            depth = np.zeros(1)
            data_full_zdim = 1

        kwargs['data_full_zdim'] = data_full_zdim

        if len(data_filenames) > 1 and 'time' not in dimensions and timestamps is None:
            raise RuntimeError('Multiple files given but no time dimension specified')

        if grid is None:
            # Concatenate time variable to determine overall dimension
            # across multiple files
            if timestamps is not None:
                dataFiles = []
                for findex in range(len(data_filenames)):
                    for f in [data_filenames[findex], ] * len(timestamps[findex]):
                        dataFiles.append(f)
                timeslices = np.array([stamp for file in timestamps for stamp in file])
                time = timeslices
            else:
                timeslices = []
                dataFiles = []
                for fname in data_filenames:
                    with NetcdfFileBuffer(fname, dimensions, indices, netcdf_engine, field_chunksize=False, lock_file=True) as filebuffer:
                        ftime = filebuffer.time
                        timeslices.append(ftime)
                        dataFiles.append([fname] * len(ftime))
                timeslices = np.array(timeslices)
                time = np.concatenate(timeslices)
                dataFiles = np.concatenate(np.array(dataFiles))
            if time.size == 1 and time[0] is None:
                time[0] = 0
            time_origin = TimeConverter(time[0])
            time = time_origin.reltime(time)

            if not np.all((time[1:]-time[:-1]) > 0):
                id_not_ordered = np.where(time[1:] < time[:-1])[0][0]
                raise AssertionError('Please make sure your netCDF files are ordered in time. First pair of non-ordered files: %s, %s'
                                     % (dataFiles[id_not_ordered], dataFiles[id_not_ordered+1]))

            grid = Grid.create_grid(lon, lat, depth, time, time_origin=time_origin, mesh=mesh)
            grid.timeslices = timeslices
            if 'field_chunksize' in kwargs.keys() and grid.master_chunksize is None:
                grid.master_chunksize = kwargs['field_chunksize']
            kwargs['dataFiles'] = dataFiles
        elif grid is not None and ('dataFiles' not in kwargs or kwargs['dataFiles'] is None):
            # ==== means: the field has a shared grid, but may have different data files, so we need to collect the
            # ==== correct file time series again.
            if timestamps is not None:
                dataFiles = []
                for findex in range(len(data_filenames)):
                    for f in [data_filenames[findex], ] * len(timestamps[findex]):
                        dataFiles.append(f)
                field_timeslices = np.array([stamp for file in timestamps for stamp in file])
                field_time = field_timeslices
            else:
                field_timeslices = []
                dataFiles = []
                for fname in data_filenames:
                    with NetcdfFileBuffer(fname, dimensions, indices, netcdf_engine, field_chunksize=False, lock_file=True) as filebuffer:
                        ftime = filebuffer.time
                        field_timeslices.append(ftime)
                        dataFiles.append([fname] * len(ftime))
                field_timeslices = np.array(field_timeslices)
                field_time = np.concatenate(field_timeslices)
                dataFiles = np.concatenate(np.array(dataFiles))
            if field_time.size == 1 and field_time[0] is None:
                field_time[0] = 0
            time_origin = TimeConverter(field_time[0])
            field_time = time_origin.reltime(field_time)
            if not np.all((field_time[1:]-field_time[:-1]) > 0):
                id_not_ordered = np.where(field_time[1:] < field_time[:-1])[0][0]
                raise AssertionError('Please make sure your netCDF files are ordered in time. First pair of non-ordered files: %s, %s'
                                     % (dataFiles[id_not_ordered], dataFiles[id_not_ordered+1]))
            if 'field_chunksize' in kwargs.keys() and grid.master_chunksize is None:
                grid.master_chunksize = kwargs['field_chunksize']
            kwargs['dataFiles'] = dataFiles

        if 'time' in indices:
            logger.warning_once('time dimension in indices is not necessary anymore. It is then ignored.')

        if 'full_load' in kwargs:  # for backward compatibility with Parcels < v2.0.0
            deferred_load = not kwargs['full_load']

        if grid.time.size <= 3 or deferred_load is False:
            # Pre-allocate data before reading files into buffer
            data_list = []
            ti = 0
            for tslice, fname in zip(grid.timeslices, data_filenames):
                with NetcdfFileBuffer(fname, dimensions, indices, netcdf_engine,
                                      interp_method=interp_method, data_full_zdim=data_full_zdim,
                                      field_chunksize=False) as filebuffer:
                    # If Field.from_netcdf is called directly, it may not have a 'data' dimension
                    # In that case, assume that 'name' is the data dimension
                    filebuffer.name = filebuffer.parse_name(variable[1])
                    buffer_data = filebuffer.data
                    if len(buffer_data.shape) == 2:
                        data_list.append(buffer_data.reshape(sum(((len(tslice), 1), buffer_data.shape), ())))
                    elif len(buffer_data.shape) == 3:
                        if len(filebuffer.indices['depth']) > 1:
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
            allow_time_extrapolation = False if 'time' in dimensions else True

        kwargs['dimensions'] = dimensions.copy()
        kwargs['indices'] = indices
        kwargs['time_periodic'] = time_periodic
        kwargs['netcdf_engine'] = netcdf_engine

        return cls(variable, data, grid=grid, timestamps=timestamps,
                   allow_time_extrapolation=allow_time_extrapolation, interp_method=interp_method, **kwargs)

    @classmethod
    def from_xarray(cls, da, name, dimensions, mesh='spherical', allow_time_extrapolation=None,
                    time_periodic=False, **kwargs):
        """Create field from xarray Variable

        :param da: Xarray DataArray
        :param name: Name of the Field
        :param dimensions: Dictionary mapping variable names for the relevant dimensions in the DataArray
        :param mesh: String indicating the type of mesh coordinates and
               units used during velocity interpolation:

               1. spherical (default): Lat and lon in degree, with a
                  correction for zonal velocity U near the poles.
               2. flat: No conversion, lat/lon are assumed to be in m.
        :param allow_time_extrapolation: boolean whether to allow for extrapolation in time
               (i.e. beyond the last available time snapshot)
               Default is False if dimensions includes time, else True
        :param time_periodic: boolean whether to loop periodically over the time component of the FieldSet
               This flag overrides the allow_time_interpolation and sets it to False
        """

        data = da.values
        interp_method = kwargs.pop('interp_method', 'linear')

        time = da[dimensions['time']].values if 'time' in dimensions else np.array([0])
        depth = da[dimensions['depth']].values if 'depth' in dimensions else np.array([0])
        lon = da[dimensions['lon']].values
        lat = da[dimensions['lat']].values

        time_origin = TimeConverter(time[0])
        time = time_origin.reltime(time)

        grid = Grid.create_grid(lon, lat, depth, time, time_origin=time_origin, mesh=mesh)
        return cls(name, data, grid=grid, allow_time_extrapolation=allow_time_extrapolation,
                   interp_method=interp_method, **kwargs)

    def reshape(self, data, transpose=False):

        # Ensure that field data is the right data type
        if not data.dtype == np.float32:
            logger.warning_once("Casting field data to np.float32")
            data = data.astype(np.float32)
        lib = np if isinstance(data, np.ndarray) else da
        if transpose:
            data = lib.transpose(data)
        if self.grid.lat_flipped:
            data = lib.flip(data, axis=-2)

        if self.grid.tdim == 1:
            if len(data.shape) < 4:
                data = data.reshape(sum(((1,), data.shape), ()))
        if self.grid.zdim == 1:
            if len(data.shape) == 4:
                data = data.reshape(sum(((data.shape[0],), data.shape[2:]), ()))
        if len(data.shape) == 4:
            assert data.shape == (self.grid.tdim, self.grid.zdim, self.grid.ydim-2*self.grid.meridional_halo, self.grid.xdim-2*self.grid.zonal_halo), \
                                 ('Field %s expecting a data shape of a [tdim, zdim, ydim, xdim]. Flag transpose=True could help to reorder the data.' % (self.name))
        else:
            assert data.shape == (self.grid.tdim, self.grid.ydim-2*self.grid.meridional_halo, self.grid.xdim-2*self.grid.zonal_halo), \
                                 ('Field %s expecting a data shape of a [ydim, xdim], [zdim, ydim, xdim], [tdim, ydim, xdim]. Flag transpose=True could help to reorder the data.' % (self.name))
        if self.grid.meridional_halo > 0 or self.grid.zonal_halo > 0:
            data = self.add_periodic_halo(zonal=self.grid.zonal_halo > 0, meridional=self.grid.meridional_halo > 0, halosize=max(self.grid.meridional_halo, self.grid.zonal_halo), data=data)
        return data

    def set_scaling_factor(self, factor):
        """Scales the field data by some constant factor.

        :param factor: scaling factor
        """

        if self._scaling_factor:
            raise NotImplementedError(('Scaling factor for field %s already defined.' % self.name))
        self._scaling_factor = factor
        if not self.grid.defer_load:
            self.data *= factor

    def set_depth_from_field(self, field):
        self.grid.depth_field = field

    def __getitem__(self, key):
        return self.eval(*key)

    def calc_cell_edge_sizes(self):
        """Method to calculate cell sizes based on numpy.gradient method
                Currently only works for Rectilinear Grids"""
        if not self.grid.cell_edge_sizes:
            if self.grid.gtype in (GridCode.RectilinearZGrid, GridCode.RectilinearSGrid):
                self.grid.cell_edge_sizes['x'] = np.zeros((self.grid.ydim, self.grid.xdim), dtype=np.float32)
                self.grid.cell_edge_sizes['y'] = np.zeros((self.grid.ydim, self.grid.xdim), dtype=np.float32)

                x_conv = GeographicPolar() if self.grid.mesh == 'spherical' else UnitConverter()
                y_conv = Geographic() if self.grid.mesh == 'spherical' else UnitConverter()
                for y, (lat, dy) in enumerate(zip(self.grid.lat, np.gradient(self.grid.lat))):
                    for x, (lon, dx) in enumerate(zip(self.grid.lon, np.gradient(self.grid.lon))):
                        self.grid.cell_edge_sizes['x'][y, x] = x_conv.to_source(dx, lon, lat, self.grid.depth[0])
                        self.grid.cell_edge_sizes['y'][y, x] = y_conv.to_source(dy, lon, lat, self.grid.depth[0])
                self.cell_edge_sizes = self.grid.cell_edge_sizes
            else:
                logger.error(('Field.cell_edge_sizes() not implemented for ', self.grid.gtype, 'grids.',
                              'You can provide Field.grid.cell_edge_sizes yourself',
                              'by in e.g. NEMO using the e1u fields etc from the mesh_mask.nc file'))
                exit(-1)

    def cell_areas(self):
        """Method to calculate cell sizes based on cell_edge_sizes
                Currently only works for Rectilinear Grids"""
        if not self.grid.cell_edge_sizes:
            self.calc_cell_edge_sizes()
        return self.grid.cell_edge_sizes['x'] * self.grid.cell_edge_sizes['y']

    def search_indices_vertical_z(self, z):
        grid = self.grid
        z = np.float32(z)
        if grid.depth[-1] > grid.depth[0]:
            if z < grid.depth[0]:
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
        zeta = (z-grid.depth[zi]) / (grid.depth[zi+1]-grid.depth[zi])
        return (zi, zeta)

    def search_indices_vertical_s(self, x, y, z, xi, yi, xsi, eta, ti, time):
        grid = self.grid
        if self.interp_method in ['bgrid_velocity', 'bgrid_w_velocity', 'bgrid_tracer']:
            xsi = 1
            eta = 1
        if time < grid.time[ti]:
            ti -= 1
        if grid.z4d:
            if ti == len(grid.time)-1:
                depth_vector = (1-xsi)*(1-eta) * grid.depth[-1, :, yi, xi] + \
                    xsi*(1-eta) * grid.depth[-1, :, yi, xi+1] + \
                    xsi*eta * grid.depth[-1, :, yi+1, xi+1] + \
                    (1-xsi)*eta * grid.depth[-1, :, yi+1, xi]
            else:
                dv2 = (1-xsi)*(1-eta) * grid.depth[ti:ti+2, :, yi, xi] + \
                    xsi*(1-eta) * grid.depth[ti:ti+2, :, yi, xi+1] + \
                    xsi*eta * grid.depth[ti:ti+2, :, yi+1, xi+1] + \
                    (1-xsi)*eta * grid.depth[ti:ti+2, :, yi+1, xi]
                tt = (time-grid.time[ti]) / (grid.time[ti+1]-grid.time[ti])
                assert tt >= 0 and tt <= 1, 'Vertical s grid is being wrongly interpolated in time'
                depth_vector = dv2[0, :] * (1-tt) + dv2[1, :] * tt
        else:
            depth_vector = (1-xsi)*(1-eta) * grid.depth[:, yi, xi] + \
                xsi*(1-eta) * grid.depth[:, yi, xi+1] + \
                xsi*eta * grid.depth[:, yi+1, xi+1] + \
                (1-xsi)*eta * grid.depth[:, yi+1, xi]
        z = np.float32(z)

        if depth_vector[-1] > depth_vector[0]:
            depth_indices = depth_vector <= z
            if z >= depth_vector[-1]:
                zi = len(depth_vector) - 2
            else:
                zi = depth_indices.argmin() - 1 if z >= depth_vector[0] else 0
            if z < depth_vector[zi]:
                raise FieldOutOfBoundSurfaceError(0, 0, z, field=self)
            elif z > depth_vector[zi+1]:
                raise FieldOutOfBoundError(x, y, z, field=self)
        else:
            depth_indices = depth_vector >= z
            if z <= depth_vector[-1]:
                zi = len(depth_vector) - 2
            else:
                zi = depth_indices.argmin() - 1 if z <= depth_vector[0] else 0
            if z > depth_vector[zi]:
                raise FieldOutOfBoundSurfaceError(0, 0, z, field=self)
            elif z < depth_vector[zi+1]:
                raise FieldOutOfBoundError(x, y, z, field=self)
        zeta = (z - depth_vector[zi]) / (depth_vector[zi+1]-depth_vector[zi])
        return (zi, zeta)

    def reconnect_bnd_indices(self, xi, yi, xdim, ydim, sphere_mesh):
        if xi < 0:
            if sphere_mesh:
                xi = xdim-2
            else:
                xi = 0
        if xi > xdim-2:
            if sphere_mesh:
                xi = 0
            else:
                xi = xdim-2
        if yi < 0:
            yi = 0
        if yi > ydim-2:
            yi = ydim-2
            if sphere_mesh:
                xi = xdim - xi
        return xi, yi

    def search_indices_rectilinear(self, x, y, z, ti=-1, time=-1, search2D=False):
        grid = self.grid
        xi = yi = -1

        if not grid.zonal_periodic:
            if x < grid.lonlat_minmax[0] or x > grid.lonlat_minmax[1]:
                raise FieldOutOfBoundError(x, y, z, field=self)
        if y < grid.lonlat_minmax[2] or y > grid.lonlat_minmax[3]:
            raise FieldOutOfBoundError(x, y, z, field=self)

        if grid.mesh != 'spherical':
            lon_index = grid.lon < x
            if lon_index.all():
                xi = len(grid.lon) - 2
            else:
                xi = lon_index.argmin() - 1 if lon_index.any() else 0
            xsi = (x-grid.lon[xi]) / (grid.lon[xi+1]-grid.lon[xi])
            if xsi < 0:
                xi -= 1
                xsi = (x-grid.lon[xi]) / (grid.lon[xi+1]-grid.lon[xi])
            elif xsi > 1:
                xi += 1
                xsi = (x-grid.lon[xi]) / (grid.lon[xi+1]-grid.lon[xi])
        else:
            lon_fixed = grid.lon.copy()
            indices = lon_fixed >= lon_fixed[0]
            if not indices.all():
                lon_fixed[indices.argmin():] += 360
            if x < lon_fixed[0]:
                lon_fixed -= 360

            lon_index = lon_fixed < x
            if lon_index.all():
                xi = len(lon_fixed) - 2
            else:
                xi = lon_index.argmin() - 1 if lon_index.any() else 0
            xsi = (x-lon_fixed[xi]) / (lon_fixed[xi+1]-lon_fixed[xi])
            if xsi < 0:
                xi -= 1
                xsi = (x-lon_fixed[xi]) / (lon_fixed[xi+1]-lon_fixed[xi])
            elif xsi > 1:
                xi += 1
                xsi = (x-lon_fixed[xi]) / (lon_fixed[xi+1]-lon_fixed[xi])

        lat_index = grid.lat < y
        if lat_index.all():
            yi = len(grid.lat) - 2
        else:
            yi = lat_index.argmin() - 1 if lat_index.any() else 0

        eta = (y-grid.lat[yi]) / (grid.lat[yi+1]-grid.lat[yi])
        if eta < 0:
            yi -= 1
            eta = (y-grid.lat[yi]) / (grid.lat[yi+1]-grid.lat[yi])
        elif eta > 1:
            yi += 1
            eta = (y-grid.lat[yi]) / (grid.lat[yi+1]-grid.lat[yi])

        if grid.zdim > 1 and not search2D:
            if grid.gtype == GridCode.RectilinearZGrid:
                # Never passes here, because in this case, we work with scipy
                try:
                    (zi, zeta) = self.search_indices_vertical_z(z)
                except FieldOutOfBoundError:
                    raise FieldOutOfBoundError(x, y, z, field=self)
                except FieldOutOfBoundSurfaceError:
                    raise FieldOutOfBoundSurfaceError(x, y, z, field=self)
            elif grid.gtype == GridCode.RectilinearSGrid:
                (zi, zeta) = self.search_indices_vertical_s(x, y, z, xi, yi, xsi, eta, ti, time)
        else:
            zi = -1
            zeta = 0

        if not ((0 <= xsi <= 1) and (0 <= eta <= 1) and (0 <= zeta <= 1)):
            raise FieldSamplingError(x, y, z, field=self)

        return (xsi, eta, zeta, xi, yi, zi)

    def search_indices_curvilinear(self, x, y, z, xi, yi, ti=-1, time=-1, search2D=False):
        xsi = eta = -1
        grid = self.grid
        invA = np.array([[1, 0, 0, 0],
                         [-1, 1, 0, 0],
                         [-1, 0, 0, 1],
                         [1, -1, 1, -1]])
        maxIterSearch = 1e6
        it = 0
        if not grid.zonal_periodic:
            if x < grid.lonlat_minmax[0] or x > grid.lonlat_minmax[1]:
                if grid.lon[0, 0] < grid.lon[0, -1]:
                    raise FieldOutOfBoundError(x, y, z, field=self)
                elif x < grid.lon[0, 0] and x > grid.lon[0, -1]:  # This prevents from crashing in [160, -160]
                    raise FieldOutOfBoundError(x, y, z, field=self)
        if y < grid.lonlat_minmax[2] or y > grid.lonlat_minmax[3]:
            raise FieldOutOfBoundError(x, y, z, field=self)

        while xsi < 0 or xsi > 1 or eta < 0 or eta > 1:
            px = np.array([grid.lon[yi, xi], grid.lon[yi, xi+1], grid.lon[yi+1, xi+1], grid.lon[yi+1, xi]])
            if grid.mesh == 'spherical':
                px[0] = px[0]+360 if px[0] < x-225 else px[0]
                px[0] = px[0]-360 if px[0] > x+225 else px[0]
                px[1:] = np.where(px[1:] - px[0] > 180, px[1:]-360, px[1:])
                px[1:] = np.where(-px[1:] + px[0] > 180, px[1:]+360, px[1:])
            py = np.array([grid.lat[yi, xi], grid.lat[yi, xi+1], grid.lat[yi+1, xi+1], grid.lat[yi+1, xi]])
            a = np.dot(invA, px)
            b = np.dot(invA, py)

            aa = a[3]*b[2] - a[2]*b[3]
            bb = a[3]*b[0] - a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + x*b[3] - y*a[3]
            cc = a[1]*b[0] - a[0]*b[1] + x*b[1] - y*a[1]
            if abs(aa) < 1e-12:  # Rectilinear cell, or quasi
                eta = -cc / bb
            else:
                det2 = bb*bb-4*aa*cc
                if det2 > 0:  # so, if det is nan we keep the xsi, eta from previous iter
                    det = np.sqrt(det2)
                    eta = (-bb+det)/(2*aa)
            if abs(a[1]+a[3]*eta) < 1e-12:  # this happens when recti cell rotated of 90deg
                xsi = ((y-py[0])/(py[1]-py[0]) + (y-py[3])/(py[2]-py[3])) * .5
            else:
                xsi = (x-a[0]-a[2]*eta) / (a[1]+a[3]*eta)
            if xsi < 0 and eta < 0 and xi == 0 and yi == 0:
                raise FieldOutOfBoundError(x, y, 0, field=self)
            if xsi > 1 and eta > 1 and xi == grid.xdim-1 and yi == grid.ydim-1:
                raise FieldOutOfBoundError(x, y, 0, field=self)
            if xsi < 0:
                xi -= 1
            elif xsi > 1:
                xi += 1
            if eta < 0:
                yi -= 1
            elif eta > 1:
                yi += 1
            (xi, yi) = self.reconnect_bnd_indices(xi, yi, grid.xdim, grid.ydim, grid.mesh)
            it += 1
            if it > maxIterSearch:
                print('Correct cell not found after %d iterations' % maxIterSearch)
                raise FieldOutOfBoundError(x, y, 0, field=self)

        if grid.zdim > 1 and not search2D:
            if grid.gtype == GridCode.CurvilinearZGrid:
                try:
                    (zi, zeta) = self.search_indices_vertical_z(z)
                except FieldOutOfBoundError:
                    raise FieldOutOfBoundError(x, y, z, field=self)
            elif grid.gtype == GridCode.CurvilinearSGrid:
                (zi, zeta) = self.search_indices_vertical_s(x, y, z, xi, yi, xsi, eta, ti, time)
        else:
            zi = -1
            zeta = 0

        if not ((0 <= xsi <= 1) and (0 <= eta <= 1) and (0 <= zeta <= 1)):
            raise FieldSamplingError(x, y, z, field=self)

        return (xsi, eta, zeta, xi, yi, zi)

    def search_indices(self, x, y, z, xi, yi, ti=-1, time=-1, search2D=False):
        if self.grid.gtype in [GridCode.RectilinearSGrid, GridCode.RectilinearZGrid]:
            return self.search_indices_rectilinear(x, y, z, ti, time, search2D=search2D)
        else:
            return self.search_indices_curvilinear(x, y, z, xi, yi, ti, time, search2D=search2D)

    def interpolator2D(self, ti, z, y, x):
        xi = 0
        yi = 0
        (xsi, eta, _, xi, yi, _) = self.search_indices(x, y, z, xi, yi)
        if self.interp_method == 'nearest':
            xii = xi if xsi <= .5 else xi+1
            yii = yi if eta <= .5 else yi+1
            return self.data[ti, yii, xii]
        elif self.interp_method in ['linear', 'bgrid_velocity']:
            val = (1-xsi)*(1-eta) * self.data[ti, yi, xi] + \
                xsi*(1-eta) * self.data[ti, yi, xi+1] + \
                xsi*eta * self.data[ti, yi+1, xi+1] + \
                (1-xsi)*eta * self.data[ti, yi+1, xi]
            return val
        elif self.interp_method in ['cgrid_tracer', 'bgrid_tracer']:
            return self.data[ti, yi+1, xi+1]
        elif self.interp_method == 'cgrid_velocity':
            raise RuntimeError("%s is a scalar field. cgrid_velocity interpolation method should be used for vector fields (e.g. FieldSet.UV)" % self.name)
        else:
            raise RuntimeError(self.interp_method+" is not implemented for 2D grids")

    def interpolator3D(self, ti, z, y, x, time):
        xi = int(self.grid.xdim / 2) - 1
        yi = int(self.grid.ydim / 2) - 1
        (xsi, eta, zeta, xi, yi, zi) = self.search_indices(x, y, z, xi, yi, ti, time)
        if self.interp_method == 'nearest':
            xii = xi if xsi <= .5 else xi+1
            yii = yi if eta <= .5 else yi+1
            zii = zi if zeta <= .5 else zi+1
            return self.data[ti, zii, yii, xii]
        elif self.interp_method == 'cgrid_velocity':
            # evaluating W velocity in c_grid
            f0 = self.data[ti, zi, yi+1, xi+1]
            f1 = self.data[ti, zi+1, yi+1, xi+1]
            return (1-zeta) * f0 + zeta * f1
        elif self.interp_method in ['linear', 'bgrid_velocity', 'bgrid_w_velocity']:
            if self.interp_method == 'bgrid_velocity':
                zeta = 0.
            elif self.interp_method == 'bgrid_w_velocity':
                eta = 1.
                xsi = 1.
            data = self.data[ti, zi, :, :]
            f0 = (1-xsi)*(1-eta) * data[yi, xi] + \
                xsi*(1-eta) * data[yi, xi+1] + \
                xsi*eta * data[yi+1, xi+1] + \
                (1-xsi)*eta * data[yi+1, xi]
            data = self.data[ti, zi+1, :, :]
            f1 = (1-xsi)*(1-eta) * data[yi, xi] + \
                xsi*(1-eta) * data[yi, xi+1] + \
                xsi*eta * data[yi+1, xi+1] + \
                (1-xsi)*eta * data[yi+1, xi]
            return (1-zeta) * f0 + zeta * f1
        elif self.interp_method in ['cgrid_tracer', 'bgrid_tracer']:
            return self.data[ti, zi, yi+1, xi+1]
        else:
            raise RuntimeError(self.interp_method+" is not implemented for 3D grids")

    def temporal_interpolate_fullfield(self, ti, time):
        """Calculate the data of a field between two snapshots,
        using linear interpolation

        :param ti: Index in time array associated with time (via :func:`time_index`)
        :param time: Time to interpolate to

        :rtype: Linearly interpolated field"""
        t0 = self.grid.time[ti]
        if time == t0:
            return self.data[ti, :]
        elif ti+1 >= len(self.grid.time):
            raise TimeExtrapolationError(time, field=self, msg='show_time')
        else:
            t1 = self.grid.time[ti+1]
            f0 = self.data[ti, :]
            f1 = self.data[ti+1, :]
            return f0 + (f1 - f0) * ((time - t0) / (t1 - t0))

    def spatial_interpolation(self, ti, z, y, x, time):
        """Interpolate horizontal field values using a SciPy interpolator"""

        if self.grid.zdim == 1:
            val = self.interpolator2D(ti, z, y, x)
        else:
            val = self.interpolator3D(ti, z, y, x, time)
        if np.isnan(val):
            # Detect Out-of-bounds sampling and raise exception
            raise FieldOutOfBoundError(x, y, z, field=self)
        else:
            if isinstance(val, da.core.Array):
                val = val.compute()
            return val

    def time_index(self, time):
        """Find the index in the time array associated with a given time

        Note that we normalize to either the first or the last index
        if the sampled value is outside the time value range.
        """
        if not self.time_periodic and not self.allow_time_extrapolation and (time < self.grid.time[0] or time > self.grid.time[-1]):
            raise TimeExtrapolationError(time, field=self)
        time_index = self.grid.time <= time
        if self.time_periodic:
            if time_index.all() or np.logical_not(time_index).all():
                periods = int(math.floor((time-self.grid.time_full[0])/(self.grid.time_full[-1]-self.grid.time_full[0])))
                if isinstance(self.grid.periods, c_int):
                    self.grid.periods.value = periods
                else:
                    self.grid.periods = periods
                time -= periods*(self.grid.time_full[-1]-self.grid.time_full[0])
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

    def depth_index(self, depth, lat, lon):
        """Find the index in the depth array associated with a given depth"""
        if depth > self.grid.depth[-1]:
            raise FieldOutOfBoundError(lon, lat, depth, field=self)
        depth_index = self.grid.depth <= depth
        if depth_index.all():
            # If given depth == largest field depth, use the second-last
            # field depth (as zidx+1 needed in interpolation)
            return len(self.grid.depth) - 2
        else:
            return depth_index.argmin() - 1 if depth_index.any() else 0

    def eval(self, time, z, y, x, applyConversion=True):
        """Interpolate field values in space and time.

        We interpolate linearly in time and apply implicit unit
        conversion to the result. Note that we defer to
        scipy.interpolate to perform spatial interpolation.
        """
        (ti, periods) = self.time_index(time)
        time -= periods*(self.grid.time_full[-1]-self.grid.time_full[0])
        if ti < self.grid.tdim-1 and time > self.grid.time[ti]:
            f0 = self.spatial_interpolation(ti, z, y, x, time)
            f1 = self.spatial_interpolation(ti + 1, z, y, x, time)
            t0 = self.grid.time[ti]
            t1 = self.grid.time[ti + 1]
            value = f0 + (f1 - f0) * ((time - t0) / (t1 - t0))
        else:
            # Skip temporal interpolation if time is outside
            # of the defined time range or if we have hit an
            # excat value in the time array.
            value = self.spatial_interpolation(ti, z, y, x, self.grid.time[ti])

        if applyConversion:
            return self.units.to_target(value, x, y, z)
        else:
            return value

    def ccode_eval(self, var, t, z, y, x):
        # Casting interp_methd to int as easier to pass on in C-code
        return "temporal_interpolation(%s, %s, %s, %s, %s, particle->cxi, particle->cyi, particle->czi, particle->cti, &%s, %s)" \
            % (x, y, z, t, self.ccode_name, var, self.interp_method.upper())

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
        # self.grid.load_chunk = np.zeros(npartitions, dtype=c_int)
        self.grid.load_chunk = np.zeros(npartitions, dtype=c_int, order='C')
        # self.grid.chunk_info format: number of dimensions (without tdim); number of chunks per dimensions;
        #      chunksizes (the 0th dim sizes for all chunk of dim[0], then so on for next dims
        self.grid.chunk_info = [[len(self.nchunks)-1], list(self.nchunks[1:]), sum(list(list(ci) for ci in chunks[1:]), [])]
        self.grid.chunk_info = sum(self.grid.chunk_info, [])
        self.chunk_set = True

    def chunk_data(self):
        if not self.chunk_set:
            self.chunk_setup()
        # self.grid.load_chunk code:
        # 0: not loaded
        # 1: was asked to load by kernel in JIT
        # 2: is loaded and was touched last C call
        # 3: is loaded
        if isinstance(self.data, da.core.Array):
            for block_id in range(len(self.grid.load_chunk)):
                if self.grid.load_chunk[block_id] == 1 or self.grid.load_chunk[block_id] > 1 and self.data_chunks[block_id] is None:
                    block = self.get_block(block_id)
                    # self.data_chunks[block_id] = np.array(self.data.blocks[(slice(self.grid.tdim),) + block])
                    self.data_chunks[block_id] = np.array(self.data.blocks[(slice(self.grid.tdim),) + block], order='C')
                elif self.grid.load_chunk[block_id] == 0:
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

            self.grid.load_chunk[0] = 2
            # self.data_chunks[0] = np.array(self.data)
            self.data_chunks[0] = np.array(self.data, order='C')

    @property
    def ctypes_struct(self):
        """Returns a ctypes struct object containing all relevant
        pointers and sizes for this field."""

        # Ctypes struct corresponding to the type definition in parcels.h
        class CField(Structure):
            _fields_ = [('xdim', c_int), ('ydim', c_int), ('zdim', c_int),
                        ('tdim', c_int), ('igrid', c_int),
                        ('allow_time_extrapolation', c_int),
                        ('time_periodic', c_int),
                        ('data_chunks', POINTER(POINTER(POINTER(c_float)))),
                        ('grid', POINTER(CGrid))]

        # Create and populate the c-struct object
        allow_time_extrapolation = 1 if self.allow_time_extrapolation else 0
        time_periodic = 1 if self.time_periodic else 0
        for i in range(len(self.grid.load_chunk)):
            if self.grid.load_chunk[i] == 1:
                raise ValueError('data_chunks should have been loaded by now if requested. grid.load_chunk[bid] cannot be 1')
            if self.grid.load_chunk[i] > 1:
                if not self.data_chunks[i].flags['C_CONTIGUOUS']:
                    # self.data_chunks[i] = self.data_chunks[i].copy()
                    self.data_chunks[i] = np.array(self.data_chunks[i], order='C')
                self.c_data_chunks[i] = self.data_chunks[i].ctypes.data_as(POINTER(POINTER(c_float)))
            else:
                self.c_data_chunks[i] = None

        cstruct = CField(self.grid.xdim, self.grid.ydim, self.grid.zdim,
                         self.grid.tdim, self.igrid, allow_time_extrapolation, time_periodic,
                         (POINTER(POINTER(c_float)) * len(self.c_data_chunks))(*self.c_data_chunks),
                         pointer(self.grid.ctypes_struct))
        return cstruct

    def show(self, animation=False, show_time=None, domain=None, depth_level=0, projection=None, land=True,
             vmin=None, vmax=None, savefile=None, **kwargs):
        """Method to 'show' a Parcels Field

        :param animation: Boolean whether result is a single plot, or an animation
        :param show_time: Time at which to show the Field (only in single-plot mode)
        :param domain: dictionary (with keys 'N', 'S', 'E', 'W') defining domain to show
        :param depth_level: depth level to be plotted (default 0)
        :param projection: type of cartopy projection to use (default PlateCarree)
        :param land: Boolean whether to show land. This is ignored for flat meshes
        :param vmin: minimum colour scale (only in single-plot mode)
        :param vmax: maximum colour scale (only in single-plot mode)
        :param savefile: Name of a file to save the plot to
        """
        from parcels.plotting import plotfield
        plt, _, _, _ = plotfield(self, animation=animation, show_time=show_time, domain=domain, depth_level=depth_level,
                                 projection=projection, land=land, vmin=vmin, vmax=vmax, savefile=savefile, **kwargs)
        if plt:
            plt.show()

    def add_periodic_halo(self, zonal, meridional, halosize=5, data=None):
        """Add a 'halo' to all Fields in a FieldSet, through extending the Field (and lon/lat)
        by copying a small portion of the field on one side of the domain to the other.
        Before adding a periodic halo to the Field, it has to be added to the Grid on which the Field depends

        :param zonal: Create a halo in zonal direction (boolean)
        :param meridional: Create a halo in meridional direction (boolean)
        :param halosize: size of the halo (in grid points). Default is 5 grid points
        :param data: if data is not None, the periodic halo will be achieved on data instead of self.data and data will be returned
        """
        dataNone = not isinstance(data, (np.ndarray, da.core.Array))
        if self.grid.defer_load and dataNone:
            return
        data = self.data if dataNone else data
        lib = np if isinstance(data, np.ndarray) else da
        if zonal:
            if len(data.shape) == 3:
                data = lib.concatenate((data[:, :, -halosize:], data,
                                       data[:, :, 0:halosize]), axis=len(data.shape)-1)
                assert data.shape[2] == self.grid.xdim, "Third dim must be x."
            else:
                data = lib.concatenate((data[:, :, :, -halosize:], data,
                                       data[:, :, :, 0:halosize]), axis=len(data.shape) - 1)
                assert data.shape[3] == self.grid.xdim, "Fourth dim must be x."
            self.lon = self.grid.lon
            self.lat = self.grid.lat
        if meridional:
            if len(data.shape) == 3:
                data = lib.concatenate((data[:, -halosize:, :], data,
                                       data[:, 0:halosize, :]), axis=len(data.shape)-2)
                assert data.shape[1] == self.grid.ydim, "Second dim must be y."
            else:
                data = lib.concatenate((data[:, :, -halosize:, :], data,
                                       data[:, :, 0:halosize, :]), axis=len(data.shape) - 2)
                assert data.shape[2] == self.grid.ydim, "Third dim must be y."
            self.lat = self.grid.lat
        if dataNone:
            self.data = data
        else:
            return data

    def write(self, filename, varname=None):
        """Write a :class:`Field` to a netcdf file

        :param filename: Basename of the file
        :param varname: Name of the field, to be appended to the filename"""
        filepath = str(Path('%s%s.nc' % (filename, self.name)))
        if varname is None:
            varname = self.name
        # Derive name of 'depth' variable for NEMO convention
        vname_depth = 'depth%s' % self.name.lower()

        # Create DataArray objects for file I/O
        if self.grid.gtype == GridCode.RectilinearZGrid:
            nav_lon = xr.DataArray(self.grid.lon + np.zeros((self.grid.ydim, self.grid.xdim), dtype=np.float32),
                                   coords=[('y', self.grid.lat), ('x', self.grid.lon)])
            nav_lat = xr.DataArray(self.grid.lat.reshape(self.grid.ydim, 1) + np.zeros(self.grid.xdim, dtype=np.float32),
                                   coords=[('y', self.grid.lat), ('x', self.grid.lon)])
        elif self.grid.gtype == GridCode.CurvilinearZGrid:
            nav_lon = xr.DataArray(self.grid.lon, coords=[('y', range(self.grid.ydim)),
                                                          ('x', range(self.grid.xdim))])
            nav_lat = xr.DataArray(self.grid.lat, coords=[('y', range(self.grid.ydim)),
                                                          ('x', range(self.grid.xdim))])
        else:
            raise NotImplementedError('Field.write only implemented for RectilinearZGrid and CurvilinearZGrid')

        attrs = {'units': 'seconds since ' + str(self.grid.time_origin)} if self.grid.time_origin.calendar else {}
        time_counter = xr.DataArray(self.grid.time,
                                    dims=['time_counter'],
                                    attrs=attrs)
        vardata = xr.DataArray(self.data.reshape((self.grid.tdim, self.grid.zdim, self.grid.ydim, self.grid.xdim)),
                               dims=['time_counter', vname_depth, 'y', 'x'])
        # Create xarray Dataset and output to netCDF format
        attrs = {'parcels_mesh': self.grid.mesh}
        dset = xr.Dataset({varname: vardata}, coords={'nav_lon': nav_lon,
                                                      'nav_lat': nav_lat,
                                                      'time_counter': time_counter,
                                                      vname_depth: self.grid.depth}, attrs=attrs)
        dset.to_netcdf(filepath)

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
            data = lib.concatenate([data_to_concat, data[tindex+1:, :]], axis=0)
        elif tindex == 1:
            data = lib.concatenate([data[:tindex, :], data_to_concat, data[tindex+1:, :]], axis=0)
        elif tindex == 2:
            data = lib.concatenate([data[:tindex, :], data_to_concat], axis=0)
        else:
            raise ValueError("data_concatenate is used for computeTimeChunk, with tindex in [0, 1, 2]")
        return data

    def advancetime(self, field_new, advanceForward):
        if isinstance(self.data) is not isinstance(field_new):
            logger.warning("[Field.advancetime] New field data and persistent field data have different types - time advance not possible.")
            return
        lib = np if isinstance(self.data, np.ndarray) else da
        if advanceForward == 1:  # forward in time, so appending at end
            self.data = lib.concatenate((self.data[1:, :, :], field_new.data[:, :, :]), 0)
            self.time = self.grid.time
        else:  # backward in time, so prepending at start
            self.data = lib.concatenate((field_new.data[:, :, :], self.data[:-1, :, :]), 0)
            self.time = self.grid.time

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

        filebuffer = NetcdfFileBuffer(self.dataFiles[g.ti + tindex], self.dimensions, self.indices,
                                      self.netcdf_engine, timestamp=timestamp,
                                      interp_method=self.interp_method,
                                      data_full_zdim=self.data_full_zdim,
                                      field_chunksize=self.field_chunksize,
                                      rechunk_callback_fields=self.chunk_setup)
        filebuffer.__enter__()
        time_data = filebuffer.time
        time_data = g.time_origin.reltime(time_data)
        filebuffer.ti = (time_data <= g.time[tindex]).argmin() - 1
        if self.netcdf_engine != 'xarray':
            filebuffer.name = filebuffer.parse_name(self.filebuffername)
        buffer_data = filebuffer.data
        lib = np if isinstance(buffer_data, np.ndarray) else da
        if len(buffer_data.shape) == 2:
            buffer_data = lib.reshape(buffer_data, sum(((1, 1), buffer_data.shape), ()))
        elif len(buffer_data.shape) == 3 and g.zdim > 1:
            buffer_data = lib.reshape(buffer_data, sum(((1, ), buffer_data.shape), ()))
        elif len(buffer_data.shape) == 3:
            buffer_data = lib.reshape(buffer_data, sum(((buffer_data.shape[0], 1, ), buffer_data.shape[1:]), ()))
        data = self.data_concatenate(data, buffer_data, tindex)
        self.filebuffers[tindex] = filebuffer
        return data

    def __add__(self, field):
        if isinstance(self, Field) and isinstance(field, Field):
            return SummedField('_SummedField', [self, field])
        elif isinstance(field, SummedField):
            assert isinstance(self, type(field[0])), 'Fields in a SummedField should be either all scalars or all vectors'
            field.insert(0, self)
            return field


class VectorField(object):
    """Class VectorField stores 2 or 3 fields which defines together a vector field.
    This enables to interpolate them as one single vector field in the kernels.

    :param name: Name of the vector field
    :param U: field defining the zonal component
    :param V: field defining the meridional component
    :param W: field defining the vertical component (default: None)
    """
    def __init__(self, name, U, V, W=None):
        self.name = name
        self.U = U
        self.V = V
        self.W = W
        self.vector_type = '3D' if W else '2D'
        if self.U.interp_method == 'cgrid_velocity':
            assert self.V.interp_method == 'cgrid_velocity', (
                'Interpolation methods of U and V are not the same.')
            assert self.U.grid is self.V.grid, (
                'Grids of U and V are not the same.')
            if self.vector_type == '3D':
                assert self.W.interp_method == 'cgrid_velocity', (
                    'Interpolation methods of U and W are not the same.')
                assert self.W.grid is self.U.grid, (
                    'Grids of U and W are not the same.')

    def dist(self, lon1, lon2, lat1, lat2, mesh, lat):
        if mesh == 'spherical':
            rad = np.pi/180.
            deg2m = 1852 * 60.
            return np.sqrt(((lon2-lon1)*deg2m*math.cos(rad * lat))**2 + ((lat2-lat1)*deg2m)**2)
        else:
            return np.sqrt((lon2-lon1)**2 + (lat2-lat1)**2)

    def jacobian(self, xsi, eta, px, py):
        dphidxsi = [eta-1, 1-eta, eta, -eta]
        dphideta = [xsi-1, -xsi, xsi, 1-xsi]

        dxdxsi = np.dot(px, dphidxsi)
        dxdeta = np.dot(px, dphideta)
        dydxsi = np.dot(py, dphidxsi)
        dydeta = np.dot(py, dphideta)
        jac = dxdxsi*dydeta - dxdeta*dydxsi
        return jac

    def spatial_c_grid_interpolation2D(self, ti, z, y, x, time):
        grid = self.U.grid
        xi = int(grid.xdim / 2) - 1
        yi = int(grid.ydim / 2) - 1
        (xsi, eta, zeta, xi, yi, zi) = self.U.search_indices(x, y, z, xi, yi, ti, time)

        if grid.gtype in [GridCode.RectilinearSGrid, GridCode.RectilinearZGrid]:
            px = np.array([grid.lon[xi], grid.lon[xi+1], grid.lon[xi+1], grid.lon[xi]])
            py = np.array([grid.lat[yi], grid.lat[yi], grid.lat[yi+1], grid.lat[yi+1]])
        else:
            px = np.array([grid.lon[yi, xi], grid.lon[yi, xi+1], grid.lon[yi+1, xi+1], grid.lon[yi+1, xi]])
            py = np.array([grid.lat[yi, xi], grid.lat[yi, xi+1], grid.lat[yi+1, xi+1], grid.lat[yi+1, xi]])

        if grid.mesh == 'spherical':
            px[0] = px[0]+360 if px[0] < x-225 else px[0]
            px[0] = px[0]-360 if px[0] > x+225 else px[0]
            px[1:] = np.where(px[1:] - px[0] > 180, px[1:]-360, px[1:])
            px[1:] = np.where(-px[1:] + px[0] > 180, px[1:]+360, px[1:])
        xx = (1-xsi)*(1-eta) * px[0] + xsi*(1-eta) * px[1] + xsi*eta * px[2] + (1-xsi)*eta * px[3]
        assert abs(xx-x) < 1e-4
        c1 = self.dist(px[0], px[1], py[0], py[1], grid.mesh, np.dot(i_u.phi2D_lin(xsi, 0.), py))
        c2 = self.dist(px[1], px[2], py[1], py[2], grid.mesh, np.dot(i_u.phi2D_lin(1., eta), py))
        c3 = self.dist(px[2], px[3], py[2], py[3], grid.mesh, np.dot(i_u.phi2D_lin(xsi, 1.), py))
        c4 = self.dist(px[3], px[0], py[3], py[0], grid.mesh, np.dot(i_u.phi2D_lin(0., eta), py))
        if grid.zdim == 1:
            U0 = self.U.data[ti, yi+1, xi] * c4
            U1 = self.U.data[ti, yi+1, xi+1] * c2
            V0 = self.V.data[ti, yi, xi+1] * c1
            V1 = self.V.data[ti, yi+1, xi+1] * c3
        else:
            U0 = self.U.data[ti, zi, yi+1, xi] * c4
            U1 = self.U.data[ti, zi, yi+1, xi+1] * c2
            V0 = self.V.data[ti, zi, yi, xi+1] * c1
            V1 = self.V.data[ti, zi, yi+1, xi+1] * c3
        U = (1-xsi) * U0 + xsi * U1
        V = (1-eta) * V0 + eta * V1
        rad = np.pi/180.
        deg2m = 1852 * 60.
        meshJac = (deg2m * deg2m * math.cos(rad * y)) if grid.mesh == 'spherical' else 1
        jac = self.jacobian(xsi, eta, px, py) * meshJac

        u = ((-(1-eta) * U - (1-xsi) * V) * px[0]
             + ((1-eta) * U - xsi * V) * px[1]
             + (eta * U + xsi * V) * px[2]
             + (-eta * U + (1-xsi) * V) * px[3]) / jac
        v = ((-(1-eta) * U - (1-xsi) * V) * py[0]
             + ((1-eta) * U - xsi * V) * py[1]
             + (eta * U + xsi * V) * py[2]
             + (-eta * U + (1-xsi) * V) * py[3]) / jac
        if isinstance(u, da.core.Array):
            u = u.compute()
            v = v.compute()
        return (u, v)

    def spatial_c_grid_interpolation3D_full(self, ti, z, y, x, time):
        grid = self.U.grid
        xi = int(grid.xdim / 2) - 1
        yi = int(grid.ydim / 2) - 1
        (xsi, eta, zet, xi, yi, zi) = self.U.search_indices(x, y, z, xi, yi, ti, time)

        if grid.gtype in [GridCode.RectilinearSGrid, GridCode.RectilinearZGrid]:
            px = np.array([grid.lon[xi], grid.lon[xi+1], grid.lon[xi+1], grid.lon[xi]])
            py = np.array([grid.lat[yi], grid.lat[yi], grid.lat[yi+1], grid.lat[yi+1]])
        else:
            px = np.array([grid.lon[yi, xi], grid.lon[yi, xi+1], grid.lon[yi+1, xi+1], grid.lon[yi+1, xi]])
            py = np.array([grid.lat[yi, xi], grid.lat[yi, xi+1], grid.lat[yi+1, xi+1], grid.lat[yi+1, xi]])

        if grid.mesh == 'spherical':
            px[0] = px[0]+360 if px[0] < x-225 else px[0]
            px[0] = px[0]-360 if px[0] > x+225 else px[0]
            px[1:] = np.where(px[1:] - px[0] > 180, px[1:]-360, px[1:])
            px[1:] = np.where(-px[1:] + px[0] > 180, px[1:]+360, px[1:])
        xx = (1-xsi)*(1-eta) * px[0] + xsi*(1-eta) * px[1] + xsi*eta * px[2] + (1-xsi)*eta * px[3]
        assert abs(xx-x) < 1e-4

        px = np.concatenate((px, px))
        py = np.concatenate((py, py))
        if grid.z4d:
            pz = np.array([grid.depth[0, zi, yi, xi], grid.depth[0, zi, yi, xi+1], grid.depth[0, zi, yi+1, xi+1], grid.depth[0, zi, yi+1, xi],
                           grid.depth[0, zi+1, yi, xi], grid.depth[0, zi+1, yi, xi+1], grid.depth[0, zi+1, yi+1, xi+1], grid.depth[0, zi+1, yi+1, xi]])
        else:
            pz = np.array([grid.depth[zi, yi, xi], grid.depth[zi, yi, xi+1], grid.depth[zi, yi+1, xi+1], grid.depth[zi, yi+1, xi],
                           grid.depth[zi+1, yi, xi], grid.depth[zi+1, yi, xi+1], grid.depth[zi+1, yi+1, xi+1], grid.depth[zi+1, yi+1, xi]])

        u0 = self.U.data[ti, zi, yi+1, xi]
        u1 = self.U.data[ti, zi, yi+1, xi+1]
        v0 = self.V.data[ti, zi, yi, xi+1]
        v1 = self.V.data[ti, zi, yi+1, xi+1]
        w0 = self.W.data[ti, zi, yi+1, xi+1]
        w1 = self.W.data[ti, zi+1, yi+1, xi+1]

        U0 = u0 * i_u.jacobian3D_lin_face(px, py, pz, 0, eta, zet, 'zonal', grid.mesh)
        U1 = u1 * i_u.jacobian3D_lin_face(px, py, pz, 1, eta, zet, 'zonal', grid.mesh)
        V0 = v0 * i_u.jacobian3D_lin_face(px, py, pz, xsi, 0, zet, 'meridional', grid.mesh)
        V1 = v1 * i_u.jacobian3D_lin_face(px, py, pz, xsi, 1, zet, 'meridional', grid.mesh)
        W0 = w0 * i_u.jacobian3D_lin_face(px, py, pz, xsi, eta, 0, 'vertical', grid.mesh)
        W1 = w1 * i_u.jacobian3D_lin_face(px, py, pz, xsi, eta, 1, 'vertical', grid.mesh)

        # Computing fluxes in half left hexahedron -> flux_u05
        xx = [px[0], (px[0]+px[1])/2, (px[2]+px[3])/2, px[3], px[4], (px[4]+px[5])/2, (px[6]+px[7])/2, px[7]]
        yy = [py[0], (py[0]+py[1])/2, (py[2]+py[3])/2, py[3], py[4], (py[4]+py[5])/2, (py[6]+py[7])/2, py[7]]
        zz = [pz[0], (pz[0]+pz[1])/2, (pz[2]+pz[3])/2, pz[3], pz[4], (pz[4]+pz[5])/2, (pz[6]+pz[7])/2, pz[7]]
        flux_u0 = u0 * i_u.jacobian3D_lin_face(xx, yy, zz, 0, .5, .5, 'zonal', grid.mesh)
        flux_v0_halfx = v0 * i_u.jacobian3D_lin_face(xx, yy, zz, .5, 0, .5, 'meridional', grid.mesh)
        flux_v1_halfx = v1 * i_u.jacobian3D_lin_face(xx, yy, zz, .5, 1, .5, 'meridional', grid.mesh)
        flux_w0_halfx = w0 * i_u.jacobian3D_lin_face(xx, yy, zz, .5, .5, 0, 'vertical', grid.mesh)
        flux_w1_halfx = w1 * i_u.jacobian3D_lin_face(xx, yy, zz, .5, .5, 1, 'vertical', grid.mesh)
        flux_u05 = flux_u0 + flux_v0_halfx - flux_v1_halfx + flux_w0_halfx - flux_w1_halfx

        # Computing fluxes in half front hexahedron -> flux_v05
        xx = [px[0], px[1], (px[1]+px[2])/2, (px[0]+px[3])/2, px[4], px[5], (px[5]+px[6])/2, (px[4]+px[7])/2]
        yy = [py[0], py[1], (py[1]+py[2])/2, (py[0]+py[3])/2, py[4], py[5], (py[5]+py[6])/2, (py[4]+py[7])/2]
        zz = [pz[0], pz[1], (pz[1]+pz[2])/2, (pz[0]+pz[3])/2, pz[4], pz[5], (pz[5]+pz[6])/2, (pz[4]+pz[7])/2]
        flux_u0_halfy = u0 * i_u.jacobian3D_lin_face(xx, yy, zz, 0, .5, .5, 'zonal', grid.mesh)
        flux_u1_halfy = u1 * i_u.jacobian3D_lin_face(xx, yy, zz, 1, .5, .5, 'zonal', grid.mesh)
        flux_v0 = v0 * i_u.jacobian3D_lin_face(xx, yy, zz, .5, 0, .5, 'meridional', grid.mesh)
        flux_w0_halfy = w0 * i_u.jacobian3D_lin_face(xx, yy, zz, .5, .5, 0, 'vertical', grid.mesh)
        flux_w1_halfy = w1 * i_u.jacobian3D_lin_face(xx, yy, zz, .5, .5, 1, 'vertical', grid.mesh)
        flux_v05 = flux_u0_halfy - flux_u1_halfy + flux_v0 + flux_w0_halfy - flux_w1_halfy

        # Computing fluxes in half lower hexahedron -> flux_w05
        xx = [px[0], px[1], px[2], px[3], (px[0]+px[4])/2, (px[1]+px[5])/2, (px[2]+px[6])/2, (px[3]+px[7])/2]
        yy = [py[0], py[1], py[2], py[3], (py[0]+py[4])/2, (py[1]+py[5])/2, (py[2]+py[6])/2, (py[3]+py[7])/2]
        zz = [pz[0], pz[1], pz[2], pz[3], (pz[0]+pz[4])/2, (pz[1]+pz[5])/2, (pz[2]+pz[6])/2, (pz[3]+pz[7])/2]
        flux_u0_halfz = u0 * i_u.jacobian3D_lin_face(xx, yy, zz, 0, .5, .5, 'zonal', grid.mesh)
        flux_u1_halfz = u1 * i_u.jacobian3D_lin_face(xx, yy, zz, 1, .5, .5, 'zonal', grid.mesh)
        flux_v0_halfz = v0 * i_u.jacobian3D_lin_face(xx, yy, zz, .5, 0, .5, 'meridional', grid.mesh)
        flux_v1_halfz = v1 * i_u.jacobian3D_lin_face(xx, yy, zz, .5, 1, .5, 'meridional', grid.mesh)
        flux_w0 = w0 * i_u.jacobian3D_lin_face(xx, yy, zz, .5, .5, 0, 'vertical', grid.mesh)
        flux_w05 = flux_u0_halfz - flux_u1_halfz + flux_v0_halfz - flux_v1_halfz + flux_w0

        surf_u05 = i_u.jacobian3D_lin_face(px, py, pz, .5, .5, .5, 'zonal', grid.mesh)
        jac_u05 = i_u.jacobian3D_lin_face(px, py, pz, .5, eta, zet, 'zonal', grid.mesh)
        U05 = flux_u05 / surf_u05 * jac_u05

        surf_v05 = i_u.jacobian3D_lin_face(px, py, pz, .5, .5, .5, 'meridional', grid.mesh)
        jac_v05 = i_u.jacobian3D_lin_face(px, py, pz, xsi, .5, zet, 'meridional', grid.mesh)
        V05 = flux_v05 / surf_v05 * jac_v05

        surf_w05 = i_u.jacobian3D_lin_face(px, py, pz, .5, .5, .5, 'vertical', grid.mesh)
        jac_w05 = i_u.jacobian3D_lin_face(px, py, pz, xsi, eta, .5, 'vertical', grid.mesh)
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

    def spatial_c_grid_interpolation3D(self, ti, z, y, x, time):
        """
          __ V1 __
        |          |
        U0         U1
        | __ V0 __ |
        The interpolation is done in the following by
        interpolating linearly U depending on the longitude coordinate and
        interpolating linearly V depending on the latitude coordinate.
        Curvilinear grids are treated properly, since the element is projected to a rectilinear parent element.
        """
        if self.U.grid.gtype in [GridCode.RectilinearSGrid, GridCode.CurvilinearSGrid]:
            (u, v, w) = self.spatial_c_grid_interpolation3D_full(ti, z, y, x, time)
        else:
            (u, v) = self.spatial_c_grid_interpolation2D(ti, z, y, x, time)
            w = self.W.eval(time, z, y, x, False)
            w = self.W.units.to_target(w, x, y, z)
        return (u, v, w)

    def eval(self, time, z, y, x):
        if self.U.interp_method != 'cgrid_velocity':
            u = self.U.eval(time, z, y, x, False)
            v = self.V.eval(time, z, y, x, False)
            u = self.U.units.to_target(u, x, y, z)
            v = self.V.units.to_target(v, x, y, z)
            if self.vector_type == '3D':
                w = self.W.eval(time, z, y, x, False)
                w = self.W.units.to_target(w, x, y, z)
                return (u, v, w)
            else:
                return (u, v)
        else:
            grid = self.U.grid
            (ti, periods) = self.U.time_index(time)
            time -= periods*(grid.time_full[-1]-grid.time_full[0])
            if ti < grid.tdim-1 and time > grid.time[ti]:
                t0 = grid.time[ti]
                t1 = grid.time[ti + 1]
                if self.vector_type == '3D':
                    (u0, v0, w0) = self.spatial_c_grid_interpolation3D(ti, z, y, x, time)
                    (u1, v1, w1) = self.spatial_c_grid_interpolation3D(ti + 1, z, y, x, time)
                    w = w0 + (w1 - w0) * ((time - t0) / (t1 - t0))
                else:
                    (u0, v0) = self.spatial_c_grid_interpolation2D(ti, z, y, x, time)
                    (u1, v1) = self.spatial_c_grid_interpolation2D(ti + 1, z, y, x, time)
                u = u0 + (u1 - u0) * ((time - t0) / (t1 - t0))
                v = v0 + (v1 - v0) * ((time - t0) / (t1 - t0))
                if self.vector_type == '3D':
                    return (u, v, w)
                else:
                    return (u, v)
            else:
                # Skip temporal interpolation if time is outside
                # of the defined time range or if we have hit an
                # excat value in the time array.
                if self.vector_type == '3D':
                    return self.spatial_c_grid_interpolation3D(ti, z, y, x, grid.time[ti])
                else:
                    return self.spatial_c_grid_interpolation2D(ti, z, y, x, grid.time[ti])

    def __getitem__(self, key):
        return self.eval(*key)

    def ccode_eval(self, varU, varV, varW, U, V, W, t, z, y, x):
        # Casting interp_methd to int as easier to pass on in C-code
        if self.vector_type == '3D':
            return "temporal_interpolationUVW(%s, %s, %s, %s, %s, %s, %s, " \
                   % (x, y, z, t, U.ccode_name, V.ccode_name, W.ccode_name) + \
                   "particle->cxi, particle->cyi, particle->czi, particle->cti, &%s, &%s, &%s, %s)" \
                   % (varU, varV, varW, U.interp_method.upper())
        else:
            return "temporal_interpolationUV(%s, %s, %s, %s, %s, %s, " \
                   % (x, y, z, t, U.ccode_name, V.ccode_name) + \
                   "particle->cxi, particle->cyi, particle->czi, particle->cti, &%s, &%s, %s)" \
                   % (varU, varV, U.interp_method.upper())


class DeferredArray():
    """Class used for throwing error when Field.data is not read in deferred loading mode"""
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
        raise RuntimeError("Field is in deferred_load mode, so can't be accessed. Use .computeTimeChunk() method to force loading of data")


class SummedField(list):
    """Class SummedField is a list of Fields over which Field interpolation
    is summed. This can e.g. be used when combining multiple flow fields,
    where the total flow is the sum of all the individual flows.
    Note that the individual Fields can be on different Grids.
    Also note that, since SummedFields are lists, the individual Fields can
    still be queried through their list index (e.g. SummedField[1]).
    SummedField is composed of either Fields or VectorFields.

    :param name: Name of the SummedField
    :param F: List of fields. F can be a scalar Field, a VectorField, or the zonal component (U) of the VectorField
    :param V: List of fields defining the meridional component of a VectorField, if F is the zonal component. (default: None)
    :param W: List of fields defining the vertical component of a VectorField, if F and V are the zonal and meridional components (default: None)
    """

    def __init__(self, name, F, V=None, W=None):
        if V is None:
            if isinstance(F[0], VectorField):
                vector_type = F[0].vector_type
            for Fi in F:
                assert isinstance(Fi, Field) or (isinstance(Fi, VectorField) and Fi.vector_type == vector_type), 'Components of a SummedField must be Field or VectorField'
                self.append(Fi)
        elif W is None:
            for (i, Fi, Vi) in zip(range(len(F)), F, V):
                assert isinstance(Fi, Field) and isinstance(Vi, Field), \
                    'F, and V components of a SummedField must be Field'
                self.append(VectorField(name+'_%d' % i, Fi, Vi))
        else:
            for (i, Fi, Vi, Wi) in zip(range(len(F)), F, V, W):
                assert isinstance(Fi, Field) and isinstance(Vi, Field) and isinstance(Wi, Field), \
                    'F, V and W components of a SummedField must be Field'
                self.append(VectorField(name+'_%d' % i, Fi, Vi, Wi))
        self.name = name

    def __getitem__(self, key):
        if isinstance(key, int):
            return list.__getitem__(self, key)
        else:
            vals = []
            val = None
            for iField in range(len(self)):
                val = list.__getitem__(self, iField).eval(*key)
                vals.append(val)
            return tuple(np.sum(vals, 0)) if isinstance(val, tuple) else np.sum(vals)

    def __add__(self, field):
        if isinstance(field, Field):
            assert isinstance(self[0], type(field)), 'Fields in a SummedField should be either all scalars or all vectors'
            self.append(field)
        elif isinstance(field, SummedField):
            assert isinstance(self[0], type(field[0])), 'Fields in a SummedField should be either all scalars or all vectors'
            for fld in field:
                self.append(fld)
        return self


class NestedField(list):
    """Class NestedField is a list of Fields from which the first one to be not declared out-of-boundaries
    at particle position is interpolated. This induces that the order of the fields in the list matters.
    Each one it its turn, a field is interpolated: if the interpolation succeeds or if an error other
    than `ErrorOutOfBounds` is thrown, the function is stopped. Otherwise, next field is interpolated.
    NestedField returns an `ErrorOutOfBounds` only if last field is as well out of boundaries.
    NestedField is composed of either Fields or VectorFields.

    :param name: Name of the NestedField
    :param F: List of fields (order matters). F can be a scalar Field, a VectorField, or the zonal component (U) of the VectorField
    :param V: List of fields defining the meridional component of a VectorField, if F is the zonal component. (default: None)
    :param W: List of fields defining the vertical component of a VectorField, if F and V are the zonal and meridional components (default: None)
    """

    def __init__(self, name, F, V=None, W=None):
        if V is None:
            if isinstance(F[0], VectorField):
                vector_type = F[0].vector_type
            for Fi in F:
                assert isinstance(Fi, Field) or (isinstance(Fi, VectorField) and Fi.vector_type == vector_type), 'Components of a NestedField must be Field or VectorField'
                self.append(Fi)
        elif W is None:
            for (i, Fi, Vi) in zip(range(len(F)), F, V):
                assert isinstance(Fi, Field) and isinstance(Vi, Field), \
                    'F, and V components of a NestedField must be Field'
                self.append(VectorField(name+'_%d' % i, Fi, Vi))
        else:
            for (i, Fi, Vi, Wi) in zip(range(len(F)), F, V, W):
                assert isinstance(Fi, Field) and isinstance(Vi, Field) and isinstance(Wi, Field), \
                    'F, V and W components of a NestedField must be Field'
                self.append(VectorField(name+'_%d' % i, Fi, Vi, Wi))
        self.name = name

    def __getitem__(self, key):
        if isinstance(key, int):
            return list.__getitem__(self, key)
        else:
            for iField in range(len(self)):
                try:
                    val = list.__getitem__(self, iField).eval(*key)
                    break
                except (FieldOutOfBoundError, FieldSamplingError):
                    if iField == len(self)-1:
                        raise
                    else:
                        pass
            return val


class NetcdfFileBuffer(object):
    _name_maps = {'lon': ['lon', 'nav_lon', 'x', 'longitude', 'lo', 'ln', 'i'],
                  'lat': ['lat', 'nav_lat', 'y', 'latitude', 'la', 'lt', 'j'],
                  'depth': ['depth', 'depthu', 'depthv', 'depthw', 'depths', 'deptht', 'depthx', 'depthy', 'depthz',
                            'z', 'z_u', 'z_v', 'z_w', 'd', 'k', 'w_dep', 'w_deps'],
                  'time': ['time', 'time_count', 'time_counter', 'timer_count', 't']}
    _min_dim_chunksize = 16

    """ Class that encapsulates and manages deferred access to file data. """
    def __init__(self, filename, dimensions, indices, netcdf_engine, timestamp=None,
                 interp_method='linear', data_full_zdim=None, field_chunksize='auto', rechunk_callback_fields=None, lock_file=True):
        self.filename = filename
        self.dimensions = dimensions  # Dict with dimension keys for file data
        self.indices = indices
        self.dataset = None
        self.netcdf_engine = netcdf_engine
        self.timestamp = timestamp
        self.ti = None
        self.interp_method = interp_method
        self.data_full_zdim = data_full_zdim
        self.field_chunksize = field_chunksize
        self.chunk_mapping = None
        self.rechunk_callback_fields = rechunk_callback_fields
        self.chunking_finalized = False
        self.lock_file = lock_file

    def __enter__(self):
        if self.netcdf_engine == 'xarray':
            self.dataset = self.filename
            return self

        if self.field_chunksize not in [False, None, 'auto'] and type(self.field_chunksize) not in [list, tuple, dict]:
            raise AttributeError("'field_chunksize' is of wrong type. Parameter is expected to be a list, tuple or dict per data dimension, or be False, None or 'auto'.")
        if isinstance(self.field_chunksize, list):
            self.field_chunksize = tuple(self.field_chunksize)

        init_chunk_dict = None
        if self.field_chunksize not in [False, None]:
            init_chunk_dict = self._get_initial_chunk_dictionary()
        try:
            # Unfortunately we need to do if-else here, cause the lock-parameter is either False or a Lock-object
            # (which we would rather want to have being auto-managed).
            # If 'lock' is not specified, the Lock-object is auto-created and managed bz xarray internally.
            if self.lock_file:
                self.dataset = xr.open_dataset(str(self.filename), decode_cf=True, engine=self.netcdf_engine, chunks=init_chunk_dict)
            else:
                self.dataset = xr.open_dataset(str(self.filename), decode_cf=True, engine=self.netcdf_engine, chunks=init_chunk_dict, lock=False)
            self.dataset['decoded'] = True
        except:
            logger.warning_once("File %s could not be decoded properly by xarray (version %s).\n         It will be opened with no decoding. Filling values might be wrongly parsed."
                                % (self.filename, xr.__version__))
            if self.lock_file:
                self.dataset = xr.open_dataset(str(self.filename), decode_cf=False, engine=self.netcdf_engine, chunks=init_chunk_dict)
            else:
                self.dataset = xr.open_dataset(str(self.filename), decode_cf=False, engine=self.netcdf_engine, chunks=init_chunk_dict, lock=False)
            self.dataset['decoded'] = False

        for inds in self.indices.values():
            if type(inds) not in [list, range]:
                raise RuntimeError('Indices for field subsetting need to be a list')
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        if self.netcdf_engine == 'xarray':
            pass
        else:
            if self.dataset is not None:
                self.dataset.close()
                self.dataset = None
        self.chunking_finalized = False
        self.chunk_mapping = None

    def _get_initial_chunk_dictionary(self):
        # ==== check-opening requested dataset to access metadata                   ==== #
        # ==== file-opening and dimension-reading does not require a decode or lock ==== #
        self.dataset = xr.open_dataset(str(self.filename), decode_cf=False, engine=self.netcdf_engine, chunks={}, lock=False)
        self.dataset['decoded'] = False
        # ==== self.dataset temporarily available ==== #
        init_chunk_dict = {}
        if isinstance(self.field_chunksize, dict):
            init_chunk_dict = self.field_chunksize
        elif isinstance(self.field_chunksize, tuple):  # and (len(self.dimensions) == len(self.field_chunksize)):
            tmp_chs = [0, ] * len(self.field_chunksize)
            chunk_index = len(self.field_chunksize)-1

            loni, lonname, _ = self._is_dimension_in_dataset('lon')
            if loni >= 0 and chunk_index >= 0:
                init_chunk_dict[lonname] = self.field_chunksize[chunk_index]
                tmp_chs[chunk_index] = self.field_chunksize[chunk_index]
            else:
                logger.warning_once(self._netcdf_DimNotFound_warning_message('lon'))
            chunk_index -= 1

            lati, latname, _ = self._is_dimension_in_dataset('lat')
            if lati >= 0 and chunk_index >= 0:
                init_chunk_dict[latname] = self.field_chunksize[chunk_index]
                tmp_chs[chunk_index] = self.field_chunksize[chunk_index]
            else:
                logger.warning_once(self._netcdf_DimNotFound_warning_message('lat'))
            chunk_index -= 1

            depthi, depthname, _ = self._is_dimension_in_dataset('depth')
            if depthi >= 0 and chunk_index >= 0:
                if self._is_dimension_available('depth'):
                    init_chunk_dict[depthname] = self.field_chunksize[chunk_index]
                    tmp_chs[chunk_index] = self.field_chunksize[chunk_index]
            else:
                logger.warning_once(self._netcdf_DimNotFound_warning_message('depth'))
            chunk_index -= 1

            timei, timename, _ = self._is_dimension_in_dataset('time')
            if timei >= 0 and chunk_index >= 0:
                if self._is_dimension_available('time'):
                    init_chunk_dict[timename] = self.field_chunksize[chunk_index]
                    tmp_chs[chunk_index] = self.field_chunksize[chunk_index]
            else:
                logger.warning_once(self._netcdf_DimNotFound_warning_message('time'))
            chunk_index -= 1

            # ==== re-arrange the tupe and correct for empty dimensions ==== #
            for chunk_index in range(len(self.field_chunksize)-1, -1, -1):
                if tmp_chs[chunk_index] < 1:
                    tmp_chs.pop(chunk_index)
            self.field_chunksize = tuple(tmp_chs)

        elif self.field_chunksize == 'auto':
            av_mem = psutil.virtual_memory().available
            chunk_cap = av_mem * (1/8) * (1/3)
            if 'array.chunk-size' in da_conf.config.keys():
                chunk_cap = da_utils.parse_bytes(da_conf.config.get('array.chunk-size'))
            else:
                predefined_cap = da_conf.get('array.chunk-size')
                if predefined_cap is not None:
                    chunk_cap = da_utils.parse_bytes(predefined_cap)
                else:
                    logger.info_once("Unable to locate chunking hints from dask, thus estimating the max. chunk size heuristically. Please consider defining the 'chunk-size' for 'array' in your local dask configuration file (see http://oceanparcels.org/faq.html#field_chunking_config and https://docs.dask.org).")
            loni, lonname, lonvalue = self._is_dimension_in_dataset('lon')
            lati, latname, latvalue = self._is_dimension_in_dataset('lat')
            if lati >= 0 and loni >= 0:
                pDim = int(math.floor(math.sqrt(chunk_cap/np.dtype(np.float64).itemsize)))
                init_chunk_dict[latname] = min(latvalue, pDim)
                init_chunk_dict[lonname] = min(lonvalue, pDim)
            timei, timename, _ = self._is_dimension_in_dataset('time')
            if timei >= 0:
                init_chunk_dict[timename] = 1
            depthi, depthname, depthvalue = self._is_dimension_in_dataset('depth')
            if depthi >= 0:
                init_chunk_dict[depthname] = max(1, depthvalue)
        # ==== closing check-opened requested dataset ==== #
        self.dataset.close()
        # ==== check if the chunksize reading is successfull. if not, load the file ONCE really into memory and ==== #
        # ==== deduce the chunking from the array dims.                                                         ==== #
        try:
            self.dataset = xr.open_dataset(str(self.filename), decode_cf=True, engine=self.netcdf_engine, chunks=init_chunk_dict, lock=False)
        except:
            # ==== fail - open it as a normal array and deduce the dimensions from the read field ==== #
            init_chunk_dict = {}
            self.dataset = ncDataset(str(self.filename))
            refdims = self.dataset.dimensions.keys()
            max_field = ""
            max_dim_names = ()
            max_overlay_dims = 0
            for vname in self.dataset.variables:
                var = self.dataset.variables[vname]
                overlay_dims = []
                for vdname in var.dimensions:
                    if vdname in refdims:
                        overlay_dims.append(vdname)
                n_overlay_dims = len(overlay_dims)
                if n_overlay_dims > max_overlay_dims:
                    max_field = vname
                    max_dim_names = tuple(overlay_dims)
                    max_overlay_dims = n_overlay_dims
            self.name = max_field
            for dname in max_dim_names:
                # if dname in self._name_maps['time'] and self.dataset.dimensions[dname].size == 1:
                #     continue
                if isinstance(self.field_chunksize, dict):
                    if dname in self.field_chunksize.keys():
                        init_chunk_dict[dname] = min(self.field_chunksize[dname], self.dataset.dimensions[dname].size)
                        continue
                init_chunk_dict[dname] = min(self._min_dim_chunksize, self.dataset.dimensions[dname].size)
            # ==== because in this case it has shown that the requested field_chunksize setup cannot be used, ==== #
            # ==== replace the requested field_chunksize with this auto-derived version.                      ==== #
            self.field_chunksize = init_chunk_dict
        finally:
            self.dataset.close()
        self.dataset = None
        # ==== self.dataset not available ==== #
        return init_chunk_dict

    def _is_dimension_available(self, dimension_name):
        if self.dimensions is None or self.dataset is None:
            return False
        return dimension_name in self.dimensions

    def _is_dimension_in_dataset(self, dimension_name):
        k, dname, dvalue = (-1, '', 0)
        if self.dimensions is None or self.dataset is None:
            return k, dname, dvalue
        dimension_name = dimension_name.lower()
        for i, name in enumerate(self._name_maps[dimension_name]):
            if name in self.dataset.dims:
                value = self.dataset.dims[name]
                k, dname, dvalue = i, name, value
                break
        return k, dname, dvalue

    def _is_dimension_in_chunksize_request(self, dimension_name):
        k, dname, dvalue = (-1, '', 0)
        if self.dimensions is None or self.dataset is None:
            return k, dname, dvalue
        dimension_name = dimension_name.lower()
        for i, name in enumerate(self._name_maps[dimension_name]):
            if name in self.field_chunksize:
                value = self.field_chunksize[name]
                k, dname, dvalue = i, name, value
                break
        return k, dname, dvalue

    def _netcdf_DimNotFound_warning_message(self, dimension_name):
        display_name = dimension_name if (dimension_name not in self.dimensions) else self.dimensions[dimension_name]
        return "Did not find {} in NetCDF dims. Please specifiy field_chunksize as dictionary for NetCDF dimension names, e.g.\n field_chunksize={{ '{}': <number>, ... }}.".format(display_name, display_name)

    def _chunksize_to_chunkmap(self):
        if self.field_chunksize in [False, 'auto', None]:
            return
        self.chunk_mapping = {}
        if isinstance(self.field_chunksize, tuple):
            for i in range(len(self.field_chunksize)):
                self.chunk_mapping[i] = self.field_chunksize[i]
        else:
            # ====== 'time' is strictly excluded from the reading dimensions as it is implicitly organized with the data ====== #
            timei, timename, timevalue = self._is_dimension_in_chunksize_request('time')
            dtimei, dtimename, dtimevalue = self._is_dimension_in_dataset('time')
            depthi, depthname, depthvalue = self._is_dimension_in_chunksize_request('depth')
            ddepthi, ddepthname, ddepthvalue = self._is_dimension_in_dataset('depth')
            lati, latname, latvalue = self._is_dimension_in_chunksize_request('lat')
            loni, lonname, lonvalue = self._is_dimension_in_chunksize_request('lon')
            dim_index = 0
            if len(self.field_chunksize) == 2:

                self.chunk_mapping[dim_index] = latvalue
                dim_index += 1
                self.chunk_mapping[dim_index] = lonvalue
                dim_index += 1
            elif len(self.field_chunksize) >= 3:
                if timei >= 0 and timevalue > 1 and dtimei >= 0 and dtimevalue > 1 and self._is_dimension_available('time'):
                    # self.chunk_mapping[dim_index] = self.field_chunksize[self.dimensions['time']]
                    self.chunk_mapping[dim_index] = 1  # still need to make sure that we only load 1 time step at a time
                    dim_index += 1

                if depthi >= 0 and depthvalue > 1 and ddepthi >= 0 and ddepthvalue > 1 and self._is_dimension_available('depth'):
                    # self.chunk_mapping[dim_index] = self.field_chunksize[self.dimensions['depth']]
                    self.chunk_mapping[dim_index] = depthvalue
                    dim_index += 1
                self.chunk_mapping[dim_index] = latvalue
                dim_index += 1
                self.chunk_mapping[dim_index] = lonvalue
                dim_index += 1

    def _chunkmap_to_chunksize(self):
        if self.field_chunksize in [False, None]:
            return
        self.field_chunksize = {}
        chunk_map = self.chunk_mapping
        timei, _, timevalue = self._is_dimension_in_dataset('time')
        depthi, _, depthvalue = self._is_dimension_in_dataset('depth')
        if len(chunk_map) == 2:
            self.field_chunksize[self.dimensions['lat']] = chunk_map[0]
            self.field_chunksize[self.dimensions['lon']] = chunk_map[1]
        elif len(chunk_map) == 3:
            chunk_dim_index = 0
            if depthi >= 0 and depthvalue > 1 and self._is_dimension_available('depth'):
                self.field_chunksize[self.dimensions['depth']] = chunk_map[chunk_dim_index]
                chunk_dim_index += 1
            elif timei >= 0 and timevalue > 1 and self._is_dimension_available('time'):
                self.field_chunksize[self.dimensions['time']] = chunk_map[chunk_dim_index]
                chunk_dim_index += 1
            self.field_chunksize[self.dimensions['lat']] = chunk_map[chunk_dim_index]
            chunk_dim_index += 1
            self.field_chunksize[self.dimensions['lon']] = chunk_map[chunk_dim_index]
        elif len(chunk_map) >= 4:
            self.field_chunksize[self.dimensions['time']] = chunk_map[0]
            self.field_chunksize[self.dimensions['depth']] = chunk_map[1]
            self.field_chunksize[self.dimensions['lat']] = chunk_map[2]
            self.field_chunksize[self.dimensions['lon']] = chunk_map[3]
            dim_index = 4
            for dim_name in self.dimensions:
                if dim_name not in ['time', 'depth', 'lat', 'lon']:
                    self.field_chunksize[self.dimensions[dim_name]] = chunk_map[dim_index]
                    dim_index += 1

    def parse_name(self, name):
        if isinstance(name, list):
            for nm in name:
                if hasattr(self.dataset, nm):
                    name = nm
                    break
        if isinstance(name, list):
            raise IOError('None of variables in list found in file')
        return name

    @property
    def read_lonlat(self):
        lon = self.dataset[self.dimensions['lon']]
        lat = self.dataset[self.dimensions['lat']]
        xdim = lon.size if len(lon.shape) == 1 else lon.shape[-1]
        ydim = lat.size if len(lat.shape) == 1 else lat.shape[-2]
        self.indices['lon'] = self.indices['lon'] if 'lon' in self.indices else range(xdim)
        self.indices['lat'] = self.indices['lat'] if 'lat' in self.indices else range(ydim)
        if len(lon.shape) == 1:
            lon_subset = np.array(lon[self.indices['lon']])
            lat_subset = np.array(lat[self.indices['lat']])
        elif len(lon.shape) == 2:
            lon_subset = np.array(lon[self.indices['lat'], self.indices['lon']])
            lat_subset = np.array(lat[self.indices['lat'], self.indices['lon']])
        elif len(lon.shape) == 3:  # some lon, lat have a time dimension 1
            lon_subset = np.array(lon[0, self.indices['lat'], self.indices['lon']])
            lat_subset = np.array(lat[0, self.indices['lat'], self.indices['lon']])
        elif len(lon.shape) == 4:  # some lon, lat have a time and depth dimension 1
            lon_subset = np.array(lon[0, 0, self.indices['lat'], self.indices['lon']])
            lat_subset = np.array(lat[0, 0, self.indices['lat'], self.indices['lon']])
        if len(lon.shape) > 1:  # Tests if lon, lat are rectilinear but were stored in arrays
            rectilinear = True
            # test if all columns and rows are the same for lon and lat (in which case grid is rectilinear)
            for xi in range(1, lon_subset.shape[0]):
                if not np.allclose(lon_subset[0, :], lon_subset[xi, :]):
                    rectilinear = False
                    break
            if rectilinear:
                for yi in range(1, lat_subset.shape[1]):
                    if not np.allclose(lat_subset[:, 0], lat_subset[:, yi]):
                        rectilinear = False
                        break
            if rectilinear:
                lon_subset = lon_subset[0, :]
                lat_subset = lat_subset[:, 0]
        return lon_subset, lat_subset

    @property
    def read_depth(self):
        if 'depth' in self.dimensions:
            depth = self.dataset[self.dimensions['depth']]
            depthsize = depth.size if len(depth.shape) == 1 else depth.shape[-3]
            self.data_full_zdim = depthsize
            self.indices['depth'] = self.indices['depth'] if 'depth' in self.indices else range(depthsize)
            if len(depth.shape) == 1:
                return np.array(depth[self.indices['depth']])
            elif len(depth.shape) == 3:
                return np.array(depth[self.indices['depth'], self.indices['lat'], self.indices['lon']])
            elif len(depth.shape) == 4:
                # raise NotImplementedError('Time varying depth data cannot be read in netcdf files yet')
                return np.array(depth[:, self.indices['depth'], self.indices['lat'], self.indices['lon']])
        else:
            self.indices['depth'] = [0]
            return np.zeros(1)

    @property
    def read_depth_dimensions(self):
        if 'depth' in self.dimensions:
            data = self.dataset[self.name]
            depthsize = data.shape[-3]
            self.data_full_zdim = depthsize
            self.indices['depth'] = self.indices['depth'] if 'depth' in self.indices else range(depthsize)
            return np.empty((0, len(self.indices['depth']), len(self.indices['lat']), len(self.indices['lon'])))

    @property
    def data(self):
        return self.data_access()

    def data_access(self):
        if self.chunk_mapping is None and self.field_chunksize not in ['auto', False, None]:
            self.chunk_mapping = {}
            if(isinstance(self.field_chunksize, tuple)):
                j = 0
                for i in range(len(self.field_chunksize)):
                    if self.field_chunksize[i] <= 1:
                        continue
                    self.chunk_mapping[j] = self.field_chunksize[i]
                    j += 1
                self.field_chunksize = tuple([self.chunk_mapping[i] for i in range(len(self.chunk_mapping))])
            else:
                self._chunksize_to_chunkmap()
        data = self.dataset[self.name]
        libcheck = data.data if isinstance(data, xr.DataArray) else data
        lib = np if isinstance(libcheck, np.ndarray) else da
        libcheck = None

        ti = range(data.shape[0]) if self.ti is None else self.ti
        if len(data.shape) == 2:
            data = data[self.indices['lat'], self.indices['lon']]
        elif len(data.shape) == 3:
            if self.indices['depth'][-1] == self.data_full_zdim-1 and data.shape[0] == self.data_full_zdim-1 and self.interp_method in ['bgrid_velocity', 'bgrid_w_velocity', 'bgrid_tracer']:
                # Add a bottom level of zeros for B-grid if missing in the data.
                # The last level is unused by B-grid interpolator (U, V, tracer) but must be there
                # to match Parcels data shape. for W, last level must be 0 for impermeability
                for dim in ['depth', 'lat', 'lon']:
                    if not isinstance(self.indices[dim], (list, range)):
                        raise NotImplementedError("For B grids, indices must be provided as a range")
                        # this is because da.concatenate needs data which are indexed using slices, not a range of indices
                d0 = self.indices['depth'][0]
                d1 = self.indices['depth'][-1]+1
                lat0 = self.indices['lat'][0]
                lat1 = self.indices['lat'][-1]+1
                lon0 = self.indices['lon'][0]
                lon1 = self.indices['lon'][-1]+1
                data = lib.concatenate((data[d0:d1-1, lat0:lat1, lon0:lon1],
                                       da.zeros((1, lat1-lat0, lon1-lon0))), axis=0)
            elif len(self.indices['depth']) > 1:
                data = data[self.indices['depth'], self.indices['lat'], self.indices['lon']]
            else:
                data = data[ti, self.indices['lat'], self.indices['lon']]
        else:
            if self.indices['depth'][-1] == self.data_full_zdim-1 and data.shape[1] == self.data_full_zdim-1 and self.interp_method in ['bgrid_velocity', 'bgrid_w_velocity', 'bgrid_tracer']:
                for dim in ['depth', 'lat', 'lon']:
                    if not isinstance(self.indices[dim], (list, range)):
                        raise NotImplementedError("For B grids, indices must be provided as a range")
                        # this is because da.concatenate needs data which are indexed using slices, not a range of indices
                d0 = self.indices['depth'][0]
                d1 = self.indices['depth'][-1]+1
                lat0 = self.indices['lat'][0]
                lat1 = self.indices['lat'][-1]+1
                lon0 = self.indices['lon'][0]
                lon1 = self.indices['lon'][-1]+1
                if(type(ti) in [list, range]):
                    t0 = ti[0]
                    t1 = ti[-1]+1
                    data = lib.concatenate((data[t0:t1, d0:d1-1, lat0:lat1, lon0:lon1],
                                           da.zeros((t1-t0, 1, lat1-lat0, lon1-lon0))), axis=1)
                else:
                    data = lib.concatenate((data[ti, d0:d1-1, lat0:lat1, lon0:lon1],
                                           da.zeros((1, lat1-lat0, lon1-lon0))), axis=0)
            else:
                data = data[ti, self.indices['depth'], self.indices['lat'], self.indices['lon']]

        if isinstance(data, xr.DataArray):
            data = data.data
        if self.field_chunksize is False:
            data = np.array(data)
            self.chunking_finalized = True
        else:
            if isinstance(data, da.core.Array):
                if not self.chunking_finalized:
                    if self.field_chunksize == 'auto':
                        if data.shape[-2:] != data.chunksize[-2:]:
                            data = data.rechunk(self.field_chunksize)
                        self.chunk_mapping = {}
                        chunkIndex = 0
                        # timei, _, timevalue = self._is_dimension_in_dataset('time')
                        # depthi, _, depthvalue = self._is_dimension_in_dataset('depth')
                        # has_time = timei >= 0 and timevalue > 1
                        # has_depth = depthi >= 0 and depthvalue > 1
                        startblock = 0
                        # startblock += 1 if has_time and not self._is_dimension_available('time') else 0
                        # startblock += 1 if has_depth and not self._is_dimension_available('depth') else 0
                        for chunkDim in data.chunksize[startblock:]:
                            self.chunk_mapping[chunkIndex] = chunkDim
                            chunkIndex += 1
                        self._chunkmap_to_chunksize()
                        if self.rechunk_callback_fields is not None:
                            self.rechunk_callback_fields()
                            self.chunking_finalized = True
                    else:
                        # ==== I think this can be "pass" too ==== #
                        data = data.rechunk(self.chunk_mapping)
                        self.chunking_finalized = True
            else:
                da_data = da.from_array(data, chunks=self.field_chunksize)
                if self.field_chunksize == 'auto' and da_data.shape[-2:] == da_data.chunksize[-2:]:
                    data = np.array(data)
                else:
                    data = da_data
                if not self.chunking_finalized and self.rechunk_callback_fields is not None:
                    self.rechunk_callback_fields()
                    self.chunking_finalized = True

        return data

    @property
    def time(self):
        return self.time_access()

    def time_access(self):
        if self.timestamp is not None:
            return self.timestamp

        if 'time' not in self.dimensions:
            return np.array([None])

        time_da = self.dataset[self.dimensions['time']]
        convert_xarray_time_units(time_da, self.dimensions['time'])
        time = np.array([time_da[self.dimensions['time']]]) if len(time_da.shape) == 0 else np.array(time_da[self.dimensions['time']])
        if isinstance(time[0], datetime.datetime):
            raise NotImplementedError('Parcels currently only parses dates ranging from 1678 AD to 2262 AD, which are stored by xarray as np.datetime64. If you need a wider date range, please open an Issue on the parcels github page.')
        return time
