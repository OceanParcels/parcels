import datetime
import math
import warnings

import dask.array as da
import numpy as np
import psutil
import xarray as xr
from dask import config as da_conf
from dask import utils as da_utils
from netCDF4 import Dataset as ncDataset

from parcels._typing import InterpMethodOption
from parcels.tools.converters import convert_xarray_time_units
from parcels.tools.statuscodes import DaskChunkingError
from parcels.tools.warnings import FileWarning


class _FileBuffer:
    def __init__(
        self,
        filename,
        dimensions,
        indices,
        timestamp=None,
        interp_method: InterpMethodOption = "linear",
        data_full_zdim=None,
        **kwargs,
    ):
        self.filename = filename
        self.dimensions = dimensions  # Dict with dimension keys for file data
        self.indices = indices
        self.dataset = None
        self.timestamp = timestamp
        self.cast_data_dtype = kwargs.pop("cast_data_dtype", np.float32)
        self.ti = None
        self.interp_method = interp_method
        self.gridindexingtype = kwargs.pop("gridindexingtype", "nemo")
        self.data_full_zdim = data_full_zdim
        if ("lon" in self.indices) or ("lat" in self.indices):
            self.nolonlatindices = False
        else:
            self.nolonlatindices = True


class NetcdfFileBuffer(_FileBuffer):
    def __init__(self, *args, **kwargs):
        self.lib = np
        self.netcdf_engine = kwargs.pop("netcdf_engine", "netcdf4")
        super().__init__(*args, **kwargs)

    def __enter__(self):
        try:
            # Unfortunately we need to do if-else here, cause the lock-parameter is either False or a Lock-object
            # (which we would rather want to have being auto-managed).
            # If 'lock' is not specified, the Lock-object is auto-created and managed by xarray internally.
            self.dataset = xr.open_dataset(str(self.filename), decode_cf=True, engine=self.netcdf_engine)
            self.dataset["decoded"] = True
        except:
            warnings.warn(
                f"File {self.filename} could not be decoded properly by xarray (version {xr.__version__}). "
                "It will be opened with no decoding. Filling values might be wrongly parsed.",
                FileWarning,
                stacklevel=2,
            )

            self.dataset = xr.open_dataset(str(self.filename), decode_cf=False, engine=self.netcdf_engine)
            self.dataset["decoded"] = False
        for inds in self.indices.values():
            if type(inds) not in [list, range]:
                raise RuntimeError("Indices for field subsetting need to be a list")
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        if self.dataset is not None:
            self.dataset.close()
            self.dataset = None

    def parse_name(self, name):
        if isinstance(name, list):
            for nm in name:
                if hasattr(self.dataset, nm):
                    name = nm
                    break
        if isinstance(name, list):
            raise OSError("None of variables in list found in file")
        return name

    @property
    def lonlat(self):
        lon = self.dataset[self.dimensions["lon"]]
        lat = self.dataset[self.dimensions["lat"]]
        if self.nolonlatindices and self.gridindexingtype not in ["croco"]:
            if len(lon.shape) < 3:
                lon_subset = np.array(lon)
                lat_subset = np.array(lat)
            elif len(lon.shape) == 3:  # some lon, lat have a time dimension 1
                lon_subset = np.array(lon[0, :, :])
                lat_subset = np.array(lat[0, :, :])
            elif len(lon.shape) == 4:  # some lon, lat have a time and depth dimension 1
                lon_subset = np.array(lon[0, 0, :, :])
                lat_subset = np.array(lat[0, 0, :, :])
        else:
            xdim = lon.size if len(lon.shape) == 1 else lon.shape[-1]
            ydim = lat.size if len(lat.shape) == 1 else lat.shape[-2]
            if self.gridindexingtype in ["croco"]:
                xdim -= 1
                ydim -= 1
            self.indices["lon"] = self.indices["lon"] if "lon" in self.indices else range(xdim)
            self.indices["lat"] = self.indices["lat"] if "lat" in self.indices else range(ydim)
            if len(lon.shape) == 1:
                lon_subset = np.array(lon[self.indices["lon"]])
                lat_subset = np.array(lat[self.indices["lat"]])
            elif len(lon.shape) == 2:
                lon_subset = np.array(lon[self.indices["lat"], self.indices["lon"]])
                lat_subset = np.array(lat[self.indices["lat"], self.indices["lon"]])
            elif len(lon.shape) == 3:  # some lon, lat have a time dimension 1
                lon_subset = np.array(lon[0, self.indices["lat"], self.indices["lon"]])
                lat_subset = np.array(lat[0, self.indices["lat"], self.indices["lon"]])
            elif len(lon.shape) == 4:  # some lon, lat have a time and depth dimension 1
                lon_subset = np.array(lon[0, 0, self.indices["lat"], self.indices["lon"]])
                lat_subset = np.array(lat[0, 0, self.indices["lat"], self.indices["lon"]])

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
    def depth(self):
        if "depth" in self.dimensions:
            depth = self.dataset[self.dimensions["depth"]]
            depthsize = depth.size if len(depth.shape) == 1 else depth.shape[-3]
            if self.gridindexingtype in ["croco"]:
                depthsize -= 1
            self.data_full_zdim = depthsize
            self.indices["depth"] = self.indices["depth"] if "depth" in self.indices else range(depthsize)
            if len(depth.shape) == 1:
                return np.array(depth[self.indices["depth"]])
            elif len(depth.shape) == 3:
                if self.nolonlatindices:
                    return np.array(depth[self.indices["depth"], :, :])
                else:
                    return np.array(depth[self.indices["depth"], self.indices["lat"], self.indices["lon"]])
            elif len(depth.shape) == 4:
                if self.nolonlatindices:
                    return np.array(depth[:, self.indices["depth"], :, :])
                else:
                    return np.array(depth[:, self.indices["depth"], self.indices["lat"], self.indices["lon"]])
        else:
            self.indices["depth"] = [0]
            return np.zeros(1)

    @property
    def depth_dimensions(self):
        if "depth" in self.dimensions:
            data = self.dataset[self.name]
            depthsize = data.shape[-3]
            self.data_full_zdim = depthsize
            self.indices["depth"] = self.indices["depth"] if "depth" in self.indices else range(depthsize)
            if self.nolonlatindices:
                return np.empty((0, len(self.indices["depth"])) + data.shape[-2:])
            else:
                return np.empty((0, len(self.indices["depth"]), len(self.indices["lat"]), len(self.indices["lon"])))

    def _check_extend_depth(self, data, di):
        return (
            self.indices["depth"][-1] == self.data_full_zdim - 1
            and data.shape[di] == self.data_full_zdim - 1
            and self.interp_method in ["bgrid_velocity", "bgrid_w_velocity", "bgrid_tracer"]
        )

    def _apply_indices(self, data, ti):
        if len(data.shape) == 2:
            if self.nolonlatindices:
                pass
            else:
                data = data[self.indices["lat"], self.indices["lon"]]
        elif len(data.shape) == 3:
            if self._check_extend_depth(data, 0):
                if self.nolonlatindices:
                    data = data[self.indices["depth"][:-1], :, :]
                else:
                    data = data[self.indices["depth"][:-1], self.indices["lat"], self.indices["lon"]]
            elif len(self.indices["depth"]) > 1:
                if self.nolonlatindices:
                    data = data[self.indices["depth"], :, :]
                else:
                    data = data[self.indices["depth"], self.indices["lat"], self.indices["lon"]]
            else:
                if self.nolonlatindices:
                    data = data[ti, :, :]
                else:
                    data = data[ti, self.indices["lat"], self.indices["lon"]]
        else:
            if self._check_extend_depth(data, 1):
                if self.nolonlatindices:
                    data = data[ti, self.indices["depth"][:-1], :, :]
                else:
                    data = data[ti, self.indices["depth"][:-1], self.indices["lat"], self.indices["lon"]]
            else:
                if self.nolonlatindices:
                    data = data[ti, self.indices["depth"], :, :]
                else:
                    data = data[ti, self.indices["depth"], self.indices["lat"], self.indices["lon"]]
        return data

    @property
    def data(self):
        return self.data_access()

    def data_access(self):
        data = self.dataset[self.name]
        ti = range(data.shape[0]) if self.ti is None else self.ti
        data = self._apply_indices(data, ti)
        return np.array(data, dtype=self.cast_data_dtype)

    @property
    def time(self):
        return self.time_access()

    def time_access(self):
        if self.timestamp is not None:
            return self.timestamp

        if "time" not in self.dimensions:
            return np.array([None])

        time_da = self.dataset[self.dimensions["time"]]
        convert_xarray_time_units(time_da, self.dimensions["time"])
        time = (
            np.array([time_da[self.dimensions["time"]].data])
            if len(time_da.shape) == 0
            else np.array(time_da[self.dimensions["time"]])
        )
        if isinstance(time[0], datetime.datetime):
            raise NotImplementedError(
                "Parcels currently only parses dates ranging from 1678 AD to 2262 AD, which are stored by xarray as np.datetime64. If you need a wider date range, please open an Issue on the parcels github page."
            )
        return time


class DeferredNetcdfFileBuffer(NetcdfFileBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DaskFileBuffer(NetcdfFileBuffer):
    _static_name_maps = {
        "time": ["time", "time_count", "time_counter", "timer_count", "t"],
        "depth": [
            "depth",
            "depthu",
            "depthv",
            "depthw",
            "depths",
            "deptht",
            "depthx",
            "depthy",
            "depthz",
            "z",
            "z_u",
            "z_v",
            "z_w",
            "d",
            "k",
            "w_dep",
            "w_deps",
            "Z",
            "Zp1",
            "Zl",
            "Zu",
            "level",
        ],
        "lat": ["lat", "nav_lat", "y", "latitude", "la", "lt", "j", "YC", "YG"],
        "lon": ["lon", "nav_lon", "x", "longitude", "lo", "ln", "i", "XC", "XG"],
    }
    _min_dim_chunksize = 16

    """ Class that encapsulates and manages deferred access to file data. """

    def __init__(self, *args, **kwargs):
        """
        Initializes this specific filebuffer type. As a result of using dask, the internal library is set to 'da'.
        The chunksize parameter is popped from the argument list, as well as the locking-parameter and the
        rechunk callback function. Also chunking-related variables are initialized.
        """
        self.lib = da
        self.chunksize = kwargs.pop("chunksize", "auto")
        self.lock_file = kwargs.pop("lock_file", True)
        self.chunk_mapping = None
        self.rechunk_callback_fields = kwargs.pop("rechunk_callback_fields", None)
        self.chunking_finalized = False
        self.autochunkingfailed = False
        super().__init__(*args, **kwargs)

    def __enter__(self):
        """
        This function enters the physical file (equivalent to a 'with open(...)' statement) and returns a file object.
        In Dask, with dynamic loading, this is the point where we have access to the header-information of the file.
        Hence, this function initializes the dynamic loading by parsing the chunksize-argument and maps the requested
        chunksizes onto the variables found in the file. For auto-chunking, educated guesses are made (e.g. with the
        dask configuration file in the background) to determine the ideal chunk sizes. This is also the point
        where - due to the chunking, the file is 'locked', meaning that it cannot be simultaneously accessed by
        another process. This is significant in a cluster setup.
        """
        if self.chunksize not in [False, None, "auto"] and type(self.chunksize) is not dict:
            raise AttributeError(
                "'chunksize' is of wrong type. Parameter is expected to be a dict per data dimension, or be False, None or 'auto'."
            )
        if isinstance(self.chunksize, list):
            self.chunksize = tuple(self.chunksize)

        init_chunk_dict = None
        if self.chunksize not in [False, None]:
            init_chunk_dict = self._get_initial_chunk_dictionary()
        try:
            # Unfortunately we need to do if-else here, cause the lock-parameter is either False or a Lock-object
            # (which we would rather want to have being auto-managed).
            # If 'lock' is not specified, the Lock-object is auto-created and managed by xarray internally.
            if self.lock_file:
                self.dataset = xr.open_dataset(
                    str(self.filename), decode_cf=True, engine=self.netcdf_engine, chunks=init_chunk_dict
                )
            else:
                self.dataset = xr.open_dataset(
                    str(self.filename), decode_cf=True, engine=self.netcdf_engine, chunks=init_chunk_dict, lock=False
                )
            self.dataset["decoded"] = True
        except:
            warnings.warn(
                f"File {self.filename} could not be decoded properly by xarray (version {xr.__version__}). "
                "It will be opened with no decoding. Filling values might be wrongly parsed.",
                FileWarning,
                stacklevel=2,
            )
            if self.lock_file:
                self.dataset = xr.open_dataset(
                    str(self.filename), decode_cf=False, engine=self.netcdf_engine, chunks=init_chunk_dict
                )
            else:
                self.dataset = xr.open_dataset(
                    str(self.filename), decode_cf=False, engine=self.netcdf_engine, chunks=init_chunk_dict, lock=False
                )
            self.dataset["decoded"] = False

        for inds in self.indices.values():
            if type(inds) not in [list, range]:
                raise RuntimeError("Indices for field subsetting need to be a list")
        return self

    def __exit__(self, type, value, traceback):
        """Function releases the file handle.

        This function releases the file handle. Hence access to the dataset and its header-information is lost. The
        previously executed chunking is lost. Furthermore, if the file access required file locking, the lock-handle
        is freed so other processes can now access the file again.
        """
        self.close()

    def close(self):
        """Teardown FileBuffer object with dask.

        This function can be called to initialise an orderly teardown of a FileBuffer object with dask, meaning
        to release the file handle, deposing the dataset, and releasing the file lock (if required).
        """
        if self.dataset is not None:
            self.dataset.close()
            self.dataset = None
        self.chunking_finalized = False
        self.chunk_mapping = None

    @classmethod
    def add_to_dimension_name_map_global(self, name_map):
        """
        [externally callable]
        This function adds entries to the name map from parcels_dim -> netcdf_dim. This is required if you want to
        use auto-chunking on large fields whose map parameters are not defined. This function must be called before
        entering the filebuffer object. Example:
        DaskFileBuffer.add_to_dimension_name_map_global({'lat': 'nydim',
                                                         'lon': 'nxdim',
                                                         'time': 'ntdim',
                                                         'depth': 'nddim'})
        fieldset = FieldSet(..., chunksize='auto')
        [...]
        Note that not all parcels dimensions need to be present in 'name_map'.
        """
        assert isinstance(name_map, dict)
        for pcls_dim_name in name_map.keys():
            if isinstance(name_map[pcls_dim_name], list):
                for nc_dim_name in name_map[pcls_dim_name]:
                    self._static_name_maps[pcls_dim_name].append(nc_dim_name)
            elif isinstance(name_map[pcls_dim_name], str):
                self._static_name_maps[pcls_dim_name].append(name_map[pcls_dim_name])

    def add_to_dimension_name_map(self, name_map):
        """
        [externally callable]
        This function adds entries to the name map from parcels_dim -> netcdf_dim. This is required if you want to
        use auto-chunking on large fields whose map parameters are not defined. This function must be called after
        constructing an filebuffer object and before entering the filebuffer. Example:
        fb = DaskFileBuffer(...)
        fb.add_to_dimension_name_map({'lat': 'nydim', 'lon': 'nxdim', 'time': 'ntdim', 'depth': 'nddim'})
        with fb:
            [do_stuff}
        Note that not all parcels dimensions need to be present in 'name_map'.
        """
        assert isinstance(name_map, dict)
        for pcls_dim_name in name_map.keys():
            self._static_name_maps[pcls_dim_name].append(name_map[pcls_dim_name])

    def _get_available_dims_indices_by_request(self):
        """Returns a dict mapping 'parcels_dimname' -> [None, int32_index_data_array].

        This dictionary is based on the information provided by the requested dimensions.
        Example: {'time': 0, 'depth': None, 'lat': 1, 'lon': 2}
        """
        result = {}
        neg_offset = 0
        tpl_offset = 0
        for name in ["time", "depth", "lat", "lon"]:
            i = list(self._static_name_maps.keys()).index(name)
            if name not in self.dimensions:
                result[name] = None
                tpl_offset += 1
                neg_offset += 1
            elif (
                (type(self.chunksize) is dict)
                and (
                    name not in self.chunksize
                    or (
                        type(self.chunksize[name]) is tuple
                        and len(self.chunksize[name]) == 2
                        and self.chunksize[name][1] <= 1
                    )
                )
            ) or (
                (type(self.chunksize) is tuple) and name in self.dimensions and (self.chunksize[i - tpl_offset] <= 1)
            ):
                result[name] = None
                neg_offset += 1
            else:
                result[name] = i - neg_offset
        return result

    def _get_available_dims_indices_by_namemap(self):
        """
        Returns a dict mapping 'parcels_dimname' -> [None, int32_index_data_array].
        This dictionary is based on the information provided by the requested dimensions.
        Example: {'time': 0, 'depth': 1, 'lat': 2, 'lon': 3}
        """
        result = {}
        for name in ["time", "depth", "lat", "lon"]:
            result[name] = list(self._static_name_maps.keys()).index(name)
        return result

    def _get_available_dims_indices_by_netcdf_file(self):
        """
        [File needs to be open (i.e. self.dataset is not None) for this to work - otherwise generating an error]
        Returns a dict mapping 'parcels_dimname' -> [None, int32_index_data_array].
        This dictionary is based on the information provided by the requested dimensions.
        Example: {'time': 0, 'depth': 5, 'lat': 3, 'lon': 1}
                 for NetCDF with dimensions:
                     timer: 1
                     x: [0 4000]
                     xr: [0 3999]
                     y: [0 2140]
                     yr: [0 2139]
                     z: [0 75]
        """
        if self.dataset is None:
            raise OSError("Trying to parse NetCDF header information before opening the file.")
        result = {}
        for pcls_dimname in ["time", "depth", "lat", "lon"]:
            for nc_dimname in self._static_name_maps[pcls_dimname]:
                if nc_dimname not in self.dataset.sizes.keys():
                    continue
                result[pcls_dimname] = list(self.dataset.sizes.keys()).index(nc_dimname)
        return result

    def _is_dimension_available(self, dimension_name):
        """
        This function returns a boolean value indicating if a certain variable (name) is available in the
        requested dimensions as well as in the actual dataset of the file. If any of the two conditions is not met,
        if returns 'False'.
        """
        if self.dimensions is None or self.dataset is None:
            return False
        return dimension_name in self.dimensions

    def _is_dimension_chunked(self, dimension_name):
        """
        This functions returns a boolean value indicating if a certain variable is available in the requested
        dimensions, the NetCDF file dataset, and is also required to be chunked according to the requested
        chunksize dictionary. If any of the two conditions is not met, if returns 'False'.
        """
        if self.dimensions is None or self.dataset is None or self.chunksize in [None, False, "auto"]:
            return False
        dim_chunked = False
        dim_chunked = (
            True
            if (not dim_chunked and type(self.chunksize) is dict and dimension_name in self.chunksize.keys())
            else False
        )
        dim_chunked = True if (not dim_chunked and type(self.chunksize) in [None, False]) else False
        return (dimension_name in self.dimensions) and dim_chunked

    def _is_dimension_in_dataset(self, parcels_dimension_name, netcdf_dimension_name=None):
        """
        [File needs to be open (i.e. self.dataset is not None) for this to work - otherwise generating an error]
        This function returns the index, the name and the size of a NetCDF dimension in the file (in order: index, name, size).
        It requires as input the name of the related parcels dimension (i.e. one of ['time', 'depth', 'lat', 'lon']. If
        no hint on its mapping to a NetCDF dimension is provided, a heuristic based on the pre-defined name dictionary
        is used. If a hint is provided, a connections is made between the designated parcels-dimension and NetCDF dimension.
        """
        if self.dataset is None:
            raise OSError("Trying to parse NetCDF header information before opening the file.")
        k, dname, dvalue = (-1, "", 0)
        dimension_name = parcels_dimension_name.lower()
        dim_indices = self._get_available_dims_indices_by_request()
        i = dim_indices[dimension_name]
        if netcdf_dimension_name is not None and netcdf_dimension_name in self.dataset.sizes.keys():
            value = self.dataset.sizes[netcdf_dimension_name]
            k, dname, dvalue = i, netcdf_dimension_name, value
        elif self.dimensions is None or self.dataset is None:
            return k, dname, dvalue
        else:
            for name in self._static_name_maps[dimension_name]:
                if name in self.dataset.sizes:
                    value = self.dataset.sizes[name]
                    k, dname, dvalue = i, name, value
                    break
        return k, dname, dvalue

    def _is_dimension_in_chunksize_request(self, parcels_dimension_name):
        """
        This function returns the dense-array index, the NetCDF dimension name and the requested chunsize of a requested
        parcels dimension(in order: index, name, size). This only works if the chunksize is provided as a dictionary
        of tuples of parcels dimensions and their chunk mapping (i.e. dict(parcels_dim_name => (netcdf_dim_name, chunksize)).
        It requires as input the name of the related parcels dimension (i.e. one of ['time', 'depth', 'lat', 'lon'].
        """
        k, dname, dvalue = (-1, "", 0)
        if self.dimensions is None or self.dataset is None:
            return k, dname, dvalue
        parcels_dimension_name = parcels_dimension_name.lower()
        dim_indices = self._get_available_dims_indices_by_request()
        i = dim_indices[parcels_dimension_name]
        name = self.chunksize[parcels_dimension_name][0]
        value = self.chunksize[parcels_dimension_name][1]
        k, dname, dvalue = i, name, value
        return k, dname, dvalue

    def _netcdf_DimNotFound_warning_message(self, dimension_name):
        """Helper function that issues a warning message if a certain requested NetCDF dimension is not found in the file."""
        display_name = dimension_name if (dimension_name not in self.dimensions) else self.dimensions[dimension_name]
        return f"Did not find {display_name} in NetCDF dims. Please specifiy chunksize as dictionary for NetCDF dimension names, e.g.\n chunksize={{ '{display_name}': <number>, ... }}."

    def _chunkmap_to_chunksize(self):
        """
        [File needs to be open via the '__enter__'-method for this to work - otherwise generating an error]
        This functions translates the array-index-to-chunksize chunk map into a proper fieldsize dictionary that
        can later be used for re-chunking, if a previously-opened file is re-opened again.
        """
        if self.chunksize in [False, None]:
            return
        self.chunksize = {}
        chunk_map = self.chunk_mapping
        timei, timename, timevalue = self._is_dimension_in_dataset("time")
        depthi, depthname, depthvalue = self._is_dimension_in_dataset("depth")
        lati, latname, latvalue = self._is_dimension_in_dataset("lat")
        loni, lonname, lonvalue = self._is_dimension_in_dataset("lon")
        if len(chunk_map) == 2:
            self.chunksize["lon"] = (latname, chunk_map[0])
            self.chunksize["lat"] = (lonname, chunk_map[1])
        elif len(chunk_map) == 3:
            chunk_dim_index = 0
            if depthi is not None and depthi >= 0 and depthvalue > 1 and self._is_dimension_available("depth"):
                self.chunksize["depth"] = (depthname, chunk_map[chunk_dim_index])
                chunk_dim_index += 1
            elif timei is not None and timei >= 0 and timevalue > 1 and self._is_dimension_available("time"):
                self.chunksize["time"] = (timename, chunk_map[chunk_dim_index])
                chunk_dim_index += 1
            self.chunksize["lat"] = (latname, chunk_map[chunk_dim_index])
            chunk_dim_index += 1
            self.chunksize["lon"] = (lonname, chunk_map[chunk_dim_index])
        elif len(chunk_map) >= 4:
            self.chunksize["time"] = (timename, chunk_map[0])
            self.chunksize["depth"] = (depthname, chunk_map[1])
            self.chunksize["lat"] = (latname, chunk_map[2])
            self.chunksize["lon"] = (lonname, chunk_map[3])
            dim_index = 4
            for dim_name in self.dimensions:
                if dim_name not in ["time", "depth", "lat", "lon"]:
                    self.chunksize[dim_name] = (self.dimensions[dim_name], chunk_map[dim_index])
                    dim_index += 1

    def _get_initial_chunk_dictionary_by_dict_(self):
        """
        [File needs to be open (i.e. self.dataset is not None) for this to work - otherwise generating an error]
        Maps and correlates the requested dictionary-style chunksize with the requested parcels dimensions, variables
        and the NetCDF-available dimensions. Thus, it takes care to remove chunksize arguments that are not in the
        Parcels- or NetCDF dimensions, or whose chunking would be omitted due to an empty chunk dimension.
        The function returns a pair of two things: corrected_chunk_dict, chunk_map
        The corrected chunk_dict is the corrected version of the requested chunksize. The chunk map maps the array index
        dimension to the requested chunksize.
        """
        chunk_dict = {}
        chunk_index_map = {}
        neg_offset = 0
        if "time" in self.chunksize.keys():
            timei, timename, timesize = self._is_dimension_in_dataset(
                parcels_dimension_name="time", netcdf_dimension_name=self.chunksize["time"][0]
            )
            timevalue = self.chunksize["time"][1]
            if timei is not None and timei >= 0 and timevalue > 1:
                timevalue = min(timesize, timevalue)
                chunk_dict[timename] = timevalue
                chunk_index_map[timei - neg_offset] = timevalue
            else:
                self.chunksize.pop("time")
        if "depth" in self.chunksize.keys():
            depthi, depthname, depthsize = self._is_dimension_in_dataset(
                parcels_dimension_name="depth", netcdf_dimension_name=self.chunksize["depth"][0]
            )
            depthvalue = self.chunksize["depth"][1]
            if depthi is not None and depthi >= 0 and depthvalue > 1:
                depthvalue = min(depthsize, depthvalue)
                chunk_dict[depthname] = depthvalue
                chunk_index_map[depthi - neg_offset] = depthvalue
            else:
                self.chunksize.pop("depth")
        if "lat" in self.chunksize.keys():
            lati, latname, latsize = self._is_dimension_in_dataset(
                parcels_dimension_name="lat", netcdf_dimension_name=self.chunksize["lat"][0]
            )
            latvalue = self.chunksize["lat"][1]
            if lati is not None and lati >= 0 and latvalue > 1:
                latvalue = min(latsize, latvalue)
                chunk_dict[latname] = latvalue
                chunk_index_map[lati - neg_offset] = latvalue
            else:
                self.chunksize.pop("lat")
        if "lon" in self.chunksize.keys():
            loni, lonname, lonsize = self._is_dimension_in_dataset(
                parcels_dimension_name="lon", netcdf_dimension_name=self.chunksize["lon"][0]
            )
            lonvalue = self.chunksize["lon"][1]
            if loni is not None and loni >= 0 and lonvalue > 1:
                lonvalue = min(lonsize, lonvalue)
                chunk_dict[lonname] = lonvalue
                chunk_index_map[loni - neg_offset] = lonvalue
            else:
                self.chunksize.pop("lon")
        return chunk_dict, chunk_index_map

    def _failsafe_parse_(self):
        """['name' need to be initialised]"""
        # ==== fail - open it as a normal array and deduce the dimensions from the variable-function names ==== #
        # ==== done by parsing ALL variables in the NetCDF, and comparing their call-parameters with the   ==== #
        # ==== name map available here.                                                                    ==== #
        init_chunk_dict = {}
        self.dataset = ncDataset(str(self.filename))
        refdims = self.dataset.dimensions.keys()
        max_field = ""
        max_dim_names = ()
        max_coincide_dims = 0
        for vname in self.dataset.variables:
            var = self.dataset.variables[vname]
            coincide_dims = []
            for vdname in var.dimensions:
                if vdname in refdims:
                    coincide_dims.append(vdname)
            n_coincide_dims = len(coincide_dims)
            if n_coincide_dims > max_coincide_dims:
                max_field = vname
                max_dim_names = tuple(coincide_dims)
                max_coincide_dims = n_coincide_dims
        self.name = max_field
        for nc_dname in max_dim_names:
            pcls_dname = None
            for dname in self._static_name_maps.keys():
                if nc_dname in self._static_name_maps[dname]:
                    pcls_dname = dname
                    break
            nc_dimsize = None
            pcls_dim_chunksize = None
            if pcls_dname is not None and pcls_dname in self.dimensions:
                pcls_dim_chunksize = self._min_dim_chunksize
            if isinstance(self.chunksize, dict) and pcls_dname is not None:
                nc_dimsize = self.dataset.dimensions[nc_dname].size
                if pcls_dname in self.chunksize.keys():
                    pcls_dim_chunksize = self.chunksize[pcls_dname][1]
            if (
                pcls_dname is not None
                and nc_dname is not None
                and nc_dimsize is not None
                and pcls_dim_chunksize is not None
            ):
                init_chunk_dict[nc_dname] = pcls_dim_chunksize

        # ==== because in this case it has shown that the requested chunksize setup cannot be used, ==== #
        # ==== replace the requested chunksize with this auto-derived version.                      ==== #
        return init_chunk_dict

    def _get_initial_chunk_dictionary(self):
        """
        Super-function that maps and correlates the requested chunksize with the requested parcels dimensions, variables
        and the NetCDF-available dimensions. Thus, it takes care to remove chunksize arguments that are not in the
        Parcels- or NetCDF dimensions, or whose chunking would be omitted due to an empty chunk dimension.
        The function returns the corrected chunksize dictionary. The function also initializes the chunk_map.
        The chunk map maps the array index dimension to the requested chunksize.
        Apart from resolving the different requested version of the chunksize, the function also test-executes the
        chunk request. If this initial test fails, as a last resort, we execute a heuristic to map the requested
        parcels dimensions to the dimension signature of the most-parameterized NetCDF variable, and heuristically
        try to map its parameters to the parcels dimensions with the class-wide name-map.
        """
        # ==== check-opening requested dataset to access metadata                   ==== #
        # ==== file-opening and dimension-reading does not require a decode or lock ==== #
        self.dataset = xr.open_dataset(
            str(self.filename), decode_cf=False, engine=self.netcdf_engine, chunks={}, lock=False
        )
        self.dataset["decoded"] = False
        # ==== self.dataset temporarily available ==== #
        init_chunk_dict = {}
        init_chunk_map = {}
        if isinstance(self.chunksize, dict):
            init_chunk_dict, init_chunk_map = self._get_initial_chunk_dictionary_by_dict_()
        elif self.chunksize == "auto":
            av_mem = psutil.virtual_memory().available
            chunk_cap = av_mem * (1 / 8) * (1 / 3)
            if "array.chunk-size" in da_conf.config.keys():
                chunk_cap = da_utils.parse_bytes(da_conf.config.get("array.chunk-size"))
            else:
                predefined_cap = da_conf.get("array.chunk-size")
                if predefined_cap is not None:
                    chunk_cap = da_utils.parse_bytes(predefined_cap)
                else:
                    warnings.warn(
                        "Unable to locate chunking hints from dask, thus estimating the max. chunk size heuristically. "
                        "Please consider defining the 'chunk-size' for 'array' in your local dask configuration file (see https://docs.oceanparcels.org/en/latest/examples/documentation_MPI.html#Chunking-the-FieldSet-with-dask and https://docs.dask.org).",
                        FileWarning,
                        stacklevel=2,
                    )
            loni, lonname, lonvalue = self._is_dimension_in_dataset("lon")
            lati, latname, latvalue = self._is_dimension_in_dataset("lat")
            if lati is not None and loni is not None and lati >= 0 and loni >= 0:
                pDim = int(math.floor(math.sqrt(chunk_cap / np.dtype(np.float64).itemsize)))
                init_chunk_dict[latname] = min(latvalue, pDim)
                init_chunk_map[lati] = min(latvalue, pDim)
                init_chunk_dict[lonname] = min(lonvalue, pDim)
                init_chunk_map[loni] = min(lonvalue, pDim)
            timei, timename, timevalue = self._is_dimension_in_dataset("time")
            if timei is not None and timei >= 0:
                init_chunk_dict[timename] = min(1, timevalue)
                init_chunk_map[timei] = min(1, timevalue)
            depthi, depthname, depthvalue = self._is_dimension_in_dataset("depth")
            if depthi is not None and depthi >= 0:
                init_chunk_dict[depthname] = max(1, depthvalue)
                init_chunk_map[depthi] = max(1, depthvalue)
        # ==== closing check-opened requested dataset ==== #
        self.dataset.close()
        # ==== check if the chunksize reading is successful. if not, load the file ONCE really into memory and ==== #
        # ==== deduce the chunking from the array dims.                                                         ==== #
        if len(init_chunk_dict) == 0 and self.chunksize not in [False, None, "auto"]:
            self.autochunkingfailed = True
            raise DaskChunkingError(
                self.__class__.__name__,
                "No correct mapping found between Parcels- and NetCDF dimensions! Please correct the 'FieldSet(..., chunksize={...})' parameter and try again.",
            )
        else:
            self.autochunkingfailed = False
        try:
            self.dataset = xr.open_dataset(
                str(self.filename), decode_cf=True, engine=self.netcdf_engine, chunks=init_chunk_dict, lock=False
            )
            if isinstance(self.chunksize, dict):
                self.chunksize = init_chunk_dict
        except:
            warnings.warn(
                f"Chunking with init_chunk_dict = {init_chunk_dict} failed - Executing Dask chunking 'failsafe'...",
                FileWarning,
                stacklevel=2,
            )
            self.autochunkingfailed = True
            if not self.autochunkingfailed:
                init_chunk_dict = self._failsafe_parse_()
            if isinstance(self.chunksize, dict):
                self.chunksize = init_chunk_dict
        finally:
            self.dataset.close()
            self.chunk_mapping = init_chunk_map
        self.dataset = None
        # ==== self.dataset not available ==== #
        return init_chunk_dict

    @property
    def data(self):
        return self.data_access()

    def data_access(self):
        data = self.dataset[self.name]

        ti = range(data.shape[0]) if self.ti is None else self.ti
        data = self._apply_indices(data, ti)
        if isinstance(data, xr.DataArray):
            data = data.data

        if isinstance(data, da.core.Array):
            if not self.chunking_finalized:
                if self.chunksize == "auto":
                    # ==== as the chunksize is not initiated, the data is chunked automatically by Dask.  ==== #
                    # ==== the resulting chunk dictionary is stored, to be re-used later. This prevents   ==== #
                    # ==== the expensive re-calculation and PHYSICAL FILE RECHUNKING on each data access. ==== #
                    if data.shape[-2:] != data.chunksize[-2:]:
                        data = data.rechunk(self.chunksize)
                    self.chunk_mapping = {}
                    chunkIndex = 0
                    startblock = 0
                    for chunkDim in data.chunksize[startblock:]:
                        self.chunk_mapping[chunkIndex] = chunkDim
                        chunkIndex += 1
                    self._chunkmap_to_chunksize()
                    if self.rechunk_callback_fields is not None:
                        self.rechunk_callback_fields()
                        self.chunking_finalized = True
                else:
                    self.chunking_finalized = True
        else:
            da_data = da.from_array(data, chunks=self.chunksize)
            if self.chunksize == "auto" and da_data.shape[-2:] == da_data.chunksize[-2:]:
                data = np.array(data)
            else:
                data = da_data
            if not self.chunking_finalized and self.rechunk_callback_fields is not None:
                self.rechunk_callback_fields()
                self.chunking_finalized = True

        return data.astype(self.cast_data_dtype)


class DeferredDaskFileBuffer(DaskFileBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
