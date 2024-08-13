import importlib.util
import os
import sys
from copy import deepcopy
from glob import glob

import dask.array as da
import numpy as np

from parcels.field import DeferredArray, Field, NestedField, VectorField
from parcels.grid import Grid
from parcels.gridset import GridSet
from parcels.tools.converters import TimeConverter, convert_xarray_time_units
from parcels.tools.loggers import logger
from parcels.tools.statuscodes import TimeExtrapolationError

try:
    from mpi4py import MPI
except ModuleNotFoundError:
    MPI = None


__all__ = ['FieldSet']


class FieldSet:
    """FieldSet class that holds hydrodynamic data needed to execute particles.

    Parameters
    ----------
    U : parcels.field.Field
        Field object for zonal velocity component
    V : parcels.field.Field
        Field object for meridional velocity component
    fields : dict mapping str to Field
        Additional fields to include in the FieldSet. These fields can be used
        in custom kernels.
    """

    def __init__(self, U, V, fields=None):
        self.gridset = GridSet()
        self.completed = False
        self.particlefile = None
        if U:
            self.add_field(U, 'U')
            self.time_origin = self.U.grid.time_origin if isinstance(self.U, Field) else self.U[0].grid.time_origin
        if V:
            self.add_field(V, 'V')

        # Add additional fields as attributes
        if fields:
            for name, field in fields.items():
                self.add_field(field, name)

        self.compute_on_defer = None
        self.add_UVfield()

    @staticmethod
    def checkvaliddimensionsdict(dims):
        for d in dims:
            if d not in ['lon', 'lat', 'depth', 'time']:
                raise NameError(f'{d} is not a valid key in the dimensions dictionary')

    @classmethod
    def from_data(cls, data, dimensions, transpose=False, mesh='spherical',
                  allow_time_extrapolation=None, time_periodic=False, **kwargs):
        """Initialise FieldSet object from raw data.

        Parameters
        ----------
        data :
            Dictionary mapping field names to numpy arrays.
            Note that at least a 'U' and 'V' numpy array need to be given, and that
            the built-in Advection kernels assume that U and V are in m/s

            1. If data shape is [xdim, ydim], [xdim, ydim, zdim], [xdim, ydim, tdim] or [xdim, ydim, zdim, tdim],
               whichever is relevant for the dataset, use the flag transpose=True
            2. If data shape is [ydim, xdim], [zdim, ydim, xdim], [tdim, ydim, xdim] or [tdim, zdim, ydim, xdim],
               use the flag transpose=False (default value)
            3. If data has any other shape, you first need to reorder it
        dimensions : dict
            Dictionary mapping field dimensions (lon,
            lat, depth, time) to numpy arrays.
            Note that dimensions can also be a dictionary of dictionaries if
            dimension names are different for each variable
            (e.g. dimensions['U'], dimensions['V'], etc).
        transpose : bool
            Whether to transpose data on read-in (Default value = False)
        mesh : str
            String indicating the type of mesh coordinates and
            units used during velocity interpolation, see also `this tutorial <../examples/tutorial_unitconverters.ipynb>`__:

            1. spherical (default): Lat and lon in degree, with a
               correction for zonal velocity U near the poles.
            2. flat: No conversion, lat/lon are assumed to be in m.
        allow_time_extrapolation : bool
            boolean whether to allow for extrapolation
            (i.e. beyond the last available time snapshot)
            Default is False if dimensions includes time, else True
        time_periodic : bool, float or datetime.timedelta
            To loop periodically over the time component of the Field. It is set to either False or the length of the period (either float in seconds or datetime.timedelta object). (Default: False)
            This flag overrides the allow_time_extrapolation and sets it to False
        **kwargs :
            Keyword arguments passed to the :class:`Field` constructor.

        Examples
        --------
        For usage examples see the following tutorials:

        * `Analytical advection <../examples/tutorial_analyticaladvection.ipynb>`__

        * `Diffusion <../examples/tutorial_diffusion.ipynb>`__

        * `Interpolation <../examples/tutorial_interpolation.ipynb>`__

        * `Unit converters <../examples/tutorial_unitconverters.ipynb>`__
        """
        fields = {}
        for name, datafld in data.items():
            # Use dimensions[name] if dimensions is a dict of dicts
            dims = dimensions[name] if name in dimensions else dimensions
            cls.checkvaliddimensionsdict(dims)

            if allow_time_extrapolation is None:
                allow_time_extrapolation = False if 'time' in dims else True

            lon = dims['lon']
            lat = dims['lat']
            depth = np.zeros(1, dtype=np.float32) if 'depth' not in dims else dims['depth']
            time = np.zeros(1, dtype=np.float64) if 'time' not in dims else dims['time']
            time = np.array(time) if not isinstance(time, np.ndarray) else time
            if isinstance(time[0], np.datetime64):
                time_origin = TimeConverter(time[0])
                time = np.array([time_origin.reltime(t) for t in time])
            else:
                time_origin = kwargs.pop('time_origin', TimeConverter(0))
            grid = Grid.create_grid(lon, lat, depth, time, time_origin=time_origin, mesh=mesh)
            if 'creation_log' not in kwargs.keys():
                kwargs['creation_log'] = 'from_data'

            fields[name] = Field(name, datafld, grid=grid, transpose=transpose,
                                 allow_time_extrapolation=allow_time_extrapolation, time_periodic=time_periodic, **kwargs)
        u = fields.pop('U', None)
        v = fields.pop('V', None)
        return cls(u, v, fields=fields)

    def add_field(self, field, name=None):
        """Add a :class:`parcels.field.Field` object to the FieldSet.

        Parameters
        ----------
        field : parcels.field.Field
            Field object to be added
        name : str
            Name of the :class:`parcels.field.Field` object to be added


        Examples
        --------
        For usage examples see the following tutorials:

        * `Nested Fields <../examples/tutorial_NestedFields.ipynb>`__

        * `Unit converters <../examples/tutorial_unitconverters.ipynb>`__ (Default value = None)

        """
        if self.completed:
            raise RuntimeError("FieldSet has already been completed. Are you trying to add a Field after you've created the ParticleSet?")
        name = field.name if name is None else name
        if hasattr(self, name):  # check if Field with same name already exists when adding new Field
            raise RuntimeError(f"FieldSet already has a Field with name '{name}'")
        if isinstance(field, NestedField):
            setattr(self, name, field)
            for fld in field:
                self.gridset.add_grid(fld)
                fld.fieldset = self
        else:
            setattr(self, name, field)
            self.gridset.add_grid(field)
            field.fieldset = self

    def add_constant_field(self, name, value, mesh='flat'):
        """Wrapper function to add a Field that is constant in space,
           useful e.g. when using constant horizontal diffusivity

        Parameters
        ----------
        name : str
            Name of the :class:`parcels.field.Field` object to be added
        value : float
            Value of the constant field (stored as 32-bit float)
        mesh : str
            String indicating the type of mesh coordinates and
            units used during velocity interpolation, see also `this tutorial <../examples/tutorial_unitconverters.ipynb>`__:

            1. spherical (default): Lat and lon in degree, with a
               correction for zonal velocity U near the poles.
            2. flat: No conversion, lat/lon are assumed to be in m.
        """
        self.add_field(Field(name, value, lon=0, lat=0, mesh=mesh))

    def add_vector_field(self, vfield):
        """Add a :class:`parcels.field.VectorField` object to the FieldSet.

        Parameters
        ----------
        vfield : parcels.VectorField
            class:`parcels.field.VectorField` object to be added
        """
        setattr(self, vfield.name, vfield)
        for v in vfield.__dict__.values():
            if isinstance(v, Field) and (v not in self.get_fields()):
                self.add_field(v)
        vfield.fieldset = self
        if isinstance(vfield, NestedField):
            for f in vfield:
                f.fieldset = self

    def add_UVfield(self):
        if not hasattr(self, 'UV') and hasattr(self, 'U') and hasattr(self, 'V'):
            if isinstance(self.U, NestedField):
                self.add_vector_field(NestedField('UV', self.U, self.V))
            else:
                self.add_vector_field(VectorField('UV', self.U, self.V))
        if not hasattr(self, 'UVW') and hasattr(self, 'W'):
            if isinstance(self.U, NestedField):
                self.add_vector_field(NestedField('UVW', self.U, self.V, self.W))
            else:
                self.add_vector_field(VectorField('UVW', self.U, self.V, self.W))

    def check_complete(self):
        assert self.U, 'FieldSet does not have a Field named "U"'
        assert self.V, 'FieldSet does not have a Field named "V"'
        for attr, value in vars(self).items():
            if type(value) is Field:
                assert value.name == attr, f'Field {value.name}.name ({attr}) is not consistent'

        def check_velocityfields(U, V, W):
            if (U.interp_method == 'cgrid_velocity' and V.interp_method != 'cgrid_velocity') or \
                    (U.interp_method != 'cgrid_velocity' and V.interp_method == 'cgrid_velocity'):
                raise ValueError("If one of U,V.interp_method='cgrid_velocity', the other should be too")

            if 'linear_invdist_land_tracer' in [U.interp_method, V.interp_method]:
                raise NotImplementedError("interp_method='linear_invdist_land_tracer' is not implemented for U and V Fields")

            if U.interp_method == 'cgrid_velocity':
                if U.grid.xdim == 1 or U.grid.ydim == 1 or V.grid.xdim == 1 or V.grid.ydim == 1:
                    raise NotImplementedError('C-grid velocities require longitude and latitude dimensions at least length 2')

            if U.gridindexingtype not in ['nemo', 'mitgcm', 'mom5', 'pop']:
                raise ValueError("Field.gridindexing has to be one of 'nemo', 'mitgcm', 'mom5' or 'pop'")

            if V.gridindexingtype != U.gridindexingtype or (W and W.gridindexingtype != U.gridindexingtype):
                raise ValueError('Not all velocity Fields have the same gridindexingtype')

            if U.cast_data_dtype != V.cast_data_dtype or (W and W.cast_data_dtype != U.cast_data_dtype):
                raise ValueError('Not all velocity Fields have the same dtype')

        if isinstance(self.U, NestedField):
            w = self.W if hasattr(self, 'W') else [None]*len(self.U)
            for U, V, W in zip(self.U, self.V, w):
                check_velocityfields(U, V, W)
        else:
            W = self.W if hasattr(self, 'W') else None
            check_velocityfields(self.U, self.V, W)

        for g in self.gridset.grids:
            g.check_zonal_periodic()
            if len(g.time) == 1:
                continue
            assert isinstance(g.time_origin.time_origin, type(self.time_origin.time_origin)), 'time origins of different grids must be have the same type'
            g.time = g.time + self.time_origin.reltime(g.time_origin)
            if g.defer_load:
                g.time_full = g.time_full + self.time_origin.reltime(g.time_origin)
            g.time_origin = self.time_origin
        self.add_UVfield()

        ccode_fieldnames = []
        counter = 1
        for fld in self.get_fields():
            if fld.name not in ccode_fieldnames:
                fld.ccode_name = fld.name
            else:
                fld.ccode_name = fld.name + str(counter)
                counter += 1
            ccode_fieldnames.append(fld.ccode_name)

        for f in self.get_fields():
            if isinstance(f, (VectorField, NestedField)) or f.dataFiles is None:
                continue
            if f.grid.depth_field is not None:
                if f.grid.depth_field == 'not_yet_set':
                    raise ValueError("If depth dimension is set at 'not_yet_set', it must be added later using Field.set_depth_from_field(field)")
                if not f.grid.defer_load:
                    depth_data = f.grid.depth_field.data
                    f.grid.depth = depth_data if isinstance(depth_data, np.ndarray) else np.array(depth_data)
        self.completed = True

    @classmethod
    def parse_wildcards(cls, paths, filenames, var):
        if not isinstance(paths, list):
            paths = sorted(glob(str(paths)))
        if len(paths) == 0:
            notfound_paths = filenames[var] if isinstance(filenames, dict) and var in filenames else filenames
            raise OSError(f"FieldSet files not found for variable {var}: {notfound_paths}")
        for fp in paths:
            if not os.path.exists(fp):
                raise OSError(f"FieldSet file not found: {fp}")
        return paths

    @classmethod
    def from_netcdf(cls, filenames, variables, dimensions, indices=None, fieldtype=None,
                    mesh='spherical', timestamps=None, allow_time_extrapolation=None, time_periodic=False,
                    deferred_load=True, chunksize=None, **kwargs):
        """Initialises FieldSet object from NetCDF files.

        Parameters
        ----------
        filenames :
            Dictionary mapping variables to file(s). The
            filepath may contain wildcards to indicate multiple files
            or be a list of file.
            filenames can be a list ``[files]``, a dictionary ``{var:[files]}``,
            a dictionary ``{dim:[files]}`` (if lon, lat, depth and/or data not stored in same files as data),
            or a dictionary of dictionaries ``{var:{dim:[files]}}``.
            time values are in ``filenames[data]``
        variables : dict
            Dictionary mapping variables to variable names in the netCDF file(s).
            Note that the built-in Advection kernels assume that U and V are in m/s
        dimensions : dict
            Dictionary mapping data dimensions (lon,
            lat, depth, time, data) to dimensions in the netCF file(s).
            Note that dimensions can also be a dictionary of dictionaries if
            dimension names are different for each variable
            (e.g. dimensions['U'], dimensions['V'], etc).
        indices :
            Optional dictionary of indices for each dimension
            to read from file(s), to allow for reading of subset of data.
            Default is to read the full extent of each dimension.
            Note that negative indices are not allowed.
        fieldtype :
            Optional dictionary mapping fields to fieldtypes to be used for UnitConverter.
            (either 'U', 'V', 'Kh_zonal', 'Kh_meridional' or None) (Default value = None)
        mesh : str
            String indicating the type of mesh coordinates and
            units used during velocity interpolation, see also `this tutorial <../examples/tutorial_unitconverters.ipynb>`__:

            1. spherical (default): Lat and lon in degree, with a
               correction for zonal velocity U near the poles.
            2. flat: No conversion, lat/lon are assumed to be in m.
        timestamps :
            list of lists or array of arrays containing the timestamps for
            each of the files in filenames. Outer list/array corresponds to files, inner
            array corresponds to indices within files.
            Default is None if dimensions includes time.
        allow_time_extrapolation : bool
            boolean whether to allow for extrapolation
            (i.e. beyond the last available time snapshot)
            Default is False if dimensions includes time, else True
        time_periodic : bool, float or datetime.timedelta
            To loop periodically over the time component of the Field. It is set to either False or the length of the period (either float in seconds or datetime.timedelta object). (Default: False)
            This flag overrides the allow_time_extrapolation and sets it to False
        deferred_load : bool
            boolean whether to only pre-load data (in deferred mode) or
            fully load them (default: True). It is advised to deferred load the data, since in
            that case Parcels deals with a better memory management during particle set execution.
            deferred_load=False is however sometimes necessary for plotting the fields.
        interp_method : str
            Method for interpolation. Options are 'linear' (default), 'nearest',
            'linear_invdist_land_tracer', 'cgrid_velocity', 'cgrid_tracer' and 'bgrid_velocity'
        gridindexingtype : str
            The type of gridindexing. Either 'nemo' (default) or 'mitgcm' are supported.
            See also the Grid indexing documentation on oceanparcels.org
        chunksize :
            size of the chunks in dask loading. Default is None (no chunking). Can be None or False (no chunking),
            'auto' (chunking is done in the background, but results in one grid per field individually), or a dict in the format
            ``{parcels_varname: {netcdf_dimname : (parcels_dimname, chunksize_as_int)}, ...}``, where ``parcels_dimname`` is one of ('time', 'depth', 'lat', 'lon')
        netcdf_engine :
            engine to use for netcdf reading in xarray. Default is 'netcdf',
            but in cases where this doesn't work, setting netcdf_engine='scipy' could help
        **kwargs :
            Keyword arguments passed to the :class:`parcels.Field` constructor.


        Examples
        --------
        For usage examples see the following tutorials:

        * `Basic Parcels setup <../examples/parcels_tutorial.ipynb>`__

        * `Argo floats <../examples/tutorial_Argofloats.ipynb>`__

        * `Timestamps <../examples/tutorial_timestamps.ipynb>`__

        * `Time-evolving depth dimensions <../examples/tutorial_timevaryingdepthdimensions.ipynb>`__

        """
        # Ensure that times are not provided both in netcdf file and in 'timestamps'.
        if timestamps is not None and 'time' in dimensions:
            logger.warning_once("Time already provided, defaulting to dimensions['time'] over timestamps.")
            timestamps = None

        fields = {}
        if 'creation_log' not in kwargs.keys():
            kwargs['creation_log'] = 'from_netcdf'
        for var, name in variables.items():
            # Resolve all matching paths for the current variable
            paths = filenames[var] if type(filenames) is dict and var in filenames else filenames
            if type(paths) is not dict:
                paths = cls.parse_wildcards(paths, filenames, var)
            else:
                for dim, p in paths.items():
                    paths[dim] = cls.parse_wildcards(p, filenames, var)

            # Use dimensions[var] and indices[var] if either of them is a dict of dicts
            dims = dimensions[var] if var in dimensions else dimensions
            cls.checkvaliddimensionsdict(dims)
            inds = indices[var] if (indices and var in indices) else indices
            fieldtype = fieldtype[var] if (fieldtype and var in fieldtype) else fieldtype
            varchunksize = chunksize[var] if (chunksize and var in chunksize) else chunksize  # <varname> -> {<netcdf_dimname>: (<parcels_dimname>, <chunksize_as_int_numeral>) }

            grid = None
            dFiles = None
            # check if grid has already been processed (i.e. if other fields have same filenames, dimensions and indices)
            for procvar, _ in fields.items():
                procdims = dimensions[procvar] if procvar in dimensions else dimensions
                procinds = indices[procvar] if (indices and procvar in indices) else indices
                procpaths = filenames[procvar] if isinstance(filenames, dict) and procvar in filenames else filenames
                procchunk = chunksize[procvar] if (chunksize and procvar in chunksize) else chunksize
                nowpaths = filenames[var] if isinstance(filenames, dict) and var in filenames else filenames
                if procdims == dims and procinds == inds:
                    possibly_samegrid = True
                    if procchunk != varchunksize:
                        for dim in varchunksize:
                            if varchunksize[dim][1] != procchunk[dim][1]:
                                possibly_samegrid &= False
                    if not possibly_samegrid:
                        break
                    if varchunksize == 'auto':
                        break
                    if 'depth' in dims and dims['depth'] == 'not_yet_set':
                        break
                    processedGrid = False
                    if ((not isinstance(filenames, dict)) or filenames[procvar] == filenames[var]):
                        processedGrid = True
                    elif isinstance(filenames[procvar], dict):
                        processedGrid = True
                        for dim in ['lon', 'lat', 'depth']:
                            if dim in dimensions:
                                processedGrid *= filenames[procvar][dim] == filenames[var][dim]
                    if processedGrid:
                        grid = fields[procvar].grid
                        if procpaths == nowpaths:
                            dFiles = fields[procvar].dataFiles
                            break
            fields[var] = Field.from_netcdf(paths, (var, name), dims, inds, grid=grid, mesh=mesh, timestamps=timestamps,
                                            allow_time_extrapolation=allow_time_extrapolation,
                                            time_periodic=time_periodic, deferred_load=deferred_load,
                                            fieldtype=fieldtype, chunksize=varchunksize, dataFiles=dFiles, **kwargs)

        u = fields.pop('U', None)
        v = fields.pop('V', None)
        return cls(u, v, fields=fields)

    @classmethod
    def from_nemo(cls, filenames, variables, dimensions, indices=None, mesh='spherical',
                  allow_time_extrapolation=None, time_periodic=False,
                  tracer_interp_method='cgrid_tracer', chunksize=None, **kwargs):
        """Initialises FieldSet object from NetCDF files of Curvilinear NEMO fields.

        See `here <../examples/tutorial_nemo_curvilinear.ipynb>`__
        for a detailed tutorial on the setup for 2D NEMO fields and `here <../examples/tutorial_nemo_3D.ipynb>`__
        for the tutorial on the setup for 3D NEMO fields.

        See `here <../examples/documentation_indexing.ipynb>`__
        for a more detailed explanation of the different methods that can be used for c-grid datasets.

        Parameters
        ----------
        filenames :
            Dictionary mapping variables to file(s). The
            filepath may contain wildcards to indicate multiple files,
            or be a list of file.
            filenames can be a list ``[files]``, a dictionary ``{var:[files]}``,
            a dictionary ``{dim:[files]}`` (if lon, lat, depth and/or data not stored in same files as data),
            or a dictionary of dictionaries ``{var:{dim:[files]}}``
            time values are in ``filenames[data]``
        variables : dict
            Dictionary mapping variables to variable names in the netCDF file(s).
            Note that the built-in Advection kernels assume that U and V are in m/s
        dimensions : dict
            Dictionary mapping data dimensions (lon,
            lat, depth, time, data) to dimensions in the netCF file(s).
            Note that dimensions can also be a dictionary of dictionaries if
            dimension names are different for each variable.
            Watch out: NEMO is discretised on a C-grid:
            U and V velocities are not located on the same nodes (see https://www.nemo-ocean.eu/doc/node19.html). ::

                +-----------------------------+-----------------------------+-----------------------------+
                |                             |         V[k,j+1,i+1]        |                             |
                +-----------------------------+-----------------------------+-----------------------------+
                |U[k,j+1,i]                   |W[k:k+2,j+1,i+1],T[k,j+1,i+1]|U[k,j+1,i+1]                 |
                +-----------------------------+-----------------------------+-----------------------------+
                |                             |         V[k,j,i+1]          |                             |
                +-----------------------------+-----------------------------+-----------------------------+

            To interpolate U, V velocities on the C-grid, Parcels needs to read the f-nodes,
            which are located on the corners of the cells.
            (for indexing details: https://www.nemo-ocean.eu/doc/img360.png )
            In 3D, the depth is the one corresponding to W nodes
            The gridindexingtype is set to 'nemo'. See also the Grid indexing documentation on oceanparcels.org
        indices :
            Optional dictionary of indices for each dimension
            to read from file(s), to allow for reading of subset of data.
            Default is to read the full extent of each dimension.
            Note that negative indices are not allowed.
        fieldtype :
            Optional dictionary mapping fields to fieldtypes to be used for UnitConverter.
            (either 'U', 'V', 'Kh_zonal', 'Kh_meridional' or None)
        mesh : str
            String indicating the type of mesh coordinates and
            units used during velocity interpolation, see also `this tutorial <../examples/tutorial_unitconverters.ipynb>`__:

            1. spherical (default): Lat and lon in degree, with a
               correction for zonal velocity U near the poles.
            2. flat: No conversion, lat/lon are assumed to be in m.
        allow_time_extrapolation : bool
            boolean whether to allow for extrapolation
            (i.e. beyond the last available time snapshot)
            Default is False if dimensions includes time, else True
        time_periodic : bool, float or datetime.timedelta
            To loop periodically over the time component of the Field. It is set to either False or the length of the period (either float in seconds or datetime.timedelta object). (Default: False)
            This flag overrides the allow_time_extrapolation and sets it to False
        tracer_interp_method : str
            Method for interpolation of tracer fields. It is recommended to use 'cgrid_tracer' (default)
            Note that in the case of from_nemo() and from_c_grid_dataset(), the velocity fields are default to 'cgrid_velocity'
        chunksize :
            size of the chunks in dask loading. Default is None (no chunking)
        **kwargs :
            Keyword arguments passed to the :func:`Fieldset.from_c_grid_dataset` constructor.

        """
        if 'creation_log' not in kwargs.keys():
            kwargs['creation_log'] = 'from_nemo'
        if kwargs.pop('gridindexingtype', 'nemo') != 'nemo':
            raise ValueError("gridindexingtype must be 'nemo' in FieldSet.from_nemo(). Use FieldSet.from_c_grid_dataset otherwise")
        fieldset = cls.from_c_grid_dataset(filenames, variables, dimensions, mesh=mesh, indices=indices, time_periodic=time_periodic,
                                           allow_time_extrapolation=allow_time_extrapolation, tracer_interp_method=tracer_interp_method,
                                           chunksize=chunksize, gridindexingtype='nemo', **kwargs)
        if hasattr(fieldset, 'W'):
            fieldset.W.set_scaling_factor(-1.)
        return fieldset

    @classmethod
    def from_mitgcm(cls, filenames, variables, dimensions, indices=None, mesh='spherical',
                    allow_time_extrapolation=None, time_periodic=False,
                    tracer_interp_method='cgrid_tracer', chunksize=None, **kwargs):
        """Initialises FieldSet object from NetCDF files of MITgcm fields.
        All parameters and keywords are exactly the same as for FieldSet.from_nemo(), except that
        gridindexing is set to 'mitgcm' for grids that have the shape::

            +-----------------------------+-----------------------------+-----------------------------+
            |                             |         V[k,j+1,i]          |                             |
            +-----------------------------+-----------------------------+-----------------------------+
            |U[k,j,i]                     |    W[k-1:k,j,i], T[k,j,i]   |U[k,j,i+1]                   |
            +-----------------------------+-----------------------------+-----------------------------+
            |                             |         V[k,j,i]            |                             |
            +-----------------------------+-----------------------------+-----------------------------+

        For indexing details: https://mitgcm.readthedocs.io/en/latest/algorithm/algorithm.html#spatial-discretization-of-the-dynamical-equations
        Note that vertical velocity (W) is assumed positive in the positive z direction (which is upward in MITgcm)
        """
        if 'creation_log' not in kwargs.keys():
            kwargs['creation_log'] = 'from_mitgcm'
        if kwargs.pop('gridindexingtype', 'mitgcm') != 'mitgcm':
            raise ValueError("gridindexingtype must be 'mitgcm' in FieldSet.from_mitgcm(). Use FieldSet.from_c_grid_dataset otherwise")
        fieldset = cls.from_c_grid_dataset(filenames, variables, dimensions, mesh=mesh, indices=indices, time_periodic=time_periodic,
                                           allow_time_extrapolation=allow_time_extrapolation, tracer_interp_method=tracer_interp_method,
                                           chunksize=chunksize, gridindexingtype='mitgcm', **kwargs)
        return fieldset

    @classmethod
    def from_c_grid_dataset(cls, filenames, variables, dimensions, indices=None, mesh='spherical',
                            allow_time_extrapolation=None, time_periodic=False,
                            tracer_interp_method='cgrid_tracer', gridindexingtype='nemo', chunksize=None, **kwargs):
        """Initialises FieldSet object from NetCDF files of Curvilinear NEMO fields.

        See `here <../examples/documentation_indexing.ipynb>`__
        for a more detailed explanation of the different methods that can be used for c-grid datasets.

        Parameters
        ----------
        filenames :
            Dictionary mapping variables to file(s). The
            filepath may contain wildcards to indicate multiple files,
            or be a list of file.
            filenames can be a list ``[files]``, a dictionary ``{var:[files]}``,
            a dictionary ``{dim:[files]}`` (if lon, lat, depth and/or data not stored in same files as data),
            or a dictionary of dictionaries ``{var:{dim:[files]}}``
            time values are in ``filenames[data]``
        variables : dict
            Dictionary mapping variables to variable
            names in the netCDF file(s).
        dimensions : dict
            Dictionary mapping data dimensions (lon,
            lat, depth, time, data) to dimensions in the netCF file(s).
            Note that dimensions can also be a dictionary of dictionaries if
            dimension names are different for each variable.
            Watch out: NEMO is discretised on a C-grid:
            U and V velocities are not located on the same nodes (see https://www.nemo-ocean.eu/doc/node19.html ). ::

                +-----------------------------+-----------------------------+-----------------------------+
                |                             |         V[k,j+1,i+1]        |                             |
                +-----------------------------+-----------------------------+-----------------------------+
                |U[k,j+1,i]                   |W[k:k+2,j+1,i+1],T[k,j+1,i+1]|U[k,j+1,i+1]                 |
                +-----------------------------+-----------------------------+-----------------------------+
                |                             |         V[k,j,i+1]          |                             |
                +-----------------------------+-----------------------------+-----------------------------+

            To interpolate U, V velocities on the C-grid, Parcels needs to read the f-nodes,
            which are located on the corners of the cells.
            (for indexing details: https://www.nemo-ocean.eu/doc/img360.png )
            In 3D, the depth is the one corresponding to W nodes.
        indices :
            Optional dictionary of indices for each dimension
            to read from file(s), to allow for reading of subset of data.
            Default is to read the full extent of each dimension.
            Note that negative indices are not allowed.
        fieldtype :
            Optional dictionary mapping fields to fieldtypes to be used for UnitConverter.
            (either 'U', 'V', 'Kh_zonal', 'Kh_meridional' or None)
        mesh : str
            String indicating the type of mesh coordinates and
            units used during velocity interpolation, see also `this tutorial <../examples/tutorial_unitconverters.ipynb>`__:

            1. spherical (default): Lat and lon in degree, with a
               correction for zonal velocity U near the poles.
            2. flat: No conversion, lat/lon are assumed to be in m.
        allow_time_extrapolation : bool
            boolean whether to allow for extrapolation
            (i.e. beyond the last available time snapshot)
            Default is False if dimensions includes time, else True
        time_periodic : bool, float or datetime.timedelta
            To loop periodically over the time component of the Field. It is set to either False or the length of the period (either float in seconds or datetime.timedelta object). (Default: False)
            This flag overrides the allow_time_extrapolation and sets it to False
        tracer_interp_method : str
            Method for interpolation of tracer fields. It is recommended to use 'cgrid_tracer' (default)
            Note that in the case of from_nemo() and from_c_grid_dataset(), the velocity fields are default to 'cgrid_velocity'
        gridindexingtype : str
            The type of gridindexing. Set to 'nemo' in FieldSet.from_nemo()
            See also the Grid indexing documentation on oceanparcels.org (Default value = 'nemo')
        chunksize :
            size of the chunks in dask loading. (Default value = None)
        **kwargs :
            Keyword arguments passed to the :func:`Fieldset.from_netcdf` constructor.
        """
        if 'U' in dimensions and 'V' in dimensions and dimensions['U'] != dimensions['V']:
            raise ValueError("On a C-grid, the dimensions of velocities should be the corners (f-points) of the cells, so the same for U and V. "
                             "See also ../examples/documentation_indexing.ipynb")
        if 'U' in dimensions and 'W' in dimensions and dimensions['U'] != dimensions['W']:
            raise ValueError("On a C-grid, the dimensions of velocities should be the corners (f-points) of the cells, so the same for U, V and W. "
                             "See also ../examples/documentation_indexing.ipynb")
        if 'interp_method' in kwargs.keys():
            raise TypeError("On a C-grid, the interpolation method for velocities should not be overridden")

        interp_method = {}
        for v in variables:
            if v in ['U', 'V', 'W']:
                interp_method[v] = 'cgrid_velocity'
            else:
                interp_method[v] = tracer_interp_method
        if 'creation_log' not in kwargs.keys():
            kwargs['creation_log'] = 'from_c_grid_dataset'

        return cls.from_netcdf(filenames, variables, dimensions, mesh=mesh, indices=indices, time_periodic=time_periodic,
                               allow_time_extrapolation=allow_time_extrapolation, interp_method=interp_method,
                               chunksize=chunksize, gridindexingtype=gridindexingtype, **kwargs)

    @classmethod
    def from_pop(cls, filenames, variables, dimensions, indices=None, mesh='spherical',
                 allow_time_extrapolation=None, time_periodic=False,
                 tracer_interp_method='bgrid_tracer', chunksize=None, depth_units='m', **kwargs):
        """Initialises FieldSet object from NetCDF files of POP fields.
            It is assumed that the velocities in the POP fields is in cm/s.

        Parameters
        ----------
        filenames :
            Dictionary mapping variables to file(s). The
            filepath may contain wildcards to indicate multiple files,
            or be a list of file.
            filenames can be a list ``[files]``, a dictionary ``{var:[files]}``,
            a dictionary ``{dim:[files]}`` (if lon, lat, depth and/or data not stored in same files as data),
            or a dictionary of dictionaries ``{var:{dim:[files]}}``
            time values are in ``filenames[data]``
        variables : dict
            Dictionary mapping variables to variable names in the netCDF file(s).
            Note that the built-in Advection kernels assume that U and V are in m/s
        dimensions : dict
            Dictionary mapping data dimensions (lon,
            lat, depth, time, data) to dimensions in the netCF file(s).
            Note that dimensions can also be a dictionary of dictionaries if
            dimension names are different for each variable.
            Watch out: POP is discretised on a B-grid:
            U and V velocity nodes are not located as W velocity and T tracer nodes (see http://www2.cesm.ucar.edu/models/cesm1.0/pop2/doc/sci/POPRefManual.pdf ). ::

                +-----------------------------+-----------------------------+-----------------------------+
                |U[k,j+1,i],V[k,j+1,i]        |                             |U[k,j+1,i+1],V[k,j+1,i+1]    |
                +-----------------------------+-----------------------------+-----------------------------+
                |                             |W[k:k+2,j+1,i+1],T[k,j+1,i+1]|                             |
                +-----------------------------+-----------------------------+-----------------------------+
                |U[k,j,i],V[k,j,i]            |                             |U[k,j,i+1],V[k,j,i+1]        |
                +-----------------------------+-----------------------------+-----------------------------+

            In 2D: U and V nodes are on the cell vertices and interpolated bilinearly as a A-grid.
            T node is at the cell centre and interpolated constant per cell as a C-grid.
            In 3D: U and V nodes are at the middle of the cell vertical edges,
            They are interpolated bilinearly (independently of z) in the cell.
            W nodes are at the centre of the horizontal interfaces.
            They are interpolated linearly (as a function of z) in the cell.
            T node is at the cell centre, and constant per cell.
            Note that Parcels assumes that the length of the depth dimension (at the W-points)
            is one larger than the size of the velocity and tracer fields in the depth dimension.
        indices :
            Optional dictionary of indices for each dimension
            to read from file(s), to allow for reading of subset of data.
            Default is to read the full extent of each dimension.
            Note that negative indices are not allowed.
        fieldtype :
            Optional dictionary mapping fields to fieldtypes to be used for UnitConverter.
            (either 'U', 'V', 'Kh_zonal', 'Kh_meridional' or None)
        mesh : str
            String indicating the type of mesh coordinates and
            units used during velocity interpolation, see also `this tutorial <../examples/tutorial_unitconverters.ipynb>`__:

            1. spherical (default): Lat and lon in degree, with a
               correction for zonal velocity U near the poles.
            2. flat: No conversion, lat/lon are assumed to be in m.
        allow_time_extrapolation : bool
            boolean whether to allow for extrapolation
            (i.e. beyond the last available time snapshot)
            Default is False if dimensions includes time, else True
        time_periodic : bool, float or datetime.timedelta
            To loop periodically over the time component of the Field. It is set to either False or the length of the period (either float in seconds or datetime.timedelta object). (Default: False)
            This flag overrides the allow_time_extrapolation and sets it to False
        tracer_interp_method : str
            Method for interpolation of tracer fields. It is recommended to use 'bgrid_tracer' (default)
            Note that in the case of from_pop() and from_b_grid_dataset(), the velocity fields are default to 'bgrid_velocity'
        chunksize :
            size of the chunks in dask loading (Default value = None)
        depth_units :
            The units of the vertical dimension. Default in Parcels is 'm',
            but many POP outputs are in 'cm'
        **kwargs :
            Keyword arguments passed to the :func:`Fieldset.from_b_grid_dataset` constructor.

        """
        if 'creation_log' not in kwargs.keys():
            kwargs['creation_log'] = 'from_pop'
        fieldset = cls.from_b_grid_dataset(filenames, variables, dimensions, mesh=mesh, indices=indices, time_periodic=time_periodic,
                                           allow_time_extrapolation=allow_time_extrapolation, tracer_interp_method=tracer_interp_method,
                                           chunksize=chunksize, gridindexingtype='pop', **kwargs)
        if hasattr(fieldset, 'U'):
            fieldset.U.set_scaling_factor(0.01)  # cm/s to m/s
        if hasattr(fieldset, 'V'):
            fieldset.V.set_scaling_factor(0.01)  # cm/s to m/s
        if hasattr(fieldset, 'W'):
            if depth_units == 'm':
                fieldset.W.set_scaling_factor(-0.01)  # cm/s to m/s and change the W direction
                logger.warning_once("Parcels assumes depth in POP output to be in 'm'. Use depth_units='cm' if the output depth is in 'cm'.")
            elif depth_units == 'cm':
                fieldset.W.set_scaling_factor(-1.)  # change the W direction but keep W in cm/s because depth is in cm
            else:
                raise SyntaxError("'depth_units' has to be 'm' or 'cm'")
        return fieldset

    @classmethod
    def from_mom5(cls, filenames, variables, dimensions, indices=None, mesh='spherical',
                  allow_time_extrapolation=None, time_periodic=False,
                  tracer_interp_method='bgrid_tracer', chunksize=None, **kwargs):
        """Initialises FieldSet object from NetCDF files of MOM5 fields.

        Parameters
        ----------
        filenames :
            Dictionary mapping variables to file(s). The
            filepath may contain wildcards to indicate multiple files,
            or be a list of file.
            filenames can be a list ``[files]``, a dictionary ``{var:[files]}``,
            a dictionary ``{dim:[files]}`` (if lon, lat, depth and/or data not stored in same files as data),
            or a dictionary of dictionaries ``{var:{dim:[files]}}``
            time values are in ``filenames[data]``
        variables : dict
            Dictionary mapping variables to variable names in the netCDF file(s).
            Note that the built-in Advection kernels assume that U and V are in m/s
        dimensions : dict
            Dictionary mapping data dimensions (lon,
            lat, depth, time, data) to dimensions in the netCF file(s).
            Note that dimensions can also be a dictionary of dictionaries if
            dimension names are different for each variable. ::

                +-------------------------------+-------------------------------+-------------------------------+
                |U[k,j+1,i],V[k,j+1,i]          |                               |U[k,j+1,i+1],V[k,j+1,i+1]      |
                +-------------------------------+-------------------------------+-------------------------------+
                |                               |W[k-1:k+1,j+1,i+1],T[k,j+1,i+1]|                               |
                +-------------------------------+-------------------------------+-------------------------------+
                |U[k,j,i],V[k,j,i]              |                               |U[k,j,i+1],V[k,j,i+1]          |
                +-------------------------------+-------------------------------+-------------------------------+

            In 2D: U and V nodes are on the cell vertices and interpolated bilinearly as a A-grid.
            T node is at the cell centre and interpolated constant per cell as a C-grid.
            In 3D: U and V nodes are at the middle of the cell vertical edges,
            They are interpolated bilinearly (independently of z) in the cell.
            W nodes are at the centre of the horizontal interfaces, but below the U and V.
            They are interpolated linearly (as a function of z) in the cell.
            Note that W is normally directed upward in MOM5, but Parcels requires W
            in the positive z-direction (downward) so W is multiplied by -1.
            T node is at the cell centre, and constant per cell.
        indices :
            Optional dictionary of indices for each dimension
            to read from file(s), to allow for reading of subset of data.
            Default is to read the full extent of each dimension.
            Note that negative indices are not allowed.
        fieldtype :
            Optional dictionary mapping fields to fieldtypes to be used for UnitConverter.
            (either 'U', 'V', 'Kh_zonal', 'Kh_meridional' or None)
        mesh : str
            String indicating the type of mesh coordinates and
            units used during velocity interpolation, see also the `Unit converters tutorial <../examples/tutorial_unitconverters.ipynb>`__:

            1. spherical (default): Lat and lon in degree, with a
               correction for zonal velocity U near the poles.
            2. flat: No conversion, lat/lon are assumed to be in m.
        allow_time_extrapolation : bool
            boolean whether to allow for extrapolation
            (i.e. beyond the last available time snapshot)
            Default is False if dimensions includes time, else True
        time_periodic:
            To loop periodically over the time component of the Field. It is set to either False or the length of the period (either float in seconds or datetime.timedelta object). (Default: False)
            This flag overrides the allow_time_extrapolation and sets it to False
        tracer_interp_method : str
            Method for interpolation of tracer fields. It is recommended to use 'bgrid_tracer' (default)
            Note that in the case of from_mom5() and from_b_grid_dataset(), the velocity fields are default to 'bgrid_velocity'
        chunksize :
            size of the chunks in dask loading (Default value = None)
        **kwargs :
            Keyword arguments passed to the :func:`Fieldset.from_b_grid_dataset` constructor.
        """
        if 'creation_log' not in kwargs.keys():
            kwargs['creation_log'] = 'from_mom5'
        fieldset = cls.from_b_grid_dataset(filenames, variables, dimensions, mesh=mesh, indices=indices, time_periodic=time_periodic,
                                           allow_time_extrapolation=allow_time_extrapolation, tracer_interp_method=tracer_interp_method,
                                           chunksize=chunksize, gridindexingtype='mom5', **kwargs)
        if hasattr(fieldset, 'W'):
            fieldset.W.set_scaling_factor(-1)
        return fieldset

    @classmethod
    def from_a_grid_dataset(cls, filenames, variables, dimensions, **kwargs):
        """
        Load a FieldSet from an A-grid dataset, which is the default grid type.

        Parameters
        ----------
        filenames :
            Path(s) to the input files.
        variables :
            Dictionary of the variables in the NetCDF file.
        dimensions :
            Dictionary of the dimensions in the NetCDF file.
        **kwargs :
            Additional keyword arguments for `from_netcdf()`.

        Returns
        -------
        FieldSet
            A FieldSet object.
        """
        return cls.from_netcdf(filenames, variables, dimensions, **kwargs)

    @classmethod
    def from_b_grid_dataset(cls, filenames, variables, dimensions, indices=None, mesh='spherical',
                            allow_time_extrapolation=None, time_periodic=False,
                            tracer_interp_method='bgrid_tracer', chunksize=None, **kwargs):
        """Initialises FieldSet object from NetCDF files of Bgrid fields.

        Parameters
        ----------
        filenames :
            Dictionary mapping variables to file(s). The
            filepath may contain wildcards to indicate multiple files,
            or be a list of file.
            filenames can be a list ``[files]``, a dictionary ``{var:[files]}``,
            a dictionary ``{dim:[files]}`` (if lon, lat, depth and/or data not stored in same files as data),
            or a dictionary of dictionaries ``{var:{dim:[files]}}``
            time values are in ``filenames[data]``
        variables : dict
            Dictionary mapping variables to variable
            names in the netCDF file(s).
        dimensions : dict
            Dictionary mapping data dimensions (lon,
            lat, depth, time, data) to dimensions in the netCF file(s).
            Note that dimensions can also be a dictionary of dictionaries if
            dimension names are different for each variable.
            U and V velocity nodes are not located as W velocity and T tracer nodes (see http://www2.cesm.ucar.edu/models/cesm1.0/pop2/doc/sci/POPRefManual.pdf ). ::

                +-----------------------------+-----------------------------+-----------------------------+
                |U[k,j+1,i],V[k,j+1,i]        |                             |U[k,j+1,i+1],V[k,j+1,i+1]    |
                +-----------------------------+-----------------------------+-----------------------------+
                |                             |W[k:k+2,j+1,i+1],T[k,j+1,i+1]|                             |
                +-----------------------------+-----------------------------+-----------------------------+
                |U[k,j,i],V[k,j,i]            |                             |U[k,j,i+1],V[k,j,i+1]        |
                +-----------------------------+-----------------------------+-----------------------------+

            In 2D: U and V nodes are on the cell vertices and interpolated bilinearly as a A-grid.
            T node is at the cell centre and interpolated constant per cell as a C-grid.
            In 3D: U and V nodes are at the midlle of the cell vertical edges,
            They are interpolated bilinearly (independently of z) in the cell.
            W nodes are at the centre of the horizontal interfaces.
            They are interpolated linearly (as a function of z) in the cell.
            T node is at the cell centre, and constant per cell.
        indices :
            Optional dictionary of indices for each dimension
            to read from file(s), to allow for reading of subset of data.
            Default is to read the full extent of each dimension.
            Note that negative indices are not allowed.
        fieldtype :
            Optional dictionary mapping fields to fieldtypes to be used for UnitConverter.
            (either 'U', 'V', 'Kh_zonal', 'Kh_meridional' or None)
        mesh : str
            String indicating the type of mesh coordinates and
            units used during velocity interpolation, see also `this tutorial <../examples/tutorial_unitconverters.ipynb>`__:

            1. spherical (default): Lat and lon in degree, with a
               correction for zonal velocity U near the poles.
            2. flat: No conversion, lat/lon are assumed to be in m.
        allow_time_extrapolation : bool
            boolean whether to allow for extrapolation
            (i.e. beyond the last available time snapshot)
            Default is False if dimensions includes time, else True
        time_periodic : bool, float or datetime.timedelta
            To loop periodically over the time component of the Field. It is set to either False or the length of the period (either float in seconds or datetime.timedelta object). (Default: False)
            This flag overrides the allow_time_extrapolation and sets it to False
        tracer_interp_method : str
            Method for interpolation of tracer fields. It is recommended to use 'bgrid_tracer' (default)
            Note that in the case of from_pop() and from_b_grid_dataset(), the velocity fields are default to 'bgrid_velocity'
        chunksize :
            size of the chunks in dask loading (Default value = None)
        **kwargs :
            Keyword arguments passed to the :func:`Fieldset.from_netcdf` constructor.
        """
        if 'U' in dimensions and 'V' in dimensions and dimensions['U'] != dimensions['V']:
            raise ValueError("On a B-grid, the dimensions of velocities should be the (top) corners of the grid cells, so the same for U and V. "
                             "See also ../examples/documentation_indexing.ipynb")
        if 'U' in dimensions and 'W' in dimensions and dimensions['U'] != dimensions['W']:
            raise ValueError("On a B-grid, the dimensions of velocities should be the (top) corners of the grid cells, so the same for U, V and W. "
                             "See also ../examples/documentation_indexing.ipynb")

        interp_method = {}
        for v in variables:
            if v in ['U', 'V']:
                interp_method[v] = 'bgrid_velocity'
            elif v in ['W']:
                interp_method[v] = 'bgrid_w_velocity'
            else:
                interp_method[v] = tracer_interp_method
        if 'creation_log' not in kwargs.keys():
            kwargs['creation_log'] = 'from_b_grid_dataset'

        return cls.from_netcdf(filenames, variables, dimensions, mesh=mesh, indices=indices, time_periodic=time_periodic,
                               allow_time_extrapolation=allow_time_extrapolation, interp_method=interp_method,
                               chunksize=chunksize, **kwargs)

    @classmethod
    def from_parcels(cls, basename, uvar='vozocrtx', vvar='vomecrty', indices=None, extra_fields=None,
                     allow_time_extrapolation=None, time_periodic=False, deferred_load=True,
                     chunksize=None, **kwargs):
        """Initialises FieldSet data from NetCDF files using the Parcels FieldSet.write() conventions.

        Parameters
        ----------
        basename : str
            Base name of the file(s); may contain
            wildcards to indicate multiple files.
        indices :
            Optional dictionary of indices for each dimension
            to read from file(s), to allow for reading of subset of data.
            Default is to read the full extent of each dimension.
            Note that negative indices are not allowed.
        fieldtype :
            Optional dictionary mapping fields to fieldtypes to be used for UnitConverter.
            (either 'U', 'V', 'Kh_zonal', 'Kh_meridional' or None)
        extra_fields :
            Extra fields to read beyond U and V (Default value = None)
        allow_time_extrapolation : bool
            boolean whether to allow for extrapolation
            (i.e. beyond the last available time snapshot)
            Default is False if dimensions includes time, else True
        time_periodic : bool, float or datetime.timedelta
            To loop periodically over the time component of the Field. It is set to either False or the length of the period (either float in seconds or datetime.timedelta object). (Default: False)
            This flag overrides the allow_time_extrapolation and sets it to False
        deferred_load : bool
            boolean whether to only pre-load data (in deferred mode) or
            fully load them (default: True). It is advised to deferred load the data, since in
            that case Parcels deals with a better memory management during particle set execution.
            deferred_load=False is however sometimes necessary for plotting the fields.
        chunksize :
            size of the chunks in dask loading (Default value = None)
        uvar :
             (Default value = 'vozocrtx')
        vvar :
             (Default value = 'vomecrty')
        **kwargs :
            Keyword arguments passed to the :func:`Fieldset.from_netcdf` constructor.
        """
        if extra_fields is None:
            extra_fields = {}
        if 'creation_log' not in kwargs.keys():
            kwargs['creation_log'] = 'from_parcels'

        dimensions = {}
        default_dims = {'lon': 'nav_lon', 'lat': 'nav_lat',
                        'depth': 'depth', 'time': 'time_counter'}
        extra_fields.update({'U': uvar, 'V': vvar})
        for vars in extra_fields:
            dimensions[vars] = deepcopy(default_dims)
            dimensions[vars]['depth'] = 'depth%s' % vars.lower()
        filenames = {v: str(f"{basename}{v}.nc") for v in extra_fields.keys()}
        return cls.from_netcdf(filenames, indices=indices, variables=extra_fields,
                               dimensions=dimensions, allow_time_extrapolation=allow_time_extrapolation,
                               time_periodic=time_periodic, deferred_load=deferred_load,
                               chunksize=chunksize, **kwargs)

    @classmethod
    def from_xarray_delft3d(cls, ds, variables, dimensions, mesh='spherical',
                            allow_time_extrapolation=None,
                            time_periodic=False, **kwargs):
        """Initialises FieldSet data from Delft3D-FLOW velocity output.
        Important note: at the moment the code is only applicable to depth-average flowfields from Delft3D-FLOW (2D flow fields).

        :param ds: Delft3D-FLOW output file.
               Note that the built-in Advection kernels assume that U and V are in m/s
        :param variables: Dictionary mapping parcels variable names to data variables in the Delft3D-FLOW output file.
        :param dimensions: Dictionary mapping data dimensions (lon,
               lat, depth, time, data) to dimensions in the xarray Dataset.
               Note that dimensions can also be a dictionary of dictionaries if
               dimension names are different for each variable
               (e.g. dimensions['U'], dimensions['V'], etc).
        :param fieldtype: Optional dictionary mapping fields to fieldtypes to be used for UnitConverter.
               (either 'U', 'V', 'Kh_zonal', 'Kh_meridional' or None)
        :param mesh: String indicating the type of mesh coordinates and
               units used during velocity interpolation, see also `this tutorial <https://nbviewer.jupyter.org/github/OceanParcels/parcels/blob/master/parcels/examples/tutorial_unitconverters.ipynb>`_:

               1. spherical (default): Lat and lon in degree, with a
                  correction for zonal velocity U near the poles. - default mesh in Delft3D-FLOW output file
               2. flat: No conversion, lat/lon are assumed to be in m.
        :param allow_time_extrapolation: boolean whether to allow for extrapolation
               (i.e. beyond the last available time snapshot)
               Default is False if dimensions includes time, else True
        :param time_periodic: To loop periodically over the time component of the Field. It is set to either False or the length of the period (either float in seconds or datetime.timedelta object). (Default: False)
               This flag overrides the allow_time_interpolation and sets it to False
        """

        # Coercing into mitgcm format
        ds = ds.assign(U1=ds.U1.swap_dims({'N': 'NC'}),
                       V1=ds.V1.swap_dims({'M': 'MC'}),
                       S1=ds.S1.swap_dims({'N': 'NC', 'M': 'MC'}),
                       XZ=ds.XZ.swap_dims({'N': 'NC', 'M': 'MC'}),
                       YZ=ds.YZ.swap_dims({'N': 'NC', 'M': 'MC'}),
                       )[['U1', 'V1', 'S1']].drop(['XZ', 'YZ']).isel(KMAXOUT_RESTR=0)

        # Transposing xarray dataset as per https://github.com/OceanParcels/parcels/issues/1180
        ds = ds.transpose('time', 'NC', 'MC', ...)

        fields = {}
        if 'creation_log' not in kwargs.keys():
            kwargs['creation_log'] = 'from_xarray_delft3d'
        if kwargs.pop('gridindexingtype', 'mitgcm') != 'mitgcm':
            raise ValueError("gridindexingtype must be 'mitgcm' in FieldSet.from_xarray_delft3d(), if not: consider changing method")
        if 'time' in dimensions:
            if 'units' not in ds[dimensions['time']].attrs and 'Unit' in ds[dimensions['time']].attrs:
                # Fix DataArrays that have time.Unit instead of expected time.units
                convert_xarray_time_units(ds, dimensions['time'])

        for var, name in variables.items():
            dims = dimensions[var] if var in dimensions else dimensions
            cls.checkvaliddimensionsdict(dims)

            fields[var] = Field.from_xarray_delft3d_field(ds[name], var, dims,
                                                          mesh=mesh,
                                                          allow_time_extrapolation=allow_time_extrapolation,
                                                          time_periodic=time_periodic, **kwargs)
        u = fields.pop('U', None)
        v = fields.pop('V', None)
        return cls(u, v, fields=fields)

    @classmethod
    def from_xarray_dataset(cls, ds, variables, dimensions, mesh='spherical', allow_time_extrapolation=None,
                            time_periodic=False, **kwargs):
        """Initialises FieldSet data from xarray Datasets.

        Parameters
        ----------
        ds : xr.Dataset
            xarray Dataset.
            Note that the built-in Advection kernels assume that U and V are in m/s
        variables : dict
            Dictionary mapping parcels variable names to data variables in the xarray Dataset.
        dimensions : dict
            Dictionary mapping data dimensions (lon,
            lat, depth, time, data) to dimensions in the xarray Dataset.
            Note that dimensions can also be a dictionary of dictionaries if
            dimension names are different for each variable
            (e.g. dimensions['U'], dimensions['V'], etc).
        fieldtype :
            Optional dictionary mapping fields to fieldtypes to be used for UnitConverter.
            (either 'U', 'V', 'Kh_zonal', 'Kh_meridional' or None)
        mesh : str
            String indicating the type of mesh coordinates and
            units used during velocity interpolation, see also `this tutorial <../examples/tutorial_unitconverters.ipynb>`__:

            1. spherical (default): Lat and lon in degree, with a
               correction for zonal velocity U near the poles.
            2. flat: No conversion, lat/lon are assumed to be in m.
        allow_time_extrapolation : bool
            boolean whether to allow for extrapolation
            (i.e. beyond the last available time snapshot)
            Default is False if dimensions includes time, else True
        time_periodic : bool, float or datetime.timedelta
            To loop periodically over the time component of the Field. It is set to either False or the length of the period (either float in seconds or datetime.timedelta object). (Default: False)
            This flag overrides the allow_time_extrapolation and sets it to False
        **kwargs :
            Keyword arguments passed to the :func:`Field.from_xarray` constructor.
        """
        fields = {}
        if 'creation_log' not in kwargs.keys():
            kwargs['creation_log'] = 'from_xarray_dataset'
        if 'time' in dimensions:
            if 'units' not in ds[dimensions['time']].attrs and 'Unit' in ds[dimensions['time']].attrs:
                # Fix DataArrays that have time.Unit instead of expected time.units
                convert_xarray_time_units(ds, dimensions['time'])

        for var, name in variables.items():
            dims = dimensions[var] if var in dimensions else dimensions
            cls.checkvaliddimensionsdict(dims)

            fields[var] = Field.from_xarray(ds[name], var, dims, mesh=mesh, allow_time_extrapolation=allow_time_extrapolation,
                                            time_periodic=time_periodic, **kwargs)
        u = fields.pop('U', None)
        v = fields.pop('V', None)
        return cls(u, v, fields=fields)

    @classmethod
    def from_modulefile(cls, filename, modulename="create_fieldset", **kwargs):
        """Initialises FieldSet data from a file containing a python module file with a create_fieldset() function.

        Parameters
        ----------
        filename: path to a python file containing at least a function which returns a FieldSet object.
        modulename: name of the function in the python file that returns a FieldSet object. Default is "create_fieldset".
        """
        # check if filename exists
        if not os.path.exists(filename):
            raise IOError(f"FieldSet module file {filename} does not exist")

        # Importing the source file directly (following https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly)
        spec = importlib.util.spec_from_file_location(modulename, filename)
        fieldset_module = importlib.util.module_from_spec(spec)
        sys.modules[modulename] = fieldset_module
        spec.loader.exec_module(fieldset_module)

        if not hasattr(fieldset_module, modulename):
            raise IOError(f"{filename} does not contain a {modulename} function")
        fieldset = getattr(fieldset_module, modulename)(**kwargs)
        if not isinstance(fieldset, FieldSet):
            raise IOError(f"Module {filename}.{modulename} does not return a FieldSet object")
        return fieldset

    def get_fields(self):
        """Returns a list of all the :class:`parcels.field.Field` and :class:`parcels.field.VectorField`
        objects associated with this FieldSet.
        """
        fields = []
        for v in self.__dict__.values():
            if type(v) in [Field, VectorField]:
                if v not in fields:
                    fields.append(v)
            elif isinstance(v, NestedField):
                if v not in fields:
                    fields.append(v)
                for v2 in v:
                    if v2 not in fields:
                        fields.append(v2)
        return fields

    def add_constant(self, name, value):
        """Add a constant to the FieldSet. Note that all constants are
        stored as 32-bit floats. While constants can be updated during
        execution in SciPy mode, they can not be updated in JIT mode.

        Parameters
        ----------
        name : str
            Name of the constant
        value :
            Value of the constant (stored as 32-bit float)


        Examples
        --------
        Tutorials using fieldset.add_constant:
        `Analytical advection <../examples/tutorial_analyticaladvection.ipynb>`__
        `Diffusion <../examples/tutorial_diffusion.ipynb>`__
        `Periodic boundaries <../examples/tutorial_periodic_boundaries.ipynb>`__
        """
        setattr(self, name, value)

    def add_periodic_halo(self, zonal=False, meridional=False, halosize=5):
        """Add a 'halo' to all :class:`parcels.field.Field` objects in a FieldSet,
        through extending the Field (and lon/lat) by copying a small portion
        of the field on one side of the domain to the other.

        Parameters
        ----------
        zonal : bool
            Create a halo in zonal direction (Default value = False)
        meridional : bool
            Create a halo in meridional direction (Default value = False)
        halosize : int
            size of the halo (in grid points). Default is 5 grid points
        """
        for grid in self.gridset.grids:
            grid.add_periodic_halo(zonal, meridional, halosize)
        for value in self.__dict__.values():
            if isinstance(value, Field):
                value.add_periodic_halo(zonal, meridional, halosize)

    def write(self, filename):
        """Write FieldSet to NetCDF file using NEMO convention.

        Parameters
        ----------
        filename : str
            Basename of the output fileset.
        """
        if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
            logger.info(f"Generating FieldSet output with basename: {filename}")

            if hasattr(self, 'U'):
                self.U.write(filename, varname='vozocrtx')
            if hasattr(self, 'V'):
                self.V.write(filename, varname='vomecrty')

            for v in self.get_fields():
                if isinstance(v, Field) and (v.name != 'U') and (v.name != 'V'):
                    v.write(filename)

    def computeTimeChunk(self, time=0., dt=1):
        """Load a chunk of three data time steps into the FieldSet.
        This is used when FieldSet uses data imported from netcdf,
        with default option deferred_load. The loaded time steps are at or immediatly before time
        and the two time steps immediately following time if dt is positive (and inversely for negative dt)

        Parameters
        ----------
        time :
            Time around which the FieldSet chunks are to be loaded.
            Time is provided as a double, relatively to Fieldset.time_origin.
            Default is 0.
        dt :
            time step of the integration scheme, needed to set the direction of time chunk loading.
            Default is 1.
        """
        signdt = np.sign(dt)
        nextTime = np.inf if dt > 0 else -np.inf

        for g in self.gridset.grids:
            g.update_status = 'not_updated'
        for f in self.get_fields():
            if isinstance(f, (VectorField, NestedField)) or not f.grid.defer_load:
                continue
            if f.grid.update_status == 'not_updated':
                nextTime_loc = f.grid.computeTimeChunk(f, time, signdt)
                if time == nextTime_loc and signdt != 0:
                    raise TimeExtrapolationError(time, field=f, msg='In fset.computeTimeChunk')
            nextTime = min(nextTime, nextTime_loc) if signdt >= 0 else max(nextTime, nextTime_loc)

        for f in self.get_fields():
            if isinstance(f, (VectorField, NestedField)) or not f.grid.defer_load or f.dataFiles is None:
                continue
            g = f.grid
            if g.update_status == 'first_updated':  # First load of data
                if f.data is not None and not isinstance(f.data, DeferredArray):
                    if not isinstance(f.data, list):
                        f.data = None
                    else:
                        for i in range(len(f.data)):
                            del f.data[i, :]

                lib = np if f.chunksize in [False, None] else da
                if f.gridindexingtype == 'pop' and g.zdim > 1:
                    zd = g.zdim - 1
                else:
                    zd = g.zdim
                data = lib.empty((g.tdim, zd, g.ydim-2*g.meridional_halo, g.xdim-2*g.zonal_halo), dtype=np.float32)
                f.loaded_time_indices = range(2)
                for tind in f.loaded_time_indices:
                    for fb in f.filebuffers:
                        if fb is not None:
                            fb.close()
                        fb = None
                    data = f.computeTimeChunk(data, tind)
                data = f.rescale_and_set_minmax(data)

                if (isinstance(f.data, DeferredArray)):
                    f.data = DeferredArray()
                f.data = f.reshape(data)
                if not f.chunk_set:
                    f.chunk_setup()
                if len(g.load_chunk) > g.chunk_not_loaded:
                    g.load_chunk = np.where(g.load_chunk == g.chunk_loaded_touched,
                                            g.chunk_loading_requested, g.load_chunk)
                    g.load_chunk = np.where(g.load_chunk == g.chunk_deprecated,
                                            g.chunk_not_loaded, g.load_chunk)

            elif g.update_status == 'updated':
                lib = np if isinstance(f.data, np.ndarray) else da
                if f.gridindexingtype == 'pop' and g.zdim > 1:
                    zd = g.zdim - 1
                else:
                    zd = g.zdim
                data = lib.empty((g.tdim, zd, g.ydim-2*g.meridional_halo, g.xdim-2*g.zonal_halo), dtype=np.float32)
                if signdt >= 0:
                    f.loaded_time_indices = [1]
                    if f.filebuffers[0] is not None:
                        f.filebuffers[0].close()
                        f.filebuffers[0] = None
                    f.filebuffers[0] = f.filebuffers[1]
                    data = f.computeTimeChunk(data, 1)
                else:
                    f.loaded_time_indices = [0]
                    if f.filebuffers[1] is not None:
                        f.filebuffers[1].close()
                        f.filebuffers[1] = None
                    f.filebuffers[1] = f.filebuffers[0]
                    data = f.computeTimeChunk(data, 0)
                data = f.rescale_and_set_minmax(data)
                if signdt >= 0:
                    data = f.reshape(data)[1, :]
                    if lib is da:
                        f.data = lib.stack([f.data[1, :], data], axis=0)
                    else:
                        if not isinstance(f.data, DeferredArray):
                            if isinstance(f.data, list):
                                del f.data[0, :]
                            else:
                                f.data[0, :] = None
                        f.data[0, :] = f.data[1, :]
                        f.data[1, :] = data
                else:
                    data = f.reshape(data)[0, :]
                    if lib is da:
                        f.data = lib.stack([data, f.data[0, :]], axis=0)
                    else:
                        if not isinstance(f.data, DeferredArray):
                            if isinstance(f.data, list):
                                del f.data[1, :]
                            else:
                                f.data[1, :] = None
                        f.data[1, :] = f.data[0, :]
                        f.data[0, :] = data
                g.load_chunk = np.where(g.load_chunk == g.chunk_loaded_touched,
                                        g.chunk_loading_requested, g.load_chunk)
                g.load_chunk = np.where(g.load_chunk == g.chunk_deprecated,
                                        g.chunk_not_loaded, g.load_chunk)
                if isinstance(f.data, da.core.Array) and len(g.load_chunk) > 0:
                    if signdt >= 0:
                        for block_id in range(len(g.load_chunk)):
                            if g.load_chunk[block_id] == g.chunk_loaded_touched:
                                if f.data_chunks[block_id] is None:
                                    # file chunks were never loaded.
                                    # happens when field not called by kernel, but shares a grid with another field called by kernel
                                    break
                                block = f.get_block(block_id)
                                f.data_chunks[block_id][0] = None
                                f.data_chunks[block_id][1] = np.array(f.data.blocks[(slice(2),)+block][1])
                    else:
                        for block_id in range(len(g.load_chunk)):
                            if g.load_chunk[block_id] == g.chunk_loaded_touched:
                                if f.data_chunks[block_id] is None:
                                    # file chunks were never loaded.
                                    # happens when field not called by kernel, but shares a grid with another field called by kernel
                                    break
                                block = f.get_block(block_id)
                                f.data_chunks[block_id][1] = None
                                f.data_chunks[block_id][0] = np.array(f.data.blocks[(slice(2),)+block][0])
        # do user-defined computations on fieldset data
        if self.compute_on_defer:
            self.compute_on_defer(self)

        # update time varying grid depth
        for f in self.get_fields():
            if isinstance(f, (VectorField, NestedField)) or not f.grid.defer_load or f.dataFiles is None:
                continue
            if f.grid.depth_field is not None:
                depth_data = f.grid.depth_field.data
                f.grid.depth = depth_data if isinstance(depth_data, np.ndarray) else np.array(depth_data)

        if abs(nextTime) == np.inf or np.isnan(nextTime):  # Second happens when dt=0
            return nextTime
        else:
            nSteps = int((nextTime - time) / dt)
            if nSteps == 0:
                return nextTime
            else:
                return time + nSteps * dt
