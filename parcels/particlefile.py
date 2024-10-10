"""Module controlling the writing of ParticleSets to Zarr file."""

import os
import warnings
from datetime import timedelta

import numpy as np
import xarray as xr
import zarr

import parcels
from parcels._compat import MPI
from parcels.tools._helpers import deprecated, deprecated_made_private
from parcels.tools.warnings import FileWarning

__all__ = ["ParticleFile"]


def _set_calendar(origin_calendar):
    if origin_calendar == "np_datetime64":
        return "standard"
    else:
        return origin_calendar


class ParticleFile:
    """Initialise trajectory output.

    Parameters
    ----------
    name : str
        Basename of the output file. This can also be a Zarr store object.
    particleset :
        ParticleSet to output
    outputdt :
        Interval which dictates the update frequency of file output
        while ParticleFile is given as an argument of ParticleSet.execute()
        It is either a timedelta object or a positive double.
    chunks :
        Tuple (trajs, obs) to control the size of chunks in the zarr output.
    create_new_zarrfile : bool
        Whether to create a new file. Default is True

    Returns
    -------
    ParticleFile
        ParticleFile object that can be used to write particle data to file
    """

    def __init__(self, name, particleset, outputdt=np.inf, chunks=None, create_new_zarrfile=True):
        self._outputdt = outputdt.total_seconds() if isinstance(outputdt, timedelta) else outputdt
        self._chunks = chunks
        self._particleset = particleset
        self.parcels_mesh = "spherical"
        if self.particleset.fieldset is not None:
            self.parcels_mesh = self.particleset.fieldset.gridset.grids[0].mesh
        self.lonlatdepth_dtype = self.particleset.particledata.lonlatdepth_dtype
        self.maxids = 0
        self.pids_written = {}
        self._create_new_zarrfile = create_new_zarrfile
        self._vars_to_write = {}
        for var in self.particleset.particledata.ptype.variables:
            if var.to_write:
                self.vars_to_write[var.name] = var.dtype
        self.mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
        self.particleset.fieldset._particlefile = self
        self.analytical = False  # Flag to indicate if ParticleFile is used for analytical trajectories

        # Reset obs_written of each particle, in case new ParticleFile created for a ParticleSet
        particleset.particledata.setallvardata("obs_written", 0)

        self.metadata = {
            "feature_type": "trajectory",
            "Conventions": "CF-1.6/CF-1.7",
            "ncei_template_version": "NCEI_NetCDF_Trajectory_Template_v2.0",
            "parcels_version": parcels.__version__,
            "parcels_mesh": self.parcels_mesh,
        }

        # Create dictionary to translate datatypes and fill_values
        self.fill_value_map = {
            np.float16: np.nan,
            np.float32: np.nan,
            np.float64: np.nan,
            np.bool_: np.iinfo(np.int8).max,
            np.int8: np.iinfo(np.int8).max,
            np.int16: np.iinfo(np.int16).max,
            np.int32: np.iinfo(np.int32).max,
            np.int64: np.iinfo(np.int64).max,
            np.uint8: np.iinfo(np.uint8).max,
            np.uint16: np.iinfo(np.uint16).max,
            np.uint32: np.iinfo(np.uint32).max,
            np.uint64: np.iinfo(np.uint64).max,
        }
        if issubclass(type(name), zarr.storage.Store):
            # If we already got a Zarr store, we won't need any of the naming logic below.
            # But we need to handle incompatibility with MPI mode for now:
            if MPI and MPI.COMM_WORLD.Get_size() > 1:
                raise ValueError("Currently, MPI mode is not compatible with directly passing a Zarr store.")
            fname = name
        else:
            extension = os.path.splitext(str(name))[1]
            if extension in [".nc", ".nc4"]:
                raise RuntimeError(
                    "Output in NetCDF is not supported anymore. Use .zarr extension for ParticleFile name."
                )
            if MPI and MPI.COMM_WORLD.Get_size() > 1:
                fname = os.path.join(name, f"proc{self.mpi_rank:02d}.zarr")
                if extension in [".zarr"]:
                    warnings.warn(
                        f"The ParticleFile name contains .zarr extension, but zarr files will be written per processor in MPI mode at {fname}",
                        FileWarning,
                        stacklevel=2,
                    )
            else:
                fname = name if extension in [".zarr"] else f"{name}.zarr"
        self._fname = fname

    @property
    def create_new_zarrfile(self):
        return self._create_new_zarrfile

    @property
    def outputdt(self):
        return self._outputdt

    @property
    def chunks(self):
        return self._chunks

    @property
    def particleset(self):
        return self._particleset

    @property
    def fname(self):
        return self._fname

    @property
    def vars_to_write(self):
        return self._vars_to_write

    @property
    def time_origin(self):
        return self.particleset.time_origin

    def _create_variables_attribute_dict(self):
        """Creates the dictionary with variable attributes.

        Notes
        -----
        For ParticleSet structures other than SoA, and structures where ID != index, this has to be overridden.
        """
        attrs = {
            "z": {"long_name": "", "standard_name": "depth", "units": "m", "positive": "down"},
            "trajectory": {
                "long_name": "Unique identifier for each particle",
                "cf_role": "trajectory_id",
                "_FillValue": self.fill_value_map[np.int64],
            },
            "time": {"long_name": "", "standard_name": "time", "units": "seconds", "axis": "T"},
            "lon": {"long_name": "", "standard_name": "longitude", "units": "degrees_east", "axis": "X"},
            "lat": {"long_name": "", "standard_name": "latitude", "units": "degrees_north", "axis": "Y"},
        }

        if self.time_origin.calendar is not None:
            attrs["time"]["units"] = "seconds since " + str(self.time_origin)
            attrs["time"]["calendar"] = _set_calendar(self.time_origin.calendar)

        for vname in self.vars_to_write:
            if vname not in ["time", "lat", "lon", "depth", "id"]:
                attrs[vname] = {
                    "_FillValue": self.fill_value_map[self.vars_to_write[vname]],
                    "long_name": "",
                    "standard_name": vname,
                    "units": "unknown",
                }

        return attrs

    @deprecated(
        "ParticleFile.metadata is a dictionary. Use `ParticleFile.metadata['key'] = ...` or other dictionary methods instead."
    )  # TODO: Remove 6 months after v3.1.0
    def add_metadata(self, name, message):
        """Add metadata to :class:`parcels.particleset.ParticleSet`.

        Parameters
        ----------
        name : str
            Name of the metadata variable
        message : str
            message to be written
        """
        self.metadata[name] = message

    def _convert_varout_name(self, var):
        if var == "depth":
            return "z"
        elif var == "id":
            return "trajectory"
        else:
            return var

    @deprecated_made_private  # TODO: Remove 6 months after v3.1.0
    def write_once(self, *args, **kwargs):
        return self._write_once(*args, **kwargs)

    def _write_once(self, var):
        return self.particleset.particledata.ptype[var].to_write == "once"

    def _extend_zarr_dims(self, Z, store, dtype, axis):
        if axis == 1:
            a = np.full((Z.shape[0], self.chunks[1]), self.fill_value_map[dtype], dtype=dtype)
            obs = zarr.group(store=store, overwrite=False)["obs"]
            if len(obs) == Z.shape[1]:
                obs.append(np.arange(self.chunks[1]) + obs[-1] + 1)
        else:
            extra_trajs = self.maxids - Z.shape[0]
            if len(Z.shape) == 2:
                a = np.full((extra_trajs, Z.shape[1]), self.fill_value_map[dtype], dtype=dtype)
            else:
                a = np.full((extra_trajs,), self.fill_value_map[dtype], dtype=dtype)
        Z.append(a, axis=axis)
        zarr.consolidate_metadata(store)

    def write(self, pset, time, indices=None):
        """Write all data from one time step to the zarr file,
        before the particle locations are updated.

        Parameters
        ----------
        pset :
            ParticleSet object to write
        time :
            Time at which to write ParticleSet
        """
        time = time.total_seconds() if isinstance(time, timedelta) else time

        if pset.particledata._ncount == 0:
            warnings.warn(
                f"ParticleSet is empty on writing as array at time {time:g}",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        if indices is None:
            indices_to_write = pset.particledata._to_write_particles(pset.particledata._data, time)
        else:
            indices_to_write = indices

        if len(indices_to_write) > 0:
            pids = pset.particledata.getvardata("id", indices_to_write)
            to_add = sorted(set(pids) - set(self.pids_written.keys()))
            for i, pid in enumerate(to_add):
                self.pids_written[pid] = self.maxids + i
            ids = np.array([self.pids_written[p] for p in pids], dtype=int)
            self.maxids = len(self.pids_written)

            once_ids = np.where(pset.particledata.getvardata("obs_written", indices_to_write) == 0)[0]
            if len(once_ids) > 0:
                ids_once = ids[once_ids]
                indices_to_write_once = indices_to_write[once_ids]

            if self.create_new_zarrfile:
                if self.chunks is None:
                    self._chunks = (len(ids), 1)
                if pset._repeatpclass is not None and self.chunks[0] < 1e4:
                    warnings.warn(
                        f"ParticleFile chunks are set to {self.chunks}, but this may lead to "
                        f"a significant slowdown in Parcels when many calls to repeatdt. "
                        f"Consider setting a larger chunk size for your ParticleFile (e.g. chunks=(int(1e4), 1)).",
                        FileWarning,
                        stacklevel=2,
                    )
                if (self.maxids > len(ids)) or (self.maxids > self.chunks[0]):
                    arrsize = (self.maxids, self.chunks[1])
                else:
                    arrsize = (len(ids), self.chunks[1])
                ds = xr.Dataset(
                    attrs=self.metadata,
                    coords={"trajectory": ("trajectory", pids), "obs": ("obs", np.arange(arrsize[1], dtype=np.int32))},
                )
                attrs = self._create_variables_attribute_dict()
                obs = np.zeros((self.maxids), dtype=np.int32)
                for var in self.vars_to_write:
                    varout = self._convert_varout_name(var)
                    if varout not in ["trajectory"]:  # because 'trajectory' is written as coordinate
                        if self._write_once(var):
                            data = np.full(
                                (arrsize[0],),
                                self.fill_value_map[self.vars_to_write[var]],
                                dtype=self.vars_to_write[var],
                            )
                            data[ids_once] = pset.particledata.getvardata(var, indices_to_write_once)
                            dims = ["trajectory"]
                        else:
                            data = np.full(
                                arrsize, self.fill_value_map[self.vars_to_write[var]], dtype=self.vars_to_write[var]
                            )
                            data[ids, 0] = pset.particledata.getvardata(var, indices_to_write)
                            dims = ["trajectory", "obs"]
                        ds[varout] = xr.DataArray(data=data, dims=dims, attrs=attrs[varout])
                        ds[varout].encoding["chunks"] = self.chunks[0] if self._write_once(var) else self.chunks
                ds.to_zarr(self.fname, mode="w")
                self._create_new_zarrfile = False
            else:
                # Either use the store that was provided directly or create a DirectoryStore:
                if issubclass(type(self.fname), zarr.storage.Store):
                    store = self.fname
                else:
                    store = zarr.DirectoryStore(self.fname)
                Z = zarr.group(store=store, overwrite=False)
                obs = pset.particledata.getvardata("obs_written", indices_to_write)
                for var in self.vars_to_write:
                    varout = self._convert_varout_name(var)
                    if self.maxids > Z[varout].shape[0]:
                        self._extend_zarr_dims(Z[varout], store, dtype=self.vars_to_write[var], axis=0)
                    if self._write_once(var):
                        if len(once_ids) > 0:
                            Z[varout].vindex[ids_once] = pset.particledata.getvardata(var, indices_to_write_once)
                    else:
                        if max(obs) >= Z[varout].shape[1]:
                            self._extend_zarr_dims(Z[varout], store, dtype=self.vars_to_write[var], axis=1)
                        Z[varout].vindex[ids, obs] = pset.particledata.getvardata(var, indices_to_write)

            pset.particledata.setvardata("obs_written", indices_to_write, obs + 1)

    def write_latest_locations(self, pset, time):
        """Write the current (latest) particle locations to zarr file.
        This can be useful at the end of a pset.execute(), when the last locations are not written yet.
        Note that this only updates the locations, not any of the other Variables. Therefore, use with care.

        Parameters
        ----------
        pset :
            ParticleSet object to write
        time :
            Time at which to write ParticleSet. Note that typically this would be pset.time_nextloop
        """
        for var in ["lon", "lat", "depth", "time"]:
            pset.particledata.setallvardata(f"{var}", pset.particledata.getvardata(f"{var}_nextloop"))

        self.write(pset, time)
