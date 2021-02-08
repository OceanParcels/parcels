"""Module controlling the writing of ParticleSets to NetCDF file"""
import os
import random
import shutil
import string
from datetime import timedelta as delta
from glob import glob

import netCDF4
import numpy as np

from parcels.tools.error import ErrorCode
from parcels.tools.loggers import logger
try:
    from mpi4py import MPI
except:
    MPI = None
try:
    from parcels._version import version as parcels_version
except:
    raise EnvironmentError('Parcels version can not be retrieved. Have you run ''python setup.py install''?')
try:
    from os import getuid
except:
    # Windows does not have getuid(), so define to simply return 'tmp'
    def getuid():
        return 'tmp'
# from parcels.particleset_node import ParticleSet
# from parcels.particleset_vectorized import ParticleSet as BadParticleSet


__all__ = ['ParticleFile']


def _is_particle_started_yet(particle, time):
    """We don't want to write a particle that is not started yet.
    Particle will be written if:
      * particle.time is equal to time argument of pfile.write()
      * particle.time is before time (in case particle was deleted between previous export and current one)
    """
    return (particle.dt*particle.time <= particle.dt*time or np.isclose(particle.time, time))


def _set_calendar(origin_calendar):
    if origin_calendar == 'np_datetime64':
        return 'standard'
    else:
        return origin_calendar

# ==== Comment: the way this class is written is really unsustainable / non-maintainable. This is because it holds
# ==== vectorized attribute copies of the particle set als well as their mapped memory areas in the NetCDF file,
# ==== but NONE of the attributes are visible in the constructor. Basically: one wanted to subclass the ParticleSet
# ==== and the NetCDF.Dataset, but instead mapped their attributes manually by lazy binding into this class.
# ==== This obviously circumvents any variable checks and is also prone to leave garbage memory open during the computation.


class ParticleFile(object):
    """Initialise trajectory output.
    :param name: Basename of the output file
    :param particleset: ParticleSet to output
    :param outputdt: Interval which dictates the update frequency of file output
                     while ParticleFile is given as an argument of ParticleSet.execute()
                     It is either a timedelta object or a positive double.
    :param write_ondelete: Boolean to write particle data only when they are deleted. Default is False
    :param convert_at_end: Boolean to convert npy files to netcdf at end of run. Default is True
    :param tempwritedir: directories to write temporary files to during executing.
                     Default is out-XXXXXX where Xs are random capitals. Files for individual
                     processors are written to subdirectories 0, 1, 2 etc under tempwritedir
    :param pset_info: dictionary of info on the ParticleSet, stored in tempwritedir/XX/pset_info.npy,
                     used to create NetCDF file from npy-files.
    """

    def __init__(self, name, particleset, outputdt=np.infty, write_ondelete=False, convert_at_end=True,
                 tempwritedir=None, pset_info=None):
        # if particleset is not None and isinstance(particleset, ParticleSet):
        #     err_msg_addendum = "Classes do not match."
        #     if isinstance(particleset, BadParticleSet):
        #         err_msg_addendum = "You tried to initialize a node-based particle file with a vectorized particle set - action forbidden."
        #     logger.error("'particleset' is not a node-based Particle Set. %s. Exiting" % (err_msg_addendum))
        #     exit()

        self.write_ondelete = write_ondelete
        self.convert_at_end = convert_at_end
        self.outputdt = outputdt
        self.lasttime_written = None  # variable to check if time has been written already

        self.dataset = None
        self.metadata = {}
        if pset_info is not None:
            for v in pset_info.keys():
                setattr(self, v, pset_info[v])
        else:
            self.name = name
            self.particleset = particleset
            self.parcels_mesh = self.particleset.fieldset.gridset.grids[0].mesh
            self.time_origin = self.particleset.time_origin
            self.lonlatdepth_dtype = self.particleset.lonlatdepth_dtype
            self.var_names = []
            self.var_names_once = []
            for v in self.particleset.ptype.variables:
                if v.to_write == 'once':
                    self.var_names_once += [v.name]
                elif v.to_write is True:
                    self.var_names += [v.name]
            if len(self.var_names_once) > 0:
                self.written_once = []
                self.file_list_once = []

            self.file_list = []
            self.time_written = []
            # self.num_ids_written = -1  # => Idea: take IDgenerator.length that always gives you the maximum number of particles!
            # self.maxid_written = -1
            self.max_index_written = -1

        if tempwritedir is None:
            tempwritedir = os.path.join(os.path.dirname(str(self.name)), "out-%s"
                                        % ''.join(random.choice(string.ascii_uppercase) for _ in range(8)))

        if MPI:
            mpi_rank = MPI.COMM_WORLD.Get_rank()
            self.tempwritedir_base = MPI.COMM_WORLD.bcast(tempwritedir, root=0)
        else:
            self.tempwritedir_base = tempwritedir
            mpi_rank = 0
        self.tempwritedir = os.path.join(self.tempwritedir_base, "%d" % mpi_rank)

        if pset_info is None:  # otherwise arrive here from convert_npydir_to_netcdf
            self.delete_tempwritedir()

    def open_netcdf_file(self, data_shape):
        """Initialise NetCDF4.Dataset for trajectory output.
        The output follows the format outlined in the Discrete Sampling Geometries
        section of the CF-conventions:
        http://cfconventions.org/cf-conventions/v1.6.0/cf-conventions.html#discrete-sampling-geometries
        The current implementation is based on the NCEI template:
        http://www.nodc.noaa.gov/data/formats/netcdf/v2.0/trajectoryIncomplete.cdl
        :param data_shape: shape of the variables in the NetCDF4 file
        """
        extension = os.path.splitext(str(self.name))[1]
        fname = self.name if extension in ['.nc', '.nc4'] else "%s.nc" % self.name
        if os.path.exists(str(fname)):
            os.remove(str(fname))
        self.dataset = netCDF4.Dataset(fname, "w", format="NETCDF4")
        self.dataset.createDimension("obs", data_shape[1])
        self.dataset.createDimension("traj", data_shape[0])
        coords = ("traj", "obs")
        self.dataset.feature_type = "trajectory"
        self.dataset.Conventions = "CF-1.6/CF-1.7"
        self.dataset.ncei_template_version = "NCEI_NetCDF_Trajectory_Template_v2.0"
        self.dataset.parcels_version = parcels_version
        self.dataset.parcels_mesh = self.parcels_mesh

        # Create ID variable according to CF conventions
        # self.id = self.dataset.createVariable("trajectory", "i4", coords, fill_value=-2**(31))  # maxint32 fill_value
        self.id = self.dataset.createVariable("trajectory", "u8", coords, fill_value=np.iinfo(np.uint64).max)
        self.id.long_name = "Unique identifier for each particle"
        self.id.cf_role = "trajectory_id"

        self.index = self.dataset.createVariable("index", "i4", coords, fill_value=-2**(31))
        self.index.long_name = "running (zero-based continuous) indexing element referring to the LOCAL index of a particle within one timestep"

        # Create time, lat, lon and z variables according to CF conventions:
        self.time = self.dataset.createVariable("time", "f8", coords, fill_value=np.nan)
        self.time.long_name = ""
        self.time.standard_name = "time"
        if self.time_origin.calendar is None:
            self.time.units = "seconds"
        else:
            self.time.units = "seconds since " + str(self.time_origin)
            self.time.calendar = 'standard' if self.time_origin.calendar == 'np_datetime64' else self.time_origin.calendar
        self.time.axis = "T"

        if self.lonlatdepth_dtype is np.float64:
            lonlatdepth_precision = "f8"
        else:
            lonlatdepth_precision = "f4"

        self.lat = self.dataset.createVariable("lat", lonlatdepth_precision, coords, fill_value=np.nan)
        self.lat.long_name = ""
        self.lat.standard_name = "latitude"
        self.lat.units = "degrees_north"
        self.lat.axis = "Y"

        self.lon = self.dataset.createVariable("lon", lonlatdepth_precision, coords, fill_value=np.nan)
        self.lon.long_name = ""
        self.lon.standard_name = "longitude"
        self.lon.units = "degrees_east"
        self.lon.axis = "X"

        self.z = self.dataset.createVariable("z", lonlatdepth_precision, coords, fill_value=np.nan)
        self.z.long_name = ""
        self.z.standard_name = "depth"
        self.z.units = "m"
        self.z.positive = "down"

        for vname in self.var_names:
            if vname not in ['time', 'lat', 'lon', 'depth', 'id', 'index']:
                # ==== Comment - hm, that looks a bit 'crooked' - certainly from getting the ptype
                # ==== you can parse the variables' bytesize - or make a "numpy-size to netcdf-size' conversion table here ...
                setattr(self, vname, self.dataset.createVariable(vname, "f4", coords, fill_value=np.nan))
                getattr(self, vname).long_name = ""
                getattr(self, vname).standard_name = vname
                getattr(self, vname).units = "unknown"

        for vname in self.var_names_once:
            setattr(self, vname, self.dataset.createVariable(vname, "f4", "traj", fill_value=np.nan))
            # ==== Comment - hm, that looks a bit 'crooked' - certainly from getting the ptype
            # ==== you can parse the variables' bytesize - or make a "numpy-size to netcdf-size' conversion table here ...
            getattr(self, vname).long_name = ""
            getattr(self, vname).standard_name = vname
            getattr(self, vname).units = "unknown"

        for name, message in self.metadata.items():
            setattr(self.dataset, name, message)

    def __del__(self):
        if self.convert_at_end:
            self.close()

    def close(self, delete_tempfiles=True):
        """Close the ParticleFile object by exporting and then deleting
        the temporary npy files"""
        self.export()
        mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
        if mpi_rank == 0:
            if delete_tempfiles:
                self.delete_tempwritedir(tempwritedir=self.tempwritedir_base)
        self.convert_at_end = False

    def add_metadata(self, name, message):
        """Add metadata to :class:`parcels.particleset.ParticleSet`
        :param name: Name of the metadata variabale
        :param message: message to be written
        """
        if self.dataset is None:
            self.metadata[name] = message
        else:
            setattr(self.dataset, name, message)

    def convert_pset_to_dict(self, pset, time, deleted_only=False):
        """Convert all Particle data from one time step to a python dictionary.
        :param pset: ParticleSet object to write
        :param time: Time at which to write ParticleSet
        :param deleted_only: Flag to write only the deleted Particles
        returns two dictionaries: one for all variables to be written each outputdt,
         and one for all variables to be written once
        """
        data_dict = {}
        data_dict_once = {}

        time = time.total_seconds() if isinstance(time, delta) else time

        if self.lasttime_written != time and \
           (self.write_ondelete is False or deleted_only is True):
            if pset.size == 0:
                logger.warning("ParticleSet is empty on writing as array at time %g" % time)
            else:
                if deleted_only:
                    pset_towrite = pset
                # == commented due to git rebase to master 27 02 2020 == #
                # elif pset[0].dt > 0:
                #     pset_towrite = [p for p in pset if time - p.dt/2 <= p.time < time + p.dt and np.isfinite(p.id)]
                # else:
                #     pset_towrite = [p for p in pset if time + p.dt < p.time <= time - p.dt/2 and np.isfinite(p.id)]
                # else:
                #     pset_towrite = [p for p in pset if time - np.abs(p.dt/2) <= p.time < time + np.abs(p.dt) and np.isfinite(p.id)]
                pset_towrite = []
                node = pset.begin()
                while node is not None:
                    if (time - np.abs(node.data.dt/2)) <= node.data.time < (time + np.abs(node.data.dt)) and np.isfinite(node.data.id):
                        pset_towrite.append(node.data)
                    node = node.next
                if len(pset_towrite) > 0:
                    for var in self.var_names:
                        # if type(getattr(pset_towrite[0], var)) in [np.uint64, np.int64, np.uint32]:
                        #     data_dict[var] = np.array([np.int32(getattr(p, var)) for p in pset_towrite])
                        # else:
                        if 1:
                            data_dict[var] = np.array([getattr(p, var) for p in pset_towrite])
                    # ---- this doesn't work anymore cause the number of particles cannot be inferred from the IDs anymore ---- #
                    # self.maxid_written = np.max([self.maxid_written, np.max(data_dict['id'])])
                    # ==== does not work because the index changes depending on the MPI environment - it needs a running overview of how many particles there are ==== #
                    self.max_index_written = np.max([self.max_index_written, np.max(data_dict['index'])])

                pset_errs = [p for p in pset_towrite if p.state != ErrorCode.Delete and abs(time-p.time) > 1e-3]
                for p in pset_errs:
                    logger.warning_once('time argument in pfile.write() is %g, but a particle has time % g.' % (time, p.time))
                pset_towrite.clear()

                if time not in self.time_written:
                    self.time_written.append(time)

                if len(self.var_names_once) > 0:
                    # ------ ------ ------ ------ ------ ------ #
                    # first_write = [p for p in pset if (np.int32(p.id) not in self.written_once) and _is_particle_started_yet(p, time)]
                    # data_dict_once['id'] = np.array([np.int32(p.id) for p in first_write])
                    # for var in self.var_names_once:
                    #     data_dict_once[var] = np.array([getattr(p, var) for p in first_write])
                    # self.written_once += data_dict_once['id'].tolist()
                    # ------ ------ ------ ------ ------ ------ #
                    first_write = []
                    node = pset.begin()
                    while node is not None:
                        if (node.data.index not in self.written_once) and _is_particle_started_yet(node.data, time):
                            first_write.append(node.data)
                        node = node.next
                    data_dict_once['id'] = np.array([p.id for p in first_write])
                    for var in self.var_names_once:
                        data_dict_once[var] = np.array([getattr(p, var) for p in first_write])
                    self.written_once += data_dict_once['index'].tolist()
                    first_write.clear()

            if not deleted_only:
                self.lasttime_written = time

        return data_dict, data_dict_once

    def dump_dict_to_npy(self, data_dict, data_dict_once):
        """Buffer data to set of temporary numpy files, using np.save"""

        if not os.path.exists(self.tempwritedir):
            os.makedirs(self.tempwritedir)

        if len(data_dict) > 0:
            tmpfilename = os.path.join(self.tempwritedir, str(len(self.file_list)) + ".npy")
            with open(tmpfilename, 'wb') as f:
                np.save(f, data_dict)
            self.file_list.append(tmpfilename)

        if len(data_dict_once) > 0:
            tmpfilename = os.path.join(self.tempwritedir, str(len(self.file_list_once)) + '_once.npy')
            with open(tmpfilename, 'wb') as f:
                np.save(f, data_dict_once)
            self.file_list_once.append(tmpfilename)

    def dump_psetinfo_to_npy(self):
        pset_info = {}
        # attrs_to_dump = ['name', 'var_names', 'var_names_once', 'time_origin', 'lonlatdepth_dtype',
        #                  'file_list', 'file_list_once', 'maxid_written', 'time_written', 'parcels_mesh',
        #                  'metadata']
        attrs_to_dump = ['name', 'var_names', 'var_names_once', 'time_origin', 'lonlatdepth_dtype',
                         'file_list', 'file_list_once', 'max_index_written', 'time_written', 'parcels_mesh',
                         'metadata']
        for a in attrs_to_dump:
            if hasattr(self, a):
                pset_info[a] = getattr(self, a)
        with open(os.path.join(self.tempwritedir, 'pset_info.npy'), 'wb') as f:
            np.save(f, pset_info)

    def write(self, pset, time, deleted_only=False):
        """Write all data from one time step to a temporary npy-file
        using a python dictionary. The data is saved in the folder 'out'.
        :param pset: ParticleSet object to write
        :param time: Time at which to write ParticleSet
        :param deleted_only: Flag to write only the deleted Particles
        """

        data_dict, data_dict_once = self.convert_pset_to_dict(pset, time, deleted_only=deleted_only)
        self.dump_dict_to_npy(data_dict, data_dict_once)
        self.dump_psetinfo_to_npy()

    def read_from_npy(self, file_list, time_steps, var):
        """Read NPY-files for one variable using a loop over all files.
        :param file_list: List that  contains all file names in the output directory
        :param time_steps: Number of time steps that were written in out directory
        :param var: name of the variable to read
        """

        # data = np.nan * np.zeros((self.maxid_written+1, time_steps))
        # time_index = np.zeros(self.maxid_written+1, dtype=int)

        data = np.nan * np.zeros((self.max_index_written+1, time_steps))
        time_index = np.zeros(self.max_index_written+1, dtype=np.int32)
        t_ind_used = np.zeros(time_steps, dtype=int)

        # loop over all files
        for npyfile in file_list:
            try:
                data_dict = np.load(npyfile, allow_pickle=True).item()
            except NameError:
                raise RuntimeError('Cannot combine npy files into netcdf file because your ParticleFile is '
                                   'still open on interpreter shutdown.\nYou can use '
                                   '"parcels_convert_npydir_to_netcdf %s" to convert these to '
                                   'a NetCDF file yourself.\nTo avoid this error, make sure you '
                                   'close() your ParticleFile at the end of your script.' % self.tempwritedir)
            # id_ind = np.array(data_dict["id"], dtype=np.int32)
            # ------ ------ ------ ------ ------ ------
            # id_ind = np.array(data_dict["id"], dtype=np.uint64)
            id_ind = np.array(data_dict['index'])
            # ------ ------ ------ ------ ------ ------
            t_ind = time_index[id_ind] if 'once' not in file_list[0] else 0
            t_ind_used[t_ind] = 1
            data[id_ind, t_ind] = data_dict[var]
            time_index[id_ind] = time_index[id_ind] + 1

        # remove rows and columns that are completely filled with nan values
        tmp = data[time_index > 0, :]
        return tmp[:, t_ind_used == 1]

    def export(self):
        """Exports outputs in temporary NPY-files to NetCDF file"""

        if MPI:
            # The export can only start when all threads are done.
            MPI.COMM_WORLD.Barrier()
            if MPI.COMM_WORLD.Get_rank() > 0:
                return  # export only on threat 0

        # Retrieve all temporary writing directories and sort them in numerical order
        temp_names = sorted(glob(os.path.join("%s" % self.tempwritedir_base, "*")),
                            key=lambda x: int(os.path.basename(x)))

        if len(temp_names) == 0:
            raise RuntimeError("No npy files found in %s" % self.tempwritedir_base)

        global_max_index_written = -1
        # global_maxid_written = -1
        global_time_written = []
        global_file_list = []
        if len(self.var_names_once) > 0:
            global_file_list_once = []
        for tempwritedir in temp_names:
            if os.path.exists(tempwritedir):
                pset_info_local = np.load(os.path.join(tempwritedir, 'pset_info.npy'), allow_pickle=True).item()
                # global_maxid_written = np.max([global_maxid_written, pset_info_local['maxid_written']])
                global_max_index_written = np.max([global_max_index_written, pset_info_local['max_index_written']])
                global_time_written += pset_info_local['time_written']
                global_file_list += pset_info_local['file_list']
                if len(self.var_names_once) > 0:
                    global_file_list_once += pset_info_local['file_list_once']
        # self.maxid_written = global_maxid_written
        self.max_index_written = global_max_index_written
        self.time_written = np.unique(global_time_written)

        for var in self.var_names:
            data = self.read_from_npy(global_file_list, len(self.time_written), var)
            if var == self.var_names[0]:
                self.open_netcdf_file(data.shape)
            varout = 'z' if var == 'depth' else var
            getattr(self, varout)[:, :] = data
        # idx_data = self.read_from_npy(global_file_list, len(self.time_written), 'index')
        # getattr(self, 'index')[:, :] = idx_data

        if len(self.var_names_once) > 0:
            for var in self.var_names_once:
                getattr(self, var)[:] = self.read_from_npy(global_file_list_once, 1, var)

        self.dataset.close()

    def delete_tempwritedir(self, tempwritedir=None):
        """Deleted all temporary npy files
        :param tempwritedir Optional path of the directory to delete
        """
        if tempwritedir is None:
            tempwritedir = self.tempwritedir
        if os.path.exists(tempwritedir):
            shutil.rmtree(tempwritedir)
