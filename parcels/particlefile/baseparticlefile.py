"""Module controlling the writing of ParticleSets to sqlite file."""
import os
import sqlite3
from abc import ABC
from datetime import timedelta as delta

import numpy as np

from parcels.tools.loggers import logger

try:
    from mpi4py import MPI
except:
    MPI = None
try:
    from parcels._version import version as parcels_version
except:
    raise OSError('Parcels version can not be retrieved. Have you run ''python setup.py install''?')


__all__ = ['BaseParticleFile']


class BaseParticleFile(ABC):
    """Initialise trajectory output.

    Parameters
    ----------
    name : str
        Basename of the output file. This can also be a Zarr store object.  # TODO make sure can also write to sqlite store?
    particleset :
        ParticleSet to output
    outputdt :
        Interval which dictates the update frequency of file output
        while ParticleFile is given as an argument of ParticleSet.execute()
        It is either a timedelta object or a positive double.

    Returns
    -------
    BaseParticleFile
        ParticleFile object that can be used to write particle data to file
    """

    outputdt = None
    lasttime_written = None
    particleset = None
    parcels_mesh = None
    time_origin = None
    lonlatdepth_dtype = None

    def __init__(self, name, particleset, outputdt=None):

        self.outputdt = outputdt
        self.lasttime_written = None  # variable to check if time has been written already

        if isinstance(self.outputdt, delta):
            self.outputdt = self.outputdt.total_seconds()
        if (not isinstance(self.outputdt, (int, float, delta))) or self.outputdt < 0:
            raise SyntaxError('outputdt in ParticleFile must be a timedelta object or a positive double')
        self.particleset = particleset
        self.parcels_mesh = 'spherical'
        if self.particleset.fieldset is not None:
            self.parcels_mesh = self.particleset.fieldset.gridset.grids[0].mesh
        self.time_origin = self.particleset.time_origin
        self.lonlatdepth_dtype = self.particleset.collection.lonlatdepth_dtype
        self.vars_to_write = {}
        self.vars_to_write['id'] = np.int64
        for var in self.particleset.collection.ptype.variables:
            if var.to_write:
                self.vars_to_write[var.name] = var.dtype
        self.analytical = False
        self.mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0

        if isinstance(name, str) and ':memory:' in name:
            # If we get a :memory: store, we won't need any of the naming logic below.
            # But we need to handle incompatibility with MPI mode for now:
            if MPI and MPI.COMM_WORLD.Get_size() > 1:
                raise ValueError("Currently, MPI mode is not compatible with directly passing a :memory: store.")
            self.fname = name
        else:
            extension = os.path.splitext(str(name))[1]
            if extension in ['.db', '.sqlite', '']:
                pass
            else:
                raise RuntimeError(f"Output format {extension} not supported. Use .sqlite extension for ParticleFile name.")

            if MPI and MPI.COMM_WORLD.Get_size() > 1:
                self.fname = os.path.join(name, f"proc{self.mpi_rank:02d}.sqlite")
                if extension in ['.db', '.sqlite']:
                    logger.warning(f'The ParticleFile name contains .sqlite extension, but sqlite files will be written per processor in MPI mode at {self.fname}')
            else:
                self.fname = name if extension in ['.db', '.sqlite'] else f"{name}.sqlite"
                try:
                    os.remove(self.fname)
                except OSError:
                    pass

        def _convert_varout_name(var):
            if var == 'depth':
                return 'z'
            elif var == 'id':
                return 'trajectory'
            else:
                return var

        self.con = sqlite3.connect(self.fname, uri=True)
        self.cur = self.con.cursor()
        varstr = ', '.join([f'{_convert_varout_name(var)}' for var in self.vars_to_write.keys()])
        self.cur.execute(f"CREATE TABLE particles({varstr})")
        self.cur.execute("PRAGMA journal_mode = WAL")
        self.cur.execute("PRAGMA synchronous = normal")

        if self.particleset.time_origin.calendar is None:
            calendar = 'timedelta64[s]'
            time_origin = self.particleset.time_origin.time_origin
        else:
            calendar = self.particleset.time_origin.calendar
            time_origin = str(self.particleset.time_origin.time_origin)
        self.metadata = {"feature_type": "trajectory",
                         "Conventions": "CF-1.6/CF-1.7",
                         "parcels_version": parcels_version,
                         "calendar": calendar,
                         "time_origin": time_origin,
                         "parcels_mesh": self.parcels_mesh}
        meta_varstr = ', '.join([f'{mvar}' for mvar in self.metadata.keys()])
        self.cur.execute(f"CREATE TABLE metadata({meta_varstr})")
        meta_str = ', '.join('?' * len(self.metadata))
        self.cur.execute(f"INSERT INTO metadata VALUES ({meta_str})", list(self.metadata.values()))
        self.con.commit()
        self.particleset.fieldset.particlefile = self

    def __del__(self):
        if hasattr(self, 'con'):
            self.con.close()

    def add_metadata(self, name, message):  # TODO check if metadata can be added in sqlite
        """Add metadata to :class:`parcels.particleset.ParticleSet`.

        Parameters
        ----------
        name : str
            Name of the metadata variabale
        message : str
            message to be written
        """
        self.cur.execute(f"alter table metadata add column {name}")
        self.cur.execute(f"UPDATE metadata SET {name} = {message} where rowid = 1")
        self.con.commit()
