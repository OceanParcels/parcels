from parcels import FieldSet, JITParticle, AdvectionRK4, Variable, StateCode, OperationCode, ErrorCode
from parcels.particleset_benchmark import ParticleSet_Benchmark
from datetime import timedelta as delta
from glob import glob
import numpy as np
import itertools
import matplotlib.pyplot as plt
import xarray as xr
import warnings
import math
import sys
import os
import gc
from  argparse import ArgumentParser
import fnmatch
import time as ostime
#import dask
warnings.simplefilter("ignore", category=xr.SerializationWarning)

try:
    from mpi4py import MPI
except:
    MPI = None


def create_galapagos_fieldset(datahead, periodic_wrap, use_stokes):
    # dask.config.set({'array.chunk-size': '16MiB'})
    ddir = os.path.join(datahead,"NEMO-MEDUSA/ORCA0083-N006/")
    ufiles = sorted(glob(ddir+'means/ORCA0083-N06_20[00-10]*d05U.nc'))
    vfiles = [u.replace('05U.nc', '05V.nc') for u in ufiles]
    meshfile = glob(ddir+'domain/coordinates.nc')
    nemo_files = {'U': {'lon': meshfile, 'lat': meshfile, 'data': ufiles},
                 'V': {'lon': meshfile, 'lat': meshfile, 'data': vfiles}}
    nemo_variables = {'U': 'uo', 'V': 'vo'}
    nemo_dimensions = {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'}
    period = delta(days=366) if periodic_wrap else False
    extrapolation = False if periodic_wrap else True
    # ==== Because the stokes data is a different grid, we actually need to define the chunking ==== #
    # fieldset_nemo = FieldSet.from_nemo(nemofiles, nemovariables, nemodimensions, field_chunksize='auto')
    nemo_chs = {'time_counter': 1, 'depthu': 75, 'depthv': 75, 'depthw': 75, 'deptht': 75, 'y': 100, 'x': 100}
    fieldset_nemo = FieldSet.from_nemo(nemo_files, nemo_variables, nemo_dimensions, field_chunksize=nemo_chs, time_periodic=period, allow_time_extrapolation=extrapolation)

    if wstokes:
        stokes_files = sorted(glob(datahead+"/WaveWatch3data/CFSR/WW3-*_uss.nc"))
        stokes_variables = {'U': 'uuss', 'V': 'vuss'}
        stokes_dimensions = {'lat': 'latitude', 'lon': 'longitude', 'time': 'time'}
        stokes_chs = {'time': 1, 'latitude': 16, 'longitude': 32}
        fieldset_stokes = FieldSet.from_netcdf(stokes_files, stokes_variables, stokes_dimensions, field_chunksize=stokes_chs, time_periodic=period, allow_time_extrapolation=extrapolation)
        fieldset_stokes.add_periodic_halo(zonal=True, meridional=False, halosize=5)

        fieldset = FieldSet(U=fieldset_nemo.U+fieldset_stokes.U, V=fieldset_nemo.V+fieldset_stokes.V)
        fU = fieldset.U[0]
    else:
        fieldset = fieldset_nemo
        fU = fieldset.U

    return fieldset, fU


def perIterGC():
    gc.collect()


class GalapagosParticle(JITParticle):
    age = Variable('age', initial=0.)
    life_expectancy = Variable('life_expectancy', dtype=np.float64, initial=14.0*86400.0)  # np.finfo(np.float64).max


def Age(particle, fieldset, time):
    if particle.state == StateCode.Evaluate:
        particle.age = particle.age + math.fabs(particle.dt)
    if particle.age > particle.life_expectancy:
        particle.delete()


def DeleteParticle(particle, fieldset, time):
    particle.delete()


def periodicBC(particle, fieldSet, time):
    dlon = -89.0 + 91.8
    dlat = 0.7 + 1.4
    if particle.lon > -89.0:
        particle.lon -= dlon
    if particle.lon < -91.8:
        particle.lon += dlon
    if particle.lat > 0.7:
        particle.lat -= dlat
    if particle.lat < -1.4:
        particle.lat += dlat


if __name__=='__main__':
    parser = ArgumentParser(description="Example of particle advection around an idealised peninsula")
    parser.add_argument("-s", "--stokes", dest="stokes", action='store_true', default=False, help="use Stokes' field data")
    parser.add_argument("-i", "--imageFileName", dest="imageFileName", type=str, default="mpiChunking_plot_MPI.png", help="image file name of the plot")
    parser.add_argument("-p", "--periodic", dest="periodic", action='store_true', default=False, help="enable/disable periodic wrapping (else: extrapolation)")
    # parser.add_argument("-t", "--time_in_days", dest="time_in_days", type=int, default=365, help="runtime in days (default: 365)")
    parser.add_argument("-t", "--time_in_days", dest="time_in_days", type=str, default="1*365", help="runtime in days (default: 1*365)")
    parser.add_argument("-G", "--GC", dest="useGC", action='store_true', default=False, help="using a garbage collector (default: false)")
    args = parser.parse_args()

    wstokes = args.stokes
    imageFileName=args.imageFileName
    periodicFlag=args.periodic
    time_in_days = int(float(eval(args.time_in_days)))
    with_GC = args.useGC

    headdir = ""
    odir = ""
    datahead = ""
    dirread_top = ""
    dirread_top_bgc = ""
    dirread_mesh = ""
    if os.uname()[1] in ['science-bs35', 'science-bs36']:  # Gemini
        # headdir = "/scratch/{}/experiments/palaeo-parcels".format(os.environ['USER'])
        headdir = "/scratch/{}/experiments/galapagos".format("ckehl")
        odir = os.path.join(headdir,"BENCHres")
        datahead = "/data/oceanparcels/input_data"
        ddir_head = os.path.join(datahead, 'NEMO-MEDUSA/ORCA0083-N006/')
    elif fnmatch.fnmatchcase(os.uname()[1], "*.bullx*"):  # Cartesius
        CARTESIUS_SCRATCH_USERNAME = 'ckehluu'
        headdir = "/scratch/shared/{}/experiments/galapagos".format(CARTESIUS_SCRATCH_USERNAME)
        odir = os.path.join(headdir, "/BENCHres")
        datahead = "/projects/0/topios/hydrodynamic_data"
        ddir_head = os.path.join(datahead, 'NEMO-MEDUSA/ORCA0083-N006/')
    else:
        headdir = "/var/scratch/galapagos"
        odir = os.path.join(headdir, "BENCHres")
        datahead = "/data"
        ddir_head = os.path.join(datahead, 'NEMO-MEDUSA/ORCA0083-N006/')




    fieldset, fU = create_galapagos_fieldset(datahead, True, wstokes)
    fname = os.path.join(odir,"galapagosparticles_bwd_wstokes_v2.nc") if wstokes else os.path.join(odir,"galapagosparticles_bwd_v2.nc")

    galapagos_extent = [-91.8, -89, -1.4, 0.7]
    startlon, startlat = np.meshgrid(np.arange(galapagos_extent[0], galapagos_extent[1], 0.2),
                                     np.arange(galapagos_extent[2], galapagos_extent[3], 0.2))

    pset = ParticleSet_Benchmark(fieldset=fieldset, pclass=GalapagosParticle, lon=startlon, lat=startlat, time=fU.grid.time[-1], repeatdt=delta(days=7))
    """ Kernal + Execution"""
    postProcessFuncs = []
    if with_GC:
        postProcessFuncs.append(perIterGC)
    outfile = pset.ParticleFile(name=fname, outputdt=delta(days=1))
    kernel = pset.Kernel(AdvectionRK4)+pset.Kernel(Age)+pset.Kernel(periodicBC)

    starttime = 0
    endtime = 0
    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        if mpi_rank==0:
            # global_t_0 = ostime.time()
            # starttime = MPI.Wtime()
            starttime = ostime.process_time()
    else:
        #starttime = ostime.time()
        starttime = ostime.process_time()

    pset.execute(kernel, dt=delta(hours=-1), output_file=outfile, recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle}, postIterationCallbacks=postProcessFuncs, callbackdt=delta(days=1))

    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        if mpi_rank==0:
            # global_t_0 = ostime.time()
            # endtime = MPI.Wtime()
            endtime = ostime.process_time()
    else:
        # endtime = ostime.time()
        endtime = ostime.process_time()

    size_Npart = len(pset.nparticle_log)
    Npart = pset.nparticle_log.get_param(size_Npart-1)
    if MPI:
        mpi_comm = MPI.COMM_WORLD
        Npart = mpi_comm.reduce(Npart, op=MPI.SUM, root=0)
        if mpi_comm.Get_rank() == 0:
            if size_Npart>0:
                sys.stdout.write("final # particles: {}\n".format( Npart ))
            sys.stdout.write("Time of pset.execute(): {} sec.\n".format(endtime-starttime))
            avg_time = np.mean(np.array(pset.total_log.get_values(), dtype=np.float64))
            sys.stdout.write("Avg. kernel update time: {} msec.\n".format(avg_time*1000.0))
    else:
        if size_Npart > 0:
            sys.stdout.write("final # particles: {}\n".format( Npart ))
        sys.stdout.write("Time of pset.execute(): {} sec.\n".format(endtime - starttime))
        avg_time = np.mean(np.array(pset.total_log.get_values(), dtype=np.float64))
        sys.stdout.write("Avg. kernel update time: {} msec.\n".format(avg_time * 1000.0))

    outfile.close()

    if MPI:
        mpi_comm = MPI.COMM_WORLD
        # mpi_comm.Barrier()
        Nparticles = mpi_comm.reduce(np.array(pset.nparticle_log.get_params()), op=MPI.SUM, root=0)
        Nmem = mpi_comm.reduce(np.array(pset.mem_log.get_params()), op=MPI.SUM, root=0)
        if mpi_comm.Get_rank() == 0:
            pset.plot_and_log(memory_used=Nmem, nparticles=Nparticles, target_N=1, imageFilePath=imageFileName, odir=odir)
    else:
        pset.plot_and_log(target_N=1, imageFilePath=imageFileName, odir=odir)
