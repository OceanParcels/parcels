"""
Author: Dr. Christian Kehl
Date: 11-02-2020
"""

from parcels import AdvectionEE, AdvectionRK45, AdvectionRK4
from parcels import FieldSet, ScipyParticle, JITParticle, Variable, AdvectionRK4, StateCode, OperationCode, ErrorCode
from parcels.particleset_benchmark import ParticleSet_Benchmark as BenchmarkParticleSet
from parcels.particleset import ParticleSet as DryParticleSet
from parcels.field import Field, VectorField, NestedField, SummedField
# from parcels import plotTrajectoriesFile_loadedField
# from parcels import rng as random
from parcels import ParcelsRandom
from datetime import timedelta as delta
import math
from argparse import ArgumentParser
import datetime
import numpy as np
import xarray as xr
import fnmatch
import sys
import gc
import os
import time as ostime
# import matplotlib.pyplot as plt
from parcels.tools import perlin3d
from parcels.tools import perlin2d

try:
    from mpi4py import MPI
except:
    MPI = None
with_GC = False

pset = None
ptype = {'scipy': ScipyParticle, 'jit': JITParticle}
method = {'RK4': AdvectionRK4, 'EE': AdvectionEE, 'RK45': AdvectionRK45}
global_t_0 = 0
Nparticle = int(math.pow(2,10)) # equals to Nparticle = 1024
#Nparticle = int(math.pow(2,11)) # equals to Nparticle = 2048
#Nparticle = int(math.pow(2,12)) # equals to Nparticle = 4096
#Nparticle = int(math.pow(2,13)) # equals to Nparticle = 8192
#Nparticle = int(math.pow(2,14)) # equals to Nparticle = 16384
#Nparticle = int(math.pow(2,15)) # equals to Nparticle = 32768
#Nparticle = int(math.pow(2,16)) # equals to Nparticle = 65536
#Nparticle = int(math.pow(2,17)) # equals to Nparticle = 131072
#Nparticle = int(math.pow(2,18)) # equals to Nparticle = 262144
#Nparticle = int(math.pow(2,19)) # equals to Nparticle = 524288

noctaves=3
#noctaves=4 # formerly
perlinres=(1,32,8)
shapescale=(4,8,8)
#shapescale=(8,6,6) # formerly
perlin_persistence=0.3
a = 1000 * 1e3
b = 1000 * 1e3
scalefac = 2.0
tsteps = 61
tscale = 6

# Idea for 4D: perlin3D creates a time-consistent 3D field
# Thus, we can use skimage to create shifted/rotated/morphed versions
# for the depth domain, so that perlin4D = [depth][time][lat][lon].
# then, we can do a transpose-op in numpy, to get [time][depth][lat][lon]

# we need to modify the kernel.execute / pset.execute so that it returns from the JIT
# in a given time WITHOUT writing to disk via outfie => introduce a pyloop_dt

def DeleteParticle(particle, fieldset, time):
    particle.delete()

def RenewParticle(particle, fieldset, time):
    particle.lat = np.random.rand() * a
    particle.lon = np.random.rand() * b

def perIterGC():
    gc.collect()

def perlin_fieldset_from_numpy(periodic_wrap=False):
    """Simulate a current from structured random noise (i.e. Perlin noise).
    we use the external package 'perlin-numpy' as field generator, see:
    https://github.com/pvigier/perlin-numpy

    Perlin noise was introduced in the literature here:
    Perlin, Ken (July 1985). "An Image Synthesizer". SIGGRAPH Comput. Graph. 19 (97–8930), p. 287–296.
    doi:10.1145/325165.325247, https://dl.acm.org/doi/10.1145/325334.325247
    """
    img_shape = (int(math.pow(2,noctaves))*perlinres[1]*shapescale[1], int(math.pow(2,noctaves))*perlinres[2]*shapescale[2])

    # Coordinates of the test fieldset (on A-grid in deg)
    lon = np.linspace(-a*0.5, a*0.5, img_shape[0], dtype=np.float32)
    # sys.stdout.write("lon field: {}\n".format(lon.size))
    lat = np.linspace(-b*0.5, b*0.5, img_shape[1], dtype=np.float32)
    # sys.stdout.write("lat field: {}\n".format(lat.size))
    totime = tsteps*tscale*24.0*60.0*60.0
    time = np.linspace(0., totime, tsteps, dtype=np.float64)
    # sys.stdout.write("time field: {}\n".format(time.size))

    # Define arrays U (zonal), V (meridional)
    U = perlin2d.generate_fractal_noise_temporal2d(img_shape, tsteps, (perlinres[1], perlinres[2]), noctaves, perlin_persistence, max_shift=((-1, 2), (-1, 2)))
    U = np.transpose(U, (0,2,1))
    # U = np.swapaxes(U, 1, 2)
    # print("U-statistics - min: {:10.7f}; max: {:10.7f}; avg. {:10.7f}; std_dev: {:10.7f}".format(U.min(initial=0), U.max(initial=0), U.mean(), U.std()))
    V = perlin2d.generate_fractal_noise_temporal2d(img_shape, tsteps, (perlinres[1], perlinres[2]), noctaves, perlin_persistence, max_shift=((-1, 2), (-1, 2)))
    V = np.transpose(V, (0,2,1))
    # V = np.swapaxes(V, 1, 2)
    # print("V-statistics - min: {:10.7f}; max: {:10.7f}; avg. {:10.7f}; std_dev: {:10.7f}".format(V.min(initial=0), V.max(initial=0), V.mean(), V.std()))

    # U = perlin3d.generate_fractal_noise_3d(img_shape, perlinres, noctaves, perlin_persistence) * scalefac
    # U = np.transpose(U, (0,2,1))
    # sys.stdout.write("U field shape: {} - [tdim][ydim][xdim]=[{}][{}][{}]\n".format(U.shape, time.shape[0], lat.shape[0], lon.shape[0]))
    # V = perlin3d.generate_fractal_noise_3d(img_shape, perlinres, noctaves, perlin_persistence) * scalefac
    # V = np.transpose(V, (0,2,1))
    # sys.stdout.write("V field shape: {} - [tdim][ydim][xdim]=[{}][{}][{}]\n".format(V.shape, time.shape[0], lat.shape[0], lon.shape[0]))

    data = {'U': U, 'V': V}
    dimensions = {'time': time, 'lon': lon, 'lat': lat}
    if periodic_wrap:
        return FieldSet.from_data(data, dimensions, mesh='flat', transpose=False, time_periodic=delta(days=1))
    else:
        return FieldSet.from_data(data, dimensions, mesh='flat', transpose=False, allow_time_extrapolation=True)


def perlin_fieldset_from_xarray(periodic_wrap=False):
    """Simulate a current from structured random noise (i.e. Perlin noise).
    we use the external package 'perlin-numpy' as field generator, see:
    https://github.com/pvigier/perlin-numpy

    Perlin noise was introduced in the literature here:
    Perlin, Ken (July 1985). "An Image Synthesizer". SIGGRAPH Comput. Graph. 19 (97–8930), p. 287–296.
    doi:10.1145/325165.325247, https://dl.acm.org/doi/10.1145/325334.325247
    """
    img_shape = (perlinres[0]*shapescale[0], int(math.pow(2,noctaves))*perlinres[1]*shapescale[1], int(math.pow(2,noctaves))*perlinres[2]*shapescale[2])

    # Coordinates of the test fieldset (on A-grid in deg)
    lon = np.linspace(0, a, img_shape[1], dtype=np.float32)
    lat = np.linspace(0, b, img_shape[2], dtype=np.float32)
    totime = img_shape[0] * 24.0 * 60.0 * 60.0
    time = np.linspace(0., totime, img_shape[0], dtype=np.float32)

    # Define arrays U (zonal), V (meridional), W (vertical) and P (sea
    # surface height) all on A-grid
    U = perlin3d.generate_fractal_noise_3d(img_shape, perlinres, noctaves, perlin_persistence) * scalefac
    U = np.transpose(U, (0,2,1))
    V = perlin3d.generate_fractal_noise_3d(img_shape, perlinres, noctaves, perlin_persistence) * scalefac
    V = np.transpose(V, (0,2,1))
    #P = perlin3d.generate_fractal_noise_3d(img_shape, perlinres, noctaves, perlin_persistence) * scalefac

    dimensions = {'time': time, 'lon': lon, 'lat': lat}
    dims = ('time', 'lat', 'lon')
    data = {'Uxr': xr.DataArray(U, coords=dimensions, dims=dims),
            'Vxr': xr.DataArray(V, coords=dimensions, dims=dims)}   #,'Pxr': xr.DataArray(P, coords=dimensions, dims=dims)
    ds = xr.Dataset(data)

    variables = {'U': 'Uxr', 'V': 'Vxr'}
    dimensions = {'time': 'time', 'lat': 'lat', 'lon': 'lon'}
    if periodic_wrap:
        return FieldSet.from_xarray_dataset(ds, variables, dimensions, mesh='flat', time_periodic=delta(days=1))
    else:
        return FieldSet.from_xarray_dataset(ds, variables, dimensions, mesh='flat', allow_time_extrapolation=True)

class AgeParticle_JIT(JITParticle):
    age = Variable('age', dtype=np.float64, initial=0.0)
    life_expectancy = Variable('life_expectancy', dtype=np.float64, initial=np.finfo(np.float64).max)
    initialized_dynamic = Variable('initialized_dynamic', dtype=np.int32, initial=0)

class AgeParticle_SciPy(ScipyParticle):
    age = Variable('age', dtype=np.float64, initial=0.0)
    life_expectancy = Variable('life_expectancy', dtype=np.float64, initial=np.finfo(np.float64).max)
    initialized_dynamic = Variable('initialized_dynamic', dtype=np.int32, initial=0)

def initialize(particle, fieldset, time):
    if particle.initialized_dynamic < 1:
        np_scaler = math.sqrt(3.0 / 2.0)
        particle.life_expectancy = time + ParcelsRandom.uniform(.0, (fieldset.life_expectancy-time) * 2.0 / np_scaler)
        # particle.life_expectancy = time + ParcelsRandom.uniform(.0, (fieldset.life_expectancy-time)*math.sqrt(3.0 / 2.0))
        # particle.life_expectancy = time + ParcelsRandom.uniform(.0, fieldset.life_expectancy) * math.sqrt(3.0 / 2.0)
        # particle.life_expectancy = time+ParcelsRandom.uniform(.0, fieldset.life_expectancy) * ((3.0/2.0)**2.0)
        particle.initialized_dynamic = 1

def Age(particle, fieldset, time):
    if particle.state == StateCode.Evaluate:
        particle.age = particle.age + math.fabs(particle.dt)
    if particle.age > particle.life_expectancy:
        particle.delete()

age_ptype = {'scipy': AgeParticle_SciPy, 'jit': AgeParticle_JIT}

if __name__=='__main__':
    parser = ArgumentParser(description="Example of particle advection using in-memory stommel test case")
    parser.add_argument("-i", "--imageFileName", dest="imageFileName", type=str, default="mpiChunking_plot_MPI.png", help="image file name of the plot")
    parser.add_argument("-b", "--backwards", dest="backwards", action='store_true', default=False, help="enable/disable running the simulation backwards")
    parser.add_argument("-p", "--periodic", dest="periodic", action='store_true', default=False, help="enable/disable periodic wrapping (else: extrapolation)")
    parser.add_argument("-r", "--release", dest="release", action='store_true', default=False, help="continuously add particles via repeatdt (default: False)")
    parser.add_argument("-rt", "--releasetime", dest="repeatdt", type=int, default=720, help="repeating release rate of added particles in Minutes (default: 720min = 12h)")
    parser.add_argument("-a", "--aging", dest="aging", action='store_true', default=False, help="Removed aging particles dynamically (default: False)")
    parser.add_argument("-t", "--time_in_days", dest="time_in_days", type=int, default=1, help="runtime in days (default: 1)")
    parser.add_argument("-x", "--xarray", dest="use_xarray", action='store_true', default=False, help="use xarray as data backend")
    parser.add_argument("-w", "--writeout", dest="write_out", action='store_true', default=False, help="write data in outfile")
    parser.add_argument("-d", "--delParticle", dest="delete_particle", action='store_true', default=False, help="switch to delete a particle (True) or reset a particle (default: False).")
    parser.add_argument("-A", "--animate", dest="animate", action='store_true', default=False, help="animate the particle trajectories during the run or not (default: False).")
    parser.add_argument("-V", "--visualize", dest="visualize", action='store_true', default=False, help="Visualize particle trajectories at the end (default: False). Requires -w in addition to take effect.")
    parser.add_argument("-N", "--n_particles", dest="nparticles", type=str, default="2**6", help="number of particles to generate and advect (default: 2e6)")
    parser.add_argument("-sN", "--start_n_particles", dest="start_nparticles", type=str, default="96", help="(optional) number of particles generated per release cycle (if --rt is set) (default: 96)")
    parser.add_argument("-m", "--mode", dest="compute_mode", choices=['jit','scipy'], default="jit", help="computation mode = [JIT, SciPp]")
    parser.add_argument("-G", "--GC", dest="useGC", action='store_true', default=False, help="using a garbage collector (default: false)")
    parser.add_argument("--dry", dest="dryrun", action="store_true", default=False, help="Start dry run (no benchmarking and its classes")
    args = parser.parse_args()

    ParticleSet = BenchmarkParticleSet
    if args.dryrun:
        ParticleSet = DryParticleSet

    imageFileName=args.imageFileName
    periodicFlag=args.periodic
    backwardSimulation = args.backwards
    repeatdtFlag=args.release
    repeatRateMinutes=args.repeatdt
    time_in_days = args.time_in_days
    use_xarray = args.use_xarray
    agingParticles = args.aging
    with_GC = args.useGC
    Nparticle = int(float(eval(args.nparticles)))
    target_N = Nparticle
    addParticleN = 1
    # np_scaler = math.sqrt(3.0/2.0)
    # np_scaler = (3.0 / 2.0)**2.0       # **
    np_scaler = 3.0 / 2.0
    # cycle_scaler = math.sqrt(3.0/2.0)
    # cycle_scaler = (3.0 / 2.0)**2.0    # **
    # cycle_scaler = 3.0 / 2.0
    cycle_scaler = 7.0 / 4.0
    start_N_particles = int(float(eval(args.start_nparticles)))
    if MPI:
        mpi_comm = MPI.COMM_WORLD
        if mpi_comm.Get_rank() == 0:
            if agingParticles and not repeatdtFlag:
                sys.stdout.write("N: {} ( {} )\n".format(Nparticle, int(Nparticle * np_scaler)))
            else:
                sys.stdout.write("N: {}\n".format(Nparticle))
    else:
        if agingParticles and not repeatdtFlag:
            sys.stdout.write("N: {} ( {} )\n".format(Nparticle, int(Nparticle * np_scaler)))
        else:
            sys.stdout.write("N: {}\n".format(Nparticle))

    dt_minutes = 60
    #dt_minutes = 20
    #random.seed(123456)
    nowtime = datetime.datetime.now()
    ParcelsRandom.seed(nowtime.microsecond)

    odir = ""
    if os.uname()[1] in ['science-bs35', 'science-bs36']:  # Gemini
        # odir = "/scratch/{}/experiments".format(os.environ['USER'])
        odir = "/scratch/{}/experiments".format("ckehl")
    elif fnmatch.fnmatchcase(os.uname()[1], "*.bullx*"):  # Cartesius
        CARTESIUS_SCRATCH_USERNAME = 'ckehluu'
        odir = "/scratch/shared/{}/experiments".format(CARTESIUS_SCRATCH_USERNAME)
    else:
        odir = "/var/scratch/experiments"

    func_time = []
    mem_used_GB = []

    np.random.seed(0)
    fieldset = None
    if use_xarray:
        fieldset = perlin_fieldset_from_xarray(periodic_wrap=periodicFlag)
    else:
        fieldset = perlin_fieldset_from_numpy(periodic_wrap=periodicFlag)

    if args.compute_mode is 'scipy':
        Nparticle = 2**10

    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        if mpi_rank==0:
            global_t_0 = ostime.process_time()
    else:
        global_t_0 = ostime.process_time()

    simStart = None
    for f in fieldset.get_fields():
        if type(f) in [VectorField, NestedField, SummedField]:  # or not f.grid.defer_load
            continue
        else:
            if backwardSimulation:
                simStart=f.grid.time_full[-1]
            else:
                simStart = f.grid.time_full[0]
            break

    if agingParticles:
        if not repeatdtFlag:
            Nparticle = int(Nparticle * np_scaler)
        fieldset.add_constant('life_expectancy', delta(days=time_in_days).total_seconds())
    if repeatdtFlag:
        addParticleN = Nparticle/2.0
        refresh_cycle = (delta(days=time_in_days).total_seconds() / (addParticleN/start_N_particles)) / cycle_scaler
        if agingParticles:
            refresh_cycle /= cycle_scaler
        repeatRateMinutes = int(refresh_cycle/60.0) if repeatRateMinutes == 720 else repeatRateMinutes

    if backwardSimulation:
        # ==== backward simulation ==== #
        if agingParticles:
            if repeatdtFlag:
                pset = ParticleSet(fieldset=fieldset, pclass=age_ptype[(args.compute_mode).lower()], lon=np.random.rand(start_N_particles, 1) * a, lat=np.random.rand(start_N_particles, 1) * b, time=simStart, repeatdt=delta(minutes=repeatRateMinutes))
                psetA = ParticleSet(fieldset=fieldset, pclass=age_ptype[(args.compute_mode).lower()], lon=np.random.rand(int(addParticleN), 1) * a, lat=np.random.rand(int(addParticleN), 1) * b, time=simStart)
                pset.add(psetA)
            else:
                pset = ParticleSet(fieldset=fieldset, pclass=age_ptype[(args.compute_mode).lower()], lon=np.random.rand(Nparticle, 1) * a, lat=np.random.rand(Nparticle, 1) * b, time=simStart)
        else:
            if repeatdtFlag:
                pset = ParticleSet(fieldset=fieldset, pclass=ptype[(args.compute_mode).lower()], lon=np.random.rand(start_N_particles, 1) * a, lat=np.random.rand(start_N_particles, 1) * b, time=simStart, repeatdt=delta(minutes=repeatRateMinutes))
                psetA = ParticleSet(fieldset=fieldset, pclass=ptype[(args.compute_mode).lower()], lon=np.random.rand(int(addParticleN), 1) * a, lat=np.random.rand(int(addParticleN), 1) * b, time=simStart)
                pset.add(psetA)
            else:
                pset = ParticleSet(fieldset=fieldset, pclass=ptype[(args.compute_mode).lower()], lon=np.random.rand(Nparticle, 1) * a, lat=np.random.rand(Nparticle, 1) * b, time=simStart)
    else:
        # ==== forward simulation ==== #
        if agingParticles:
            if repeatdtFlag:
                pset = ParticleSet(fieldset=fieldset, pclass=age_ptype[(args.compute_mode).lower()], lon=np.random.rand(start_N_particles, 1) * a, lat=np.random.rand(start_N_particles, 1) * b, time=simStart, repeatdt=delta(minutes=repeatRateMinutes))
                psetA = ParticleSet(fieldset=fieldset, pclass=age_ptype[(args.compute_mode).lower()], lon=np.random.rand(int(addParticleN), 1) * a, lat=np.random.rand(int(addParticleN), 1) * b, time=simStart)
                pset.add(psetA)
            else:
                pset = ParticleSet(fieldset=fieldset, pclass=age_ptype[(args.compute_mode).lower()], lon=np.random.rand(Nparticle, 1) * a, lat=np.random.rand(Nparticle, 1) * b, time=simStart)
        else:
            if repeatdtFlag:
                pset = ParticleSet(fieldset=fieldset, pclass=ptype[(args.compute_mode).lower()], lon=np.random.rand(start_N_particles, 1) * a, lat=np.random.rand(start_N_particles, 1) * b, time=simStart, repeatdt=delta(minutes=repeatRateMinutes))
                psetA = ParticleSet(fieldset=fieldset, pclass=ptype[(args.compute_mode).lower()], lon=np.random.rand(int(addParticleN), 1) * a, lat=np.random.rand(int(addParticleN), 1) * b, time=simStart)
                pset.add(psetA)
            else:
                pset = ParticleSet(fieldset=fieldset, pclass=ptype[(args.compute_mode).lower()], lon=np.random.rand(Nparticle, 1) * a, lat=np.random.rand(Nparticle, 1) * b, time=simStart)

    output_file = None
    out_fname = "benchmark_perlin"
    if args.write_out:
        if MPI and (MPI.COMM_WORLD.Get_size()>1):
            out_fname += "_MPI"
        else:
            out_fname += "_noMPI"
        out_fname += "_n"+str(Nparticle)
        if backwardSimulation:
            out_fname += "_bwd"
        else:
            out_fname += "_fwd"
        if repeatdtFlag:
            out_fname += "_add"
        if agingParticles:
            out_fname += "_age"
        output_file = pset.ParticleFile(name=os.path.join(odir,out_fname+".nc"), outputdt=delta(hours=24))
    delete_func = RenewParticle
    if args.delete_particle:
        delete_func=DeleteParticle
    postProcessFuncs = []

    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        if mpi_rank==0:
            starttime = ostime.process_time()
    else:
        starttime = ostime.process_time()
    kernels = pset.Kernel(AdvectionRK4,delete_cfiles=True)
    if agingParticles:
        kernels += pset.Kernel(initialize, delete_cfiles=True)
        kernels += pset.Kernel(Age, delete_cfiles=True)
    if with_GC:
        postProcessFuncs.append(perIterGC)
    if backwardSimulation:
        # ==== backward simulation ==== #
        if args.animate:
            pset.execute(kernels, runtime=delta(days=time_in_days), dt=delta(minutes=-dt_minutes), output_file=output_file, recovery={ErrorCode.ErrorOutOfBounds: delete_func}, postIterationCallbacks=postProcessFuncs, callbackdt=delta(hours=12), moviedt=delta(hours=6), movie_background_field=fieldset.U)
        else:
            pset.execute(kernels, runtime=delta(days=time_in_days), dt=delta(minutes=-dt_minutes), output_file=output_file, recovery={ErrorCode.ErrorOutOfBounds: delete_func}, postIterationCallbacks=postProcessFuncs, callbackdt=delta(hours=12))
    else:
        # ==== forward simulation ==== #
        if args.animate:
            pset.execute(kernels, runtime=delta(days=time_in_days), dt=delta(minutes=dt_minutes), output_file=output_file, recovery={ErrorCode.ErrorOutOfBounds: delete_func}, postIterationCallbacks=postProcessFuncs, callbackdt=delta(hours=12), moviedt=delta(hours=6), movie_background_field=fieldset.U)
        else:
            pset.execute(kernels, runtime=delta(days=time_in_days), dt=delta(minutes=dt_minutes), output_file=output_file, recovery={ErrorCode.ErrorOutOfBounds: delete_func}, postIterationCallbacks=postProcessFuncs, callbackdt=delta(hours=12))
    if MPI:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
        if mpi_rank==0:
            endtime = ostime.process_time()
    else:
        endtime = ostime.process_time()

    if not args.dryrun:
        if MPI:
            mpi_comm = MPI.COMM_WORLD
            # mpi_comm.Barrier()
            size_Npart = len(pset.nparticle_log)
            Npart = pset.nparticle_log.get_param(size_Npart-1)
            Npart = mpi_comm.reduce(Npart, op=MPI.SUM, root=0)
            if mpi_comm.Get_rank() == 0:
                if size_Npart>0:
                    sys.stdout.write("final # particles: {}\n".format( Npart ))
                sys.stdout.write("Time of pset.execute(): {} sec.\n".format(endtime-starttime))
                avg_time = np.mean(np.array(pset.total_log.get_values(), dtype=np.float64))
                sys.stdout.write("Avg. kernel update time: {} msec.\n".format(avg_time*1000.0))
        else:
            size_Npart = len(pset.nparticle_log)
            Npart = pset.nparticle_log.get_param(size_Npart-1)
            if size_Npart > 0:
                sys.stdout.write("final # particles: {}\n".format( Npart ))
            sys.stdout.write("Time of pset.execute(): {} sec.\n".format(endtime - starttime))
            avg_time = np.mean(np.array(pset.total_log.get_values(), dtype=np.float64))
            sys.stdout.write("Avg. kernel update time: {} msec.\n".format(avg_time * 1000.0))

        if MPI:
            mpi_comm = MPI.COMM_WORLD
            Nparticles = mpi_comm.reduce(np.array(pset.nparticle_log.get_params()), op=MPI.SUM, root=0)
            Nmem = mpi_comm.reduce(np.array(pset.mem_log.get_params()), op=MPI.SUM, root=0)
            if mpi_comm.Get_rank() == 0:
                pset.plot_and_log(memory_used=Nmem, nparticles=Nparticles, target_N=target_N, imageFilePath=imageFileName, odir=odir, xlim_range=[0, 730], ylim_range=[0, 150])
        else:
            pset.plot_and_log(target_N=target_N, imageFilePath=imageFileName, odir=odir, xlim_range=[0, 730], ylim_range=[0, 150])


