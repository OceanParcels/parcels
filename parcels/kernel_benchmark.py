# import _ctypes
# import inspect
# import math  # noqa
# import random  # noqa
# import re
# import time
from ast import FunctionDef
# from ast import parse
# from copy import deepcopy
from ctypes import byref
from ctypes import c_double
from ctypes import c_int
# from ctypes import c_void_p
# from hashlib import md5
# from os import path
# from os import remove
# from sys import platform
# from sys import version_info

import numpy as np
# import numpy.ctypeslib as npct
try:
    from mpi4py import MPI
except:
    MPI = None

# from parcels.field import Field
from parcels.field import FieldOutOfBoundError
from parcels.field import FieldOutOfBoundSurfaceError
from parcels.field import TimeExtrapolationError
from parcels.field import NestedField
from parcels.field import SummedField
from parcels.field import VectorField
from parcels.tools.statuscodes import StateCode, OperationCode, ErrorCode
from parcels.tools.loggers import logger

from parcels.kernel import Kernel
from parcels.tools.performance_logger import TimingLog

__all__ = ['Kernel_Benchmark']


class Kernel_Benchmark(Kernel):
    """Kernel object that encapsulates auto-generated code.

    :arg fieldset: FieldSet object providing the field information
    :arg ptype: PType object for the kernel particle
    :param delete_cfiles: Boolean whether to delete the C-files after compilation in JIT mode (default is True)

    Note: A Kernel is either created from a compiled <function ...> object
    or the necessary information (funcname, funccode, funcvars) is provided.
    The py_ast argument may be derived from the code string, but for
    concatenation, the merged AST plus the new header definition is required.
    """

    def __init__(self, fieldset, ptype, pyfunc=None, funcname=None,
                 funccode=None, py_ast=None, funcvars=None, c_include="", delete_cfiles=True):
        super(Kernel_Benchmark, self).__init__(fieldset, ptype, pyfunc, funcname, funccode, py_ast, funcvars, c_include, delete_cfiles)
        self._compute_timings = TimingLog()
        self._io_timings = TimingLog()
        self._mem_io_timings = TimingLog()

    @property
    def io_timings(self):
        return self._io_timings

    @property
    def mem_io_timings(self):
        return self._mem_io_timings

    @property
    def compute_timings(self):
        return self._compute_timings

    def __del__(self):
        super(Kernel_Benchmark, self).__del__()

    def execute_jit(self, pset, endtime, dt):
        """Invokes JIT engine to perform the core update loop"""
        if len(pset) > 0 and pset.particle_data['xi'].ndim == 2 and pset.fieldset is not None:
            assert pset.fieldset.gridset.size == pset.particle_data['xi'].shape[1], \
                'FieldSet has different number of grids than Particle.xi. Have you added Fields after creating the ParticleSet?'

        if pset.fieldset is not None:
            self._io_timings.start_timing()
            for g in pset.fieldset.gridset.grids:
                g.cstruct = None  # This force to point newly the grids from Python to C

            # Make a copy of the transposed array to enforce
            # C-contiguous memory layout for JIT mode.
            for f in pset.fieldset.get_fields():
                if type(f) in [VectorField, NestedField, SummedField]:
                    continue
                if f in self.field_args.values():
                    f.chunk_data()
                else:
                    for block_id in range(len(f.data_chunks)):
                        f.data_chunks[block_id] = None
                        f.c_data_chunks[block_id] = None
            self._io_timings.stop_timing()
            self._io_timings.accumulate_timing()

            self._mem_io_timings.start_timing()
            for g in pset.fieldset.gridset.grids:
                g.load_chunk = np.where(g.load_chunk == 1, 2, g.load_chunk)
                if len(g.load_chunk) > 0:  # not the case if a field in not called in the kernel
                    if not g.load_chunk.flags.c_contiguous:
                        g.load_chunk = g.load_chunk.copy()
                if not g.depth.flags.c_contiguous:
                    g.depth = g.depth.copy()
                if not g.lon.flags.c_contiguous:
                    g.lon = g.lon.copy()
                if not g.lat.flags.c_contiguous:
                    g.lat = g.lat.copy()
            self._mem_io_timings.stop_timing()
            self._mem_io_timings.accumulate_timing()

        self._compute_timings.start_timing()
        fargs = []
        if pset.fieldset is not None:
            fargs += [byref(f.ctypes_struct) for f in self.field_args.values()]
        fargs += [c_double(f) for f in self.const_args.values()]
        # particle_data = pset._particle_data.ctypes.data_as(c_void_p)
        particle_data = byref(pset.ctypes_struct)
        result = self._function(c_int(len(pset)), particle_data,
                                c_double(endtime),
                                c_double(dt),
                                *fargs)
        self._compute_timings.stop_timing()
        self._compute_timings.accumulate_timing()

        self._io_timings.advance_iteration()
        self._mem_io_timings.advance_iteration()
        self._compute_timings.advance_iteration()
        return result

    def execute_python(self, pset, endtime, dt):
        """Performs the core update loop via Python"""
        sign_dt = np.sign(dt)

        if 'AdvectionAnalytical' in self.pyfunc.__name__:
            analytical = True
            if not np.isinf(dt):
                logger.warning_once('dt is not used in AnalyticalAdvection, so is set to np.inf')
            dt = np.inf
        else:
            analytical = False

        particles = pset.data_accessor()

        # back up variables in case of ErrorCode.Repeat
        p_var_back = {}

        for f in self.fieldset.get_fields():
            if type(f) in [VectorField, NestedField, SummedField]:
                continue

            self._io_timings.start_timing()
            loaded_data = f.data
            self._io_timings.stop_timing()
            self._io_timings.accumulate_timing()
            self._mem_io_timings.start_timing()
            f.data = np.array(loaded_data)
            self._mem_io_timings.stop_timing()
            self._mem_io_timings.accumulate_timing()

        self._compute_timings.start_timing()
        # for p in pset.particles:
        for p in range(pset.size):
            particles.set_index(p)

            ptype = pset.ptype
            # Don't execute particles that aren't started yet
            # sign_end_part = np.sign(endtime - p.time)
            sign_end_part = np.sign(endtime - particles.time)

            # Compute min/max dt for first timestep
            # dt_pos = min(abs(p.dt), abs(endtime - p.time))
            dt_pos = min(abs(particles.dt), abs(endtime - particles.time))

            # ==== numerically stable; also making sure that continuously-recovered particles do end successfully,
            # as they fulfil the condition here on entering at the final calculation here. ==== #
            if ((sign_end_part != sign_dt) or np.isclose(dt_pos, 0)) and not np.isclose(dt, 0):
                if abs(particles.time) >= abs(endtime):
                    particles.set_state(StateCode.Success)
                # if abs(p.time) >= abs(endtime):
                #     p.state = ErrorCode.Success
                continue

            # while p.state in [ErrorCode.Evaluate, ErrorCode.Repeat] or np.isclose(dt, 0):
            while particles.state in [StateCode.Evaluate, OperationCode.Repeat] or np.isclose(dt, 0):

                for var in ptype.variables:
                    # p_var_back[var.name] = getattr(p, var.name)
                    p_var_back[var.name] = getattr(particles, var.name)
                try:
                    pdt_prekernels = sign_dt * dt_pos
                    p.dt = pdt_prekernels
                    state_prev = p.state
                    # res = self.pyfunc(p, pset.fieldset, p.time)
                    res = self.pyfunc(particles, pset.fieldset, particles.time)
                    if res is None:
                        res = StateCode.Success

                    # if res is StateCode.Success and p.state != state_prev:
                    #     res = p.state
                    if res is StateCode.Success and particles.state != state_prev:
                        res = particles.state

                    # if res == ErrorCode.Success and not np.isclose(p.dt, pdt_prekernels):
                    #     res = ErrorCode.Repeat
                    if not analytical and res == StateCode.Success and not np.isclose(particles.dt, pdt_prekernels):
                        res = OperationCode.Repeat

                except FieldOutOfBoundError as fse_xy:
                    res = ErrorCode.ErrorOutOfBounds
                    particles.exception = fse_xy
                except FieldOutOfBoundSurfaceError as fse_z:
                    res = ErrorCode.ErrorThroughSurface
                    particles.exception = fse_z
                except TimeExtrapolationError as fse_t:
                    res = ErrorCode.ErrorTimeExtrapolation
                    particles.exception = fse_t
                except Exception as e:
                    res = ErrorCode.Error
                    particles.exception = e

                # Handle particle time and time loop
                # if res in [ErrorCode.Success, ErrorCode.Delete]:
                if res in [StateCode.Success, OperationCode.Delete]:
                    # Update time and repeat
                    particles.time += p.dt
                    particles.update_next_dt()
                    if analytical:
                        particles.dt = np.inf
                    dt_pos = min(abs(particles.dt), abs(endtime - particles.time))

                    sign_end_part = np.sign(endtime - particles.time)
                    # if res != ErrorCode.Delete and not np.isclose(dt_pos, 0) and (sign_end_part == sign_dt):
                    #     res = ErrorCode.Evaluate
                    if res != OperationCode.Delete and not np.isclose(dt_pos, 0) and (sign_end_part == sign_dt):
                        res = StateCode.Evaluate
                    if sign_end_part != sign_dt:
                        dt_pos = 0

                    p.state = res
                    if np.isclose(dt, 0):
                        break
                else:
                    # p.state = res
                    particles.set_state(res)
                    # Try again without time update
                    for var in ptype.variables:
                        if var.name not in ['dt', 'state']:
                            setattr(particles, var.name, p_var_back[var.name])
                    dt_pos = min(abs(particles.dt), abs(endtime - particles.time))

                    sign_end_part = np.sign(endtime - particles.time)
                    if sign_end_part != sign_dt:
                        dt_pos = 0
                    break
        self._compute_timings.stop_timing()
        self._compute_timings.accumulate_timing()

        self._io_timings.advance_iteration()
        self._mem_io_timings.advance_iteration()
        self._compute_timings.advance_iteration()

    def remove_deleted(self, pset, output_file, endtime):
        """Utility to remove all particles that signalled deletion"""
        self._mem_io_timings.start_timing()
        super(Kernel_Benchmark, self).remove_deleted(pset=pset, output_file=output_file, endtime=endtime)
        self._mem_io_timings.stop_timing()
        self._mem_io_timings.accumulate_timing()
        self._mem_io_timings.advance_iteration()

    def merge(self, kernel):
        funcname = self.funcname + kernel.funcname
        func_ast = FunctionDef(name=funcname, args=self.py_ast.args,
                               body=self.py_ast.body + kernel.py_ast.body,
                               decorator_list=[], lineno=1, col_offset=0)
        delete_cfiles = self.delete_cfiles and kernel.delete_cfiles
        return Kernel_Benchmark(self.fieldset, self.ptype, pyfunc=None,
                                funcname=funcname, funccode=self.funccode + kernel.funccode,
                                py_ast=func_ast, funcvars=self.funcvars + kernel.funcvars,
                                delete_cfiles=delete_cfiles)

    def __add__(self, kernel):
        if not isinstance(kernel, Kernel):
            kernel = Kernel_Benchmark(self.fieldset, self.ptype, pyfunc=kernel)
        return self.merge(kernel)

    def __radd__(self, kernel):
        if not isinstance(kernel, Kernel):
            kernel = Kernel_Benchmark(self.fieldset, self.ptype, pyfunc=kernel)
        return kernel.merge(self)
