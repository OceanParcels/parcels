import inspect
import re
import math  # noga
import random  # noga
from ast import parse
from copy import deepcopy
from ctypes import byref
from ctypes import c_double
from ctypes import c_int
# from ctypes import c_void_p
from ctypes import pointer
from os import path
from sys import version_info
import numpy as np

from parcels import Field, NestedField, SummedField, VectorField
from parcels import ErrorCode
from parcels.field import FieldOutOfBoundError, FieldOutOfBoundSurfaceError, TimeExtrapolationError
from parcels.tools.global_statics import get_cache_dir, get_package_dir
from parcels.wrapping import KernelGenerator, NodeLoopGenerator
from parcels.tools.error import recovery_map as recovery_base_map
from parcels import AdvectionRK4_3D, logger

from parcels.kernel_node import Kernel
from parcels.tools.performance_logger import TimingLog

__all__ = ['Kernel_Benchmark']


class Kernel_Benchmark(Kernel):
    def __init__(self, fieldset, ptype, pyfunc=None, funcname=None, funccode=None, py_ast=None, funcvars=None,
                 c_include="", delete_cfiles=True):
        super(Kernel_Benchmark, self).__init__(fieldset, ptype, pyfunc=pyfunc, funcname=funcname, funccode=funccode, py_ast=py_ast, funcvars=funcvars, c_include=c_include, delete_cfiles=delete_cfiles)
        self._compute_timings = TimingLog()
        self._io_timings = TimingLog()
        self._mem_io_log = TimingLog()

    def __del__(self):
        # Clean-up the in-memory dynamic linked libraries.
        # This is not really necessary, as these programs are not that large, but with the new random
        # naming scheme which is required on Windows OS'es to deal with updates to a Parcels' kernel.
        super(Kernel_Benchmark, self).__del__()

    @property
    def io_timings(self):
        return self._io_timings

    @property
    def mem_io_timings(self):
        return self._mem_io_log

    @property
    def compute_timings(self):
        return self._compute_timings

    def __add__(self, kernel):
        if not isinstance(kernel, Kernel):
            kernel = Kernel_Benchmark(self.fieldset, self.ptype, pyfunc=kernel)
        return self.merge(kernel, Kernel_Benchmark)

    def __radd__(self, kernel):
        if not isinstance(kernel, Kernel):
            kernel = Kernel_Benchmark(self.fieldset, self.ptype, pyfunc=kernel)
        return kernel.merge(self, Kernel_Benchmark)

    def execute_jit(self, pset, endtime, dt):
        """Invokes JIT engine to perform the core update loop"""
        self._io_timings.start_timing()
        if len(pset) > 0:
            assert pset.fieldset.gridset.size == len(pset[0].data.xi), \
                'FieldSet has different amount of grids than Particle.xi. Have you added Fields after creating the ParticleSet?'
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

        self._mem_io_log.start_timing()
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

        fargs = []
        if self.field_args is not None:
            # ==== major runtime expense in this function, for plain advection ==== #
            fargs += [byref(f.ctypes_struct) for f in self.field_args.values()]
        if self.const_args is not None:
            fargs += [c_double(f) for f in self.const_args.values()]
        self._mem_io_log.stop_timing()
        self._mem_io_log.accumulate_timing()

        self._compute_timings.start_timing()
        # particle_data = pset._particle_data.ctypes.data_as(c_void_p)
        node_data = pset.begin()
        if len(fargs) > 0:
            self._function(c_int(len(pset)), pointer(node_data), c_double(endtime), c_double(dt), *fargs)
        else:
            self._function(c_int(len(pset)), pointer(node_data), c_double(endtime), c_double(dt))
        self._compute_timings.stop_timing()
        self._compute_timings.accumulate_timing()

        self._io_timings.advance_iteration()
        self._mem_io_log.advance_iteration()
        self._compute_timings.advance_iteration()

    def execute_python(self, pset, endtime, dt):
        """Performs the core update loop via Python"""
        sign_dt = np.sign(dt)

        # back up variables in case of ErrorCode.Repeat
        p_var_back = {}

        for f in self.fieldset.get_fields():
            if type(f) in [VectorField, NestedField, SummedField]:
                continue

            self._io_timings.start_timing()
            loaded_data = f.data
            self._io_timings.stop_timing()
            self._io_timings.accumulate_timing()
            self._mem_io_log.start_timing()
            f.data = np.array(loaded_data)
            self._mem_io_log.stop_timing()
            self._mem_io_log.accumulate_timing()

        self._compute_timings.start_timing()
        # ========= OLD ======= #
        # for p in pset.particles:
        # ===================== #
        node = pset.begin()
        while node is not None:
            p = node.data
            ptype = p.getPType()
            # Don't execute particles that aren't started yet
            sign_end_part = np.sign(endtime - p.time)
            dt_pos = min(abs(p.dt), abs(endtime - p.time))

            # ==== numerically stable; also making sure that continuously-recovered particles do end successfully,
            # as they fulfil the condition here on entering at the final calculation here. ==== #
            if ((sign_end_part != sign_dt) or np.isclose(dt_pos, 0)) and not np.isclose(dt, 0):
                if abs(p.time) >= abs(endtime):
                    p.state = ErrorCode.Success
                node = node.next
                continue

            # Compute min/max dt for first timestep
            # while dt_pos > 1e-6 or dt == 0:
            while p.state in [ErrorCode.Evaluate, ErrorCode.Repeat] or np.isclose(dt, 0):
                for var in ptype.variables:
                    p_var_back[var.name] = getattr(p, var.name)
                try:
                    pdt_prekernels = sign_dt * dt_pos
                    p.dt = pdt_prekernels
                    state_prev = p.state
                    # res = self.pyfunc(p, None, p.time)
                    res = self.pyfunc(p, pset.fieldset, p.time)
                    if res is None:
                        res = ErrorCode.Success

                    if res is ErrorCode.Success and p.state != state_prev:
                        res = p.state

                    if res == ErrorCode.Success and not np.isclose(p.dt, pdt_prekernels):
                        res = ErrorCode.Repeat

                except FieldOutOfBoundError as fse_xy:
                    res = ErrorCode.ErrorOutOfBounds
                    p.exception = fse_xy
                except FieldOutOfBoundSurfaceError as fse_z:
                    res = ErrorCode.ErrorThroughSurface
                    p.exception = fse_z
                except TimeExtrapolationError as fse_t:
                    res = ErrorCode.ErrorTimeExtrapolation
                    p.exception = fse_t
                except Exception as e:
                    res = ErrorCode.Error
                    p.exception = e

                # Handle particle time and time loop
                if res in [ErrorCode.Success, ErrorCode.Delete]:
                    # Update time and repeat
                    p.time += p.dt
                    p.update_next_dt()
                    dt_pos = min(abs(p.dt), abs(endtime - p.time))

                    sign_end_part = np.sign(endtime - p.time)
                    if res != ErrorCode.Delete and not np.isclose(dt_pos, 0) and (sign_end_part == sign_dt):
                        res = ErrorCode.Evaluate
                    if sign_end_part != sign_dt:
                        dt_pos = 0

                    p.state = res
                    if np.isclose(dt, 0):
                        break
                else:
                    p.state = res
                    # Try again without time update
                    for var in ptype.variables:
                        if var.name not in ['dt', 'state']:
                            setattr(p, var.name, p_var_back[var.name])
                    dt_pos = min(abs(p.dt), abs(endtime - p.time))

                    sign_end_part = np.sign(endtime - p.time)
                    if sign_end_part != sign_dt:
                        dt_pos = 0
                    break
            node = node.next
        self._compute_timings.stop_timing()
        self._compute_timings.accumulate_timing()

        self._io_timings.advance_iteration()
        self._mem_io_log.advance_iteration()
        self._compute_timings.advance_iteration()

    def remove_deleted(self, pset, output_file, endtime):
        """Utility to remove all particles that signalled deletion"""
        self._mem_io_log.start_timing()
        super(Kernel_Benchmark, self).remove_deleted(pset=pset, output_file=output_file, endtime=endtime)
        self._mem_io_log.stop_timing()
        self._mem_io_log.accumulate_timing()
        self._mem_io_log.advance_iteration()

