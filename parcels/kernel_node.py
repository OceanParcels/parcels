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
from parcels import AdvectionRK4_3D, logger
from parcels.kernelbase import BaseKernel
from parcels.tools.global_statics import get_cache_dir, get_package_dir
from parcels.wrapping import KernelGenerator, NodeLoopGenerator
from parcels.tools.error import recovery_map as recovery_base_map

__all__ = ['Kernel']
DEBUG_MODE = False


re_indent = re.compile(r"^(\s+)")


def fix_indentation(string):
    """Fix indentation to allow in-lined kernel definitions"""
    lines = string.split('\n')
    indent = re_indent.match(lines[0])
    if indent:
        lines = [l.replace(indent.groups()[0], '', 1) for l in lines]
    return "\n".join(lines)


class Kernel(BaseKernel):
    def __init__(self, fieldset, ptype, pyfunc=None, funcname=None, funccode=None, py_ast=None, funcvars=None,
                 c_include="", delete_cfiles=True):
        super(Kernel, self).__init__(fieldset, ptype, pyfunc=pyfunc, funcname=funcname, funccode=funccode, py_ast=py_ast, funcvars=funcvars, c_include=c_include, delete_cfiles=delete_cfiles)

        # Derive meta information from pyfunc, if not given
        if pyfunc is AdvectionRK4_3D:   # would be better if the idea of a Kernel being '2D', '3D, '4D' or 'uncertain' is captured as Attribute or as class stucture
            warning = False
            if isinstance(self.fieldset.W, Field) and self.fieldset.W.creation_log != 'from_nemo' and \
               self.fieldset.W.scaling_factor is not None and self.fieldset.W.scaling_factor > 0:
                warning = True
            if type(self.fieldset.W) in [SummedField, NestedField]:
                for f in self.fieldset.W:
                    if f.creation_log != 'from_nemo' and f.scaling_factor is not None and f.scaling_factor > 0:
                        warning = True
            if warning:
                logger.warning_once('Note that in AdvectionRK4_3D, vertical velocity is assumed positive towards increasing z.\n'
                                    '\tIf z increases downward and w is positive upward you can re-orient it downwards by setting fieldset.W.set_scaling_factor(-1.)')
        if funcvars is not None:
            self.funcvars = funcvars
        elif hasattr(pyfunc, '__code__'):
            self.funcvars = list(pyfunc.__code__.co_varnames)
        else:
            self.funcvars = None
        self.funccode = funccode or inspect.getsource(pyfunc.__code__)
        # Parse AST if it is not provided explicitly
        self.py_ast = py_ast or parse(fix_indentation(self.funccode)).body[0]
        if pyfunc is None:
            # Extract user context by inspecting the call stack
            stack = inspect.stack()
            try:
                user_ctx = stack[-1][0].f_globals
                user_ctx['math'] = globals()['math']
                user_ctx['random'] = globals()['random']
                user_ctx['ErrorCode'] = globals()['ErrorCode']
            except:
                logger.warning("Could not access user context when merging kernels")
                user_ctx = globals()
            finally:
                del stack  # Remove cyclic references
            # Compile and generate Python function from AST
            py_mod = parse("")
            py_mod.body = [self.py_ast]
            exec(compile(py_mod, "<ast>", "exec"), user_ctx)
            self.pyfunc = user_ctx[self.funcname]
        else:
            self.pyfunc = pyfunc

        if version_info[0] < 3:
            numkernelargs = len(inspect.getargspec(self.pyfunc).args)
        else:
            numkernelargs = len(inspect.getfullargspec(self.pyfunc).args)

        assert numkernelargs == 3, \
            'Since Parcels v2.0, kernels do only take 3 arguments: particle, fieldset, time !! AND !! Argument order in field interpolation is time, depth, lat, lon.'

        self.name = "%s%s" % (ptype.name, self.funcname)

        # ======== THIS NEEDS TO BE REFACTORED BASED ON THE TYPE OF PARTICLE BEING USED ======== #
        # Generate the kernel function and add the outer loop
        if self.ptype.uses_jit:
            kernelgen = KernelGenerator(ptype, self.fieldset)
            kernel_ccode = kernelgen.generate(deepcopy(self.py_ast), self.funcvars)
            self.field_args = kernelgen.field_args
            self.vector_field_args = kernelgen.vector_field_args
            for f in self.vector_field_args.values():
                Wname = f.W.ccode_name if f.W else 'not_defined'
                for sF_name, sF_component in zip([f.U.ccode_name, f.V.ccode_name, Wname], ['U', 'V', 'W']):
                    if sF_name not in self.field_args:
                        if sF_name != 'not_defined':
                            self.field_args[sF_name] = getattr(f, sF_component)
            self.const_args = kernelgen.const_args

            # loopgen = VectorizedLoopGenerator(ptype)
            loopgen = NodeLoopGenerator(ptype, fieldset=self.fieldset)

            if path.isfile(c_include):
                with open(c_include, 'r') as f:
                    c_include_str = f.read()
            else:
                c_include_str = c_include
            self.ccode = loopgen.generate(self.funcname, self.field_args, self.const_args,
                                          kernel_ccode, c_include_str)
            # self.src_file, self.lib_file, self.log_file = self.get_kernel_compile_files()
            self.dyn_srcs, self.lib_file, self.log_file = self.get_kernel_compile_files()
            static_srcs = [path.join(get_package_dir(), 'nodes', 'node.c'), ]
            self.static_srcs = static_srcs
            self.src_file = [self.dyn_srcs, ] + self.static_srcs

    def __del__(self):
        # Clean-up the in-memory dynamic linked libraries.
        # This is not really necessary, as these programs are not that large, but with the new random
        # naming scheme which is required on Windows OS'es to deal with updates to a Parcels' kernel.
        super(Kernel, self).__del__()

    def __add__(self, kernel):
        if not isinstance(kernel, Kernel):
            kernel = Kernel(self.fieldset, self.ptype, pyfunc=kernel)
        return self.merge(kernel, Kernel)

    def __radd__(self, kernel):
        if not isinstance(kernel, Kernel):
            kernel = Kernel(self.fieldset, self.ptype, pyfunc=kernel)
        return kernel.merge(self, Kernel)

    def execute_jit(self, pset, endtime, dt):
        """Invokes JIT engine to perform the core update loop"""
        # if len(pset.particles) > 0:
        #     assert pset.fieldset.gridset.size == len(pset.particles[0].xi), 'FieldSet has different amount of grids than Particle.xi. Have you added Fields after creating the ParticleSet?'
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
            fargs += [byref(f.ctypes_struct) for f in self.field_args.values()]
        if self.const_args is not None:
            fargs += [c_double(f) for f in self.const_args.values()]

        # particle_data = pset._particle_data.ctypes.data_as(c_void_p)
        node_data = pset.begin()
        if len(fargs) > 0:
            self._function(c_int(len(pset)), pointer(node_data), c_double(endtime), c_double(dt), *fargs)
        else:
            self._function(c_int(len(pset)), pointer(node_data), c_double(endtime), c_double(dt))

    def execute_python(self, pset, endtime, dt):
        """Performs the core update loop via Python"""
        sign_dt = np.sign(dt)

        # back up variables in case of ErrorCode.Repeat
        p_var_back = {}

        for f in self.fieldset.get_fields():
            if type(f) in [VectorField, NestedField, SummedField]:
                continue
            f.data = np.array(f.data)

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

    def remove_deleted(self, pset, output_file, endtime):
        """Utility to remove all particles that signalled deletion"""
        indices = pset.get_deleted_item_indices()
        if len(indices) > 0:
            pdata = [pset[i].data for i in indices]
            if len(pdata) > 0 and output_file is not None:
                output_file.write(pdata, endtime, deleted_only=True)
            pset.remove_deleted_items_by_indices(indices)
        return pset

    def execute(self, pset, endtime, dt, recovery=None, output_file=None, execute_once=False):
        """Execute this Kernel over a ParticleSet for several timesteps"""
        node = pset.begin()
        while node is not None:
            node.data.reset_state()
            node = node.next

        if abs(dt) < 1e-6 and not execute_once:
            logger.warning_once("'dt' is too small, causing numerical accuracy limit problems. Please chose a higher 'dt' and rather scale the 'time' axis of the field accordingly. (related issue #762)")

        if recovery is None:
            recovery = {}
        elif ErrorCode.ErrorOutOfBounds in recovery and ErrorCode.ErrorThroughSurface not in recovery:
            recovery[ErrorCode.ErrorThroughSurface] = recovery[ErrorCode.ErrorOutOfBounds]
        recovery_map = recovery_base_map.copy()
        recovery_map.update(recovery)

        for g in pset.fieldset.gridset.grids:
            if len(g.load_chunk) > 0:  # not the case if a field in not called in the kernel
                g.load_chunk = np.where(g.load_chunk == 2, 3, g.load_chunk)

        # Execute the kernel over the particle set
        if self.ptype.uses_jit:
            self.execute_jit(pset, endtime, dt)
        else:
            self.execute_python(pset, endtime, dt)

        # Remove all particles that signalled deletion
        self.remove_deleted(pset=pset, output_file=output_file, endtime=endtime)

        # Identify particles that threw errors
        # error_particles = [p for p in pset.particles if p.state not in [ErrorCode.Success, ErrorCode.Evaluate]]
        error_particles = [n.data for n in pset.data if n.data.state not in [ErrorCode.Success, ErrorCode.Evaluate]]

        while len(error_particles) > 0:
            # Apply recovery kernel
            for p in error_particles:
                if p.state == ErrorCode.StopExecution:
                    return
                if p.state == ErrorCode.Repeat:
                    p.reset_state()
                elif p.state == ErrorCode.Delete:
                    pass
                elif p.state in recovery_map:
                    recovery_kernel = recovery_map[p.state]
                    p.state = ErrorCode.Success
                    recovery_kernel(p, self.fieldset, p.time)
                    if(p.isComputed()):
                        p.reset_state()
                else:
                    logger.warning_once('Deleting particle {} because of non-recoverable error'.format(p.id))
                    # logger.warning('Deleting particle because of bug in #749 and #737 - particle state: {}'.format(ErrorCode.toString(p.state)))
                    p.delete()

            # Remove all particles that signalled deletion
            self.remove_deleted(pset=pset, output_file=output_file, endtime=endtime)

            # Execute core loop again to continue interrupted particles
            if self.ptype.uses_jit:
                self.execute_jit(pset, endtime, dt)
            else:
                self.execute_python(pset, endtime, dt)

            # error_particles = [p for p in pset.particles if p.state not in [ErrorCode.Success, ErrorCode.Evaluate]]
            error_particles = [n.data for n in pset.data if n.data.state not in [ErrorCode.Success, ErrorCode.Evaluate]]
