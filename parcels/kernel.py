import _ctypes
import abc
import ast
import functools
import hashlib
import inspect
import math  # noqa
import os
import random  # noqa
import sys
import textwrap
import types
import warnings
from copy import deepcopy
from ctypes import byref, c_double, c_int
from time import time as ostime

import numpy as np
import numpy.ctypeslib as npct
from numpy import ndarray

import parcels.rng as ParcelsRandom  # noqa
from parcels import rng  # noqa
from parcels._compat import MPI
from parcels.application_kernels.advection import (
    AdvectionAnalytical,
    AdvectionRK4_3D,
    AdvectionRK4_3D_CROCO,
    AdvectionRK45,
)
from parcels.compilation.codegenerator import KernelGenerator, LoopGenerator
from parcels.field import Field, NestedField, VectorField
from parcels.grid import GridType
from parcels.tools.global_statics import get_cache_dir
from parcels.tools.loggers import logger
from parcels.tools.statuscodes import (
    FieldOutOfBoundError,
    FieldOutOfBoundSurfaceError,
    FieldSamplingError,
    StatusCode,
    TimeExtrapolationError,
)
from parcels.tools.warnings import KernelWarning

__all__ = ["Kernel", "BaseKernel"]


class BaseKernel(abc.ABC):
    """Superclass for 'normal' and Interactive Kernels"""

    def __init__(
        self,
        fieldset,
        ptype,
        pyfunc=None,
        funcname=None,
        funccode=None,
        py_ast=None,
        funcvars=None,
        c_include="",
        delete_cfiles=True,
    ):
        self._fieldset = fieldset
        self.field_args = None
        self.const_args = None
        self._ptype = ptype
        self._lib = None
        self.delete_cfiles = delete_cfiles
        self._c_include = c_include

        # Derive meta information from pyfunc, if not given
        self._pyfunc = None
        self.funcname = funcname or pyfunc.__name__
        self.name = f"{ptype.name}{self.funcname}"
        self.ccode = ""
        self.funcvars = funcvars
        self.funccode = funccode
        self.py_ast = py_ast
        self.dyn_srcs = []
        self.static_srcs = []
        self.src_file = None
        self.lib_file = None
        self.log_file = None
        self.scipy_positionupdate_kernels_added = False

        # Generate the kernel function and add the outer loop
        if self._ptype.uses_jit:
            src_file_or_files, self.lib_file, self.log_file = self.get_kernel_compile_files()
            if type(src_file_or_files) in (list, dict, tuple, ndarray):
                self.dyn_srcs = src_file_or_files
            else:
                self.src_file = src_file_or_files

    def __del__(self):
        # Clean-up the in-memory dynamic linked libraries.
        # This is not really necessary, as these programs are not that large, but with the new random
        # naming scheme which is required on Windows OS'es to deal with updates to a Parcels' kernel.
        try:
            self.remove_lib()
        except:
            pass
        self._fieldset = None
        self.field_args = None
        self.const_args = None
        self.funcvars = None
        self.funccode = None

    @property
    def ptype(self):
        return self._ptype

    @property
    def pyfunc(self):
        return self._pyfunc

    @property
    def fieldset(self):
        return self._fieldset

    @property
    def c_include(self):
        return self._c_include

    @property
    def _cache_key(self):
        field_keys = ""
        if self.field_args is not None:
            field_keys = "-".join(
                [f"{name}:{field.units.__class__.__name__}" for name, field in self.field_args.items()]
            )
        key = self.name + self.ptype._cache_key + field_keys + (f"TIME:{ostime():f}")
        return hashlib.md5(key.encode("utf-8")).hexdigest()

    def remove_deleted(self, pset):
        """Utility to remove all particles that signalled deletion."""
        bool_indices = pset.particledata.state == StatusCode.Delete
        indices = np.where(bool_indices)[0]
        if len(indices) > 0 and self.fieldset.particlefile is not None:
            self.fieldset.particlefile.write(pset, None, indices=indices)
        pset.remove_indices(indices)

    @abc.abstractmethod
    def get_kernel_compile_files(self): ...


class Kernel(BaseKernel):
    """Kernel object that encapsulates auto-generated code.

    Parameters
    ----------
    fieldset : parcels.Fieldset
        FieldSet object providing the field information (possibly None)
    ptype :
        PType object for the kernel particle
    pyfunc :
        (aggregated) Kernel function
    funcname : str
        function name
    delete_cfiles : bool
        Whether to delete the C-files after compilation in JIT mode (default is True)

    Notes
    -----
    A Kernel is either created from a compiled <function ...> object
    or the necessary information (funcname, funccode, funcvars) is provided.
    The py_ast argument may be derived from the code string, but for
    concatenation, the merged AST plus the new header definition is required.
    """

    def __init__(
        self,
        fieldset,
        ptype,
        pyfunc=None,
        funcname=None,
        funccode=None,
        py_ast=None,
        funcvars=None,
        c_include="",
        delete_cfiles=True,
    ):
        super().__init__(
            fieldset=fieldset,
            ptype=ptype,
            pyfunc=pyfunc,
            funcname=funcname,
            funccode=funccode,
            py_ast=py_ast,
            funcvars=funcvars,
            c_include=c_include,
            delete_cfiles=delete_cfiles,
        )

        # Derive meta information from pyfunc, if not given
        self.check_fieldsets_in_kernels(pyfunc)

        if (pyfunc is AdvectionRK4_3D) and fieldset.U.gridindexingtype == "croco":
            pyfunc = AdvectionRK4_3D_CROCO
            self.funcname = "AdvectionRK4_3D_CROCO"

        if funcvars is not None:
            self.funcvars = funcvars
        elif hasattr(pyfunc, "__code__"):
            self.funcvars = list(pyfunc.__code__.co_varnames)
        else:
            self.funcvars = None
        self.funccode = funccode or inspect.getsource(pyfunc.__code__)
        self.funccode = (  # Remove parcels. prefix (see #1608)
            self.funccode.replace("parcels.rng", "rng")
            .replace("parcels.ParcelsRandom", "ParcelsRandom")
            .replace("parcels.StatusCode", "StatusCode")
        )

        # Parse AST if it is not provided explicitly
        self.py_ast = (
            py_ast or ast.parse(textwrap.dedent(self.funccode)).body[0]
        )  # Dedent allows for in-lined kernel definitions
        if pyfunc is None:
            # Extract user context by inspecting the call stack
            stack = inspect.stack()
            try:
                user_ctx = stack[-1][0].f_globals
                user_ctx["math"] = globals()["math"]
                user_ctx["ParcelsRandom"] = globals()["ParcelsRandom"]
                user_ctx["rng"] = globals()["rng"]
                user_ctx["random"] = globals()["random"]
                user_ctx["StatusCode"] = globals()["StatusCode"]
            except:
                warnings.warn(
                    "Could not access user context when merging kernels",
                    KernelWarning,
                    stacklevel=2,
                )
                user_ctx = globals()
            finally:
                del stack  # Remove cyclic references
            # Compile and generate Python function from AST
            py_mod = ast.parse("")
            py_mod.body = [self.py_ast]
            exec(compile(py_mod, "<ast>", "exec"), user_ctx)
            self._pyfunc = user_ctx[self.funcname]
        else:
            self._pyfunc = pyfunc

        numkernelargs = self.check_kernel_signature_on_version()

        if numkernelargs != 3:
            raise ValueError(
                "Since Parcels v2.0, kernels do only take 3 arguments: particle, fieldset, time !! AND !! Argument order in field interpolation is time, depth, lat, lon."
            )

        self.name = f"{ptype.name}{self.funcname}"

        # Generate the kernel function and add the outer loop
        if self.ptype.uses_jit:
            kernelgen = KernelGenerator(fieldset, ptype)
            kernel_ccode = kernelgen.generate(deepcopy(self.py_ast), self.funcvars)
            self.field_args = kernelgen.field_args
            self.vector_field_args = kernelgen.vector_field_args
            fieldset = self.fieldset
            for f in self.vector_field_args.values():
                Wname = f.W.ccode_name if f.W else "not_defined"
                for sF_name, sF_component in zip([f.U.ccode_name, f.V.ccode_name, Wname], ["U", "V", "W"], strict=True):
                    if sF_name not in self.field_args:
                        if sF_name != "not_defined":
                            self.field_args[sF_name] = getattr(f, sF_component)
            self.const_args = kernelgen.const_args
            loopgen = LoopGenerator(fieldset, ptype)
            if os.path.isfile(self._c_include):
                with open(self._c_include) as f:
                    c_include_str = f.read()
            else:
                c_include_str = self._c_include
            self.ccode = loopgen.generate(self.funcname, self.field_args, self.const_args, kernel_ccode, c_include_str)

            src_file_or_files, self.lib_file, self.log_file = self.get_kernel_compile_files()
            if type(src_file_or_files) in (list, dict, tuple, np.ndarray):
                self.dyn_srcs = src_file_or_files
            else:
                self.src_file = src_file_or_files

    def __del__(self):
        # Clean-up the in-memory dynamic linked libraries.
        # This is not really necessary, as these programs are not that large, but with the new random
        # naming scheme which is required on Windows OS'es to deal with updates to a Parcels' kernel.
        try:
            self.remove_lib()
        except:
            pass
        self._fieldset = None
        self.field_args = None
        self.const_args = None
        self.funcvars = None
        self.funccode = None

    @property
    def ptype(self):
        return self._ptype

    @property
    def pyfunc(self):
        return self._pyfunc

    @property
    def fieldset(self):
        return self._fieldset

    @property
    def c_include(self):
        return self._c_include

    @property
    def _cache_key(self):
        field_keys = ""
        if self.field_args is not None:
            field_keys = "-".join(
                [f"{name}:{field.units.__class__.__name__}" for name, field in self.field_args.items()]
            )
        key = self.name + self.ptype._cache_key + field_keys + (f"TIME:{ostime():f}")
        return hashlib.md5(key.encode("utf-8")).hexdigest()

    def add_scipy_positionupdate_kernels(self):
        # Adding kernels that set and update the coordinate changes
        def Setcoords(particle, fieldset, time):
            particle_dlon = 0  # noqa
            particle_dlat = 0  # noqa
            particle_ddepth = 0  # noqa
            particle.lon = particle.lon_nextloop
            particle.lat = particle.lat_nextloop
            particle.depth = particle.depth_nextloop
            particle.time = particle.time_nextloop

        def Updatecoords(particle, fieldset, time):
            particle.lon_nextloop = particle.lon + particle_dlon  # noqa
            particle.lat_nextloop = particle.lat + particle_dlat  # noqa
            particle.depth_nextloop = particle.depth + particle_ddepth  # noqa
            particle.time_nextloop = particle.time + particle.dt

        self._pyfunc = (Setcoords + self + Updatecoords)._pyfunc

    def check_fieldsets_in_kernels(self, pyfunc):
        """
        Checks the integrity of the fieldset with the kernels.

        This function is to be called from the derived class when setting up the 'pyfunc'.
        """
        if self.fieldset is not None:
            if pyfunc is AdvectionRK4_3D:
                warning = False
                if (
                    isinstance(self._fieldset.W, Field)
                    and self._fieldset.W.creation_log != "from_nemo"
                    and self._fieldset.W._scaling_factor is not None
                    and self._fieldset.W._scaling_factor > 0
                ):
                    warning = True
                if isinstance(self._fieldset.W, NestedField):
                    for f in self._fieldset.W:
                        if f.creation_log != "from_nemo" and f._scaling_factor is not None and f._scaling_factor > 0:
                            warning = True
                if warning:
                    warnings.warn(
                        "Note that in AdvectionRK4_3D, vertical velocity is assumed positive towards increasing z. "
                        "If z increases downward and w is positive upward you can re-orient it downwards by setting fieldset.W.set_scaling_factor(-1.)",
                        KernelWarning,
                        stacklevel=2,
                    )
            elif pyfunc is AdvectionAnalytical:
                if self.fieldset.particlefile is not None:
                    self.fieldset.particlefile.analytical = True
                if self._ptype.uses_jit:
                    raise NotImplementedError("Analytical Advection only works in Scipy mode")
                if self._fieldset.U.interp_method != "cgrid_velocity":
                    raise NotImplementedError("Analytical Advection only works with C-grids")
                if self._fieldset.U.grid.gtype not in [GridType.CurvilinearZGrid, GridType.RectilinearZGrid]:
                    raise NotImplementedError("Analytical Advection only works with Z-grids in the vertical")
            elif pyfunc is AdvectionRK45:
                if not hasattr(self.fieldset, "RK45_tol"):
                    warnings.warn(
                        "Setting RK45 tolerance to 10 m. Use fieldset.add_constant('RK45_tol', [distance]) to change.",
                        KernelWarning,
                        stacklevel=2,
                    )
                    self.fieldset.add_constant("RK45_tol", 10)
                if self.fieldset.U.grid.mesh == "spherical":
                    self.fieldset.RK45_tol /= (
                        1852 * 60
                    )  # TODO does not account for zonal variation in meter -> degree conversion
                if not hasattr(self.fieldset, "RK45_min_dt"):
                    warnings.warn(
                        "Setting RK45 minimum timestep to 1 s. Use fieldset.add_constant('RK45_min_dt', [timestep]) to change.",
                        KernelWarning,
                        stacklevel=2,
                    )
                    self.fieldset.add_constant("RK45_min_dt", 1)
                if not hasattr(self.fieldset, "RK45_max_dt"):
                    warnings.warn(
                        "Setting RK45 maximum timestep to 1 day. Use fieldset.add_constant('RK45_max_dt', [timestep]) to change.",
                        KernelWarning,
                        stacklevel=2,
                    )
                    self.fieldset.add_constant("RK45_max_dt", 60 * 60 * 24)

    def check_kernel_signature_on_version(self):
        """Returns number of arguments in a Python function."""
        if self._pyfunc is None:
            return 0
        return len(inspect.getfullargspec(self._pyfunc).args)

    def remove_lib(self):
        if self._lib is not None:
            self.cleanup_unload_lib(self._lib)
            del self._lib
            self._lib = None

        all_files_array = []
        if self.src_file is None:
            if self.dyn_srcs is not None:
                [all_files_array.append(fpath) for fpath in self.dyn_srcs]
        else:
            if self.src_file is not None:
                all_files_array.append(self.src_file)
        if self.log_file is not None:
            all_files_array.append(self.log_file)
        if self.lib_file is not None and all_files_array is not None and self.delete_cfiles is not None:
            self.cleanup_remove_files(self.lib_file, all_files_array, self.delete_cfiles)

        # If file already exists, pull new names. This is necessary on a Windows machine, because
        # Python's ctype does not deal in any sort of manner well with dynamic linked libraries on this OS.
        if self._ptype.uses_jit:
            src_file_or_files, self.lib_file, self.log_file = self.get_kernel_compile_files()
            if type(src_file_or_files) in (list, dict, tuple, ndarray):
                self.dyn_srcs = src_file_or_files
            else:
                self.src_file = src_file_or_files

    def get_kernel_compile_files(self):
        """Returns the correct src_file, lib_file, log_file for this kernel."""
        if MPI:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()
            cache_name = (
                self._cache_key
            )  # only required here because loading is done by Kernel class instead of Compiler class
            dyn_dir = get_cache_dir() if mpi_rank == 0 else None
            dyn_dir = mpi_comm.bcast(dyn_dir, root=0)
            basename = cache_name if mpi_rank == 0 else None
            basename = mpi_comm.bcast(basename, root=0)
            basename = basename + "_%d" % mpi_rank
        else:
            cache_name = (
                self._cache_key
            )  # only required here because loading is done by Kernel class instead of Compiler class
            dyn_dir = get_cache_dir()
            basename = f"{cache_name}_0"
        lib_path = "lib" + basename
        src_file_or_files = None
        if type(basename) in (list, dict, tuple, ndarray):
            src_file_or_files = [""] * len(basename)
            for i, src_file in enumerate(basename):
                src_file_or_files[i] = f"{os.path.join(dyn_dir, src_file)}.c"
        else:
            src_file_or_files = f"{os.path.join(dyn_dir, basename)}.c"
        lib_file = f"{os.path.join(dyn_dir, lib_path)}.{'dll' if sys.platform == 'win32' else 'so'}"
        log_file = f"{os.path.join(dyn_dir, basename)}.log"
        return src_file_or_files, lib_file, log_file

    def compile(self, compiler):
        """Writes kernel code to file and compiles it."""
        all_files_array = []
        if self.src_file is None:
            if self.dyn_srcs is not None:
                for dyn_src in self.dyn_srcs:
                    with open(dyn_src, "w") as f:
                        f.write(self.ccode)
                    all_files_array.append(dyn_src)
                compiler.compile(self.dyn_srcs, self.lib_file, self.log_file)
        else:
            if self.src_file is not None:
                with open(self.src_file, "w") as f:
                    f.write(self.ccode)
                if self.src_file is not None:
                    all_files_array.append(self.src_file)
                compiler.compile(self.src_file, self.lib_file, self.log_file)
        if len(all_files_array) > 0:
            if self.delete_cfiles is False:
                logger.info(f"Compiled {self.name} ==> {self.src_file}")
            if self.log_file is not None:
                all_files_array.append(self.log_file)

    def load_lib(self):
        self._lib = npct.load_library(self.lib_file, ".")
        self._function = self._lib.particle_loop

    def merge(self, kernel, kclass):
        funcname = self.funcname + kernel.funcname
        func_ast = None
        if self.py_ast is not None:
            func_ast = ast.FunctionDef(
                name=funcname,
                args=self.py_ast.args,
                body=self.py_ast.body + kernel.py_ast.body,
                decorator_list=[],
                lineno=1,
                col_offset=0,
            )
        delete_cfiles = self.delete_cfiles and kernel.delete_cfiles
        return kclass(
            self.fieldset,
            self.ptype,
            pyfunc=None,
            funcname=funcname,
            funccode=self.funccode + kernel.funccode,
            py_ast=func_ast,
            funcvars=self.funcvars + kernel.funcvars,
            c_include=self._c_include + kernel.c_include,
            delete_cfiles=delete_cfiles,
        )

    def __add__(self, kernel):
        if not isinstance(kernel, type(self)):
            kernel = type(self)(self.fieldset, self.ptype, pyfunc=kernel)
        return self.merge(kernel, type(self))

    def __radd__(self, kernel):
        if not isinstance(kernel, type(self)):
            kernel = type(self)(self.fieldset, self.ptype, pyfunc=kernel)
        return kernel.merge(self, type(self))

    @classmethod
    def from_list(cls, fieldset, ptype, pyfunc_list, *args, **kwargs):
        """Create a combined kernel from a list of functions.

        Takes a list of functions, converts them to kernels, and joins them
        together.

        Parameters
        ----------
        fieldset : parcels.Fieldset
            FieldSet object providing the field information (possibly None)
        ptype :
            PType object for the kernel particle
        pyfunc_list : list of functions
            List of functions to be combined into a single kernel.
        *args :
            Additional arguments passed to first kernel during construction.
        **kwargs :
            Additional keyword arguments passed to first kernel during construction.
        """
        if not isinstance(pyfunc_list, list):
            raise TypeError(f"Argument function_list should be a list of functions. Got {type(pyfunc_list)}")
        if len(pyfunc_list) == 0:
            raise ValueError("Argument function_list should have at least one function.")
        if not all([isinstance(f, types.FunctionType) for f in pyfunc_list]):
            raise ValueError("Argument function_lst should be a list of functions.")

        pyfunc_list = pyfunc_list.copy()
        pyfunc_list[0] = cls(fieldset, ptype, pyfunc_list[0], *args, **kwargs)
        return functools.reduce(lambda x, y: x + y, pyfunc_list)

    @staticmethod
    def cleanup_remove_files(lib_file, all_files_array, delete_cfiles):
        if lib_file is not None:
            if os.path.isfile(lib_file):  # and delete_cfiles
                [os.remove(s) for s in [lib_file] if os.path is not None and os.path.exists(s)]
            if delete_cfiles and len(all_files_array) > 0:
                [os.remove(s) for s in all_files_array if os.path is not None and os.path.exists(s)]

    @staticmethod
    def cleanup_unload_lib(lib):
        # Clean-up the in-memory dynamic linked libraries.
        # This is not really necessary, as these programs are not that large, but with the new random
        # naming scheme which is required on Windows OS'es to deal with updates to a Parcels' kernel.
        if lib is not None:
            try:
                _ctypes.FreeLibrary(lib._handle) if sys.platform == "win32" else _ctypes.dlclose(lib._handle)
            except:
                pass

    def load_fieldset_jit(self, pset):
        """Updates the loaded fields of pset's fieldset according to the chunk information within their grids."""
        if pset.fieldset is not None:
            for g in pset.fieldset.gridset.grids:
                g.cstruct = None  # This force to point newly the grids from Python to C
            # Make a copy of the transposed array to enforce
            # C-contiguous memory layout for JIT mode.
            for f in pset.fieldset.get_fields():
                if isinstance(f, (VectorField, NestedField)):
                    continue
                if f.data.dtype != np.float32:
                    raise RuntimeError(f"Field {f.name} data needs to be float32 in JIT mode")
                if f in self.field_args.values():
                    f.chunk_data()
                else:
                    for block_id in range(len(f.data_chunks)):
                        f.data_chunks[block_id] = None
                        f.c_data_chunks[block_id] = None

            for g in pset.fieldset.gridset.grids:
                g.load_chunk = np.where(g.load_chunk == g.chunk_loading_requested, g.chunk_loaded_touched, g.load_chunk)
                if len(g.load_chunk) > g.chunk_not_loaded:  # not the case if a field in not called in the kernel
                    if not g.load_chunk.flags["C_CONTIGUOUS"]:
                        g.load_chunk = np.array(g.load_chunk, order="C")
                if not g.depth.flags.c_contiguous:
                    g.depth = np.array(g.depth, order="C")
                if not g.lon.flags.c_contiguous:
                    g.lon = np.array(g.lon, order="C")
                if not g.lat.flags.c_contiguous:
                    g.lat = np.array(g.lat, order="C")

    def execute_jit(self, pset, endtime, dt):
        """Invokes JIT engine to perform the core update loop."""
        self.load_fieldset_jit(pset)

        fargs = [byref(f.ctypes_struct) for f in self.field_args.values()]
        fargs += [c_double(f) for f in self.const_args.values()]
        particle_data = byref(pset.ctypes_struct)
        return self._function(c_int(len(pset)), particle_data, c_double(endtime), c_double(dt), *fargs)

    def execute_python(self, pset, endtime, dt):
        """Performs the core update loop via Python."""
        if self.fieldset is not None:
            for f in self.fieldset.get_fields():
                if isinstance(f, (VectorField, NestedField)):
                    continue
                f.data = np.array(f.data)

        if not self.scipy_positionupdate_kernels_added:
            self.add_scipy_positionupdate_kernels()
            self.scipy_positionupdate_kernels_added = True

        for p in pset:
            self.evaluate_particle(p, endtime)
            if p.state == StatusCode.StopAllExecution:
                return StatusCode.StopAllExecution

    def execute(self, pset, endtime, dt):
        """Execute this Kernel over a ParticleSet for several timesteps."""
        pset.particledata.state[:] = StatusCode.Evaluate

        if abs(dt) < 1e-6:
            warnings.warn(
                "'dt' is too small, causing numerical accuracy limit problems. Please chose a higher 'dt' and rather scale the 'time' axis of the field accordingly. (related issue #762)",
                RuntimeWarning,
                stacklevel=2,
            )

        if pset.fieldset is not None:
            for g in pset.fieldset.gridset.grids:
                if len(g.load_chunk) > g.chunk_not_loaded:  # not the case if a field in not called in the kernel
                    g.load_chunk = np.where(g.load_chunk == g.chunk_loaded_touched, g.chunk_deprecated, g.load_chunk)

        # Execute the kernel over the particle set
        if self.ptype.uses_jit:
            self.execute_jit(pset, endtime, dt)
        else:
            self.execute_python(pset, endtime, dt)

        # Remove all particles that signalled deletion
        self.remove_deleted(pset)

        # Identify particles that threw errors
        n_error = pset.num_error_particles

        while n_error > 0:
            error_pset = pset.error_particles
            # Check for StatusCodes
            for p in error_pset:
                if p.state == StatusCode.StopExecution:
                    return
                if p.state == StatusCode.StopAllExecution:
                    return StatusCode.StopAllExecution
                if p.state == StatusCode.Repeat:
                    p.state = StatusCode.Evaluate
                elif p.state == StatusCode.ErrorTimeExtrapolation:
                    raise TimeExtrapolationError(p.time)
                elif p.state == StatusCode.ErrorOutOfBounds:
                    raise FieldOutOfBoundError(p.lon, p.lat, p.depth)
                elif p.state == StatusCode.ErrorThroughSurface:
                    raise FieldOutOfBoundSurfaceError(p.lon, p.lat, p.depth)
                elif p.state == StatusCode.Error:
                    raise FieldSamplingError(p.lon, p.lat, p.depth)
                elif p.state == StatusCode.Delete:
                    pass
                else:
                    warnings.warn(
                        f"Deleting particle {p.id} because of non-recoverable error",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    p.delete()

            # Remove all particles that signalled deletion
            self.remove_deleted(pset)  # Generalizable version!

            # Execute core loop again to continue interrupted particles
            if self.ptype.uses_jit:
                self.execute_jit(pset, endtime, dt)
            else:
                self.execute_python(pset, endtime, dt)

            n_error = pset.num_error_particles

    def evaluate_particle(self, p, endtime):
        """Execute the kernel evaluation of for an individual particle.

        Parameters
        ----------
        p :
            object of (sub-)type (ScipyParticle, JITParticle)
        endtime :
            endtime of this overall kernel evaluation step
        dt :
            computational integration timestep
        """
        while p.state in [StatusCode.Evaluate, StatusCode.Repeat]:
            pre_dt = p.dt

            sign_dt = np.sign(p.dt)
            if sign_dt * p.time_nextloop >= sign_dt * endtime:
                return p

            try:  # Use next_dt from AdvectionRK45 if it is set
                if abs(endtime - p.time_nextloop) < abs(p.next_dt) - 1e-6:
                    p.next_dt = abs(endtime - p.time_nextloop) * sign_dt
            except KeyError:
                if abs(endtime - p.time_nextloop) < abs(p.dt) - 1e-6:
                    p.dt = abs(endtime - p.time_nextloop) * sign_dt
            res = self._pyfunc(p, self._fieldset, p.time_nextloop)

            if res is None:
                if sign_dt * p.time < sign_dt * endtime and p.state == StatusCode.Success:
                    p.state = StatusCode.Evaluate
            else:
                p.state = res

            p.dt = pre_dt
        return p
