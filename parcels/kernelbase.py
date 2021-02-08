import _ctypes
import numpy.ctypeslib as npct
from time import time as ostime
from os import path
from os import remove
from sys import platform
from ast import FunctionDef
from hashlib import md5
from parcels.tools.loggers import logger

try:
    from mpi4py import MPI
except:
    MPI = None

from parcels.tools import get_cache_dir


__all__ = ['BaseKernel']


class BaseKernel(object):
    """Base super class for base Kernel objects that encapsulates auto-generated code.

    :arg fieldset: FieldSet object providing the field information (possibly None)
    :arg ptype: PType object for the kernel particle
    :arg pyfunc: (aggregated) Kernel function
    :arg funcname: function name
    :param delete_cfiles: Boolean whether to delete the C-files after compilation in JIT mode (default is True)

    Note: A Kernel is either created from a compiled <function ...> object
    or the necessary information (funcname, funccode, funcvars) is provided.
    The py_ast argument may be derived from the code string, but for
    concatenation, the merged AST plus the new header definition is required.
    """

    def __init__(self, fieldset, ptype, pyfunc=None, funcname=None, funccode=None, py_ast=None, funcvars=None,
                 c_include="", delete_cfiles=True):
        self.fieldset = fieldset
        self.field_args = None
        self.const_args = None
        self.ptype = ptype
        self._lib = None
        self.delete_cfiles = delete_cfiles

        # Derive meta information from pyfunc, if not given
        self.funcname = funcname or pyfunc.__name__
        self.name = "%s%s" % (ptype.name, self.funcname)
        self.ccode = ""
        self.funcvars = funcvars
        self.funccode = funccode
        self.py_ast = py_ast
        self.dyn_srcs = []
        self.static_srcs = []
        self.src_file = None
        self.lib_file = None
        self.log_file = None

        # Generate the kernel function and add the outer loop
        if self.ptype.uses_jit:
            self.dyn_srcs, self.lib_file, self.log_file = self.get_kernel_compile_files()
            self.src_file = self.dyn_srcs

    def __del__(self):
        # Clean-up the in-memory dynamic linked libraries.
        # This is not really necessary, as these programs are not that large, but with the new random
        # naming scheme which is required on Windows OS'es to deal with updates to a Parcels' kernel.
        # if self._lib is not None:
        #     _ctypes.FreeLibrary(self._lib._handle) if platform == 'win32' else _ctypes.dlclose(self._lib._handle)
        #     del self._lib
        #     self._lib = None
        #     if path.isfile(self.lib_file) and self.delete_cfiles:
        #         [remove(s) for s in [self.src_file, self.lib_file, self.log_file]]
        self.remove_lib()
        self.fieldset = None
        self.field_args = None
        self.const_args = None
        self.funcvars = None
        self.funccode = None

    @property
    def _cache_key(self):
        field_keys = ""
        if self.field_args is not None:
            field_keys = "-".join(
                ["%s:%s" % (name, field.units.__class__.__name__) for name, field in self.field_args.items()])
        key = self.name + self.ptype._cache_key + field_keys + ('TIME:%f' % ostime())
        return md5(key.encode('utf-8')).hexdigest()

    def remove_lib(self):
        # Unload the currently loaded dynamic linked library to be secure
        if self._lib is not None:
            _ctypes.FreeLibrary(self._lib._handle) if platform == 'win32' else _ctypes.dlclose(self._lib._handle)
            del self._lib
            self._lib = None
        # If file already exists, pull new names. This is necessary on a Windows machine, because
        # Python's ctype does not deal in any sort of manner well with dynamic linked libraries on this OS.
        if self.lib_file is not None and path.isfile(self.lib_file):
            [remove(s) for s in [self.lib_file, ]]
            [remove(s) for s in [self.dyn_srcs, self.log_file, ] if self.delete_cfiles]

    def get_kernel_compile_files(self):
        """
        Returns the correct src_file, lib_file, log_file for this kernel
        """
        if MPI:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()
            # cache_name = "lib"+self._cache_key    # only required here because loading is done by Kernel class instead of Compiler class
            cache_name = self._cache_key  # only required here because loading is done by Kernel class instead of Compiler class
            dyn_dir = get_cache_dir() if mpi_rank == 0 else None
            dyn_dir = mpi_comm.bcast(dyn_dir, root=0)
            ## basename = path.join(get_cache_dir(), self._cache_key) if mpi_rank == 0 else None
            # basename = path.join(get_cache_dir(), cache_name) if mpi_rank == 0 else None
            basename = cache_name if mpi_rank == 0 else None
            basename = mpi_comm.bcast(basename, root=0)
            basename = basename + "_%d" % mpi_rank
        else:
            # cache_name = "lib"+self._cache_key    # only required here because loading is done by Kernel class instead of Compiler class
            cache_name = self._cache_key    # only required here because loading is done by Kernel class instead of Compiler class
            dyn_dir = get_cache_dir()
            # basename = path.join(get_cache_dir(), "%s_0" % self._cache_key)
            basename = path.join(get_cache_dir(), "%s_0" % cache_name)
        lib_path = "lib" + basename
        src_file = "%s.c" % path.join(dyn_dir, basename)
        lib_file = "%s.%s" % (path.join(dyn_dir, lib_path), 'dll' if platform == 'win32' else 'so')
        log_file = "%s.log" % path.join(dyn_dir, basename)
        return src_file, lib_file, log_file

    def compile(self, compiler):
        """ Writes kernel code to file and compiles it."""
        # with open(self.src_file, 'w') as f:
        with open(self.dyn_srcs, 'w') as f:
            f.write(self.ccode)
        compiler.compile(self.src_file, self.lib_file, self.log_file)
        logger.info("Compiled %s ==> %s" % (self.name, self.lib_file))

    def load_lib(self):
        self._lib = npct.load_library(self.lib_file, '.')
        # self._lib = npct.load_library(self.lib_file, get_cache_dir())
        self._function = self._lib.particle_loop

    def merge(self, kernel, kclass):
        funcname = self.funcname + kernel.funcname
        func_ast = None
        if self.py_ast is not None:
            func_ast = FunctionDef(name=funcname, args=self.py_ast.args, body=self.py_ast.body + kernel.py_ast.body,
                                   decorator_list=[], lineno=1, col_offset=0)
        delete_cfiles = self.delete_cfiles and kernel.delete_cfiles
        return kclass(self.fieldset, self.ptype, pyfunc=None,
                      funcname=funcname, funccode=self.funccode + kernel.funccode,
                      py_ast=func_ast, funcvars=self.funcvars + kernel.funcvars,
                      delete_cfiles=delete_cfiles)

    def __add__(self, kernel):
        if not isinstance(kernel, BaseKernel):
            kernel = BaseKernel(self.fieldset, self.ptype, pyfunc=kernel)
        return self.merge(kernel, BaseKernel)

    def __radd__(self, kernel):
        if not isinstance(kernel, BaseKernel):
            kernel = BaseKernel(self.fieldset, self.ptype, pyfunc=kernel)
        return kernel.merge(self, BaseKernel)

    def remove_deleted(self, pset, output_file, endtime):
        pass

    def execute_jit(self, pset, endtime, dt):
        pass

    def execute_python(self, pset, endtime, dt):
        pass

    def execute(self, pset, endtime, dt, recovery=None, output_file=None):
        pass
