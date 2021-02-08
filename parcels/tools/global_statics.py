import os
import sys
import _ctypes
from tempfile import gettempdir
from pathlib import Path

try:
    from os import getuid
except:
    # Windows does not have getuid(), so define to simply return 'tmp'
    def getuid():
        return 'tmp'


def cleanup_remove_files(lib_file, log_file):
    if os.path.isfile(lib_file):
        [os.remove(s) for s in [lib_file, log_file]]


def cleanup_unload_lib(lib):
    # Clean-up the in-memory dynamic linked libraries.
    # This is not really necessary, as these programs are not that large, but with the new random
    # naming scheme which is required on Windows OS'es to deal with updates to a Parcels' kernel.
    if lib is not None:
        _ctypes.FreeLibrary(lib._handle) if sys.platform == 'win32' else _ctypes.dlclose(lib._handle)


def get_package_dir():
    fpath = Path(__file__)
    return fpath.parent.parent


def get_cache_dir():
    directory = os.path.join(gettempdir(), "parcels-%s" % getuid())
    Path(directory).mkdir(exist_ok=True)
    return directory
