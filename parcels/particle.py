from ctypes import c_void_p
from operator import attrgetter

import numpy as np

from parcels.field import Field
from parcels.tools.error import ErrorCode
from parcels.tools.loggers import logger

__all__ = ['ScipyParticle', 'JITParticle', 'Variable']

indicators_64bit = [np.float64, np.int64, c_void_p]


class Variable(object):
    """Descriptor class that delegates data access to particle data

    :param name: Variable name as used within kernels
    :param dtype: Data type (numpy.dtype) of the variable
    :param initial: Initial value of the variable. Note that this can also be a Field object,
             which will then be sampled at the location of the particle
    :param to_write: Boolean to control whether Variable is written to NetCDF file
    """
    def __init__(self, name, dtype=np.float32, initial=0, to_write=True):
        if name == 'z':
            raise NotImplementedError("Custom Variable name 'z' is not allowed, as it is used for depth in ParticleFile")
        self.name = name
        self.dtype = dtype
        self.initial = initial
        self.to_write = to_write

    def __get__(self, instance, cls):
        if instance is None:
            return self
        if issubclass(cls, JITParticle):
            return instance._cptr.__getitem__(self.name)
        else:
            return getattr(instance, "_%s" % self.name, self.initial)

    def __set__(self, instance, value):
        if isinstance(instance, JITParticle):
            instance._cptr.__setitem__(self.name, value)
        else:
            setattr(instance, "_%s" % self.name, value)

    def __repr__(self):
        return "PVar<%s|%s>" % (self.name, self.dtype)

    def is64bit(self):
        """Check whether variable is 64-bit"""
        return True if self.dtype in indicators_64bit else False


class ParticleType(object):
    """Class encapsulating the type information for custom particles

    :param user_vars: Optional list of (name, dtype) tuples for custom variables
    """

    def __init__(self, pclass):
        if not isinstance(pclass, type):
            raise TypeError("Class object required to derive ParticleType")
        if not issubclass(pclass, ScipyParticle):
            raise TypeError("Class object does not inherit from parcels.ScipyParticle")

        self.name = pclass.__name__
        self.uses_jit = issubclass(pclass, JITParticle)
        # Pick Variable objects out of __dict__.
        self.variables = [v for v in pclass.__dict__.values() if isinstance(v, Variable)]
        for cls in pclass.__bases__:
            if issubclass(cls, ScipyParticle):
                # Add inherited particle variables
                ptype = cls.getPType()
                self.variables = ptype.variables + self.variables
        # Sort variables with all the 64-bit first so that they are aligned for the JIT cptr
        self.variables = [v for v in self.variables if v.is64bit()] + \
                         [v for v in self.variables if not v.is64bit()]

    def __repr__(self):
        return "PType<%s>::%s" % (self.name, self.variables)

    @property
    def _cache_key(self):
        return "-".join(["%s:%s" % (v.name, v.dtype) for v in self.variables])

    @property
    def dtype(self):
        """Numpy.dtype object that defines the C struct"""
        type_list = [(v.name, v.dtype) for v in self.variables]
        for v in self.variables:
            if v.dtype not in self.supported_dtypes:
                raise RuntimeError(str(v.dtype) + " variables are not implemented in JIT mode")
        if self.size % 8 > 0:
            # Add padding to be 64-bit aligned
            type_list += [('pad', np.float32)]
        return np.dtype(type_list)

    @property
    def size(self):
        """Size of the underlying particle struct in bytes"""
        return sum([8 if v.is64bit() else 4 for v in self.variables])

    @property
    def supported_dtypes(self):
        """List of all supported numpy dtypes. All others are not supported"""

        # Developer note: other dtypes (mostly 2-byte ones) are not supported now
        # because implementing and aligning them in cgen.GenerableStruct is a
        # major headache. Perhaps in a later stage
        return [np.int32, np.int64, np.float32, np.double, np.float64, c_void_p]


class _Particle(object):
    """Private base class for all particle types"""
    lastID = 0  # class-level variable keeping track of last Particle ID used

    def __init__(self):
        ptype = self.getPType()
        # Explicit initialisation of all particle variables
        for v in ptype.variables:
            if isinstance(v.initial, attrgetter):
                initial = v.initial(self)
            elif isinstance(v.initial, Field):
                lon = self.getInitialValue(ptype, name='lon')
                lat = self.getInitialValue(ptype, name='lat')
                depth = self.getInitialValue(ptype, name='depth')
                time = self.getInitialValue(ptype, name='time')
                if time is None:
                    raise RuntimeError('Cannot initialise a Variable with a Field if no time provided. '
                                       'Add a "time=" to ParticleSet construction')
                v.initial.fieldset.computeTimeChunk(time, 0)
                initial = v.initial[time, depth, lat, lon]
                logger.warning_once("Particle initialisation from field can be very slow as it is computed in scipy mode.")
            else:
                initial = v.initial
            # Enforce type of initial value
            if v.dtype != c_void_p:
                setattr(self, v.name, v.dtype(initial))

        # Placeholder for explicit error handling
        self.exception = None

    @classmethod
    def getPType(cls):
        return ParticleType(cls)

    @classmethod
    def getInitialValue(cls, ptype, name):
        return next((v.initial for v in ptype.variables if v.name is name), None)

    @classmethod
    def setLastID(cls, offset):
        _Particle.lastID = offset


class ScipyParticle(_Particle):
    """Class encapsulating the basic attributes of a particle,
    to be executed in SciPy mode

    :param lon: Initial longitude of particle
    :param lat: Initial latitude of particle
    :param depth: Initial depth of particle
    :param fieldset: :mod:`parcels.fieldset.FieldSet` object to track this particle on
    :param time: Current time of the particle

    Additional Variables can be added via the :Class Variable: objects
    """

    lon = Variable('lon', dtype=np.float32)
    lat = Variable('lat', dtype=np.float32)
    depth = Variable('depth', dtype=np.float32)
    time = Variable('time', dtype=np.float64)
    id = Variable('id', dtype=np.int32)
    dt = Variable('dt', dtype=np.float64, to_write=False)
    state = Variable('state', dtype=np.int32, initial=ErrorCode.Evaluate, to_write=False)

    def __init__(self, lon, lat, pid, fieldset, depth=0., time=0., cptr=None):

        # Enforce default values through Variable descriptor
        type(self).lon.initial = lon
        type(self).lat.initial = lat
        type(self).depth.initial = depth
        type(self).time.initial = time
        type(self).id.initial = pid
        _Particle.lastID = max(_Particle.lastID, pid)
        type(self).dt.initial = None
        super(ScipyParticle, self).__init__()
        self._next_dt = None

    def __repr__(self):
        time_string = 'not_yet_set' if self.time is None or np.isnan(self.time) else "{:f}".format(self.time)
        str = "P[%d](lon=%f, lat=%f, depth=%f, " % (self.id, self.lon, self.lat, self.depth)
        for var in vars(type(self)):
            if type(getattr(type(self), var)) is Variable and getattr(type(self), var).to_write is True:
                str += "%s=%f, " % (var, getattr(self, var))
        return str + "time=%s)" % time_string

    def delete(self):
        self.state = ErrorCode.Delete

    def succeeded(self):
        self.state = ErrorCode.Success

    def isComputed(self):
        return self.state == ErrorCode.Success

    def reset_state(self):
        self.state = ErrorCode.Evaluate

    @classmethod
    def set_lonlatdepth_dtype(cls, dtype):
        cls.lon.dtype = dtype
        cls.lat.dtype = dtype
        cls.depth.dtype = dtype

    def update_next_dt(self, next_dt=None):
        if next_dt is None:
            if self._next_dt is not None:
                self.dt = self._next_dt
                self._next_dt = None
        else:
            self._next_dt = next_dt


class JITParticle(ScipyParticle):
    """Particle class for JIT-based (Just-In-Time) Particle objects

    :param lon: Initial longitude of particle
    :param lat: Initial latitude of particle
    :param fieldset: :mod:`parcels.fieldset.FieldSet` object to track this particle on
    :param dt: Execution timestep for this particle
    :param time: Current time of the particle

    Additional Variables can be added via the :Class Variable: objects

    Users should use JITParticles for faster advection computation.

    """

    cxi = Variable('cxi', dtype=np.dtype(c_void_p), to_write=False)
    cyi = Variable('cyi', dtype=np.dtype(c_void_p), to_write=False)
    czi = Variable('czi', dtype=np.dtype(c_void_p), to_write=False)
    cti = Variable('cti', dtype=np.dtype(c_void_p), to_write=False)

    def __init__(self, *args, **kwargs):
        self._cptr = kwargs.pop('cptr', None)
        if self._cptr is None:
            # Allocate data for a single particle
            ptype = self.getPType()
            self._cptr = np.empty(1, dtype=ptype.dtype)[0]
        super(JITParticle, self).__init__(*args, **kwargs)

        fieldset = kwargs.get('fieldset')
        for index in ['xi', 'yi', 'zi', 'ti']:
            if index != 'ti':
                setattr(self, index, np.zeros((fieldset.gridset.size), dtype=np.int32))
            else:
                setattr(self, index, -1*np.ones((fieldset.gridset.size), dtype=np.int32))
            setattr(self, index+'p', getattr(self, index).ctypes.data_as(c_void_p))
            setattr(self, 'c'+index, getattr(self, index+'p').value)

    def update_cptr(self, new_ptr):
        # np.copyto(new_ptr, self._cptr)
        self._cptr = new_ptr
