import random as py_random
#from parcels import random as parcels_random => this doesn't work, and we should not 'ghost' up which random module is used; the module shoudl really get a different name
from parcels import random as parcels_random
from parcels.rng import parcels_random as pa_random
import numpy as np
from parcels import FieldSet, ScipyParticle, JITParticle, Variable, ErrorCode
from parcels.particleset_vectorized import ParticleSet
from parcels.kernel_vectorized import Kernel
from parcels.kernels.seawaterdensity import polyTEOS10_bsq, UNESCO_Density
import pytest
from os import path
import sys


ptype = {'scipy': ScipyParticle, 'jit': JITParticle}


def expr_kernel(name, pset, expr):
    pycode = """def %s(particle, fieldset, time):
    particle.p = %s
    return ErrorCode.Success""" % (name, expr)
    return Kernel(pset.fieldset, pset.ptype, pyfunc=None,
                  funccode=pycode, funcname=name,
                  funcvars=['particle'])


def fieldset(xdim=20, ydim=20):
    """ Standard unit mesh fieldset """
    lon = np.linspace(0., 1., xdim, dtype=np.float32)
    lat = np.linspace(0., 1., ydim, dtype=np.float32)
    U, V = np.meshgrid(lat, lon)
    data = {'U': np.array(U, dtype=np.float32), 'V': np.array(V, dtype=np.float32)}
    dimensions = {'lat': lat, 'lon': lon}
    return FieldSet.from_data(data, dimensions, mesh='flat', transpose=True)


@pytest.fixture(name="fieldset")
def fieldset_fixture(xdim=20, ydim=20):
    return fieldset(xdim=xdim, ydim=ydim)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('name, expr, result', [
    ('Add', '2 + 5', 7),
    ('Sub', '6 - 2', 4),
    ('Mul', '3 * 5', 15),
    ('Div', '24 / 4', 6),
])
def test_expression_int(fieldset, mode, name, expr, result, npart=10):
    """ Test basic arithmetic expressions """
    class TestParticle(ptype[mode]):
        p = Variable('p', dtype=np.float32)
    pset = ParticleSet(fieldset, pclass=TestParticle,
                       lon=np.linspace(0., 1., npart),
                       lat=np.zeros(npart) + 0.5)
    pset.execute(expr_kernel('Test%s' % name, pset, expr), endtime=1., dt=1.)
    assert(np.array([result == particle.p for particle in pset]).all())


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('name, expr, result', [
    ('Add', '2. + 5.', 7),
    ('Sub', '6. - 2.', 4),
    ('Mul', '3. * 5.', 15),
    ('Div', '24. / 4.', 6),
    ('Pow', '2 ** 3', 8),
])
def test_expression_float(fieldset, mode, name, expr, result, npart=10):
    """ Test basic arithmetic expressions """
    class TestParticle(ptype[mode]):
        p = Variable('p', dtype=np.float32)
    pset = ParticleSet(fieldset, pclass=TestParticle,
                       lon=np.linspace(0., 1., npart),
                       lat=np.zeros(npart) + 0.5)
    pset.execute(expr_kernel('Test%s' % name, pset, expr), endtime=1., dt=1.)
    assert(np.array([result == particle.p for particle in pset]).all())


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('name, expr, result', [
    ('True', 'True', True),
    ('False', 'False', False),
    ('And', 'True and False', False),
    ('Or', 'True or False', True),
    ('Equal', '5 == 5', True),
    ('Lesser', '5 < 3', False),
    ('LesserEq', '3 <= 5', True),
    ('Greater', '4 > 2', True),
    ('GreaterEq', '2 >= 4', False),
])
def test_expression_bool(fieldset, mode, name, expr, result, npart=10):
    """ Test basic arithmetic expressions """
    class TestParticle(ptype[mode]):
        p = Variable('p', dtype=np.float32)
    pset = ParticleSet(fieldset, pclass=TestParticle,
                       lon=np.linspace(0., 1., npart),
                       lat=np.zeros(npart) + 0.5)
    pset.execute(expr_kernel('Test%s' % name, pset, expr), endtime=1., dt=1.)
    if mode == 'jit':
        assert(np.array([result == (particle.p == 1) for particle in pset]).all())
    else:
        assert(np.array([result == particle.p for particle in pset]).all())


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_while_if_break(fieldset, mode):
    """Test while, if and break commands"""
    class TestParticle(ptype[mode]):
        p = Variable('p', dtype=np.float32, initial=0.)
    pset = ParticleSet(fieldset, pclass=TestParticle, lon=[0], lat=[0])

    def kernel(particle, fieldset, time):
        while particle.p < 30:
            if particle.p > 9:
                break
            particle.p += 1
        if particle.p > 5:
            particle.p *= 2.
    pset.execute(kernel, endtime=1., dt=1.)
    assert np.allclose(np.array([p.p for p in pset]), 20., rtol=1e-12)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_nested_if(fieldset, mode):
    """Test nested if commands"""
    class TestParticle(ptype[mode]):
        p0 = Variable('p0', dtype=np.int32, initial=0)
        p1 = Variable('p1', dtype=np.int32, initial=1)
    pset = ParticleSet(fieldset, pclass=TestParticle, lon=0, lat=0)

    def kernel(particle, fieldset, time):
        if particle.p1 >= particle.p0:
            var = particle.p0
            if var + 1 < particle.p1:
                particle.p1 = -1

    pset.execute(kernel, endtime=10, dt=1.)
    assert np.allclose([pset[0].p0, pset[0].p1], [0, 1])


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pass(fieldset, mode):
    """Test pass commands"""
    class TestParticle(ptype[mode]):
        p = Variable('p', dtype=np.int32, initial=0)
    pset = ParticleSet(fieldset, pclass=TestParticle, lon=0, lat=0)

    def kernel(particle, fieldset, time):
        particle.p = -1
        pass

    pset.execute(kernel, endtime=10, dt=1.)
    assert np.allclose(pset[0].p, -1)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_dt_as_variable_in_kernel(fieldset, mode):
    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=0, lat=0)

    def kernel(particle, fieldset, time):
        dt = 1.  # noqa

    pset.execute(kernel, endtime=10, dt=1.)


def test_parcels_tmpvar_in_kernel(fieldset):
    """Tests for error thrown if vartiable with 'tmp' defined in custom kernel"""
    error_thrown = False
    pset = ParticleSet(fieldset, pclass=JITParticle, lon=0, lat=0)

    def kernel_tmpvar(particle, fieldset, time):
        parcels_tmpvar0 = 0  # noqa

    try:
        pset.execute(kernel_tmpvar, endtime=1, dt=1.)
    except NotImplementedError:
        error_thrown = True
    assert error_thrown


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_if_withfield(fieldset, mode):
    """Test combination of if and Field sampling commands"""
    class TestParticle(ptype[mode]):
        p = Variable('p', dtype=np.float32, initial=0.)
    pset = ParticleSet(fieldset, pclass=TestParticle, lon=[0], lat=[0])

    def kernel(particle, fieldset, time):
        u = fieldset.U[time, 0, 0, 1.]
        particle.p = 0
        if fieldset.U[time, 0, 0, 1.] == u:
            particle.p += 1
        if fieldset.U[time, 0, 0, 1.] == fieldset.U[time, 0, 0, 1.]:
            particle.p += 1
        if True:
            particle.p += 1
        if fieldset.U[time, 0, 0, 1.] == u and 1 == 1:
            particle.p += 1
        if fieldset.U[time, 0, 0, 1.] == fieldset.U[time, 0, 0, 1.] and fieldset.U[time, 0, 0, 1.] == fieldset.U[time, 0, 0, 1.]:
            particle.p += 1
        if fieldset.U[time, 0, 0, 1.] == u:
            particle.p += 1
        else:
            particle.p += 1000
        if fieldset.U[time, 0, 0, 1.] == 3:
            particle.p += 1000
        else:
            particle.p += 1

    pset.execute(kernel, endtime=1., dt=1.)
    assert np.allclose(np.array([p.p for p in pset]), 7., rtol=1e-12)


@pytest.mark.parametrize(
    'mode',
    ['scipy',
     pytest.param('jit',
                  marks=pytest.mark.xfail(
                      (sys.version_info >= (3, 0)) or (sys.platform == 'win32'),
                      reason="py.test FD capturing does not work for jit on python3 or Win"))
     ])
def test_print(fieldset, mode, capfd):
    """Test print statements"""
    class TestParticle(ptype[mode]):
        p = Variable('p', dtype=np.float32, initial=0.)
    pset = ParticleSet(fieldset, pclass=TestParticle, lon=[0.5], lat=[0.5])

    def kernel(particle, fieldset, time):
        particle.p = fieldset.U[time, particle.depth, particle.lat, particle.lon]
        tmp = 5
        print("%d %f %f" % (particle.id, particle.p, tmp))
    pset.execute(kernel, endtime=1., dt=1.)
    out, err = capfd.readouterr()
    lst = out.split(' ')
    tol = 1e-8
    assert abs(float(lst[0]) - pset[0].id) < tol and abs(float(lst[1]) - pset[0].p) < tol and abs(float(lst[2]) - 5) < tol

    def kernel2(particle, fieldset, time):
        tmp = 3
        print("%f" % (tmp))
    pset.execute(kernel2, endtime=1., dt=1.)
    out, err = capfd.readouterr()
    lst = out.split(' ')
    assert abs(float(lst[0]) - 3) < tol


def random_series(npart, rngfunc, rngargs, mode):
    random_mod = parcels_random if mode == 'jit' else py_random
    print(repr(random_mod))
    random_mod.seed(1234)
    func = getattr(random_mod, rngfunc)
    series = [func(*rngargs) for _ in range(npart)]
    random_mod.seed(1234)  # Reset the RNG seed
    return series


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('rngfunc, rngargs', [
    ('random', []),
    ('uniform', [0., 20.]),
    ('randint', [0, 20]),
])
def test_random_float(fieldset, mode, rngfunc, rngargs, npart=10):
    """ Test basic random number generation """
    class TestParticle(ptype[mode]):
        p = Variable('p', dtype=np.int32 if rngfunc == 'randint' else np.float32)
    pset = ParticleSet(fieldset, pclass=TestParticle,
                       lon=np.linspace(0., 1., npart),
                       lat=np.zeros(npart) + 0.5)
    series = random_series(npart, rngfunc, rngargs, mode)
    kernel = expr_kernel('TestRandom_%s' % rngfunc, pset,
                         'random.%s(%s)' % (rngfunc, ', '.join([str(a) for a in rngargs])))
    pset.execute(kernel, endtime=1., dt=1.)
    assert np.allclose(np.array([p.p for p in pset]), series, atol=1e-9)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('c_inc', ['str', 'file'])
def test_c_kernel(fieldset, mode, c_inc):
    coord_type = np.float32 if c_inc == 'str' else np.float64
    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=[0.5], lat=[0],
                       lonlatdepth_dtype=coord_type)

    def func(U, lon, dt):
        u = U.data[0, 2, 1]
        return lon + u * dt

    if c_inc == 'str':
        c_include = """
                 static inline ErrorCode func(CField *f, float *lon, double *dt)
                 {
                   float data2D[2][2][2];
                   ErrorCode err = getCell2D(f, 1, 2, 0, data2D, 1); CHECKERROR(err);
                   float u = data2D[0][0][0];
                   *lon += u * *dt;
                   return SUCCESS;
                 }
                 """
    else:
        c_include = path.join(path.dirname(__file__), 'customed_header.h')

    def ckernel(particle, fieldset, time):
        func('parcels_customed_Cfunc_pointer_args', fieldset.U, particle.lon, particle.dt)

    def pykernel(particle, fieldset, time):
        particle.lon = func(fieldset.U, particle.lon, particle.dt)

    if mode == 'scipy':
        kernel = pset.Kernel(pykernel)
    else:
        kernel = pset.Kernel(ckernel, c_include=c_include)
    pset.execute(kernel, endtime=3., dt=3.)
    assert np.allclose(pset[0].lon, 0.81578948)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_dt_modif_by_kernel(fieldset, mode):
    class TestParticle(ptype[mode]):
        age = Variable('age', dtype=np.float32)
    pset = ParticleSet(fieldset, pclass=TestParticle, lon=[0.5], lat=[0])

    def modif_dt(particle, fieldset, time):
        particle.age += particle.dt
        particle.dt = 2

    endtime = 4
    pset.execute(modif_dt, endtime=endtime, dt=1.)
    assert np.isclose(pset[0].age, endtime)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('dt', [1e-2, 1e-6])
def test_small_dt(fieldset, mode, dt, npart=10):
    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=np.zeros(npart),
                       lat=np.zeros(npart), time=np.arange(0, npart)*dt*10)

    def DoNothing(particle, fieldset, time):
        return ErrorCode.Success

    pset.execute(DoNothing, dt=dt, runtime=dt*100)
    assert np.allclose([p.time for p in pset], dt*100)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_seawaterdensity_kernels(mode):

    def generate_fieldset(xdim=2, ydim=2, zdim=2, tdim=1):
        lon = np.linspace(0., 10., xdim, dtype=np.float32)
        lat = np.linspace(0., 10., ydim, dtype=np.float32)
        depth = np.linspace(0, 2000, zdim, dtype=np.float32)
        time = np.zeros(tdim, dtype=np.float64)
        U = np.ones((tdim, zdim, ydim, xdim))
        V = np.ones((tdim, zdim, ydim, xdim))
        abs_salinity = 30 * np.ones((tdim, zdim, ydim, xdim))
        cons_temperature = 10 * np.ones((tdim, zdim, ydim, xdim))
        dimensions = {'lat': lat, 'lon': lon, 'depth': depth, 'time': time}
        data = {'U': np.array(U, dtype=np.float32), 'V': np.array(V, dtype=np.float32),
                'abs_salinity': np.array(abs_salinity, dtype=np.float32),
                'cons_temperature': np.array(cons_temperature, dtype=np.float32)}
        return (data, dimensions)

    data, dimensions = generate_fieldset()
    fieldset = FieldSet.from_data(data, dimensions)

    class DensParticle(ptype[mode]):
        density = Variable('density', dtype=np.float32)

    pset = ParticleSet(fieldset, pclass=DensParticle, lon=5, lat=5, depth=1000)

    pset.execute(polyTEOS10_bsq, runtime=0, dt=0)
    assert np.allclose(pset[0].density, 1022.85377)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('pressure', [0, 10])
def test_UNESCOdensity_kernel(mode, pressure):

    def generate_fieldset(p, xdim=2, ydim=2, zdim=2, tdim=1):
        lon = np.linspace(0., 10., xdim, dtype=np.float32)
        lat = np.linspace(0., 10., ydim, dtype=np.float32)
        depth = np.linspace(0, 2000, zdim, dtype=np.float32)
        time = np.zeros(tdim, dtype=np.float64)
        U = np.ones((tdim, zdim, ydim, xdim))
        V = np.ones((tdim, zdim, ydim, xdim))
        psu_salinity = 8 * np.ones((tdim, zdim, ydim, xdim))
        cons_temperature = 10 * np.ones((tdim, zdim, ydim, xdim))
        cons_pressure = p * np.ones((tdim, zdim, ydim, xdim))
        dimensions = {'lat': lat, 'lon': lon, 'depth': depth, 'time': time}
        data = {'U': np.array(U, dtype=np.float32), 'V': np.array(V, dtype=np.float32),
                'psu_salinity': np.array(psu_salinity, dtype=np.float32),
                'cons_pressure': np.array(cons_pressure, dtype=np.float32),
                'cons_temperature': np.array(cons_temperature, dtype=np.float32)}
        return (data, dimensions)

    data, dimensions = generate_fieldset(pressure)
    fieldset = FieldSet.from_data(data, dimensions)

    class DensParticle(ptype[mode]):
        density = Variable('density', dtype=np.float32)

    pset = ParticleSet(fieldset, pclass=DensParticle, lon=5, lat=5, depth=1000)

    pset.execute(UNESCO_Density, runtime=0, dt=0)

    if(pressure == 0):
        assert np.allclose(pset[0].density, 1005.9465)
    elif(pressure == 10):
        assert np.allclose(pset[0].density, 1006.4179)