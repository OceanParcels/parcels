import math
import os
from datetime import timedelta

import numpy as np
import pytest
import xarray as xr

from parcels import (
    AdvectionAnalytical,
    AdvectionDiffusionEM,
    AdvectionDiffusionM1,
    AdvectionEE,
    AdvectionRK4,
    AdvectionRK4_3D,
    AdvectionRK45,
    Field,
    FieldSet,
    JITParticle,
    ParticleSet,
    ScipyParticle,
    StatusCode,
    Variable,
)

ptype = {"scipy": ScipyParticle, "jit": JITParticle}
kernel = {
    "EE": AdvectionEE,
    "RK4": AdvectionRK4,
    "RK45": AdvectionRK45,
    "AA": AdvectionAnalytical,
    "AdvDiffEM": AdvectionDiffusionEM,
    "AdvDiffM1": AdvectionDiffusionM1,
}

# Some constants
f = 1.0e-4
u_0 = 0.3
u_g = 0.04
gamma = 1 / (86400.0 * 2.89)
gamma_g = 1 / (86400.0 * 28.9)


@pytest.fixture
def lon():
    xdim = 200
    return np.linspace(-170, 170, xdim, dtype=np.float32)


@pytest.fixture
def lat():
    ydim = 100
    return np.linspace(-80, 80, ydim, dtype=np.float32)


@pytest.fixture
def depth():
    zdim = 2
    return np.linspace(0, 30, zdim, dtype=np.float32)


@pytest.mark.parametrize("mode", ["scipy", "jit"])
def test_advection_zonal(lon, lat, depth, mode):
    """Particles at high latitude move geographically faster due to the pole correction in `GeographicPolar`."""
    npart = 10
    data2D = {
        "U": np.ones((lon.size, lat.size), dtype=np.float32),
        "V": np.zeros((lon.size, lat.size), dtype=np.float32),
    }
    data3D = {
        "U": np.ones((lon.size, lat.size, depth.size), dtype=np.float32),
        "V": np.zeros((lon.size, lat.size, depth.size), dtype=np.float32),
    }
    dimensions = {"lon": lon, "lat": lat}
    fieldset2D = FieldSet.from_data(data2D, dimensions, mesh="spherical", transpose=True)
    assert fieldset2D.U.creation_log == "from_data"

    pset2D = ParticleSet(fieldset2D, pclass=ptype[mode], lon=np.zeros(npart) + 20.0, lat=np.linspace(0, 80, npart))
    pset2D.execute(AdvectionRK4, runtime=timedelta(hours=2), dt=timedelta(seconds=30))
    assert (np.diff(pset2D.lon) > 1.0e-4).all()

    dimensions["depth"] = depth
    fieldset3D = FieldSet.from_data(data3D, dimensions, mesh="spherical", transpose=True)
    pset3D = ParticleSet(
        fieldset3D,
        pclass=ptype[mode],
        lon=np.zeros(npart) + 20.0,
        lat=np.linspace(0, 80, npart),
        depth=np.zeros(npart) + 10.0,
    )
    pset3D.execute(AdvectionRK4, runtime=timedelta(hours=2), dt=timedelta(seconds=30))
    assert (np.diff(pset3D.lon) > 1.0e-4).all()


@pytest.mark.parametrize("mode", ["scipy", "jit"])
def test_advection_meridional(lon, lat, mode):
    """Particles at high latitude move geographically faster due to the pole correction in `GeographicPolar`."""
    npart = 10
    data = {"U": np.zeros((lon.size, lat.size), dtype=np.float32), "V": np.ones((lon.size, lat.size), dtype=np.float32)}
    dimensions = {"lon": lon, "lat": lat}
    fieldset = FieldSet.from_data(data, dimensions, mesh="spherical", transpose=True)

    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=np.linspace(-60, 60, npart), lat=np.linspace(0, 30, npart))
    delta_lat = np.diff(pset.lat)
    pset.execute(AdvectionRK4, runtime=timedelta(hours=2), dt=timedelta(seconds=30))
    assert np.allclose(np.diff(pset.lat), delta_lat, rtol=1.0e-4)


@pytest.mark.parametrize("mode", ["jit", "scipy"])
def test_advection_3D(mode):
    """Flat 2D zonal flow that increases linearly with depth from 0 m/s to 1 m/s."""
    xdim = ydim = zdim = 2
    npart = 11
    dimensions = {
        "lon": np.linspace(0.0, 1e4, xdim, dtype=np.float32),
        "lat": np.linspace(0.0, 1e4, ydim, dtype=np.float32),
        "depth": np.linspace(0.0, 1.0, zdim, dtype=np.float32),
    }
    data = {"U": np.ones((xdim, ydim, zdim), dtype=np.float32), "V": np.zeros((xdim, ydim, zdim), dtype=np.float32)}
    data["U"][:, :, 0] = 0.0
    fieldset = FieldSet.from_data(data, dimensions, mesh="flat", transpose=True)

    pset = ParticleSet(
        fieldset, pclass=ptype[mode], lon=np.zeros(npart), lat=np.zeros(npart) + 1e2, depth=np.linspace(0, 1, npart)
    )
    time = timedelta(hours=2).total_seconds()
    pset.execute(AdvectionRK4, runtime=time, dt=timedelta(seconds=30))
    assert np.allclose(pset.depth * pset.time, pset.lon, atol=1.0e-1)


@pytest.mark.parametrize("mode", ["jit", "scipy"])
@pytest.mark.parametrize("direction", ["up", "down"])
@pytest.mark.parametrize("wErrorThroughSurface", [True, False])
def test_advection_3D_outofbounds(mode, direction, wErrorThroughSurface):
    xdim = ydim = zdim = 2
    dimensions = {
        "lon": np.linspace(0.0, 1, xdim, dtype=np.float32),
        "lat": np.linspace(0.0, 1, ydim, dtype=np.float32),
        "depth": np.linspace(0.0, 1, zdim, dtype=np.float32),
    }
    wfac = -1.0 if direction == "up" else 1.0
    data = {
        "U": 0.01 * np.ones((xdim, ydim, zdim), dtype=np.float32),
        "V": np.zeros((xdim, ydim, zdim), dtype=np.float32),
        "W": wfac * np.ones((xdim, ydim, zdim), dtype=np.float32),
    }
    fieldset = FieldSet.from_data(data, dimensions, mesh="flat")

    def DeleteParticle(particle, fieldset, time):
        if particle.state == StatusCode.ErrorOutOfBounds or particle.state == StatusCode.ErrorThroughSurface:
            particle.delete()

    def SubmergeParticle(particle, fieldset, time):
        if particle.state == StatusCode.ErrorThroughSurface:
            (u, v) = fieldset.UV[particle]
            particle_dlon = u * particle.dt  # noqa
            particle_dlat = v * particle.dt  # noqa
            particle_ddepth = 0.0  # noqa
            particle.depth = 0
            particle.state = StatusCode.Evaluate

    kernels = [AdvectionRK4_3D]
    if wErrorThroughSurface:
        kernels.append(SubmergeParticle)
    kernels.append(DeleteParticle)

    pset = ParticleSet(fieldset=fieldset, pclass=ptype[mode], lon=0.5, lat=0.5, depth=0.9)
    pset.execute(kernels, runtime=11.0, dt=1)

    if direction == "up" and wErrorThroughSurface:
        assert np.allclose(pset.lon[0], 0.6)
        assert np.allclose(pset.depth[0], 0)
    else:
        assert len(pset) == 0


@pytest.mark.parametrize("mode", ["scipy", "jit"])
@pytest.mark.parametrize("rk45_tol", [10, 100])
def test_advection_RK45(lon, lat, mode, rk45_tol):
    npart = 10
    data2D = {
        "U": np.ones((lon.size, lat.size), dtype=np.float32),
        "V": np.zeros((lon.size, lat.size), dtype=np.float32),
    }
    dimensions = {"lon": lon, "lat": lat}
    fieldset = FieldSet.from_data(data2D, dimensions, mesh="spherical", transpose=True)
    fieldset.add_constant("RK45_tol", rk45_tol)

    dt = timedelta(seconds=30).total_seconds()
    RK45Particles = ptype[mode].add_variable("next_dt", dtype=np.float32, initial=dt)
    pset = ParticleSet(fieldset, pclass=RK45Particles, lon=np.zeros(npart) + 20.0, lat=np.linspace(0, 80, npart))
    pset.execute(AdvectionRK45, runtime=timedelta(hours=2), dt=dt)
    assert (np.diff(pset.lon) > 1.0e-4).all()
    assert np.isclose(fieldset.RK45_tol, rk45_tol / (1852 * 60))
    print(fieldset.RK45_tol)


@pytest.mark.parametrize("mode", ["scipy", "jit"])
def test_advection_3DCROCO(mode):
    data_path = os.path.join(os.path.dirname(__file__), "test_data/")
    fieldset = FieldSet.from_modulefile(data_path + "fieldset_CROCO3D.py")
    assert fieldset.U.creation_log == "from_croco"

    runtime = 1e4
    X, Z = np.meshgrid([40e3, 80e3, 120e3], [-10, -130])
    Y = np.ones(X.size) * 100e3

    pclass = ptype[mode].add_variable(Variable("w"))
    pset = ParticleSet(fieldset=fieldset, pclass=pclass, lon=X, lat=Y, depth=Z)

    def SampleW(particle, fieldset, time):
        particle.w = fieldset.W[time, particle.depth, particle.lat, particle.lon]

    pset.execute([AdvectionRK4_3D, SampleW], runtime=runtime, dt=100)
    assert np.allclose(pset.depth, Z.flatten(), atol=5)  # TODO lower this atol
    assert np.allclose(pset.lon_nextloop, [x + runtime for x in X.flatten()], atol=1e-3)


@pytest.mark.parametrize("mode", ["scipy", "jit"])
def test_advection_2DCROCO(mode):
    data_path = os.path.join(os.path.dirname(__file__), "test_data/")
    fieldset = FieldSet.from_modulefile(data_path + "fieldset_CROCO2D.py")
    assert fieldset.U.creation_log == "from_croco"

    runtime = 1e4
    X = np.array([40e3, 80e3, 120e3])
    Y = np.ones(X.size) * 100e3
    Z = np.zeros(X.size)
    pset = ParticleSet(fieldset=fieldset, pclass=ptype[mode], lon=X, lat=Y, depth=Z)

    pset.execute([AdvectionRK4], runtime=runtime, dt=100)
    assert np.allclose(pset.depth, Z.flatten(), atol=1e-3)
    assert np.allclose(pset.lon_nextloop, [x + runtime for x in X], atol=1e-3)


def create_periodic_fieldset(xdim, ydim, uvel, vvel):
    dimensions = {
        "lon": np.linspace(0.0, 1.0, xdim + 1, dtype=np.float32)[1:],  # don't include both 0 and 1, for periodic b.c.
        "lat": np.linspace(0.0, 1.0, ydim + 1, dtype=np.float32)[1:],
    }

    data = {"U": uvel * np.ones((xdim, ydim), dtype=np.float32), "V": vvel * np.ones((xdim, ydim), dtype=np.float32)}
    return FieldSet.from_data(data, dimensions, mesh="spherical", transpose=True)


def periodicBC(particle, fieldset, time):
    particle.lon = math.fmod(particle.lon, 1)
    particle.lat = math.fmod(particle.lat, 1)


@pytest.mark.parametrize("mode", ["scipy", "jit"])
def test_advection_periodic_zonal(mode):
    xdim, ydim, halosize = 100, 100, 3
    fieldset = create_periodic_fieldset(xdim, ydim, uvel=1.0, vvel=0.0)
    fieldset.add_periodic_halo(zonal=True, halosize=halosize)
    assert len(fieldset.U.lon) == xdim + 2 * halosize

    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=[0.5], lat=[0.5])
    pset.execute(AdvectionRK4 + pset.Kernel(periodicBC), runtime=timedelta(hours=20), dt=timedelta(seconds=30))
    assert abs(pset.lon[0] - 0.15) < 0.1


@pytest.mark.parametrize("mode", ["scipy", "jit"])
def test_advection_periodic_meridional(mode):
    xdim, ydim = 100, 100
    fieldset = create_periodic_fieldset(xdim, ydim, uvel=0.0, vvel=1.0)
    fieldset.add_periodic_halo(meridional=True)
    assert len(fieldset.U.lat) == ydim + 10  # default halo size is 5 grid points

    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=[0.5], lat=[0.5])
    pset.execute(AdvectionRK4 + pset.Kernel(periodicBC), runtime=timedelta(hours=20), dt=timedelta(seconds=30))
    assert abs(pset.lat[0] - 0.15) < 0.1


@pytest.mark.parametrize("mode", ["scipy", "jit"])
def test_advection_periodic_zonal_meridional(mode):
    xdim, ydim = 100, 100
    fieldset = create_periodic_fieldset(xdim, ydim, uvel=1.0, vvel=1.0)
    fieldset.add_periodic_halo(zonal=True, meridional=True)
    assert len(fieldset.U.lat) == ydim + 10  # default halo size is 5 grid points
    assert len(fieldset.U.lon) == xdim + 10  # default halo size is 5 grid points
    assert np.allclose(np.diff(fieldset.U.lat), fieldset.U.lat[1] - fieldset.U.lat[0], rtol=0.001)
    assert np.allclose(np.diff(fieldset.U.lon), fieldset.U.lon[1] - fieldset.U.lon[0], rtol=0.001)

    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=[0.4], lat=[0.5])
    pset.execute(AdvectionRK4 + pset.Kernel(periodicBC), runtime=timedelta(hours=20), dt=timedelta(seconds=30))
    assert abs(pset.lon[0] - 0.05) < 0.1
    assert abs(pset.lat[0] - 0.15) < 0.1


@pytest.mark.parametrize("mode", ["scipy", "jit"])
@pytest.mark.parametrize("u", [-0.3, np.array(0.2)])
@pytest.mark.parametrize("v", [0.2, np.array(1)])
@pytest.mark.parametrize("w", [None, -0.2, np.array(0.7)])
def test_length1dimensions(mode, u, v, w):
    (lon, xdim) = (np.linspace(-10, 10, 21), 21) if isinstance(u, np.ndarray) else (0, 1)
    (lat, ydim) = (np.linspace(-15, 15, 31), 31) if isinstance(v, np.ndarray) else (-4, 1)
    (depth, zdim) = (np.linspace(-5, 5, 11), 11) if (isinstance(w, np.ndarray) and w is not None) else (3, 1)
    dimensions = {"lon": lon, "lat": lat, "depth": depth}

    dims = []
    if zdim > 1:
        dims.append(zdim)
    if ydim > 1:
        dims.append(ydim)
    if xdim > 1:
        dims.append(xdim)
    if len(dims) > 0:
        U = u * np.ones(dims, dtype=np.float32)
        V = v * np.ones(dims, dtype=np.float32)
        if w is not None:
            W = w * np.ones(dims, dtype=np.float32)
    else:
        U, V, W = u, v, w

    data = {"U": U, "V": V}
    if w is not None:
        data["W"] = W
    fieldset = FieldSet.from_data(data, dimensions, mesh="flat")

    x0, y0, z0 = 2, 8, -4
    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=x0, lat=y0, depth=z0)
    pfunc = AdvectionRK4 if w is None else AdvectionRK4_3D
    kernel = pset.Kernel(pfunc)
    pset.execute(kernel, runtime=5, dt=1)

    assert len(pset.lon) == len([p.lon for p in pset])
    assert ((np.array([p.lon - x0 for p in pset]) - 4 * u) < 1e-6).all()
    assert ((np.array([p.lat - y0 for p in pset]) - 4 * v) < 1e-6).all()
    if w:
        assert ((np.array([p.depth - y0 for p in pset]) - 4 * w) < 1e-6).all()


def truth_stationary(x_0, y_0, t):
    lat = y_0 - u_0 / f * (1 - math.cos(f * t))
    lon = x_0 + u_0 / f * math.sin(f * t)
    return lon, lat


def create_fieldset_stationary(xdim=100, ydim=100, maxtime=timedelta(hours=6)):
    """Generate a FieldSet encapsulating the flow field of a stationary eddy.

    Reference: N. Fabbroni, 2009, "Numerical simulations of passive
    tracers dispersion in the sea"
    """
    time = np.arange(0.0, maxtime.total_seconds() + 1e-5, 60.0, dtype=np.float64)
    dimensions = {
        "lon": np.linspace(0, 25000, xdim, dtype=np.float32),
        "lat": np.linspace(0, 25000, ydim, dtype=np.float32),
        "time": time,
    }
    data = {
        "U": np.ones((xdim, ydim, 1), dtype=np.float32) * u_0 * np.cos(f * time),
        "V": np.ones((xdim, ydim, 1), dtype=np.float32) * -u_0 * np.sin(f * time),
    }
    fieldset = FieldSet.from_data(data, dimensions, mesh="flat", transpose=True)
    # setting some constants for AdvectionRK45 kernel
    fieldset.RK45_min_dt = 1e-3
    fieldset.RK45_max_dt = 1e2
    fieldset.RK45_tol = 1e-5
    return fieldset


@pytest.fixture
def fieldset_stationary():
    return create_fieldset_stationary()


@pytest.mark.parametrize("mode", ["scipy", "jit"])
@pytest.mark.parametrize(
    "method, rtol, diffField",
    [
        ("EE", 1e-2, False),
        ("AdvDiffEM", 1e-2, True),
        ("AdvDiffM1", 1e-2, True),
        ("RK4", 1e-5, False),
        ("RK45", 1e-5, False),
    ],
)
def test_stationary_eddy(fieldset_stationary, mode, method, rtol, diffField):
    npart = 1
    fieldset = fieldset_stationary
    if diffField:
        fieldset.add_field(Field("Kh_zonal", np.zeros(fieldset.U.data.shape), grid=fieldset.U.grid))
        fieldset.add_field(Field("Kh_meridional", np.zeros(fieldset.V.data.shape), grid=fieldset.V.grid))
        fieldset.add_constant("dres", 0.1)
    lon = np.linspace(12000, 21000, npart)
    lat = np.linspace(12500, 12500, npart)
    dt = timedelta(minutes=3).total_seconds()
    endtime = timedelta(hours=6).total_seconds()

    RK45Particles = ptype[mode].add_variable("next_dt", dtype=np.float32, initial=dt)

    pclass = RK45Particles if method == "RK45" else ptype[mode]
    pset = ParticleSet(fieldset, pclass=pclass, lon=lon, lat=lat)
    pset.execute(kernel[method], dt=dt, endtime=endtime)

    exp_lon = [truth_stationary(x, y, pset[0].time)[0] for x, y in zip(lon, lat, strict=True)]
    exp_lat = [truth_stationary(x, y, pset[0].time)[1] for x, y in zip(lon, lat, strict=True)]
    assert np.allclose(pset.lon, exp_lon, rtol=rtol)
    assert np.allclose(pset.lat, exp_lat, rtol=rtol)


@pytest.mark.parametrize("mode", ["scipy", "jit"])
def test_stationary_eddy_vertical(mode):
    npart = 1
    lon = np.linspace(12000, 21000, npart)
    lat = np.linspace(10000, 20000, npart)
    depth = np.linspace(12500, 12500, npart)
    endtime = timedelta(hours=6).total_seconds()
    dt = timedelta(minutes=3).total_seconds()

    xdim = ydim = 100
    lon_data = np.linspace(0, 25000, xdim, dtype=np.float32)
    lat_data = np.linspace(0, 25000, ydim, dtype=np.float32)
    time_data = np.arange(0.0, 6 * 3600 + 1e-5, 60.0, dtype=np.float64)
    fld1 = np.ones((xdim, ydim, 1), dtype=np.float32) * u_0 * np.cos(f * time_data)
    fld2 = np.ones((xdim, ydim, 1), dtype=np.float32) * -u_0 * np.sin(f * time_data)
    fldzero = np.zeros((xdim, ydim, 1), dtype=np.float32) * time_data

    dimensions = {"lon": lon_data, "lat": lat_data, "time": time_data}
    data = {"U": fld1, "V": fldzero, "W": fld2}
    fieldset = FieldSet.from_data(data, dimensions, mesh="flat", transpose=True)

    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=lon, lat=lat, depth=depth)
    pset.execute(AdvectionRK4_3D, dt=dt, endtime=endtime)
    exp_lon = [truth_stationary(x, z, pset[0].time)[0] for x, z in zip(lon, depth, strict=True)]
    exp_depth = [truth_stationary(x, z, pset[0].time)[1] for x, z in zip(lon, depth, strict=True)]
    print(pset, exp_lon)
    assert np.allclose(pset.lon, exp_lon, rtol=1e-5)
    assert np.allclose(pset.lat, lat, rtol=1e-5)
    assert np.allclose(pset.depth, exp_depth, rtol=1e-5)

    data = {"U": fldzero, "V": fld2, "W": fld1}
    fieldset = FieldSet.from_data(data, dimensions, mesh="flat", transpose=True)

    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=lon, lat=lat, depth=depth)
    pset.execute(AdvectionRK4_3D, dt=dt, endtime=endtime)
    exp_depth = [truth_stationary(z, y, pset[0].time)[0] for z, y in zip(depth, lat, strict=True)]
    exp_lat = [truth_stationary(z, y, pset[0].time)[1] for z, y in zip(depth, lat, strict=True)]
    assert np.allclose(pset.lon, lon, rtol=1e-5)
    assert np.allclose(pset.lat, exp_lat, rtol=1e-5)
    assert np.allclose(pset.depth, exp_depth, rtol=1e-5)


def truth_moving(x_0, y_0, t):
    lat = y_0 - (u_0 - u_g) / f * (1 - math.cos(f * t))
    lon = x_0 + u_g * t + (u_0 - u_g) / f * math.sin(f * t)
    return lon, lat


def create_fieldset_moving(xdim=100, ydim=100, maxtime=timedelta(hours=6)):
    """Generate a FieldSet encapsulating the flow field of a moving eddy.

    Reference: N. Fabbroni, 2009, "Numerical simulations of passive
    tracers dispersion in the sea"
    """
    time = np.arange(0.0, maxtime.total_seconds() + 1e-5, 60.0, dtype=np.float64)
    dimensions = {
        "lon": np.linspace(0, 25000, xdim, dtype=np.float32),
        "lat": np.linspace(0, 25000, ydim, dtype=np.float32),
        "time": time,
    }
    data = {
        "U": np.ones((xdim, ydim, 1), dtype=np.float32) * u_g + (u_0 - u_g) * np.cos(f * time),
        "V": np.ones((xdim, ydim, 1), dtype=np.float32) * -(u_0 - u_g) * np.sin(f * time),
    }
    return FieldSet.from_data(data, dimensions, mesh="flat", transpose=True)


@pytest.fixture
def fieldset_moving():
    return create_fieldset_moving()


@pytest.mark.parametrize("mode", ["scipy", "jit"])
@pytest.mark.parametrize(
    "method, rtol, diffField",
    [
        ("EE", 1e-2, False),
        ("AdvDiffEM", 1e-2, True),
        ("AdvDiffM1", 1e-2, True),
        ("RK4", 1e-5, False),
        ("RK45", 1e-5, False),
    ],
)
def test_moving_eddy(fieldset_moving, mode, method, rtol, diffField):
    npart = 1
    fieldset = fieldset_moving
    if diffField:
        fieldset.add_field(Field("Kh_zonal", np.zeros(fieldset.U.data.shape), grid=fieldset.U.grid))
        fieldset.add_field(Field("Kh_meridional", np.zeros(fieldset.V.data.shape), grid=fieldset.V.grid))
        fieldset.add_constant("dres", 0.1)
    lon = np.linspace(12000, 21000, npart)
    lat = np.linspace(12500, 12500, npart)
    dt = timedelta(minutes=3).total_seconds()
    endtime = timedelta(hours=6).total_seconds()

    RK45Particles = ptype[mode].add_variable("next_dt", dtype=np.float32, initial=dt)

    pclass = RK45Particles if method == "RK45" else ptype[mode]
    pset = ParticleSet(fieldset, pclass=pclass, lon=lon, lat=lat)
    pset.execute(kernel[method], dt=dt, endtime=endtime)

    exp_lon = [truth_moving(x, y, t)[0] for x, y, t in zip(lon, lat, pset.time, strict=True)]
    exp_lat = [truth_moving(x, y, t)[1] for x, y, t in zip(lon, lat, pset.time, strict=True)]
    assert np.allclose(pset.lon, exp_lon, rtol=rtol)
    assert np.allclose(pset.lat, exp_lat, rtol=rtol)


def truth_decaying(x_0, y_0, t):
    lat = y_0 - (
        (u_0 - u_g) * f / (f**2 + gamma**2) * (1 - np.exp(-gamma * t) * (np.cos(f * t) + gamma / f * np.sin(f * t)))
    )
    lon = x_0 + (
        u_g / gamma_g * (1 - np.exp(-gamma_g * t))
        + (u_0 - u_g)
        * f
        / (f**2 + gamma**2)
        * (gamma / f + np.exp(-gamma * t) * (math.sin(f * t) - gamma / f * math.cos(f * t)))
    )
    return lon, lat


def create_fieldset_decaying(xdim=100, ydim=100, maxtime=timedelta(hours=6)):
    """Generate a FieldSet encapsulating the flow field of a decaying eddy.

    Reference: N. Fabbroni, 2009, "Numerical simulations of passive
    tracers dispersion in the sea"
    """
    time = np.arange(0.0, maxtime.total_seconds() + 1e-5, 60.0, dtype=np.float64)
    dimensions = {
        "lon": np.linspace(0, 25000, xdim, dtype=np.float32),
        "lat": np.linspace(0, 25000, ydim, dtype=np.float32),
        "time": time,
    }
    data = {
        "U": np.ones((xdim, ydim, 1), dtype=np.float32) * u_g * np.exp(-gamma_g * time)
        + (u_0 - u_g) * np.exp(-gamma * time) * np.cos(f * time),
        "V": np.ones((xdim, ydim, 1), dtype=np.float32) * -(u_0 - u_g) * np.exp(-gamma * time) * np.sin(f * time),
    }
    return FieldSet.from_data(data, dimensions, mesh="flat", transpose=True)


@pytest.fixture
def fieldset_decaying():
    return create_fieldset_decaying()


@pytest.mark.parametrize("mode", ["scipy", "jit"])
@pytest.mark.parametrize(
    "method, rtol, diffField",
    [
        ("EE", 1e-2, False),
        ("AdvDiffEM", 1e-2, True),
        ("AdvDiffM1", 1e-2, True),
        ("RK4", 1e-5, False),
        ("RK45", 1e-5, False),
        ("AA", 1e-3, False),
    ],
)
def test_decaying_eddy(fieldset_decaying, mode, method, rtol, diffField):
    npart = 1
    fieldset = fieldset_decaying
    if method == "AA":
        if mode == "jit":
            return  # AnalyticalAdvection not implemented in JIT
        else:
            # needed for AnalyticalAdvection to work, but comes at expense of accuracy
            fieldset.U.interp_method = "cgrid_velocity"
            fieldset.V.interp_method = "cgrid_velocity"

    if diffField:
        fieldset.add_field(Field("Kh_zonal", np.zeros(fieldset.U.data.shape), grid=fieldset.U.grid))
        fieldset.add_field(Field("Kh_meridional", np.zeros(fieldset.V.data.shape), grid=fieldset.V.grid))
        fieldset.add_constant("dres", 0.1)
    lon = np.linspace(12000, 21000, npart)
    lat = np.linspace(12500, 12500, npart)
    dt = timedelta(minutes=3).total_seconds()
    endtime = timedelta(hours=6).total_seconds()

    RK45Particles = ptype[mode].add_variable("next_dt", dtype=np.float32, initial=dt)

    pclass = RK45Particles if method == "RK45" else ptype[mode]
    pset = ParticleSet(fieldset, pclass=pclass, lon=lon, lat=lat)
    pset.execute(kernel[method], dt=dt, endtime=endtime)

    exp_lon = [truth_decaying(x, y, t)[0] for x, y, t in zip(lon, lat, pset.time, strict=True)]
    exp_lat = [truth_decaying(x, y, t)[1] for x, y, t in zip(lon, lat, pset.time, strict=True)]
    assert np.allclose(pset.lon, exp_lon, rtol=rtol)
    assert np.allclose(pset.lat, exp_lat, rtol=rtol)


@pytest.mark.parametrize("mode", ["scipy", "jit"])
def test_analyticalAgrid(mode):
    lon = np.arange(0, 15, dtype=np.float32)
    lat = np.arange(0, 15, dtype=np.float32)
    U = np.ones((lat.size, lon.size), dtype=np.float32)
    V = np.ones((lat.size, lon.size), dtype=np.float32)
    fieldset = FieldSet.from_data({"U": U, "V": V}, {"lon": lon, "lat": lat}, mesh="flat")
    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=1, lat=1)

    with pytest.raises(NotImplementedError):
        pset.execute(AdvectionAnalytical, runtime=1)


@pytest.mark.parametrize("mode", ["scipy"])  # JIT not implemented
@pytest.mark.parametrize("u", [1, -0.2, -0.3, 0])
@pytest.mark.parametrize("v", [1, -0.3, 0, -1])
@pytest.mark.parametrize("w", [None, 1, -0.3, 0, -1])
@pytest.mark.parametrize("direction", [1, -1])
def test_uniform_analytical(mode, u, v, w, direction, tmpdir):
    lon = np.arange(0, 15, dtype=np.float32)
    lat = np.arange(0, 15, dtype=np.float32)
    if w is not None:
        depth = np.arange(0, 40, 2, dtype=np.float32)
        U = u * np.ones((depth.size, lat.size, lon.size), dtype=np.float32)
        V = v * np.ones((depth.size, lat.size, lon.size), dtype=np.float32)
        W = w * np.ones((depth.size, lat.size, lon.size), dtype=np.float32)
        fieldset = FieldSet.from_data({"U": U, "V": V, "W": W}, {"lon": lon, "lat": lat, "depth": depth}, mesh="flat")
        fieldset.W.interp_method = "cgrid_velocity"
    else:
        U = u * np.ones((lat.size, lon.size), dtype=np.float32)
        V = v * np.ones((lat.size, lon.size), dtype=np.float32)
        fieldset = FieldSet.from_data({"U": U, "V": V}, {"lon": lon, "lat": lat}, mesh="flat")
    fieldset.U.interp_method = "cgrid_velocity"
    fieldset.V.interp_method = "cgrid_velocity"

    x0, y0, z0 = 6.1, 6.2, 20
    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=x0, lat=y0, depth=z0)

    outfile_path = tmpdir.join("uniformanalytical.zarr")
    outfile = pset.ParticleFile(name=outfile_path, outputdt=1, chunks=(1, 1))
    pset.execute(AdvectionAnalytical, runtime=4, dt=direction, output_file=outfile)
    assert np.abs(pset.lon - x0 - pset.time * u) < 1e-6
    assert np.abs(pset.lat - y0 - pset.time * v) < 1e-6
    if w is not None:
        assert np.abs(pset.depth - z0 - pset.time * w) < 1e-4

    ds = xr.open_zarr(outfile_path)
    times = (direction * ds["time"][:]).values.astype("timedelta64[s]")[0]
    timeref = np.arange(1, 5).astype("timedelta64[s]")
    assert np.allclose(times, timeref, atol=np.timedelta64(1, "ms"))
    lons = ds["lon"][:].values
    assert np.allclose(lons, x0 + direction * u * np.arange(1, 5))
