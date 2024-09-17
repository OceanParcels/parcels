import os
from datetime import timedelta
from math import cos, pi

import numpy as np
import pytest
import xarray as xr

from parcels import (
    AdvectionRK4,
    AdvectionRK4_3D,
    Field,
    FieldSet,
    Geographic,
    JITParticle,
    NestedField,
    ParticleSet,
    ScipyParticle,
    StatusCode,
    Variable,
)
from tests.utils import create_fieldset_global

ptype = {"scipy": ScipyParticle, "jit": JITParticle}


def pclass(mode):
    return ptype[mode].add_variables(
        [Variable("u", dtype=np.float32), Variable("v", dtype=np.float32), Variable("p", dtype=np.float32)]
    )


def SampleUV(particle, fieldset, time):
    (particle.u, particle.v) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]


def SampleUVNoConvert(particle, fieldset, time):
    (particle.u, particle.v) = fieldset.UV.eval(time, particle.depth, particle.lat, particle.lon, applyConversion=False)


def SampleP(particle, fieldset, time):
    particle.p = fieldset.P[particle]


@pytest.fixture
def fieldset():
    return create_fieldset_global()


def create_fieldset_geometric(xdim=200, ydim=100):
    """Standard earth fieldset with U and V equivalent to lon/lat in m."""
    lon = np.linspace(-180, 180, xdim, dtype=np.float32)
    lat = np.linspace(-90, 90, ydim, dtype=np.float32)
    U, V = np.meshgrid(lat, lon)
    U *= 1000.0 * 1.852 * 60.0
    V *= 1000.0 * 1.852 * 60.0
    data = {"U": U, "V": V}
    dimensions = {"lon": lon, "lat": lat}
    fieldset = FieldSet.from_data(data, dimensions, transpose=True)
    fieldset.U.units = Geographic()
    fieldset.V.units = Geographic()
    return fieldset


@pytest.fixture
def fieldset_geometric():
    return create_fieldset_geometric()


def create_fieldset_geometric_polar(xdim=200, ydim=100):
    """Standard earth fieldset with U and V equivalent to lon/lat in m
    and the inversion of the pole correction applied to U.
    """
    lon = np.linspace(-180, 180, xdim, dtype=np.float32)
    lat = np.linspace(-90, 90, ydim, dtype=np.float32)
    U, V = np.meshgrid(lat, lon)
    # Apply inverse of pole correction to U
    for i, y in enumerate(lat):
        U[:, i] *= cos(y * pi / 180)
    U *= 1000.0 * 1.852 * 60.0
    V *= 1000.0 * 1.852 * 60.0
    data = {"U": U, "V": V}
    dimensions = {"lon": lon, "lat": lat}
    return FieldSet.from_data(data, dimensions, mesh="spherical", transpose=True)


@pytest.fixture
def fieldset_geometric_polar():
    return create_fieldset_geometric_polar()


def test_fieldset_sample(fieldset):
    """Sample the fieldset using indexing notation."""
    xdim, ydim = 120, 80
    lon = np.linspace(-170, 170, xdim, dtype=np.float32)
    lat = np.linspace(-80, 80, ydim, dtype=np.float32)
    v_s = np.array([fieldset.UV[0, 0.0, 70.0, x][1] for x in lon])
    u_s = np.array([fieldset.UV[0, 0.0, y, -45.0][0] for y in lat])
    assert np.allclose(
        v_s, lon, rtol=1e-5
    )  # Tolerances were rtol=1e-7, increased due to numpy v2 float32 changes (see #1603)
    assert np.allclose(u_s, lat, rtol=1e-5)


def test_fieldset_sample_eval(fieldset):
    """Sample the fieldset using the explicit eval function."""
    xdim, ydim = 60, 60
    lon = np.linspace(-170, 170, xdim, dtype=np.float32)
    lat = np.linspace(-80, 80, ydim, dtype=np.float32)
    v_s = np.array([fieldset.UV.eval(0, 0.0, 70.0, x)[1] for x in lon])
    u_s = np.array([fieldset.UV.eval(0, 0.0, y, 0.0)[0] for y in lat])
    assert np.allclose(
        v_s, lon, rtol=1e-5
    )  # Tolerances were rtol=1e-7, increased due to numpy v2 float32 changes (see #1603)
    assert np.allclose(u_s, lat, rtol=1e-5)


@pytest.mark.parametrize("mode", ["scipy", "jit"])
def test_fieldset_polar_with_halo(fieldset_geometric_polar, mode):
    fieldset_geometric_polar.add_periodic_halo(zonal=5)
    pset = ParticleSet(fieldset_geometric_polar, pclass=pclass(mode), lon=0, lat=0)
    pset.execute(runtime=1)
    assert pset.lon[0] == 0.0


@pytest.mark.parametrize("mode", ["scipy", "jit"])
@pytest.mark.parametrize("zdir", [-1, 1])
def test_verticalsampling(mode, zdir):
    dims = (4, 2, 2)
    dimensions = {
        "lon": np.linspace(0.0, 1.0, dims[2], dtype=np.float32),
        "lat": np.linspace(0.0, 1.0, dims[1], dtype=np.float32),
        "depth": np.linspace(0.0, 1 * zdir, dims[0], dtype=np.float32),
    }
    data = {"U": np.zeros(dims, dtype=np.float32), "V": np.zeros(dims, dtype=np.float32)}
    fieldset = FieldSet.from_data(data, dimensions, mesh="flat")
    pset = ParticleSet(fieldset, pclass=pclass(mode), lon=0, lat=0, depth=0.7 * zdir)
    pset.execute(AdvectionRK4, dt=1.0, runtime=1.0)
    assert pset[0].zi == [2]


@pytest.mark.parametrize("mode", ["scipy", "jit"])
def test_pset_from_field(mode):
    xdim = 10
    ydim = 20
    npart = 10000

    np.random.seed(123456)
    dimensions = {
        "lon": np.linspace(0.0, 1.0, xdim, dtype=np.float32),
        "lat": np.linspace(0.0, 1.0, ydim, dtype=np.float32),
    }
    startfield = np.ones((xdim, ydim), dtype=np.float32)
    for x in range(xdim):
        startfield[x, :] = x
    data = {
        "U": np.zeros((xdim, ydim), dtype=np.float32),
        "V": np.zeros((xdim, ydim), dtype=np.float32),
        "start": startfield,
    }
    fieldset = FieldSet.from_data(data, dimensions, mesh="flat", transpose=True)

    densfield = Field(
        name="densfield",
        data=np.zeros((xdim + 1, ydim + 1), dtype=np.float32),
        lon=np.linspace(-1.0 / (xdim * 2), 1.0 + 1.0 / (xdim * 2), xdim + 1, dtype=np.float32),
        lat=np.linspace(-1.0 / (ydim * 2), 1.0 + 1.0 / (ydim * 2), ydim + 1, dtype=np.float32),
        transpose=True,
    )

    fieldset.add_field(densfield)
    pset = ParticleSet.from_field(fieldset, size=npart, pclass=pclass(mode), start_field=fieldset.start)
    pdens = np.histogram2d(pset.lon, pset.lat, bins=[np.linspace(0.0, 1.0, xdim + 1), np.linspace(0.0, 1.0, ydim + 1)])[
        0
    ]
    assert np.allclose(pdens / sum(pdens.flatten()), startfield / sum(startfield.flatten()), atol=1e-2)


@pytest.mark.parametrize("mode", ["scipy", "jit"])
def test_nearest_neighbor_interpolation2D(mode):
    npart = 81
    dims = (2, 2)
    dimensions = {
        "lon": np.linspace(0.0, 1.0, dims[0], dtype=np.float32),
        "lat": np.linspace(0.0, 1.0, dims[1], dtype=np.float32),
    }
    data = {
        "U": np.zeros(dims, dtype=np.float32),
        "V": np.zeros(dims, dtype=np.float32),
        "P": np.zeros(dims, dtype=np.float32),
    }
    data["P"][0, 1] = 1.0
    fieldset = FieldSet.from_data(data, dimensions, mesh="flat", transpose=True)
    fieldset.P.interp_method = "nearest"
    xv, yv = np.meshgrid(np.linspace(0.0, 1.0, int(np.sqrt(npart))), np.linspace(0.0, 1.0, int(np.sqrt(npart))))
    pset = ParticleSet(fieldset, pclass=pclass(mode), lon=xv.flatten(), lat=yv.flatten())
    pset.execute(SampleP, endtime=1, dt=1)
    assert np.allclose(pset.p[(pset.lon < 0.5) & (pset.lat > 0.5)], 1.0, rtol=1e-5)
    assert np.allclose(pset.p[(pset.lon > 0.5) | (pset.lat < 0.5)], 0.0, rtol=1e-5)


@pytest.mark.parametrize("mode", ["scipy", "jit"])
def test_nearest_neighbor_interpolation3D(mode):
    npart = 81
    dims = (2, 2, 2)
    dimensions = {
        "lon": np.linspace(0.0, 1.0, dims[0], dtype=np.float32),
        "lat": np.linspace(0.0, 1.0, dims[1], dtype=np.float32),
        "depth": np.linspace(0.0, 1.0, dims[2], dtype=np.float32),
    }
    data = {
        "U": np.zeros(dims, dtype=np.float32),
        "V": np.zeros(dims, dtype=np.float32),
        "P": np.zeros(dims, dtype=np.float32),
    }
    data["P"][0, 1, 1] = 1.0
    fieldset = FieldSet.from_data(data, dimensions, mesh="flat", transpose=True)
    fieldset.P.interp_method = "nearest"
    xv, yv = np.meshgrid(np.linspace(0, 1.0, int(np.sqrt(npart))), np.linspace(0, 1.0, int(np.sqrt(npart))))
    # combine a pset at 0m with pset at 1m, as meshgrid does not do 3D
    pset = ParticleSet(fieldset, pclass=pclass(mode), lon=xv.flatten(), lat=yv.flatten(), depth=np.zeros(npart))
    pset2 = ParticleSet(fieldset, pclass=pclass(mode), lon=xv.flatten(), lat=yv.flatten(), depth=np.ones(npart))
    pset.add(pset2)
    pset.execute(SampleP, endtime=1, dt=1)
    assert np.allclose(pset.p[(pset.lon < 0.5) & (pset.lat > 0.5) & (pset.depth > 0.5)], 1.0, rtol=1e-5)
    assert np.allclose(pset.p[(pset.lon > 0.5) | (pset.lat < 0.5) & (pset.depth < 0.5)], 0.0, rtol=1e-5)


@pytest.mark.parametrize("mode", ["scipy", "jit"])
@pytest.mark.parametrize("withDepth", [True, False])
@pytest.mark.parametrize("arrtype", ["ones", "rand"])
def test_inversedistance_nearland(mode, withDepth, arrtype):
    npart = 81
    dims = (4, 4, 6) if withDepth else (4, 6)
    dimensions = {
        "lon": np.linspace(0.0, 1.0, dims[-1], dtype=np.float32),
        "lat": np.linspace(0.0, 1.0, dims[-2], dtype=np.float32),
    }
    if withDepth:
        dimensions["depth"] = np.linspace(0.0, 1.0, dims[0], dtype=np.float32)
        P = np.random.rand(dims[0], dims[1], dims[2]) + 2 if arrtype == "rand" else np.ones(dims, dtype=np.float32)
        P[1, 1:2, 1:6] = np.nan  # setting some values to land (NaN)
    else:
        P = np.random.rand(dims[0], dims[1]) + 2 if arrtype == "rand" else np.ones(dims, dtype=np.float32)
        P[1:2, 1:6] = np.nan  # setting some values to land (NaN)

    data = {"U": np.zeros(dims, dtype=np.float32), "V": np.zeros(dims, dtype=np.float32), "P": P}
    fieldset = FieldSet.from_data(data, dimensions, mesh="flat")
    fieldset.P.interp_method = "linear_invdist_land_tracer"

    xv, yv = np.meshgrid(np.linspace(0.1, 0.9, int(np.sqrt(npart))), np.linspace(0.1, 0.9, int(np.sqrt(npart))))
    pset = ParticleSet(fieldset, pclass=pclass(mode), lon=xv.flatten(), lat=yv.flatten(), depth=np.zeros(npart))
    if withDepth:
        pset2 = ParticleSet(fieldset, pclass=pclass(mode), lon=xv.flatten(), lat=yv.flatten(), depth=np.ones(npart))
        pset.add(pset2)
    pset.execute(SampleP, endtime=1, dt=1)
    if arrtype == "rand":
        assert np.all((pset.p > 2) & (pset.p < 3))
    else:
        assert np.allclose(pset.p, 1.0, rtol=1e-5)

    success = False
    try:
        fieldset.U.interp_method = "linear_invdist_land_tracer"
        fieldset.check_complete()
    except NotImplementedError:
        success = True
    assert success


@pytest.mark.parametrize("mode", ["scipy", "jit"])
@pytest.mark.parametrize("boundaryslip", ["freeslip", "partialslip"])
@pytest.mark.parametrize("withW", [False, True])
@pytest.mark.parametrize("withT", [False, True])
def test_partialslip_nearland_zonal(mode, boundaryslip, withW, withT):
    npart = 20
    dims = (3, 9, 3)
    U = 0.1 * np.ones(dims, dtype=np.float32)
    U[:, 0, :] = np.nan
    U[:, -1, :] = np.nan
    V = np.zeros(dims, dtype=np.float32)
    V[:, 0, :] = np.nan
    V[:, -1, :] = np.nan
    dimensions = {
        "lon": np.linspace(-10, 10, dims[2]),
        "lat": np.linspace(0.0, 4.0, dims[1], dtype=np.float32),
        "depth": np.linspace(-10, 10, dims[0]),
    }
    if withT:
        dimensions["time"] = [0, 2]
        U = np.tile(U, (2, 1, 1, 1))
        V = np.tile(V, (2, 1, 1, 1))
    if withW:
        W = 0.1 * np.ones(dims, dtype=np.float32)
        W[:, 0, :] = np.nan
        W[:, -1, :] = np.nan
        if withT:
            W = np.tile(W, (2, 1, 1, 1))
        data = {"U": U, "V": V, "W": W}
    else:
        data = {"U": U, "V": V}
    fieldset = FieldSet.from_data(data, dimensions, mesh="flat", interp_method=boundaryslip)

    pset = ParticleSet(
        fieldset, pclass=pclass(mode), lon=np.zeros(npart), lat=np.linspace(0.1, 3.9, npart), depth=np.zeros(npart)
    )
    kernel = AdvectionRK4_3D if withW else AdvectionRK4
    pset.execute(kernel, endtime=2, dt=1)
    if boundaryslip == "partialslip":
        assert np.allclose([p.lon for p in pset if p.lat >= 0.5 and p.lat <= 3.5], 0.1)
        assert np.allclose([pset[0].lon, pset[-1].lon], 0.06)
        assert np.allclose([pset[1].lon, pset[-2].lon], 0.08)
        if withW:
            assert np.allclose([p.depth for p in pset if p.lat >= 0.5 and p.lat <= 3.5], 0.1)
            assert np.allclose([pset[0].depth, pset[-1].depth], 0.06)
            assert np.allclose([pset[1].depth, pset[-2].depth], 0.08)
    else:
        assert np.allclose([p.lon for p in pset], 0.1)
        if withW:
            assert np.allclose([p.depth for p in pset], 0.1)


@pytest.mark.parametrize("mode", ["scipy", "jit"])
@pytest.mark.parametrize("boundaryslip", ["freeslip", "partialslip"])
@pytest.mark.parametrize("withW", [False, True])
def test_partialslip_nearland_meridional(mode, boundaryslip, withW):
    npart = 20
    dims = (1, 1, 9)
    U = np.zeros(dims, dtype=np.float32)
    U[:, :, 0] = np.nan
    U[:, :, -1] = np.nan
    V = 0.1 * np.ones(dims, dtype=np.float32)
    V[:, :, 0] = np.nan
    V[:, :, -1] = np.nan
    dimensions = {"lon": np.linspace(0.0, 4.0, dims[2], dtype=np.float32), "lat": 0, "depth": 0}
    if withW:
        W = 0.1 * np.ones(dims, dtype=np.float32)
        W[:, :, 0] = np.nan
        W[:, :, -1] = np.nan
        data = {"U": U, "V": V, "W": W}
        interp_method = {"U": boundaryslip, "V": boundaryslip, "W": boundaryslip}
    else:
        data = {"U": U, "V": V}
        interp_method = {"U": boundaryslip, "V": boundaryslip}
    fieldset = FieldSet.from_data(data, dimensions, mesh="flat", interp_method=interp_method)

    pset = ParticleSet(
        fieldset, pclass=pclass(mode), lat=np.zeros(npart), lon=np.linspace(0.1, 3.9, npart), depth=np.zeros(npart)
    )
    kernel = AdvectionRK4_3D if withW else AdvectionRK4
    pset.execute(kernel, endtime=2, dt=1)
    if boundaryslip == "partialslip":
        assert np.allclose([p.lat for p in pset if p.lon >= 0.5 and p.lon <= 3.5], 0.1)
        assert np.allclose([pset[0].lat, pset[-1].lat], 0.06)
        assert np.allclose([pset[1].lat, pset[-2].lat], 0.08)
        if withW:
            assert np.allclose([p.depth for p in pset if p.lon >= 0.5 and p.lon <= 3.5], 0.1)
            assert np.allclose([pset[0].depth, pset[-1].depth], 0.06)
            assert np.allclose([pset[1].depth, pset[-2].depth], 0.08)
    else:
        assert np.allclose([p.lat for p in pset], 0.1)
        if withW:
            assert np.allclose([p.depth for p in pset], 0.1)


@pytest.mark.parametrize("mode", ["scipy", "jit"])
@pytest.mark.parametrize("boundaryslip", ["freeslip", "partialslip"])
def test_partialslip_nearland_vertical(mode, boundaryslip):
    npart = 20
    dims = (9, 1, 1)
    U = 0.1 * np.ones(dims, dtype=np.float32)
    U[0, :, :] = np.nan
    U[-1, :, :] = np.nan
    V = 0.1 * np.ones(dims, dtype=np.float32)
    V[0, :, :] = np.nan
    V[-1, :, :] = np.nan
    dimensions = {"lon": 0, "lat": 0, "depth": np.linspace(0.0, 4.0, dims[0], dtype=np.float32)}
    data = {"U": U, "V": V}
    fieldset = FieldSet.from_data(data, dimensions, mesh="flat", interp_method={"U": boundaryslip, "V": boundaryslip})

    pset = ParticleSet(
        fieldset, pclass=pclass(mode), lon=np.zeros(npart), lat=np.zeros(npart), depth=np.linspace(0.1, 3.9, npart)
    )
    pset.execute(AdvectionRK4, endtime=2, dt=1)
    if boundaryslip == "partialslip":
        assert np.allclose([p.lon for p in pset if p.depth >= 0.5 and p.depth <= 3.5], 0.1)
        assert np.allclose([p.lat for p in pset if p.depth >= 0.5 and p.depth <= 3.5], 0.1)
        assert np.allclose([pset[0].lon, pset[-1].lon, pset[0].lat, pset[-1].lat], 0.06)
        assert np.allclose([pset[1].lon, pset[-2].lon, pset[1].lat, pset[-2].lat], 0.08)
    else:
        assert np.allclose([p.lon for p in pset], 0.1)
        assert np.allclose([p.lat for p in pset], 0.1)


@pytest.mark.parametrize("mode", ["scipy", "jit"])
@pytest.mark.parametrize("lat_flip", [False, True])
def test_fieldset_sample_particle(mode, lat_flip):
    """Sample the fieldset using an array of particles.

    Note that the low tolerances (1.e-6) are due to the first-order
    interpolation in JIT mode and give an indication of the
    corresponding sampling error.
    """
    npart = 120
    lon = np.linspace(-180, 180, 200, dtype=np.float32)
    if lat_flip:
        lat = np.linspace(90, -90, 100, dtype=np.float32)
    else:
        lat = np.linspace(-90, 90, 100, dtype=np.float32)
    U, V = np.meshgrid(lat, lon)
    data = {"U": U, "V": V}
    dimensions = {"lon": lon, "lat": lat}

    fieldset = FieldSet.from_data(data, dimensions, mesh="flat", transpose=True)

    lon = np.linspace(-170, 170, npart)
    lat = np.linspace(-80, 80, npart)
    pset = ParticleSet(fieldset, pclass=pclass(mode), lon=lon, lat=np.zeros(npart) + 70.0)
    pset.execute(pset.Kernel(SampleUV), endtime=1.0, dt=1.0)
    assert np.allclose(pset.v, lon, rtol=1e-6)

    pset = ParticleSet(fieldset, pclass=pclass(mode), lat=lat, lon=np.zeros(npart) - 45.0)
    pset.execute(pset.Kernel(SampleUV), endtime=1.0, dt=1.0)
    assert np.allclose(pset.u, lat, rtol=1e-6)


@pytest.mark.parametrize("mode", ["scipy", "jit"])
def test_fieldset_sample_geographic(fieldset_geometric, mode):
    """Sample a fieldset with conversion to geographic units (degrees)."""
    npart = 120
    fieldset = fieldset_geometric
    lon = np.linspace(-170, 170, npart)
    lat = np.linspace(-80, 80, npart)

    pset = ParticleSet(fieldset, pclass=pclass(mode), lon=lon, lat=np.zeros(npart) + 70.0)
    pset.execute(pset.Kernel(SampleUV), endtime=1.0, dt=1.0)
    assert np.allclose(pset.v, lon, rtol=1e-6)

    pset = ParticleSet(fieldset, pclass=pclass(mode), lat=lat, lon=np.zeros(npart) - 45.0)
    pset.execute(pset.Kernel(SampleUV), endtime=1.0, dt=1.0)
    assert np.allclose(pset.u, lat, rtol=1e-6)


@pytest.mark.parametrize("mode", ["scipy", "jit"])
def test_fieldset_sample_geographic_noconvert(fieldset_geometric, mode):
    """Sample a fieldset without conversion to geographic units."""
    npart = 120
    fieldset = fieldset_geometric
    lon = np.linspace(-170, 170, npart)
    lat = np.linspace(-80, 80, npart)

    pset = ParticleSet(fieldset, pclass=pclass(mode), lon=lon, lat=np.zeros(npart) + 70.0)
    pset.execute(pset.Kernel(SampleUVNoConvert), endtime=1.0, dt=1.0)
    assert np.allclose(pset.v, lon * 1000 * 1.852 * 60, rtol=1e-6)

    pset = ParticleSet(fieldset, pclass=pclass(mode), lat=lat, lon=np.zeros(npart) - 45.0)
    pset.execute(pset.Kernel(SampleUVNoConvert), endtime=1.0, dt=1.0)
    assert np.allclose(pset.u, lat * 1000 * 1.852 * 60, rtol=1e-6)


@pytest.mark.parametrize("mode", ["scipy", "jit"])
def test_fieldset_sample_geographic_polar(fieldset_geometric_polar, mode):
    """Sample a fieldset with conversion to geographic units and a pole correction."""
    npart = 120
    fieldset = fieldset_geometric_polar
    lon = np.linspace(-170, 170, npart)
    lat = np.linspace(-80, 80, npart)

    pset = ParticleSet(fieldset, pclass=pclass(mode), lon=lon, lat=np.zeros(npart) + 70.0)
    pset.execute(pset.Kernel(SampleUV), endtime=1.0, dt=1.0)
    assert np.allclose(pset.v, lon, rtol=1e-6)

    pset = ParticleSet(fieldset, pclass=pclass(mode), lat=lat, lon=np.zeros(npart) - 45.0)
    pset.execute(pset.Kernel(SampleUV), endtime=1.0, dt=1.0)
    # Note: 1.e-2 is a very low rtol, so there seems to be a rather
    # large sampling error for the JIT correction.
    assert np.allclose(pset.u, lat, rtol=1e-2)


@pytest.mark.parametrize("mode", ["scipy", "jit"])
def test_meridionalflow_spherical(mode):
    """Create uniform NORTHWARD flow on spherical earth and advect particles.

    As flow is so simple, it can be directly compared to analytical solution.
    """
    xdim = 100
    ydim = 200

    maxvel = 1.0
    dimensions = {
        "lon": np.linspace(-180, 180, xdim, dtype=np.float32),
        "lat": np.linspace(-90, 90, ydim, dtype=np.float32),
    }
    data = {"U": np.zeros([xdim, ydim]), "V": maxvel * np.ones([xdim, ydim])}

    fieldset = FieldSet.from_data(data, dimensions, mesh="spherical", transpose=True)

    lonstart = [0, 45]
    latstart = [0, 45]
    runtime = timedelta(hours=24)
    pset = ParticleSet(fieldset, pclass=pclass(mode), lon=lonstart, lat=latstart)
    pset.execute(pset.Kernel(AdvectionRK4), runtime=runtime, dt=timedelta(hours=1))

    assert pset.lat[0] - (latstart[0] + runtime.total_seconds() * maxvel / 1852 / 60) < 1e-4
    assert pset.lon[0] - lonstart[0] < 1e-4
    assert pset.lat[1] - (latstart[1] + runtime.total_seconds() * maxvel / 1852 / 60) < 1e-4
    assert pset.lon[1] - lonstart[1] < 1e-4


@pytest.mark.parametrize("mode", ["scipy", "jit"])
def test_zonalflow_spherical(mode):
    """Create uniform EASTWARD flow on spherical earth and advect particles.

    As flow is so simple, it can be directly compared to analytical solution
    Note that in this case the cosine conversion is needed
    """
    xdim, ydim = 100, 200

    maxvel = 1.0
    p_fld = 10
    dimensions = {
        "lon": np.linspace(-180, 180, xdim, dtype=np.float32),
        "lat": np.linspace(-90, 90, ydim, dtype=np.float32),
    }
    data = {"U": maxvel * np.ones([xdim, ydim]), "V": np.zeros([xdim, ydim]), "P": p_fld * np.ones([xdim, ydim])}

    fieldset = FieldSet.from_data(data, dimensions, mesh="spherical", transpose=True)

    lonstart = [0, 45]
    latstart = [0, 45]
    runtime = timedelta(hours=24)
    pset = ParticleSet(fieldset, pclass=pclass(mode), lon=lonstart, lat=latstart)
    pset.execute(pset.Kernel(AdvectionRK4) + SampleP, runtime=runtime, dt=timedelta(hours=1))

    assert pset.lat[0] - latstart[0] < 1e-4
    assert (
        pset.lon[0] - (lonstart[0] + runtime.total_seconds() * maxvel / 1852 / 60 / cos(latstart[0] * pi / 180)) < 1e-4
    )
    assert abs(pset.p[0] - p_fld) < 1e-4
    assert pset.lat[1] - latstart[1] < 1e-4
    assert (
        pset.lon[1] - (lonstart[1] + runtime.total_seconds() * maxvel / 1852 / 60 / cos(latstart[1] * pi / 180)) < 1e-4
    )
    assert abs(pset.p[1] - p_fld) < 1e-4


@pytest.mark.parametrize("mode", ["scipy", "jit"])
def test_random_field(mode):
    """Sampling test that tests for overshoots by sampling a field of random numbers between 0 and 1."""
    xdim, ydim = 20, 20
    npart = 100

    np.random.seed(123456)
    dimensions = {
        "lon": np.linspace(0.0, 1.0, xdim, dtype=np.float32),
        "lat": np.linspace(0.0, 1.0, ydim, dtype=np.float32),
    }
    data = {
        "U": np.zeros((xdim, ydim), dtype=np.float32),
        "V": np.zeros((xdim, ydim), dtype=np.float32),
        "P": np.random.uniform(0, 1.0, size=(xdim, ydim)),
        "start": np.ones((xdim, ydim), dtype=np.float32),
    }

    fieldset = FieldSet.from_data(data, dimensions, mesh="flat", transpose=True)
    pset = ParticleSet.from_field(fieldset, size=npart, pclass=pclass(mode), start_field=fieldset.start)
    pset.execute(SampleP, endtime=1.0, dt=1.0)
    sampled = pset.p
    assert (sampled >= 0.0).all()


@pytest.mark.parametrize("mode", ["scipy", "jit"])
@pytest.mark.parametrize("allow_time_extrapolation", [True, False])
def test_sampling_out_of_bounds_time(mode, allow_time_extrapolation):
    xdim, ydim, tdim = 10, 10, 10

    dimensions = {
        "lon": np.linspace(0.0, 1.0, xdim, dtype=np.float32),
        "lat": np.linspace(0.0, 1.0, ydim, dtype=np.float32),
        "time": np.linspace(0.0, 1.0, tdim, dtype=np.float64),
    }
    data = {
        "U": np.zeros((xdim, ydim, tdim), dtype=np.float32),
        "V": np.zeros((xdim, ydim, tdim), dtype=np.float32),
        "P": np.ones((xdim, ydim, 1), dtype=np.float32) * dimensions["time"],
    }

    fieldset = FieldSet.from_data(
        data, dimensions, mesh="flat", allow_time_extrapolation=allow_time_extrapolation, transpose=True
    )
    pset = ParticleSet(fieldset, pclass=pclass(mode), lon=[0.5], lat=[0.5], time=-1.0)
    if allow_time_extrapolation:
        pset.execute(SampleP, endtime=-0.9, dt=0.1)
        assert np.allclose(pset.p, 0.0, rtol=1e-5)
    else:
        with pytest.raises(RuntimeError):
            pset.execute(SampleP, endtime=-0.9, dt=0.1)

    pset = ParticleSet(fieldset, pclass=pclass(mode), lon=[0.5], lat=[0.5], time=0)
    pset.execute(SampleP, runtime=0.1, dt=0.1)
    assert np.allclose(pset.p, 0.0, rtol=1e-5)

    pset = ParticleSet(fieldset, pclass=pclass(mode), lon=[0.5], lat=[0.5], time=0.5)
    pset.execute(SampleP, runtime=0.1, dt=0.1)
    assert np.allclose(pset.p, 0.5, rtol=1e-5)

    pset = ParticleSet(fieldset, pclass=pclass(mode), lon=[0.5], lat=[0.5], time=1.0)
    pset.execute(SampleP, runtime=0.1, dt=0.1)
    assert np.allclose(pset.p, 1.0, rtol=1e-5)

    pset = ParticleSet(fieldset, pclass=pclass(mode), lon=[0.5], lat=[0.5], time=2.0)
    if allow_time_extrapolation:
        pset.execute(SampleP, runtime=0.1, dt=0.1)
        assert np.allclose(pset.p, 1.0, rtol=1e-5)
    else:
        with pytest.raises(RuntimeError):
            pset.execute(SampleP, runtime=0.1, dt=0.1)


@pytest.mark.parametrize("mode", ["scipy", "jit"])
def test_sampling_3DCROCO(mode, npart=10):
    data_path = os.path.join(os.path.dirname(__file__), "test_data/")
    fieldset = FieldSet.from_modulefile(data_path + "fieldset_CROCO3D.py")

    SampleP = ptype[mode].add_variable("p", initial=0.0)

    def SampleU(particle, fieldset, time):
        particle.p = fieldset.U[time, particle.depth, particle.lat, particle.lon, particle]

    pset = ParticleSet(fieldset, pclass=SampleP, lon=120e3, lat=50e3, depth=-0.4)
    pset.execute(SampleU, endtime=1, dt=1)
    assert np.isclose(pset.p, 1.0)


@pytest.mark.parametrize("mode", ["jit", "scipy"])
@pytest.mark.parametrize("npart", [1, 10])
@pytest.mark.parametrize("chs", [False, "auto", {"lat": ("y", 10), "lon": ("x", 10)}])
def test_sampling_multigrids_non_vectorfield_from_file(mode, npart, tmpdir, chs):
    xdim, ydim = 100, 200
    filepath = tmpdir.join("test_subsets")
    U = Field(
        "U",
        np.zeros((ydim, xdim), dtype=np.float32),
        lon=np.linspace(0.0, 1.0, xdim, dtype=np.float32),
        lat=np.linspace(0.0, 1.0, ydim, dtype=np.float32),
    )
    V = Field(
        "V",
        np.zeros((ydim, xdim), dtype=np.float32),
        lon=np.linspace(0.0, 1.0, xdim, dtype=np.float32),
        lat=np.linspace(0.0, 1.0, ydim, dtype=np.float32),
    )
    B = Field(
        "B",
        np.ones((3 * ydim, 4 * xdim), dtype=np.float32),
        lon=np.linspace(0.0, 1.0, 4 * xdim, dtype=np.float32),
        lat=np.linspace(0.0, 1.0, 3 * ydim, dtype=np.float32),
    )
    fieldset = FieldSet(U, V)
    fieldset.add_field(B, "B")
    fieldset.write(filepath)
    fieldset = None

    ufiles = [filepath + "U.nc"] * 4
    vfiles = [filepath + "V.nc"] * 4
    bfiles = [filepath + "B.nc"] * 4
    timestamps = np.arange(0, 4, 1) * 86400.0
    timestamps = np.expand_dims(timestamps, 1)
    files = {"U": ufiles, "V": vfiles, "B": bfiles}
    variables = {"U": "vozocrtx", "V": "vomecrty", "B": "B"}
    dimensions = {"lon": "nav_lon", "lat": "nav_lat"}
    fieldset = FieldSet.from_netcdf(
        files, variables, dimensions, timestamps=timestamps, allow_time_extrapolation=True, chunksize=chs
    )

    fieldset.add_constant("sample_depth", 2.5)
    if chs == "auto":
        assert fieldset.U.grid != fieldset.V.grid
    else:
        assert fieldset.U.grid is fieldset.V.grid
    assert fieldset.U.grid is not fieldset.B.grid

    TestParticle = ptype[mode].add_variable("sample_var", initial=0.0)

    pset = ParticleSet.from_line(fieldset, pclass=TestParticle, start=[0.3, 0.3], finish=[0.7, 0.7], size=npart)

    def test_sample(particle, fieldset, time):
        particle.sample_var += fieldset.B[time, fieldset.sample_depth, particle.lat, particle.lon]

    kernels = pset.Kernel(AdvectionRK4) + pset.Kernel(test_sample)
    pset.execute(kernels, runtime=10, dt=1)
    assert np.allclose(pset.sample_var, 10.0)
    if mode == "jit":
        assert len(pset.xi.shape) == 2
        assert pset.xi.shape[0] == len(pset.lon)
        assert pset.xi.shape[1] == fieldset.gridset.size
        assert np.all(pset.xi >= 0)
        assert np.all(pset.xi[:, fieldset.B.igrid] < xdim * 4)
        assert np.all(pset.xi[:, 0] < xdim)
        assert pset.yi.shape[0] == len(pset.lon)
        assert pset.yi.shape[1] == fieldset.gridset.size
        assert np.all(pset.yi >= 0)
        assert np.all(pset.yi[:, fieldset.B.igrid] < ydim * 3)
        assert np.all(pset.yi[:, 0] < ydim)


@pytest.mark.parametrize("mode", ["jit", "scipy"])
@pytest.mark.parametrize("npart", [1, 10])
def test_sampling_multigrids_non_vectorfield(mode, npart):
    xdim, ydim = 100, 200
    U = Field(
        "U",
        np.zeros((ydim, xdim), dtype=np.float32),
        lon=np.linspace(0.0, 1.0, xdim, dtype=np.float32),
        lat=np.linspace(0.0, 1.0, ydim, dtype=np.float32),
    )
    V = Field(
        "V",
        np.zeros((ydim, xdim), dtype=np.float32),
        lon=np.linspace(0.0, 1.0, xdim, dtype=np.float32),
        lat=np.linspace(0.0, 1.0, ydim, dtype=np.float32),
    )
    B = Field(
        "B",
        np.ones((3 * ydim, 4 * xdim), dtype=np.float32),
        lon=np.linspace(0.0, 1.0, 4 * xdim, dtype=np.float32),
        lat=np.linspace(0.0, 1.0, 3 * ydim, dtype=np.float32),
    )
    fieldset = FieldSet(U, V)
    fieldset.add_field(B, "B")
    fieldset.add_constant("sample_depth", 2.5)
    assert fieldset.U.grid is fieldset.V.grid
    assert fieldset.U.grid is not fieldset.B.grid

    TestParticle = ptype[mode].add_variable("sample_var", initial=0.0)

    pset = ParticleSet.from_line(fieldset, pclass=TestParticle, start=[0.3, 0.3], finish=[0.7, 0.7], size=npart)

    def test_sample(particle, fieldset, time):
        particle.sample_var += fieldset.B[time, fieldset.sample_depth, particle.lat, particle.lon]

    kernels = pset.Kernel(AdvectionRK4) + pset.Kernel(test_sample)
    pset.execute(kernels, runtime=10, dt=1)
    assert np.allclose(pset.sample_var, 10.0)
    if mode == "jit":
        assert len(pset.xi.shape) == 2
        assert pset.xi.shape[0] == len(pset.lon)
        assert pset.xi.shape[1] == fieldset.gridset.size
        assert np.all(pset.xi >= 0)
        assert np.all(pset.xi[:, fieldset.B.igrid] < xdim * 4)
        assert np.all(pset.xi[:, 0] < xdim)
        assert pset.yi.shape[0] == len(pset.lon)
        assert pset.yi.shape[1] == fieldset.gridset.size
        assert np.all(pset.yi >= 0)
        assert np.all(pset.yi[:, fieldset.B.igrid] < ydim * 3)
        assert np.all(pset.yi[:, 0] < ydim)


@pytest.mark.parametrize("mode", ["jit", "scipy"])
@pytest.mark.parametrize("ugridfactor", [1, 10])
def test_sampling_multiple_grid_sizes(mode, ugridfactor):
    xdim, ydim = 10, 20
    U = Field(
        "U",
        np.zeros((ydim * ugridfactor, xdim * ugridfactor), dtype=np.float32),
        lon=np.linspace(0.0, 1.0, xdim * ugridfactor, dtype=np.float32),
        lat=np.linspace(0.0, 1.0, ydim * ugridfactor, dtype=np.float32),
    )
    V = Field(
        "V",
        np.zeros((ydim, xdim), dtype=np.float32),
        lon=np.linspace(0.0, 1.0, xdim, dtype=np.float32),
        lat=np.linspace(0.0, 1.0, ydim, dtype=np.float32),
    )
    fieldset = FieldSet(U, V)
    pset = ParticleSet(fieldset, pclass=pclass(mode), lon=[0.8], lat=[0.9])

    if ugridfactor > 1:
        assert fieldset.U.grid is not fieldset.V.grid
    else:
        assert fieldset.U.grid is fieldset.V.grid
    pset.execute(AdvectionRK4, runtime=10, dt=1)
    assert np.isclose(pset.lon[0], 0.8)
    assert np.all((0 <= pset.xi) & (pset.xi < xdim * ugridfactor))


def test_multiple_grid_addlater_error():
    xdim, ydim = 10, 20
    U = Field(
        "U",
        np.zeros((ydim, xdim), dtype=np.float32),
        lon=np.linspace(0.0, 1.0, xdim, dtype=np.float32),
        lat=np.linspace(0.0, 1.0, ydim, dtype=np.float32),
    )
    V = Field(
        "V",
        np.zeros((ydim, xdim), dtype=np.float32),
        lon=np.linspace(0.0, 1.0, xdim, dtype=np.float32),
        lat=np.linspace(0.0, 1.0, ydim, dtype=np.float32),
    )
    fieldset = FieldSet(U, V)

    pset = ParticleSet(fieldset, pclass=pclass("jit"), lon=[0.8], lat=[0.9])  # noqa ; to trigger fieldset.check_complete

    P = Field(
        "P",
        np.zeros((ydim * 10, xdim * 10), dtype=np.float32),
        lon=np.linspace(0.0, 1.0, xdim * 10, dtype=np.float32),
        lat=np.linspace(0.0, 1.0, ydim * 10, dtype=np.float32),
    )

    fail = False
    try:
        fieldset.add_field(P)
    except RuntimeError:
        fail = True
    assert fail


@pytest.mark.parametrize("mode", ["jit", "scipy"])
def test_nestedfields(mode):
    xdim = 10
    ydim = 20

    U1 = Field(
        "U1",
        0.1 * np.ones((ydim, xdim), dtype=np.float32),
        lon=np.linspace(0.0, 1.0, xdim, dtype=np.float32),
        lat=np.linspace(0.0, 1.0, ydim, dtype=np.float32),
    )
    V1 = Field(
        "V1",
        0.2 * np.ones((ydim, xdim), dtype=np.float32),
        lon=np.linspace(0.0, 1.0, xdim, dtype=np.float32),
        lat=np.linspace(0.0, 1.0, ydim, dtype=np.float32),
    )
    U2 = Field(
        "U2",
        0.3 * np.ones((ydim, xdim), dtype=np.float32),
        lon=np.linspace(0.0, 2.0, xdim, dtype=np.float32),
        lat=np.linspace(0.0, 2.0, ydim, dtype=np.float32),
    )
    V2 = Field(
        "V2",
        0.4 * np.ones((ydim, xdim), dtype=np.float32),
        lon=np.linspace(0.0, 2.0, xdim, dtype=np.float32),
        lat=np.linspace(0.0, 2.0, ydim, dtype=np.float32),
    )
    U = NestedField("U", [U1, U2])
    V = NestedField("V", [V1, V2])
    fieldset = FieldSet(U, V)

    P1 = Field(
        "P1",
        0.1 * np.ones((ydim, xdim), dtype=np.float32),
        lon=np.linspace(0.0, 1.0, xdim, dtype=np.float32),
        lat=np.linspace(0.0, 1.0, ydim, dtype=np.float32),
    )
    P2 = Field(
        "P2",
        0.2 * np.ones((ydim, xdim), dtype=np.float32),
        lon=np.linspace(0.0, 2.0, xdim, dtype=np.float32),
        lat=np.linspace(0.0, 2.0, ydim, dtype=np.float32),
    )
    P = NestedField("P", [P1, P2])
    fieldset.add_field(P)

    def Recover(particle, fieldset, time):
        if particle.state == StatusCode.ErrorOutOfBounds:
            particle_dlon = 0  # noqa
            particle_dlat = 0  # noqa
            particle_ddepth = 0  # noqa
            particle.lon = 0
            particle.lat = 0
            particle.p = 999
            particle.state = StatusCode.Evaluate

    pset = ParticleSet(fieldset, pclass=pclass(mode), lon=[0], lat=[0.3])
    pset.execute(AdvectionRK4 + pset.Kernel(SampleP), runtime=2, dt=1)
    assert np.isclose(pset.lat[0], 0.5)
    assert np.isclose(pset.p[0], 0.1)
    pset = ParticleSet(fieldset, pclass=pclass(mode), lon=[0], lat=[1.1])
    pset.execute(AdvectionRK4 + pset.Kernel(SampleP), runtime=2, dt=1)
    assert np.isclose(pset.lat[0], 1.5)
    assert np.isclose(pset.p[0], 0.2)
    pset = ParticleSet(fieldset, pclass=pclass(mode), lon=[0], lat=[2.3])
    pset.execute(pset.Kernel(AdvectionRK4) + SampleP + Recover, runtime=1, dt=1)
    assert np.isclose(pset.lat[0], 0)
    assert np.isclose(pset.p[0], 999)
    assert np.allclose(fieldset.UV[0][0, 0, 0, 0], [0.1, 0.2])


@pytest.mark.parametrize("mode", ["jit", "scipy"])
def test_fieldset_sampling_updating_order(mode, tmpdir):
    def calc_p(t, y, x):
        return 10 * t + x + 0.2 * y

    dims = [2, 4, 5]
    dimensions = {
        "lon": np.linspace(0.0, 1.0, dims[2], dtype=np.float32),
        "lat": np.linspace(0.0, 1.0, dims[1], dtype=np.float32),
        "time": np.arange(dims[0], dtype=np.float32),
    }

    p = np.zeros(dims, dtype=np.float32)
    for i, x in enumerate(dimensions["lon"]):
        for j, y in enumerate(dimensions["lat"]):
            for n, t in enumerate(dimensions["time"]):
                p[n, j, i] = calc_p(t, y, x)

    data = {
        "U": 0.5 * np.ones(dims, dtype=np.float32),
        "V": np.zeros(dims, dtype=np.float32),
        "P": p,
    }
    fieldset = FieldSet.from_data(data, dimensions, mesh="flat")

    xv, yv = np.meshgrid(np.arange(0, 1, 0.5), np.arange(0, 1, 0.5))
    pset = ParticleSet(fieldset, pclass=pclass(mode), lon=xv.flatten(), lat=yv.flatten())

    def SampleP(particle, fieldset, time):
        particle.p = fieldset.P[time, particle.depth, particle.lat, particle.lon]

    kernels = [AdvectionRK4, SampleP]

    filename = tmpdir.join("interpolation_offset.zarr")
    pfile = pset.ParticleFile(filename, outputdt=1)
    pset.execute(kernels, endtime=1, dt=1, output_file=pfile)

    ds = xr.open_zarr(filename)
    for t in range(len(ds["obs"])):
        for i in range(len(ds["trajectory"])):
            assert np.isclose(
                ds["p"].values[i, t],
                calc_p(float(ds["time"].values[i, t]) / 1e9, ds["lat"].values[i, t], ds["lon"].values[i, t]),
            )
