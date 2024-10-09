import inspect
from typing import Literal

import numpy as np
import pytest

from parcels import Field, FieldSet, JITParticle, ParticleFile, ParticleSet
from parcels.grid import (
    CurvilinearGrid,
    CurvilinearSGrid,
    CurvilinearZGrid,
    Grid,
    RectilinearGrid,
    RectilinearSGrid,
    RectilinearZGrid,
)
from tests.utils import create_fieldset_unit_mesh

Classes = Literal[
    "Field",
    "FieldSet",
    "ParticleSet",
    "Grid",
    "RectilinearGrid",
    "RectilinearZGrid",
    "RectilinearSGrid",
    "CurvilinearGrid",
    "CurvilinearZGrid",
    "CurvilinearSGrid",
    "ParticleData",
    "ParticleFile",
]


class Action:
    """Utility class to help manage, document, and test deprecations."""

    def __init__(
        self,
        class_: Classes,
        name: str,
        type_: Literal["read_only", "make_private", "remove"],
        *,
        skip_reason: str = "",
    ):
        if name.startswith("_"):
            raise ValueError("name should not start with an underscore")

        self.class_ = class_
        self._raw_name = name
        self.type_ = type_
        self.skip_reason = skip_reason

        if type_ == "read_only" and self.is_method:
            raise ValueError("read_only attributes should not be methods")

    @property
    def public_name(self):
        return self._raw_name.strip("()")

    @property
    def private_name(self):
        if self.type_ == "make_private":
            return f"_{self.public_name}"
        return None

    @property
    def is_method(self):
        if self._raw_name.endswith("()"):
            return True
        return False

    @property
    def skip(self):
        return bool(self.skip_reason)

    def __str__(self):
        return f"{self.class_}.{self.public_name}"

    def __repr__(self):
        return f"Action(class_={self.class_!r}, name={self._raw_name!r}, type_={self.type_!r})"


def test_testing_action_class():
    """Testing the Action class used for testing."""
    action = Action("MyClass", "my_attribute", "make_private")
    assert not action.is_method
    assert action.public_name == "my_attribute"
    assert action.private_name == "_my_attribute"
    assert action.class_ == "MyClass"
    assert action.type_ == "make_private"

    action = Action("Field", "my_attribute", "read_only")
    assert not action.is_method
    assert action.public_name == "my_attribute"
    assert action.private_name is None
    assert not action.is_method

    action = Action("Field", "my_method()", "make_private")
    assert action.is_method
    assert action.public_name == "my_method"
    assert action.private_name == "_my_method"

    with pytest.raises(ValueError):  # Can't have underscore in name
        Action("Field", "_my_attribute", "make_private")

    with pytest.raises(ValueError):  # Can't have read-only method
        Action("Field", "my_method()", "read_only")

    action = Action("Field", "my_method()", "make_private", skip_reason="Reason")
    assert action.skip


# fmt: off
actions = [
    # 1709
    Action("Field",           "dataFiles",                       "make_private"  ),
    Action("Field",           "netcdf_engine",                   "read_only"     ),
    Action("Field",           "loaded_time_indices",             "make_private"  ),
    Action("Field",           "creation_log",                    "make_private"  ),
    Action("Field",           "data_chunks",                     "make_private"  ),
    Action("Field",           "c_data_chunks",                   "make_private"  ),
    Action("Field",           "chunk_set",                       "make_private"  ),
    Action("Field",           "cell_edge_sizes",                 "read_only"     ),
    Action("Field",           "get_dim_filenames()",             "make_private"  ),
    Action("Field",           "collect_timeslices()",            "make_private"  ),
    Action("Field",           "reshape()",                       "make_private"  ),
    Action("Field",           "calc_cell_edge_sizes()",          "make_private"  ),
    Action("Field",           "search_indices_vertical_z()",     "make_private"  ),
    Action("Field",           "search_indices_vertical_s()",     "make_private"  ),
    Action("Field",           "reconnect_bnd_indices()",         "make_private"  ),
    Action("Field",           "search_indices_rectilinear()",    "make_private"  ),
    Action("Field",           "search_indices_curvilinear()",    "make_private"  ),
    Action("Field",           "search_indices()",                "make_private"  ),
    Action("Field",           "interpolator2D()",                "make_private"  ),
    Action("Field",           "interpolator3D()",                "make_private"  ),
    Action("Field",           "spatial_interpolation()",         "make_private"  ),
    Action("Field",           "time_index()",                    "make_private"  ),
    Action("Field",           "ccode_eval()",                    "make_private"  ),
    Action("Field",           "ccode_convert()",                 "make_private"  ),
    Action("Field",           "get_block_id()",                  "make_private"  ),
    Action("Field",           "get_block()",                     "make_private"  ),
    Action("Field",           "chunk_setup()",                   "make_private"  ),
    Action("Field",           "chunk_data()",                    "make_private"  ),
    Action("Field",           "rescale_and_set_minmax()",        "make_private"  ),
    Action("Field",           "data_concatenate()",              "make_private"  ),
    Action("FieldSet",        "completed",                       "make_private"  ),
    Action("FieldSet",        "particlefile",                    "read_only"     ),
    Action("FieldSet",        "add_UVfield()",                   "make_private"  ),
    Action("FieldSet",        "check_complete()",                "make_private"  ),
    Action("FieldSet",        "parse_wildcards()",               "make_private"  ),

    # 1713
    Action("ParticleSet",      "repeat_starttime",               "make_private"  ),
    Action("ParticleSet",      "repeatlon",                      "make_private"  ),
    Action("ParticleSet",      "repeatlat",                      "make_private"  ),
    Action("ParticleSet",      "repeatdepth",                    "make_private"  ),
    Action("ParticleSet",      "repeatpclass",                   "make_private"  ),
    Action("ParticleSet",      "repeatkwargs",                   "make_private"  ),
    Action("ParticleSet",      "kernel",                         "make_private"  ),
    Action("ParticleSet",      "interaction_kernel",             "make_private"  ),
    Action("ParticleSet",      "repeatpid",                      "make_private"  ),
    Action("ParticleSet",      "active_particles_mask()",        "make_private"  ),
    Action("ParticleSet",      "compute_neighbor_tree()",        "make_private"  ),
    Action("ParticleSet",      "neighbors_by_index()",           "make_private"  ),
    Action("ParticleSet",      "neighbors_by_coor()",            "make_private"  ),
    Action("ParticleSet",      "monte_carlo_sample()",           "make_private"  ),
    Action("ParticleSet",      "error_particles",                "make_private"  ),
    Action("ParticleSet",      "num_error_particles",            "make_private"  ),
    Action("Grid",             "xi",                             "remove"        ),
    Action("Grid",             "yi",                             "remove"        ),
    Action("Grid",             "zi",                             "remove"        ),
    Action("Grid",             "ti",                             "make_private"  ),
    Action("Grid",             "lon",                            "read_only"     ),
    Action("Grid",             "lat",                            "read_only"     ),
    Action("Grid",             "time_origin",                    "read_only"     ),
    Action("Grid",             "mesh",                           "read_only"     ),
    Action("Grid",             "cstruct",                        "make_private"  ),
    Action("Grid",             "cell_edge_sizes",                "read_only"     ),
    Action("Grid",             "zonal_periodic",                 "read_only"     ),
    Action("Grid",             "zonal_halo",                     "read_only"     ),
    Action("Grid",             "meridional_halo",                "read_only"     ),
    Action("Grid",             "lat_flipped",                    "make_private"  ),
    Action("Grid",             "defer_load",                     "read_only"     ),
    Action("Grid",             "lonlat_minmax",                  "read_only"     ),
    Action("RectilinearGrid",  "lonlat_minmax",                  "read_only"     ),
    Action("Grid",             "load_chunk",                     "make_private"  ),
    Action("Grid",             "cgrid",                          "make_private"  ),
    Action("Grid",             "child_ctypes_struct",            "make_private"  ),
    Action("Grid",             "gtype",                          "make_private"  ),
    Action("Grid",             "xdim",                           "read_only"     ),
    Action("Grid",             "ydim",                           "read_only"     ),
    Action("Grid",             "zdim",                           "read_only"     ),
    Action("Grid",             "z4d",                            "make_private"  ),
    Action("Grid",             "depth",                          "read_only"     ),
    Action("Grid",             "check_zonal_periodic()",         "make_private"  ),
    Action("Grid",             "add_Sdepth_periodic_halo()",     "make_private"  ),
    Action("Grid",             "computeTimeChunk()",             "make_private"  ),
    Action("Grid",             "update_status",                  "make_private"  ),
    Action("Grid",             "chunk_not_loaded",               "make_private"  ),
    Action("Grid",             "chunk_loading_requested",        "make_private"  ),
    Action("Grid",             "chunk_loaded_touched",           "make_private"  ),
    Action("Grid",             "chunk_deprecated",               "make_private"  ),
    Action("Grid",             "chunk_loaded",                   "make_private"  ),
    Action("RectilinearGrid",  "lon",                            "read_only"     ),
    Action("RectilinearGrid",  "xdim",                           "read_only"     ),
    Action("RectilinearGrid",  "lat",                            "read_only"     ),
    Action("RectilinearGrid",  "ydim",                           "read_only"     ),
    Action("RectilinearGrid",  "lat_flipped",                    "make_private"  ),
    Action("RectilinearGrid",  "zonal_periodic",                 "read_only"     ),
    Action("RectilinearGrid",  "zonal_halo",                     "read_only"     ),
    Action("RectilinearGrid",  "meridional_halo",                "read_only"     ),
    Action("RectilinearZGrid", "gtype",                          "make_private"  ),
    Action("RectilinearZGrid", "depth",                          "read_only"     ),
    Action("RectilinearZGrid", "zdim",                           "read_only"     ),
    Action("RectilinearZGrid", "z4d",                            "make_private"  ),
    Action("RectilinearSGrid", "gtype",                          "make_private"  ),
    Action("RectilinearSGrid", "depth",                          "read_only"     ),
    Action("RectilinearSGrid", "zdim",                           "read_only"     ),
    Action("RectilinearSGrid", "z4d",                            "make_private"  ),
    Action("RectilinearSGrid", "xdim",                           "read_only"     ),
    Action("RectilinearSGrid", "ydim",                           "read_only"     ),
    Action("RectilinearSGrid", "lat_flipped",                    "make_private"  ),
    Action("CurvilinearGrid",  "lon",                            "read_only"     ),
    Action("CurvilinearGrid",  "xdim",                           "read_only"     ),
    Action("CurvilinearGrid",  "ydim",                           "read_only"     ),
    Action("CurvilinearGrid",  "lat",                            "read_only"     ),
    Action("CurvilinearGrid",  "zonal_periodic",                 "read_only"     ),
    Action("CurvilinearGrid",  "zonal_halo",                     "read_only"     ),
    Action("CurvilinearGrid",  "meridional_halo",                "read_only"     ),
    Action("CurvilinearZGrid", "gtype",                          "make_private"  ),
    Action("CurvilinearZGrid", "depth",                          "read_only"     ),
    Action("CurvilinearZGrid", "zdim",                           "read_only"     ),
    Action("CurvilinearZGrid", "z4d",                            "make_private"  ),
    Action("CurvilinearSGrid", "gtype",                          "make_private"  ),
    Action("CurvilinearSGrid", "depth",                          "read_only"     ),
    Action("CurvilinearSGrid", "zdim",                           "read_only"     ),
    Action("CurvilinearSGrid", "z4d",                            "make_private"  ),
    Action("CurvilinearSGrid", "xdim",                           "read_only"     ),
    Action("CurvilinearSGrid", "ydim",                           "read_only"     ),

    # 1727
    Action("ParticleSet",      "iterator()",                     "remove"        ),
    Action("ParticleData",     "iterator()",                     "remove"        ),
    Action("ParticleFile",     "add_metadata()",                 "remove"        ),
    Action("ParticleFile",     "write_once()",                   "make_private"  ),




]
# fmt: on
assert len({str(a) for a in actions}) == len(actions)  # Check that all actions are unique

actions = list(filter(lambda action: not action.skip, actions))


def create_test_data():
    """Creates and returns the test data dictionary."""
    fieldset = create_fieldset_unit_mesh()
    field = fieldset.U

    npart = 100
    pset = ParticleSet(
        fieldset,
        lon=np.linspace(0, 1, npart, dtype=np.float32),
        lat=np.linspace(1, 0, npart, dtype=np.float32),
        pclass=JITParticle,
    )

    lon_g0 = np.linspace(0, 1000, 11, dtype=np.float32)
    lat_g0 = np.linspace(0, 1000, 11, dtype=np.float32)
    time_g0 = np.linspace(0, 1000, 2, dtype=np.float64)
    grid = RectilinearZGrid(lon_g0, lat_g0, time=time_g0)

    pfile = ParticleFile("test.zarr", pset, outputdt=1)

    return {
        "Field": {
            "class": Field,
            "object": field,
        },
        "FieldSet": {
            "class": FieldSet,
            "object": fieldset,
        },
        "ParticleSet": {
            "class": ParticleSet,
            "object": pset,
        },
        "Grid": {
            "class": Grid,
            "object": grid,
        },
        "RectilinearGrid": {
            "class": RectilinearGrid,
            "object": grid,  # not exactly right but good enough
        },
        "RectilinearZGrid": {
            "class": RectilinearZGrid,
            "object": grid,  # not exactly right but good enough
        },
        "RectilinearSGrid": {
            "class": RectilinearSGrid,
            "object": grid,  # not exactly right but good enough
        },
        "CurvilinearZGrid": {
            "class": CurvilinearZGrid,
            "object": grid,  # not exactly right but good enough
        },
        "CurvilinearGrid": {
            "class": CurvilinearGrid,
            "object": grid,  # not exactly right but good enough
        },
        "CurvilinearSGrid": {
            "class": CurvilinearSGrid,
            "object": grid,  # not exactly right but good enough
        },
        "ParticleFile": {
            "class": ParticleFile,
            "object": pfile,
        },
    }


test_data = create_test_data()


@pytest.mark.parametrize(
    "private_attribute_action",
    filter(lambda action: not action.is_method and action.type_ == "make_private", actions),
    ids=str,
)
def test_private_attrib(private_attribute_action: Action):
    """Checks that the public attribute is equivalent to the private attribute."""
    action = private_attribute_action

    obj = test_data[action.class_]["object"]

    with pytest.raises(DeprecationWarning):
        assert hasattr(obj, action.public_name)
        assert hasattr(obj, action.private_name)
        assert getattr(obj, action.public_name) is getattr(obj, action.private_name)


@pytest.mark.parametrize(
    "private_method_action",
    filter(lambda action: action.is_method and action.type_ == "make_private", actions),
    ids=str,
)
def test_private_method(private_method_action: Action):
    """Looks at the source code to ensure that `public_method` calls `private_method`.

    Looks for the string `.{method_name}(` in the source code of `public_method`.
    """
    action = private_method_action

    class_ = test_data[action.class_]["class"]

    public_method = getattr(class_, action.public_name)
    private_method = getattr(class_, action.private_name)

    assert callable(public_method)
    assert callable(private_method)
    assert f".{action.private_name}(" in inspect.getsource(public_method)


@pytest.mark.parametrize(
    "read_only_attribute_action",
    filter(lambda action: not action.is_method and action.type_ == "read_only", actions),
    ids=str,
)
def test_read_only_attr(read_only_attribute_action: Action):
    """Tries to store a variable in the read-only attribute."""
    action = read_only_attribute_action
    obj = test_data[action.class_]["object"]

    assert hasattr(obj, action.public_name)
    with pytest.raises(AttributeError):
        setattr(obj, action.public_name, None)


@pytest.mark.parametrize(
    "removed_attribute_action",
    filter(lambda action: not action.is_method and action.type_ == "remove", actions),
    ids=str,
)
def test_removed_attrib(removed_attribute_action: Action):
    """Checks that attribute has been deleted."""
    action = removed_attribute_action

    obj = test_data[action.class_]["object"]

    with pytest.raises(AttributeError):
        getattr(obj, action.public_name)
