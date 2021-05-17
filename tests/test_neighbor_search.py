import numpy as np

from parcels.interaction.brute_force import BruteFlatNeighborSearch
from parcels.interaction.brute_force import BruteSphericalNeighborSearch
from parcels.interaction.hash_flat import HashFlatNeighborSearch
from parcels.interaction.hash_spherical import HashSphericalNeighborSearch
from parcels.interaction.scipy_flat import ScipyFlatNeighborSearch


def compare_results_by_idx(instances, particle_idx, active_idx=None):
    res = {}
    for instance in instances:
        cur_neigh, _ = instance.find_neighbors_by_idx(particle_idx)
        assert instance.name != "unknown"
        res[instance.name] = cur_neigh
        if active_idx is None:
            active_idx = np.arange(instance._values.shape[1])
        if instance.name == "hash":
            instance.consistency_check()
        for neigh in cur_neigh:
            assert neigh in active_idx, f"Failed on {instance.name}"
        assert set(cur_neigh) <= set(active_idx), f"Failed on {instance.name}"
        neigh_by_coor, _ = instance.find_neighbors_by_coor(
            instance._values[:, particle_idx])
        assert np.allclose(cur_neigh, neigh_by_coor)

    assert len(res) == len(instances)
    instance_zero = instances[0]
    result_zero = res[instance_zero.name]
    assert len(result_zero) == len(set(result_zero))
    assert isinstance(result_zero, np.ndarray), f"type: {type(result_zero)}"
    for instance in instances[1:]:
        cur_result = res[instance.name]
        assert isinstance(cur_result, np.ndarray), f"Failed on {instance.name}"
        assert set(result_zero) == set(cur_result), f"Failed on {instance.name}"
        assert len(cur_result) == len(set(cur_result))


def test_flat_neighbors():
    np.random.seed(129873)
    neighbor_classes = [
        ScipyFlatNeighborSearch, BruteFlatNeighborSearch, HashFlatNeighborSearch
    ]

    instances = []
    n_particle = 1000
    positions = np.random.rand(n_particle*3).reshape(3, n_particle)
    for cur_class in neighbor_classes:
        cur_instance = cur_class(interaction_distance=0.3, interaction_depth=0.3)
        cur_instance.rebuild(positions)
        instances.append(cur_instance)

    for particle_idx in np.random.choice(positions.shape[1], 100, replace=False):
        compare_results_by_idx(instances, particle_idx)


def create_spherical_positions(n_particles, max_depth=100000):
    yrange = 2*np.random.rand(n_particles)
    lat = 180*(np.arccos(1-yrange)-0.5*np.pi)/np.pi
    long = 360*np.random.rand(n_particles)
    depth = max_depth*np.random.rand(n_particles)
    return np.array((lat, long, depth))


def create_flat_positions(n_particle):
    return np.random.rand(n_particle*3).reshape(3, n_particle)


def test_spherical_neighbors():
    np.random.seed(9837452)
    neighbor_classes = [
        BruteSphericalNeighborSearch, HashSphericalNeighborSearch
    ]

    instances = []
    positions = create_spherical_positions(10000, max_depth=100000)
    for cur_class in neighbor_classes:
        cur_instance = cur_class(interaction_distance=1000000, interaction_depth=100000)
        cur_instance.rebuild(positions)
        instances.append(cur_instance)

    for particle_idx in np.random.choice(positions.shape[1], 100, replace=False):
        compare_results_by_idx(instances, particle_idx)


def test_flat_update():
    np.random.seed(9182741)
    n_particle = 1000
    n_test_particle = 10
    n_active_mask = 10
    neighbor_classes = [
        ScipyFlatNeighborSearch, BruteFlatNeighborSearch, HashFlatNeighborSearch
    ]

    instances = []
    for cur_class in neighbor_classes:
        cur_instance = cur_class(interaction_distance=0.3, interaction_depth=0.3)
        instances.append(cur_instance)

    for i in range(n_active_mask):
        positions = create_flat_positions(n_particle) + 10*np.random.rand()
        if i == 0:
            active_mask = None
        else:
            active_mask = np.random.rand(n_particle) > 0.5
        for cur_instance in instances:
            cur_instance.update_values(positions, active_mask)
        active_idx = np.where(active_mask)[0]
        if len(active_idx) == 0:
            continue
        test_particles = np.random.choice(
            active_idx, size=min(n_test_particle, len(active_idx)), replace=False)
        for particle_idx in test_particles:
            compare_results_by_idx(instances, particle_idx, active_idx=active_idx)


def test_spherical_update():
    np.random.seed(9182741)
    n_particle = 1000
    n_test_particle = 10
    n_active_mask = 10
    neighbor_classes = [
        BruteSphericalNeighborSearch, HashSphericalNeighborSearch
    ]

    instances = []
    for cur_class in neighbor_classes:
        cur_instance = cur_class(interaction_distance=1000000, interaction_depth=100000)
        instances.append(cur_instance)

    for _ in range(n_active_mask):
        positions = create_spherical_positions(n_particle)
        active_mask = np.random.rand(n_particle) > 0.5
        for cur_instance in instances:
            cur_instance.update_values(positions, active_mask)
        active_idx = np.where(active_mask)[0]
        if len(active_idx) == 0:
            continue
        test_particles = np.random.choice(
            active_idx, size=min(n_test_particle, len(active_idx)), replace=False)
        for particle_idx in test_particles:
            compare_results_by_idx(instances, particle_idx, active_idx=active_idx)