import os
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import urlretrieve

import platformdirs

__all__ = ["download_example_dataset", "get_data_home", "list_example_datasets"]

example_data_files = {
    "MovingEddies_data": [
        "moving_eddiesP.nc",
        "moving_eddiesU.nc",
        "moving_eddiesV.nc",
    ],
    "MITgcm_example_data": ["mitgcm_UV_surface_zonally_reentrant.nc"],
    "OFAM_example_data": ["OFAM_simple_U.nc", "OFAM_simple_V.nc"],
    "Peninsula_data": [
        "peninsulaU.nc",
        "peninsulaV.nc",
        "peninsulaP.nc",
        "peninsulaT.nc",
    ],
    "GlobCurrent_example_data": [
        f"{date.strftime('%Y%m%d')}000000-GLOBCURRENT-L4-CUReul_hs-ALT_SUM-v02.0-fv01.0.nc"
        for date in ([datetime(2002, 1, 1) + timedelta(days=x) for x in range(0, 365)] + [datetime(2003, 1, 1)])
    ],
    "DecayingMovingEddy_data": [
        "decaying_moving_eddyU.nc",
        "decaying_moving_eddyV.nc",
    ],
    "NemoCurvilinear_data": [
        "U_purely_zonal-ORCA025_grid_U.nc4",
        "V_purely_zonal-ORCA025_grid_V.nc4",
        "mesh_mask.nc4",
    ],
    "NemoNorthSeaORCA025-N006_data": [
        "ORCA025-N06_20000104d05U.nc",
        "ORCA025-N06_20000109d05U.nc",
        "ORCA025-N06_20000114d05U.nc",
        "ORCA025-N06_20000119d05U.nc",
        "ORCA025-N06_20000124d05U.nc",
        "ORCA025-N06_20000129d05U.nc",
        "ORCA025-N06_20000104d05V.nc",
        "ORCA025-N06_20000109d05V.nc",
        "ORCA025-N06_20000114d05V.nc",
        "ORCA025-N06_20000119d05V.nc",
        "ORCA025-N06_20000124d05V.nc",
        "ORCA025-N06_20000129d05V.nc",
        "ORCA025-N06_20000104d05W.nc",
        "ORCA025-N06_20000109d05W.nc",
        "ORCA025-N06_20000114d05W.nc",
        "ORCA025-N06_20000119d05W.nc",
        "ORCA025-N06_20000124d05W.nc",
        "ORCA025-N06_20000129d05W.nc",
        "coordinates.nc",
    ],
    "POPSouthernOcean_data": [
        "t.x1_SAMOC_flux.169000.nc",
        "t.x1_SAMOC_flux.169001.nc",
        "t.x1_SAMOC_flux.169002.nc",
        "t.x1_SAMOC_flux.169003.nc",
        "t.x1_SAMOC_flux.169004.nc",
        "t.x1_SAMOC_flux.169005.nc",
    ],
    "SWASH_data": [
        "field_0065532.nc",
        "field_0065537.nc",
        "field_0065542.nc",
        "field_0065548.nc",
        "field_0065552.nc",
        "field_0065557.nc",
    ],
    "WOA_data": [f"woa18_decav_t{m:02d}_04.nc" for m in range(1, 13)],
    "CROCOidealized_data": ["CROCO_idealized.nc"],
}


example_data_url = "http://oceanparcels.org/examples-data"


def get_data_home(data_home=None):
    """Return a path to the cache directory for example datasets.

    This directory is used by :func:`load_dataset`.

    If the ``data_home`` argument is not provided, it will use a directory
    specified by the ``PARCELS_EXAMPLE_DATA`` environment variable (if it exists)
    or otherwise default to an OS-appropriate user cache location.
    """
    if data_home is None:
        data_home = os.environ.get("PARCELS_EXAMPLE_DATA", platformdirs.user_cache_dir("parcels"))
    data_home = os.path.expanduser(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    return data_home


def list_example_datasets() -> list[str]:
    """List the available example datasets.

    Use :func:`download_example_dataset` to download one of the datasets.

    Returns
    -------
    datasets : list of str
        The names of the available example datasets.
    """
    return list(example_data_files.keys())


def download_example_dataset(dataset: str, data_home=None):
    """Load an example dataset from the parcels website.

    This function provides quick access to a small number of example datasets
    that are useful in documentation and testing in parcels.

    Parameters
    ----------
    dataset : str
        Name of the dataset to load.
    data_home : pathlike, optional
        The directory in which to cache data. If not specified, the value
        of the ``PARCELS_EXAMPLE_DATA`` environment variable, if any, is used.
        Otherwise the default location is assigned by :func:`get_data_home`.

    Returns
    -------
    dataset_folder : Path
        Path to the folder containing the downloaded dataset files.
    """
    # Dev note: `dataset` is assumed to be a folder name with netcdf files
    if dataset not in example_data_files:
        raise ValueError(
            f"Dataset {dataset!r} not found. Available datasets are: " + ", ".join(example_data_files.keys())
        )

    cache_folder = get_data_home(data_home)
    dataset_folder = Path(cache_folder) / dataset

    if not dataset_folder.exists():
        dataset_folder.mkdir(parents=True)

    for filename in example_data_files[dataset]:
        filepath = dataset_folder / filename
        if not filepath.exists():
            url = f"{example_data_url}/{dataset}/{filename}"
            urlretrieve(url, str(filepath))

    return dataset_folder
