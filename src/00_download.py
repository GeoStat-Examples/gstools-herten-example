from pathlib import Path
import os
import shutil
import zipfile
import urllib.request
import numpy as np
import pyvista as pv


def download_herten(path):
    """Download the data, warning: its about 250MB."""
    print("Downloading Herten data")
    data_filename = "data.zip"
    data_url = (
        "http://store.pangaea.de/Publications/"
        "Bayer_et_al_2015/Herten-analog.zip"
    )
    urllib.request.urlretrieve(data_url, "data.zip")
    # extract the "big" simulation
    with zipfile.ZipFile(data_filename, "r") as zf:
        zf.extract(str(path))


def generate_transmissivity(path):
    """Generate a file with a transmissivity field from the HERTEN data."""
    print("Loading Herten data with pyvista")
    mesh = pv.read(path)
    herten = mesh.point_arrays["facies"].reshape(mesh.dimensions, order="F")
    # conductivity values per fazies from the supplementary data
    cond = 1e-4 * np.array(
        [2.5, 2.3, 0.61, 260, 1300, 950, 0.43, 0.006, 23, 1.4]
    )
    # asign the conductivities to the facies
    herten_cond = cond[herten]
    # Next, we are going to calculate the transmissivity,
    # by integrating over the vertical axis
    herten_trans = np.sum(herten_cond, axis=2) * mesh.spacing[2]
    # saving some grid informations
    grid = [mesh.dimensions[:2], mesh.origin[:2], mesh.spacing[:2]]
    print("Saving the transmissivity field and grid information")
    np.savetxt(Path("../data/herten_transmissivity.gz"), herten_trans)
    np.savetxt(Path("../data/grid_dim_origin_spacing.txt"), grid)
    # Some cleanup. You can comment out these lines to keep the downloaded data
    os.remove("data.zip")
    shutil.rmtree("Herten-analog")


# Downloading and Preprocessing
path = Path("Herten-analog/sim-unc_320x200x140/sim_011.vtk")
download_herten(path)
generate_transmissivity(path)
