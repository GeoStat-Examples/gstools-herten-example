[![GS-Frame](https://img.shields.io/badge/github-GeoStat_Framework-468a88?logo=github&style=flat)](https://github.com/GeoStat-Framework)
[![Gitter](https://badges.gitter.im/GeoStat-Examples/community.svg)](https://gitter.im/GeoStat-Examples/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5159657.svg)](https://doi.org/10.5281/zenodo.5159657)

# Analyzing the Herten Aquifer with GSTools

## Description

We are going to analyse the Herten aquifer, which is situated in Southern
Germany. Multiple outcrop faces where surveyed and interpolated to a 3D
dataset. In these publications, you can find more information about the data:

> Bayer, Peter; Comunian, Alessandro; Höyng, Dominik; Mariethoz, Gregoire (2015):
> Physicochemical properties and 3D geostatistical simulations of the Herten and the Descalvado aquifer analogs.
> PANGAEA, https://doi.org/10.1594/PANGAEA.844167,
> Supplement to: Bayer, P et al. (2015):
> Three-dimensional multi-facies realizations of sedimentary reservoir and aquifer analogs.
> Scientific Data, 2, 150033, https://doi.org/10.1038/sdata.2015.33

## Structure

The workflow is organized by the following structure:

- `data/`
  - contains a single realization of the herten aquifer downloaded by `00_download.py`
- `src/`
  - `00_download.py` - downloading the herten aquifer and deriving a single transmissivity realization
  - `01_herten.py` - analyzing the herten aquifer and generating conditioned random fields
- `results/` - all produced results


## Python environment

Main Python dependencies are stored in `requirements.txt`:

```
gstools==1.3.1
pyvista
matplotlib
seaborn
```

You can install them with `pip` (potentially in a virtual environment):

```bash
pip install -r requirements.txt
```

## Contact

You can contact us via <info@geostat-framework.org>.


## License

MIT © 2021
