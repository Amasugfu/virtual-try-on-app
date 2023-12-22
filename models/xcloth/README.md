# 3D Garment Reconstruction

This the the source code of the xCloth-based [(Srivastava et. al, 2022)](https://arxiv.org/pdf/2208.12934.pdf) garment reconstruction network.


## 1. Prerequisite

### Environment Setup

First create a new conda environment using the `xcloth_env.yml`:
```bash
conda env create -f xcloth_env.yml
```

If you want to integrate the dependency into your existing environment, run the following:
```bash
conda env update --file xcloth_env.yml
```

To overwrite the current environment, run the following:
```bash
conda env update --file xcloth_env.yml --prune
```

<br/>

Then install `pytorch3d` manually:
```
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```
Other installation methods can be found [here](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)


