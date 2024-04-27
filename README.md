# Real-Time Virtual Try-On Using Computer Vision and Augmented Reality

This is the source code of the final year project - **Real-Time Virtual Try-On Using Comuopter Vision and Augmented Reality**. This includes:

1. The xCloth-based model for predicting meshes [(Srivastava et. al, 2022)](https://arxiv.org/pdf/2208.12934.pdf) [[source]](./models/xcloth);

2. The gRPC server for handling heavy computation [[source]](./server);

3. The mobile application built on Google Filament [[source]](./virtualtryonapp/);


## Project Website

The background, objectives, paper works and more can be found [[here]](https://wp2023.cs.hku.hk/fyp23084/)

## Installation

1. Create a new conda environment using the `environment.yml`:
```bash
conda env create -f environment.yml
```
2. Install SMPL models [[offical website]](https://smpl.is.tue.mpg.de/) and put the models under `models/smpl/` 

3. [Optional] Download the pretrained weights [[here]](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/u3578889_connect_hku_hk/El_JImBOplpGum-D-d5mlC0BjSVqOWO7btgbjOe8HKmMJw?e=sM3CAH)

### Notes
1. If the conda installation failed, try removing the package hash in `environment.yml`
2. This environment was created on windows, if linux version is desired, run `linux_clean_env.sh environment.yml` first to remove windows based packages. It may not remove all possible packages so manual checking is encouraged.

## How to run

For running the server
```bash
python -m server -p <port> -c <checkpoint path>
```

## How to train

The current version was trained on DeepFashion3D V2 dataset [[download raw data here]](https://github.com/GAP-LAB-CUHK-SZ/deepFashion3D). 
New script should be created for processing extra data [[reference code]](./models/xcloth/train/preprocessing.py). 

<br/>

This command was used for processing the data
```bash
# suppose the data was downloaded to data/Deep_Fashion3D_V2
python models/scripts/make_data.py -i data/Deep_Fashion3D_V2/filtered_registered_mesh data/Deep_Fashion3D_V2/pose_estimation/packed_pose -o data/pose data/mesh --smpl-path <smpl path>
```
the smpl-path should direct to a folder containing `basicModel_f_lbs_10_207_0_v1.0.0.pkl` and `basicModel_m_lbs_10_207_0_v1.0.0.pkl`

<br/>

This command was used for training the current model
```bash
python models/scripts/train_model.py -p data/ -c res/xcloth_tm.pt --verbose --plot_path res/xcloth_tm_loss.png --log_file res/xcloth_tm.log -s 0.9 -t res/test_results -d 0.5
```
where `-s` is the split ratio for train data, `-t` is the destination storing test result, `-d` is the depth offset added to the data

