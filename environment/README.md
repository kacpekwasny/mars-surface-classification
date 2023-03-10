

We found that using the docker container was the best, and easiest solution: <br>

 - [CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
 - [CUDA on docker on Linux](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)



--- 
How to setup the environment for the model

- Install Anaconda: [Anaconda](https://www.anaconda.com)
- [Install `CUDA` 11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local)
- go into a CLI that has access to `conda` and:
    - `conda env create -f environment/env.yml`

We'll be using [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) as our model.



<br><br>

---

Quick test run

Setup the environment by following the above steps.

Then:
- `conda activate mars`
- `cd UNet`
- Sign up to 'kaggle.com'
- [Download `train_mask.zip` and `train_hq.zip`](https://www.kaggle.com/competitions/carvana-image-masking-challenge/data?select=train_masks.zip)
- Move images from `train_hq` to `UNet/imgs`
- Move masks from `train_masks` to `UNet/masks`
- run `python train.py --amp`


