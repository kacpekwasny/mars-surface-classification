# How to setup the environment for the model

- Install Anaconda: [Anaconda](https://www.anaconda.com)
- [Install `CUDA` 11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local)
- go into a CLI that has access to `conda` and:
    - `conda env create --file=environment/env.yml`

We'll be using [Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) as our model.



<br><br>

---
- `conda create -n mars python=3.10.4`
- `conda activate mars`
- `git clone https://github.com/milesial/Pytorch-UNet.git`
- `cd Pytorch-UNet`
- `python -m pip install -r requirements.txt`
- `conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia` Takes a looong while.
- 


## NOTES / DEVLOG of the env:

- `python` `3.11.0` and `3.10.7` dont meet the requirements of some of libriaries.
- `python 3.10.4` successfuly installed everything, but then an error occured:
```
(mars) PS C:\Users\quatr\IT\Code\mars-surface-classification\Pytorch-UNet> python .\predict.py -i ..\..\images\piotrf\DSLR_RAW\DSC_0001.JPG -o out.jpg
D:\Programs\anaconda3\envs\mars\lib\site-packages\numpy\__init__.py:138: UserWarning: mkl-service package failed to import, therefore Intel(R) MKL initialization ensuring its correct out-of-the box operation under condition when Gnu OpenMP had already been loaded by Python process is not assured. Please install mkl-service package, see http://github.com/IntelPython/mkl-service
  from . import _distributor_init
Traceback (most recent call last):
  File "C:\Users\quatr\IT\Code\mars-surface-classification\Pytorch-UNet\predict.py", line 5, in <module>
    import numpy as np
  File "D:\Programs\anaconda3\envs\mars\lib\site-packages\numpy\__init__.py", line 417, in <module>
    import mkl
  File "D:\Programs\anaconda3\envs\mars\lib\site-packages\mkl\__init__.py", line 48, in <module> 
    with RTLD_for_MKL():
  File "D:\Programs\anaconda3\envs\mars\lib\site-packages\mkl\__init__.py", line 33, in __enter__
    import ctypes
  File "D:\Programs\anaconda3\envs\mars\lib\ctypes\__init__.py", line 8, in <module>
    from _ctypes import Union, Structure, Array
ImportError: DLL load failed while importing _ctypes: The specified module could not be found.  
```

Google searches left me with only solution to be copying `libffi-7.dll` to different destinations like root folder of the `env` or `lib`.

I've tried with a version downloaded, and a version copied form `D:\Programs\Python\Python310\DLLs\\libffi-7.dll`.

So Ive tried to run it some on one of my other pytorch envs, and it seems to work.
The env is on python `3.9.13`



