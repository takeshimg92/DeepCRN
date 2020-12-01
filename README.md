# Deep Chemical Reaction Networks

# Overview

TBD

# Requirements

This library supposes you have a local GPU with `tensorflow-gpu` installed.
* We recommend you create an environment `tf_gpu` via
 
```
conda create --name tf_gpu tensorflow-gpu 
```
(reference: https://towardsdatascience.com/tensorflow-gpu-installation-made-easy-use-conda-instead-of-pip-52e5249374bc)
* If you want to use a CPU instead, make sure you only have `tensorflow` installed (not `tensorflow-gpu`) and set 
```{python}
USE_GPU = False
```
on `settings.py`.
# Use

## For a new Pareto simulation run
* Define a new name for this run on `settings.py`, on field `TRAIN_BASELINE_MODELS_JOB_ID`
* Run: `python pareto_simulator.py`
* After the results are done, visualizations can be done on `notebooks/Pareto Analysis.ipynb` 

## Troubleshooting
### Running out of GPU memory

Based on the response to [this issue](https://github.com/tensorflow/tensorflow/issues/36465#issuecomment-729705169).
On a terminal, run `nvidia-smi` and identify the process ID (PID) associated with Python (there might be more than one).
In the example below, the code is 125022:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 455.28       Driver Version: 455.28       CUDA Version: 11.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  GeForce RTX 206...  Off  | 00000000:01:00.0 Off |                  N/A |
| N/A   53C    P0    31W /  N/A |   2815MiB /  5934MiB |      4%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      2975      G   /usr/lib/xorg/Xorg                295MiB |
|    0   N/A  N/A      3160      G   /usr/bin/gnome-shell               89MiB |
|    0   N/A  N/A     17302      G   /usr/lib/firefox/firefox            2MiB |
|    0   N/A  N/A     17309      G   /usr/lib/firefox/firefox            2MiB |
|    0   N/A  N/A     17327      G   /usr/lib/firefox/firefox           18MiB |
|    0   N/A  N/A     17354      G   /usr/lib/firefox/firefox            2MiB |
|    0   N/A  N/A     17381      G   /usr/lib/firefox/firefox            2MiB |
|    0   N/A  N/A     17399      G   /usr/lib/firefox/firefox            2MiB |
|    0   N/A  N/A     47887      G   ...gAAAAAAAAA --shared-files       27MiB |
|    0   N/A  N/A     97563      G   ...AAAAAAAA== --shared-files       25MiB |
|    0   N/A  N/A    125022      C   python                           2339MiB |
+-----------------------------------------------------------------------------+
``` 
Then run `kill -9 125022`, substituting 125022 by the code you identify in your computer.
# Authors
TBD