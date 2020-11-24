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
# Authors
TBD