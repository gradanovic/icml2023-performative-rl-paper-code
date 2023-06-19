# Performative Reinforcement Learning [ICML'23]

This repository contains code for the paper [Performative Reinforcement Learning](https://arxiv.org/abs/2207.00046).

### Overview

The repository is structured as follows:
- ```src/``` : This folder contains all the source code files required for generating the experiments' data and figures.
- ```data/``` : This folder is where all the experiments' data will be generated.
- ```figures/``` : This folder is where all the experiments' figures will be generated.
- ```limiting_envs/``` : This folder is for storing visualizations of the environment

Before running the scripts, please install the following prerequisites. 

## Prerequisites:
```
Python3
matplotlib
numpy
copy
itertools
time
cvxpy
cvxopt
click
multiprocessing
statistics
json
contextlib
joblib
tqdm
os
cmath
```

## Running the code
To recreate the results of our paper, you will need to run the following scripts. Each of these scripts implements one of the methods described in the paper.

### Repeated Policy Optimization (Fig. 2.a and 2.b)
```
python run_experiment.py --fbeta=10
```

### Repeated Gradient Ascent (Fig. 2.c and 2.d)
```
python run_experiment.py --gradient
```

### Repeated Policy Optimization with Finite Samples (Fig. 2.e)
```
python run_experiment.py --sampling
```

### Repeated Gradient Ascent with Finite Samples (Fig. 2.f)
```
python run_experiment.py --gradient --sampling --etas 1
```

### Repeated Policy Gradient
```
python run_experiment.py --policy_gradient
```

### Solving Lagrangian
```
python run_experiment.py --sampling --lagrangian
```

## Results

After running the above scripts, new plots will be generated in the figures directory.

## Additional Notes

The following are not included in the paper:
* For the experiment *repeated gradient ascent with finite samples* the corresponding suboptimality gap is also computed

## Contact Details
For any questions or comments, contact strianta@mpi-sws.org.
