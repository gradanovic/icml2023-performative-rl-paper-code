Performative Reinforcement Learning

## Prerequisites:
```
Python3
matplotlib
numpy
copy
itertools
time
cvxpy
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
To recreate results, you will need to run the following scripts:

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