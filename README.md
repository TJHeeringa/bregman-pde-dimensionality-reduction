# Code for paper "Sparsifying dimensionality reduction of PDE solution data with Bregman learning"

This repository contains the code used to produce the results shown in the paper "Sparsifying dimensionality reduction 
of PDE solution data with Bregman learning". 

The code is created for a cluster using Slurm and relies on Weight and Biases (Wandb). Wandb is used to track training 
metrics and control the hyperparameter optimisation process. Slurm is used to run the code on a cluster and parallelize 
the execution.

## Datasets 
The folder `src/data` contains code to generate solutions to the three PDEs used in this project. These PDEs are 
 - Diffusion 1D with Gaussian initial condition
 - Advection 1D with Gaussian initial condition
 - Reaction Diffusion 2D with Spiral initial condition

The functions in `src/data/databases/generate_datasets.py` are called to create these datasets prior to training, if 
they have not been generated before.

## Training
The training is performed using the `train` function in `src/training/train.py`. It is a default PyTorch training run 
augmented with Wandb config and logging. 

Each of the three PDEs has a folder in `src/training/`. These folders contain file `train.py` with a function `train` 
that prepares the train and testsets and subsequently calls the `train` function in `src/training/train.py` to do the 
actual training. Next, to this `train.py` file, there are folders for each optimizer containing the `.yaml` files for 
the sweeps with the respective optimizer.  

The training code is called once per Wandb agent created during a sweep, following Wandb recommendations. A sweep is 
started by calling
```console
foo@bar: .../project_folder$ wandb sweep path/to/sweep/file.yaml
```
The sweep long form id returned `SWEEP_ID` is used to start the agents by calling 
```console
foo@bar: .../project_folder$ ./src/training/agents_for_sweep.sh --sweep-long-id="<SWEEP_ID>" -c <CONCURRENT> -t <TOTAL>
```
with `CONCURRENT` being the number of runs executed in parallel and `TOTAL` being the total number of runs for the sweep. 
It will write output to `.../project_folder/sweep_<SWEEP_ID>.log`, and it will block the terminal from interaction until 
it is ready. To keep track of the current progress, it can be started in a `screen`. After detaching the output can be 
continuously streamed using a command like 
```console
foo@bar: .../project_folder$ watch -n5 -d "cat sweep_<SWEEP_ID>.log"
```

