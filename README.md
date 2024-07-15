# TWIG: Topologically Weighted Intelligence Generation
Jeffrey Seathrún Sardina

ORCID: 0000-0003-0654-2938

## What is TWIG?
TWIG (Topologically-Weighted Intelligence Generation) is a novel, embedding-free paradigm for simulating Knowledge Graphs Embedding. TWIG learns weights from inputs that consist of topological features of the graph data, with no coding for latent representations of entities or edges.

If you use TWIG in your work, please cite:
```
TODO: citation if accepted
```

## How do I install TWIG?
The easiest way to install TWIG is in a Docker container -- simply pull this repo and run `docker compose up` to use the Docker configuration we provide. The container will automatically run all install instructions (in the `install/install.sh` file; a full log will be output to `install/docker_up.log`).

You can also use TWIG with a manual install. We ran TWIG in the following environment:
```
Ubuntu
python 3.9
pykeen 1.10.1
torch 2.0.1
```

While TWIG should, in principle, work on other operating systems, most of its high-level functionality is exposed as bash scripts. As a result, we suggest running it on Linux (in a Docker container or not) or on the Windows Subsystem for Linux (WSL: https://learn.microsoft.com/en-us/windows/wsl/install). Both of these settings have worked for the authors; while this should work on MAC in general as well we have not tested TWIG in a MacOS environment.

Similarly, other python environments and package versions likely will work, but we have not tested them. Note: we *highly* recommend using conda, and all instructions here will assume conda is being used.

**1. Install Linux Packages**
```
apt-get update
apt install -y wget # to download conda
apt install -y git #needed for the PyKEEN library
apt install sqlite #needed for the PyKEEN library
```

**2. Install Conda**
```
miniconda_name="Miniconda3-latest-Linux-x86_64.sh"
wget https://repo.anaconda.com/miniconda/$miniconda_name
chmod u+x $miniconda_name
./$miniconda_name -b
rm $miniconda_name
export PATH=$PATH:~/miniconda3/bin
```

**3. Create a Conda env for TWIG**
```
conda create -n "twig" python=3.9 pip
conda run --no-capture-output -n twig pip install torch torchvision torchaudio
conda run --no-capture-output -n twig pip install pykeen
conda run --no-capture-output -n twig pip install torcheval
conda init bash #or whatever shell you use
```

After running `conda init <your-shell>`, restart your shell so the changes take effect.

**4. Load the TWIG environment**
```
conda activate twig
cd twig/
```

At this point, TWIG is fully installed, and you have a shell in TWIG's working directory with all required dependencies. At this point -- congrats! You are now ready to use TWIG!

## Project Layout
This project is divided into several components. In the project root, `install/` and `docker-compose.yml` are used to install TWIG in Docker (see the above section on installing TWIG).

The `twig` folder contains all novel code and data used in the paper that presented TWIG. In `twig/` you will see the following files and directory structure:

```
twig/
    analysis/ -- contains the Jupyer notebook used to create stats and visualisations for the paper
    output/ -- contains the raw KGE-output data that TWIG learns to simulate
    rec_construct_1/ -- a mini-library for running TWIG itself
    ---
    pipeline.py -- implementation of the KGE models TWIG learns to simulate in PyKEEN
    pipeline.sh -- high-level access to pipeline.py to run batch experiments
    rec_pipeline.sh -- high-level access to training and evaluating TWIG itself
```

This code has two major components: code used to create the data TWIG learns on, and code to run the TWIG NN training and inference itself.

### Re-Creating the Data used by TWIG
TWIG is not a Knowledge Graph Embedding method -- it is embedding free, and learns to *simulate* KGE methods. As such, it needs to be trained on the *output* of KGE models, not a KG itself directly. In order to obtain this data, we must first run our KGE models. 

We run these models using `pipeline.sh`. In short, it
- loads the UMLS KG dataset
- defines a grid of 1215 hyperparameters to search over
- outputs the results (ranks lists and performance metrics) to `output/`

To re-run all KGEs on UMLS, you can run
```
run_nums="1 2 3 4"
num_processes=3
./pipeline.sh TWM $run_nums $num_processes
```

TWM stands for Topologically-Weighted Mapping, and is a keyword that TWIG will expect to be in the names of all output files (so please keep it -- so please do not change it!).

This will populate the `output/` folder with the results. Note that this can take a bit of time -- between a day to several days depending on your GPU. To help reduce compute-driven emissions, please try to run this at night (when the power grid is substantially less used) as much as possible.

### Re-Train TWIG on your new Data
Once you have your new data, you will want to actually run TWIG. TWIG itself is fully implemented in `rec_construct_1/`, which is a mini-library of sorts. `rec_construct_1/` contains the following directories and files:
```
rec_construct_1/
    data_save/ -- a directory used to save loaded PyTorch tensors to avoid re-computing each time TWIG is run
    results_save/ -- a directory that stores the raw output and evaluation results of TWIG itself
    ---
    load_data.py -- a Python module for loading the data (generated in the previous step) that TWIG trains on
    run_exp.py -- a high-level access point for defining and running TWIG experiments in Python
    trainer.py -- a Python module for running TWIG's training and evaluation loops
    twig_nn.py -- a Python module that defined TWIG's neural architecture. Two versions are defined; version 2 is much better and is the one reported in our paper.
    utils.py -- a Pytho module contianing useful miscellaneous functions
```

To train TWIG on your own custom-generated data, and ever higher-level interface is given in `twig/rec_pipeline.py`. You can run it as:
```
./rec_pipeline.py
```

If you wish to modify the settings on which TWIG is run (i.e. the version of the TWIG NN) you will need to change the version number used in `./rec_pipeline.py`.

### Extending TWIG
Unfortunately, this is not a fully-fledged library. The existing codebase should work out-of-the-box for new KG datasets defined in PyKEEN (all you have to change is the UMLS tag in various files to the name of your new dataset).

However, if you are looking for mode advanced functionality than reproduction of experiments, or want to use a KG dataset that is not part of PyKEEN (see their datasets here: https://github.com/pykeen/pykeen#datasets), then you will unfortunately have to go source-code diving. While I intend to create a full, extensible, applications-focussed library in the future, that is currently waiting on the conclusion of several more experiments on TWIG that may allow the inclusion of several more features.

If you have any questions, or would like to contact me, please raise an issue on GitHub and I'd be very happy to help!
