# TWIG: Topologically Weighted Intelligence Generation

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

## TWIG Quickstart
If you want to use TWIG, it's quite simple! We'll start simple -- reproducing on of TWIG's early experiments. You'll need to download TWIG's dataset (https://figshare.com/s/7b2da136e05f3548399f) and copy its contents into the src/output/ folder. This data contains a large (1215-many) hyperparameter grid, with all outputs (in terms of ranked lists and MRR values) for ComplEx, DistMult, and TransE when trained to do link prediction on any of the following 5 KGs: CoDExSmall, DBepdia50, Kinships, OpenEA, and UMLS.

Suppose that we want to train TWIG to simulate the KGEM ComplEx on UMLS -- teh flagship experiment that started it all! We need to tell TWIG what data to load: (note that since multiple replicates were run, we have to specify which replicate to use. Using more replicates gets better results, but takes longer, so we'll just use one for now.)
```
data_to_load = {
    "ComplEx": {
        "UMLS": ["2.1"]
    }
}

```

Next, we need to load the TWIG model and tell it to learn. This is very easy.  Using TWIG's default settings, we can run:

```
from twig_twm import do_job
do_job(data_to_load)
```

TWIG will run and automatically save checkpoints to src/checkpoints/.

That's it!

### Zero-shot Hyperparamter Prediction
Suppose you have a TWIG model trained on one KG (or set of KGs) and want to use it to predict how well CompLEx will perform on an entirely new KG, and what hyperparameters to use for it there.

We'll take a very simple example where we ask TWIG to predict on Kinships, based on what it learned from UMLS. This is also very simple. To do this, we use the finetune_job() function as so:
```
from twig_twm import finetune_job

data_to_load = {
    "ComplEx": {
        "Kinships": ["2.1"]
    }
}
model_save_path = "checkpoints/chkpt-ID_48310380606809_tag_XXX_e5-e10.pt" # for example
model_config_path = "checkpoints/chkpt-ID_48310380606809_tag_XXX.pkl" # for example

finetune_job(
    data_to_load=data_to_load,
    model_save_path=model_save_path,
    model_config_path=model_config_path,
    epochs=[0, 0]
)
```

If we wanted to instead finetune TWIG, we just need to change the number of epochs. Note that TWIG learns in two phases -- the first typically never needs more than 30 epochs even on large datasets (often 5 - 10 works very well), and the second usually is good to set around 10 - 50. 

### Custom Data for TWIG
