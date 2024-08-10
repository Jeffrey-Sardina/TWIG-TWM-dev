# TWIG imports
from load_data import _do_load
from trainer import _train_and_eval
from twig_nn import *

# external imports
import sys
import torch
import random
import time
import inspect

# Reproducibility
torch.manual_seed(17)
random.seed(17)


def load_nn(version, twig_data):
    '''
    Load the specified TWIG neural network implementation.

    The arguments it accepts are:
        - version (str): the name of the TWIG model to load

    The values it returns are:
        - model (torch.nn.Module): the initialised TWIG neural netowrk model
    '''
    n_struct = twig_data.num_struct_fts
    n_hps = twig_data.num_hyp_fts
    if version == 'base':
        model = TWIG_Base(
            n_struct=n_struct,
            n_hps=n_hps
        )
    else:
        assert False, f"Invald NN version given: {version}"
    return model

def load_optimizer(optimizer, model, optimizer_args):
    '''
    load_optimizer() loads a PyTorch optimizer for use during learning. 

    The arguments it accepts are:
        - optimizer_name (str): the name of the optimizer that should be used.
        - model (torch.nn.Module): the PyTorch NN Model object containing TWIG's neural architecture.

    The values it returns are:
        - optimizer (torch.optim.Optimizer): the optimizer object to be used
    '''
    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_args)
    elif type(optimizer) == str:
        if optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(model.parameters(), **optimizer_args)
        elif optimizer.lower() == "adagrad":
            optimizer = torch.optim.Adagrad(model.parameters(), **optimizer_args)
        else:
            assert False, f"Unrecognised optimizer: {optimizer}"
    elif inspect.isclass(optimizer):
        optimizer = optimizer(model.parameters(), **optimizer_args)
    else:
        pass #user-defined negative sampler (already instantiated)
    return optimizer

def do_job(
        datasets_to_load,
        test_ratio=0.1,
        valid_ratio=0.0,
        normalisation='zscore',
        n_bins=30,
        version='base',
        optimizer='adam',
        optimizer_args={'lr': 5e-3},
        epochs=[2,3],
        mrr_loss_coeffs=[0, 10],
        rank_dist_loss_coeffs=[1, 1],
        verbose=True,
        tag='TWIG-job'
    ):
    start = time.time()
    twig_data = _do_load(
        datasets_to_load=datasets_to_load,
        test_ratio=test_ratio,
        valid_ratio=valid_ratio,
        normalisation=normalisation,
        n_bins=n_bins
    )
    model = load_nn(
        version=version,
        twig_data=twig_data
    )
    optimizer = load_optimizer(
        optimizer=optimizer,
        model=model,
        optimizer_args=optimizer_args
    )
    checkpoint_id = str(int(random.random() * 10**16))
    model_name_prefix = f'chkpt-ID_{checkpoint_id}_tag_{tag}_{"-".join(d for d in datasets_to_load.keys())}'
    train_start = time.time()
    r2_scores, test_losses, mrr_preds_all, mrr_trues_all = _train_and_eval(
        model=model,
        twig_data=twig_data,
        mrr_loss_coeffs=mrr_loss_coeffs,
        rank_dist_loss_coeffs=rank_dist_loss_coeffs,
        epochs=epochs,
        optimizer=optimizer,
        model_name_prefix=model_name_prefix,
        checkpoint_every_n=5,
        do_print=verbose
    )
    end = time.time()
    print(f'total time taken: {end - start}')
    print(f'training time taken: {end - train_start}')
    return r2_scores, test_losses, mrr_preds_all, mrr_trues_all

if __name__ == '__main__':
    do_job(
        datasets_to_load={
            "UMLS": ["2.1", "2.2"]
        },
        test_ratio=0.5,
        valid_ratio=0.0,
        normalisation='zscore',
        n_bins=30,
        version='base',
        optimizer='adam',
        optimizer_args={'lr': 5e-3},
        epochs=[1, 1],
        mrr_loss_coeffs=[0, 10],
        rank_dist_loss_coeffs=[1, 1],
        verbose=True,
        tag='TWIG-job'
    )
