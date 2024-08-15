# TWIG imports
from load_data import _do_load
from trainer import _train_and_eval
from twig_nn import *

# external imports
import torch
import random
import time
import inspect
import pickle
import os

# Reproducibility
torch.manual_seed(17)
random.seed(17)

checkpoint_dir = 'checkpoints/'


def load_nn(model_or_version, twig_data, model_kwargs):
    '''
    Load the specified TWIG neural network implementation.

    The arguments it accepts are:
        - version (str): the name of the TWIG model to load

    The values it returns are:
        - model (torch.nn.Module): the initialised TWIG neural netowrk model
    '''
    n_struct = twig_data.num_struct_fts
    n_hps = twig_data.num_hyp_fts
    if type(model_or_version) is str:
        if model_or_version == 'base':
            model = TWIG_Base(
                n_struct=n_struct,
                n_hps=n_hps
            )
        elif model_or_version == 'large':
            model = TWIG_Large(
                n_struct=n_struct,
                n_hps=n_hps
            )
        elif model_or_version == 'small':
            model = TWIG_Small(
                n_struct=n_struct,
                n_hps=n_hps
            )
        elif model_or_version == 'tiny':
            model = TWIG_Tiny(
                n_struct=n_struct,
                n_hps=n_hps
            )
        else:
            assert False, f"Invald NN version given: {model_or_version}"
    elif inspect.isclass(model_or_version):
        model = model_or_version(
            n_struct=n_struct,
            n_hps=n_hps,
            **model_kwargs
        )
    else:
        pass #user defined model, we will assume

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
        model_or_version='base',
        model_kwargs=None,
        optimizer='adam',
        optimizer_args={'lr': 5e-3},
        epochs=[2, 3],
        mrr_loss_coeffs=[0, 10],
        rank_dist_loss_coeffs=[1, 1],
        verbose=True,
        tag='TWIG-job'
    ):
    if verbose:
        print('Starting TWIG!')
    start = time.time()
    twig_data = _do_load(
        datasets_to_load=datasets_to_load,
        test_ratio=test_ratio,
        valid_ratio=valid_ratio,
        normalisation=normalisation,
        n_bins=n_bins,
        do_print=verbose
    )
    # load all nedded data
    model = load_nn(
        model_or_version=model_or_version,
        twig_data=twig_data,
        model_kwargs=model_kwargs
    )
    optimizer = load_optimizer(
        optimizer=optimizer,
        model=model,
        optimizer_args=optimizer_args
    )
    checkpoint_id = str(int(random.random() * 10**16))
    model_name_prefix = f'chkpt-ID_{checkpoint_id}_tag_{tag}_{"-".join(d for d in datasets_to_load.keys())}'

    # save settinggs
    checkpoint_config_name = os.path.join(checkpoint_dir, model_name_prefix + '.pkl')
    with open(checkpoint_config_name, 'wb') as cache:
        to_save = {
            'test_ratio': test_ratio,
            'valid_ratio': valid_ratio,
            'normalisation': normalisation,
            'n_bins': n_bins,
            'test_ratio': test_ratio,
            'optimiser': optimizer,
            'optimizer_args': optimizer_args,
            'mrr_loss_coeffs': mrr_loss_coeffs,
            'rank_dist_loss_coeffs': rank_dist_loss_coeffs
        }
        pickle.dump(to_save, cache)

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
    if verbose:
        print(f'total time taken: {end - start}')
        print(f'training time taken: {end - train_start}')
        print('TWIG out ;))')
    return r2_scores, test_losses, mrr_preds_all, mrr_trues_all

def load_config(model_config_path):
    with open(model_config_path, 'rb') as cache:
        print('loading model settings from cache:', model_config_path)
        model_config = pickle.load(cache)
    return model_config

def finetune_job(
        datasets_to_load,
        model_save_path,
        model_config_path,
        test_ratio=None,
        valid_ratio=None,
        optimizer=None,
        optimizer_args=None,
        epochs=[2, 3],
        mrr_loss_coeffs=None,
        rank_dist_loss_coeffs=None,
        verbose=True,
        tag='TWIG-job'
):
    # load the pretrained model
    pretrained_model = torch.load(model_save_path)
    model_config = load_config(model_config_path)

    # allow some items to be overwritten (and apply defaults if needed)
    if test_ratio:
        model_config['test_ratio'] = test_ratio
    if valid_ratio
        model_config['valid_ratio'] = valid_ratio
    if optimizer:
        model_config['optimizer'] = optimizer
    if optimizer_args:
        if not 'lr' in optimizer_args:
            optimizer_args['lr'] = 5e-3
        model_config['optimizer_args'] = optimizer_args
    if mrr_loss_coeffs:
        model_config['mrr_loss_coeffs'] = mrr_loss_coeffs
    if rank_dist_loss_coeffs:
        model_config['rank_dist_loss_coeffs'] = rank_dist_loss_coeffs

    # run job
    do_job(
        datasets_to_load=datasets_to_load,
        test_ratio=model_config['test_ratio'],
        valid_ratio=model_config['valid_ratio'],
        normalisation=model_config['normalisation'],
        n_bins=model_config['n_bins'],
        model_or_version=pretrained_model,
        model_kwargs=None,
        optimizer=model_config['optimizer'],
        optimizer_args=model_config['optimizer_args'],
        epochs=epochs,
        mrr_loss_coeffs=model_config['mrr_loss_coeffs'],
        rank_dist_loss_coeffs=model_config['rank_dist_loss_coeffs'],
        verbose=verbose,
        tag=tag
    )

if __name__ == '__main__':
    do_job(
        datasets_to_load={
            "UMLS": ["2.1", "2.2"],
            "CoDExSmall": ["2.1", "2.2"],
            "DBpedia50": ["2.1", "2.2"],
            "Kinships": ["2.1", "2.3"],
            "OpenEA": ["2.1", "2.2"],
        },
        test_ratio=0.5,
        valid_ratio=0.0,
        normalisation='zscore',
        n_bins=30,
        version='base',
        optimizer='adam',
        optimizer_args={'lr': 5e-3},
        epochs=[5, 10],
        mrr_loss_coeffs=[0, 10],
        rank_dist_loss_coeffs=[1, 1],
        verbose=True,
        tag='TWIG-job'
    )
