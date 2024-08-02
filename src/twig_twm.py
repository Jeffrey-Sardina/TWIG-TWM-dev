# TWIG imports
from load_data import _do_load
from trainer import _train_and_eval
from twig_nn import *

# external imports
import sys
import torch
import random
import time

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

if __name__ == '__main__':
    start = time.time()
    datasets_to_load={
        "UMLS": ["2.1", "2.2"]
    }
    twig_data = _do_load(
        datasets_to_load=datasets_to_load,
        test_ratio=0.1,
        valid_ratio=0.01,
        normalisation='zscore'
    )

    # load model
    version = 'base'
    model = load_nn(
        version=version,
        twig_data=twig_data
    )

    # load optimizer
    optimizer_args = {
        "lr": 5e-3
    }
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_args)

    # get tagging info
    checkpoint_id = str(int(random.random() * 10**16))
    model_name_prefix = f'chkpt-ID_{checkpoint_id}_v{version}_{"-".join(d for d in datasets_to_load.keys())}'

    train_start = time.time()
    r2_scores, test_losses, mrr_preds_all, mrr_trues_all = _train_and_eval(
        model=model,
        twig_data=twig_data,
        mrr_loss_coeffs = [0, 10],
        rank_dist_loss_coeffs = [1, 1],
        epochs=[2, 3],
        n_bins=30,
        optimizer=optimizer,
        model_name_prefix=model_name_prefix,
        checkpoint_every_n=5,
        do_print=True
    )
    end = time.time()
    print(f'total time taken: {end - start}')
    print(f'training time taken: {end - train_start}')
