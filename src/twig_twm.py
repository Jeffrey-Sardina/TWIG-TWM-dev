# TWIG imports
from load_data import _do_load
from trainer import _train_and_eval
from twig_nn import *
from kge_pipeline import run_kge_pipeline
from twm_app import start_twm_app

# external imports
import torch
import random
import time
import inspect
import pickle
import os
import itertools
import copy

# Reproducibility
def set_seed(seed):
    print(f'Using random seed: {seed}')
    global random_seed
    random_seed = seed
    torch.manual_seed(seed)
    random.seed(seed)
    return seed

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
        model = model_or_version #user defined model, we will assume

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
        data_to_load,
        test_ratio=0.1,
        valid_ratio=0.0,
        normalisation='zscore',
        n_bins=30,
        model_or_version='base',
        model_kwargs=None,
        optimizer='adam',
        optimizer_args={'lr': 5e-3},
        epochs=[10, 0],
        mrr_loss_coeffs=[10, 10],
        rank_dist_loss_coeffs=[1, 1],
        rescale_mrr_loss=False,
        rescale_rank_dist_loss=False,
        verbose=True,
        tag='TWIG-job',
        seed=None
    ):
    # configure seed
    if type(seed) == int:
        set_seed(seed)
    else:
        seed = set_seed(int(random.random() * 10**16))

    if verbose:
        print('Starting TWIG!')
    start = time.time()
    twig_data = _do_load(
        data_to_load=data_to_load,
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
    model_name_prefix = f'chkpt-ID_{checkpoint_id}_tag_{tag}'
    if verbose:
        print('the checkpoint ID for this run is: ', checkpoint_id)
        print('the save name prefix for this run is: ', model_name_prefix)

    # save settinggs
    checkpoint_config_name = os.path.join(checkpoint_dir, model_name_prefix + '.pkl')
    with open(checkpoint_config_name, 'wb') as cache:
        to_save = {
            'data_to_load': data_to_load,
            'test_ratio': test_ratio,
            'valid_ratio': valid_ratio,
            'normalisation': normalisation,
            'n_bins': n_bins,
            'optimizer': optimizer,
            'optimizer_args': optimizer_args,
            'mrr_loss_coeffs': mrr_loss_coeffs,
            'rank_dist_loss_coeffs': rank_dist_loss_coeffs,
            'rescale_mrr_loss': rescale_mrr_loss,
            'rescale_rank_dist_loss': rescale_rank_dist_loss
        }
        pickle.dump(to_save, cache)
    print('running TWIG with settings:')
    for key in to_save:
        print(f'{key}: {to_save[key]}')

    train_start = time.time()
    metric_results, mrr_preds_all, mrr_trues_all = _train_and_eval(
        model=model,
        twig_data=twig_data,
        mrr_loss_coeffs=mrr_loss_coeffs,
        rank_dist_loss_coeffs=rank_dist_loss_coeffs,
        rescale_mrr_loss=rescale_mrr_loss,
        rescale_rank_dist_loss=rescale_rank_dist_loss,
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
    return metric_results, mrr_preds_all, mrr_trues_all

def load_config(model_config_path):
    with open(model_config_path, 'rb') as cache:
        print('loading model settings from cache:', model_config_path)
        model_config = pickle.load(cache)
    return model_config

def finetune_job(
        data_to_load,
        model_save_path,
        model_config_path,
        test_ratio=None,
        valid_ratio=None,
        optimizer=None,
        optimizer_args=None,
        epochs=[10, 0],
        mrr_loss_coeffs=None,
        rank_dist_loss_coeffs=None,
        rescale_mrr_loss=None,
        rescale_rank_dist_loss=None,
        verbose=True,
        tag='Finetune-job'
):
    # load the pretrained model
    pretrained_model = torch.load(model_save_path)
    model_config = load_config(model_config_path)

    # allow some items to be overwritten (and apply defaults if needed)
    if data_to_load:
        model_config['data_to_load'] = data_to_load
    if test_ratio:
        model_config['test_ratio'] = test_ratio
    if valid_ratio:
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
    if rescale_mrr_loss:
        model_config['rescale_mrr_loss'] = rescale_mrr_loss
    if rescale_rank_dist_loss:
        model_config['rescale_rank_dist_loss'] = rescale_rank_dist_loss

    # run job
    metric_results, mrr_preds_all, mrr_trues_all = do_job(
        data_to_load=data_to_load,
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
        rescale_mrr_loss=model_config['rescale_mrr_loss'],
        rescale_rank_dist_loss=model_config['rescale_rank_dist_loss'],
        verbose=verbose,
        tag=tag
    )
    return metric_results, mrr_preds_all, mrr_trues_all

def ablation_job(
        data_to_load,
        test_ratio=0.1,
        valid_ratio=0.0,
        normalisation=['zscore', 'minmax'],
        n_bins=[15, 30, 60],
        model_or_version=['base'],
        model_kwargs=[None],
        optimizer=['adam'],
        optimizer_args=[
            {'lr': 5e-3},
            {'lr': 5e-4},
            {'lr': 5e-5}
        ],
        epochs=[
            [10, 0]
        ],
        mrr_loss_coeffs=[
            [0, 10]
        ],
        rank_dist_loss_coeffs=[
            [1, 1]
        ],
        rescale_mrr_loss=[True, False],
        rescale_rank_dist_loss=[True, False],
        verbose=True,
        tag='Ablation-job',
        ablation_metric='r2_mrr', #r_mrr, r2_mrr, spearmanr_mrr@5, 10, 50, 100, or All
        ablation_type=None, #full or rand, if given (default full)
        timeout=-1, #seconds
        max_iterations=-1,
        train_and_eval_after=False,
        train_and_eval_args={
            'epochs': [2, 3],
            'verbose': True,
            'tag': 'Post-Ablation-Train-job'
        }
    ):
    # correct input
    if type(normalisation) == str:
        normalisation = [normalisation]
    if type(n_bins) == int:
        n_bins = [n_bins]
    if type(model_or_version) != list:
        model_or_version = [model_or_version]
    if type(model_kwargs) != list:
        model_kwargs = [model_kwargs]
    if type(optimizer) != list:
        optimizer = [optimizer]
    if type(optimizer_args) != list:
        optimizer_args = [optimizer_args]
    if type(epochs[0]) != list:
        epochs = [epochs]
    if type(mrr_loss_coeffs[0]) != list:
        mrr_loss_coeffs = [mrr_loss_coeffs]
    if type(rank_dist_loss_coeffs[0]) != list:
        rank_dist_loss_coeffs = [rank_dist_loss_coeffs]

    # input validation
    assert ablation_metric in (
        'r_mrr',
        'r2_mrr',
        'spearmanr_mrr@5',
        'spearmanr_mrr@10',
        'spearmanr_mrr@50',
        'spearmanr_mrr@100',
        'spearmanr_mrr@All'
    )
    if ablation_type == 'full':
        assert timeout <= 0 or timeout == None, "If 'full' ablations are done, timeout cannot be set"
        assert max_iterations <= 0 or max_iterations == None, "If 'full' ablations are done, max_iterations cannot be set"

    grid = list(itertools.product(
        normalisation,
        n_bins,
        model_or_version,
        model_kwargs,
        optimizer,
        optimizer_args,
        epochs,
        mrr_loss_coeffs,
        rank_dist_loss_coeffs,
        rescale_mrr_loss,
        rescale_rank_dist_loss
    ))

    # configure ablation type settings
    if ablation_type == 'rand':
        random.shuffle(grid)
        grid = grid[:max_iterations]
    elif not ablation_type or ablation_type == 'full':
        pass # nothing to do
    else:
        assert False, f"Invalid ablation type: {ablation_type}. Must be either 'full' or 'rand'"  

    # run ablations on the grid
    best_metric = 0.0
    best_results = None
    best_settings = {}
    start_time = time.time()
    print(f'Running on a grid of size {len(grid)}')
    for settings in grid:
        # unpack (same order as put into itertools)
        (
            normalisation_val,
            n_bins_val,
            model_or_version_val,
            model_kwargs_val,
            optimizer_val,
            optimizer_args_val,
            epochs_val,
            mrr_loss_coeffs_val,
            rank_dist_loss_coeffs_val,
            rescale_mrr_loss_val,
            rescale_rank_dist_loss_val
        ) = settings

        # need this so if a model type is ever re-used, we train it from its initial point, not from the last ablation!
        # TODO: should verify manually that this is the case as well!
        model_copy = copy.deepcopy(model_or_version_val)

        settings_dict = {
            "normalisation": normalisation_val,
            "n_bins": n_bins_val,
            "model_or_version": model_copy,
            "model_kwargs": model_kwargs_val,
            "optimizer": optimizer_val,
            "optimizer_args": optimizer_args_val,
            "epochs": epochs_val,
            "mrr_loss_coeffs": mrr_loss_coeffs_val,
            "rank_dist_loss_coeffs": rank_dist_loss_coeffs_val,
            "rescale_mrr_loss": rescale_mrr_loss_val,
            "rescale_rank_dist_loss": rescale_rank_dist_loss_val
        }


        # run the experiment
        metric_results, mrr_preds_all, mrr_trues_all = do_job(
            data_to_load=data_to_load,
            test_ratio=test_ratio,
            valid_ratio=valid_ratio,
            normalisation=normalisation_val,
            n_bins=n_bins_val,
            model_or_version=model_copy,
            model_kwargs=model_kwargs_val,
            optimizer=optimizer_val,
            optimizer_args=optimizer_args_val,
            epochs=epochs_val,
            mrr_loss_coeffs=mrr_loss_coeffs_val,
            rank_dist_loss_coeffs=rank_dist_loss_coeffs_val,
            rescale_mrr_loss=rescale_mrr_loss_val,
            rescale_rank_dist_loss=rescale_rank_dist_loss_val,
            verbose=verbose,
            tag=tag
        )

        # process results (and record them!)
        metrics = []
        for dataset_name in datasets_to_load:
            metrics.append(metric_results[ablation_metric][dataset_name])
        metric_avg = sum(metrics) / len(metrics)
        if metric_avg > best_metric:
            best_metric = metric_avg
            best_results = metric_results
            best_settings = settings_dict
        
        # check if we have reached or exceeded the timeout
        end_time = time.time()
        if timeout and timeout > 0 and end_time - start_time >= timeout:
            print('Ablation timeout reached; stopping')
            break

    # print ablation results
    print('Ablation done!')
    print(f'The best results were: {best_results}')
    print('The best settings found were:')
    for key in best_settings:
        print(f'{key}: {best_settings[key]}')
    print()

    if train_and_eval_after:
        # fill in missing input to defaults
        if not "epochs" in train_and_eval_args:
            train_and_eval_args["epochs"] = [2,3]
        if not "verbose" in train_and_eval_args:
            train_and_eval_args["verbose"] = True
        if not "tag" in train_and_eval_args:
            train_and_eval_args["tag"] = 'Post-Ablation-Train-job'

        # use given epochs only if we did not validate for number of epochs
        if len(epochs) == 1:
            epochs_to_run = train_and_eval_args["epochs"]
        else:
            epochs_to_run = best_settings["epochs"]

        # train and eval final model
        print('Now training your final model!')
        metric_results, mrr_preds_all, mrr_trues_all = do_job(
            data_to_load=data_to_load,
            test_ratio=test_ratio,
            valid_ratio=valid_ratio,
            normalisation=best_settings['normalisation'],
            n_bins=best_settings['n_bins'],
            model_or_version=best_settings['model_or_version'],
            model_kwargs=best_settings['model_kwargs'],
            optimizer=best_settings['optimizer'],
            optimizer_args=best_settings['optimizer_args'],
            epochs=epochs_to_run,
            mrr_loss_coeffs=best_settings['mrr_loss_coeffs'],
            rank_dist_loss_coeffs=best_settings['rank_dist_loss_coeffs'],
            rescale_mrr_loss=best_settings['rescale_mrr_loss'],
            rescale_rank_dist_loss=best_settings['rescale_rank_dist_loss'],
            verbose=train_and_eval_args['verbose'],
            tag=train_and_eval_args['tag']
        )
        return metric_results, mrr_preds_all, mrr_trues_all
        
def finetune_ablation_job(
        data_to_load,
        model_save_path,
        model_config_path,
        test_ratio=0.1,
        valid_ratio=0.0,
        normalisation=None,
        n_bins=[15, 30, 60],
        optimizer=['adam'],
        optimizer_args=[
            {'lr': 5e-3},
            {'lr': 5e-4},
            {'lr': 5e-5}
        ],
        epochs=[
            [10, 0]
        ],
        mrr_loss_coeffs=[
            [0, 10]
        ],
        rank_dist_loss_coeffs=[
            [1, 1]
        ],
        rescale_mrr_loss=[True, False],
        rescale_rank_dist_loss=[True, False],
        verbose=True,
        tag='Ablation-job',
        ablation_metric='r2_mrr', #r_mrr, r2_mrr, spearmanr_mrr@5, 10, 50, 100, or All
        ablation_type=None, #full or rand, if given (default full)
        timeout=-1, #seconds
        max_iterations=-1,
        train_and_eval_after=False,
        train_and_eval_args={
            'epochs': [2, 3],
            'verbose': True,
            'tag': 'Post-Ablation-Train-job'
        }
    ):
    checkpoint_id = model_save_path.split('_')[1]
    tag += "-from-" + checkpoint_id

    # load the pretrained model
    pretrained_model = torch.load(model_save_path)
    model_config = load_config(model_config_path)

    # replace None with previous value
    if data_to_load == None:
        data_to_load = model_config['data_to_load']
    if test_ratio == None:
        test_ratio = model_config['test_ratio']
    if valid_ratio == None:
        valid_ratio = model_config['valid_ratio']
    if normalisation == None:
        normalisation = model_config['normalisation']
    if n_bins == None:
        n_bins = model_config['n_bins']
    if optimizer == None:
        optimizer = model_config['optimizer']
    if optimizer_args == None:
        optimizer_args = model_config['optimizer_args']
    if epochs == None:
        epochs = model_config['epochs']
    if mrr_loss_coeffs == None:
        mrr_loss_coeffs = model_config['mrr_loss_coeffs']
    if rank_dist_loss_coeffs == None:
        rank_dist_loss_coeffs = model_config['rank_dist_loss_coeffs']
    if rescale_mrr_loss == None:
        rescale_mrr_loss = model_config['rescale_mrr_loss']
    if rescale_rank_dist_loss == None:
        rescale_rank_dist_loss = model_config['rescale_rank_dist_loss']
    
    # run the ablation
    results = ablation_job(
        data_to_load=data_to_load,
        test_ratio=test_ratio,
        valid_ratio=valid_ratio,
        normalisation=normalisation,
        n_bins=n_bins,
        model_or_version=pretrained_model,
        model_kwargs=None,
        optimizer=optimizer,
        optimizer_args=optimizer_args,
        epochs=epochs,
        mrr_loss_coeffs=mrr_loss_coeffs,
        rank_dist_loss_coeffs=rank_dist_loss_coeffs,
        rescale_mrr_loss=rescale_mrr_loss,
        rescale_rank_dist_loss=rescale_rank_dist_loss,
        verbose=verbose,
        tag=tag,
        ablation_metric=ablation_metric,
        ablation_type=ablation_type,
        timeout=timeout,
        max_iterations=max_iterations,
        train_and_eval_after=train_and_eval_after,
        train_and_eval_args=train_and_eval_args
    )
    return results

def kge_data_job(
        dataset_name,
        kge_model,
        run_id,
        num_processes=1,
        grid_override_idxs=None,
        seed=None
):
    # configure seed
    if type(seed) == int:
        set_seed(seed)
    else:
        seed = set_seed(int(random.random() * 10**16))

    run_name = f"{dataset_name}-{kge_model}-TWM-run{run_id}"
    run_dir = os.path.join('output/', dataset_name, run_name)
    try:
        os.makedirs(run_dir)
    except:
        pass
    grid_file = os.path.join(run_dir, run_name + '.grid')
    results_file = os.path.join(run_dir, run_name + '.res')
    with open(results_file, 'w') as results_write_obj:
        run_kge_pipeline(
            grid_file=grid_file,
            results_write_obj=results_write_obj,
            out_dir=run_dir,
            num_processes=num_processes,
            dataset=dataset_name,
            model=kge_model,
            seed=seed,
            grid_override_idxs=grid_override_idxs
        )

def do_app_job(
        hyps_dict_path,
        kge_model_name,
        run_id,
        model_save_path,
        model_config_path
):
    start_twm_app(
        hyps_dict_path=hyps_dict_path,
        kge_model_name_local=kge_model_name,
        run_id_local=run_id,
        model_save_path_local=model_save_path,
        model_config_path_local=model_config_path
    )

if __name__ == '__main__':
    do_app_job(
        hyps_dict_path="output/CoDExSmall/CoDExSmall-ComplEx-TWM-run2.1/CoDExSmall-TWM-run2.1.grid",
        kge_model_name='ComplEx',
        run_id='2.1',
        model_save_path="checkpoints/chkpt-ID_8606280476482954_tag_TWIG-job_UMLS_e1-e0.pt",
        model_config_path="checkpoints/chkpt-ID_8606280476482954_tag_TWIG-job_UMLS.pkl"
    )
