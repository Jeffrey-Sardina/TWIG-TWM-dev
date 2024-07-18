# twig-twm imports
from load_data import do_load
from twig_nn import *
from trainer import run_training

# external imports
import sys
import glob
import os
import torch
import random
import pickle

'''
=========
Constants
=========
'''
checkpoint_dir = 'checkpoints/'


def load_nn(version):
    '''
    Load the specified TWIG neural network implementation.

    The arguments it accepts are:
        - version (str): the name of the TWIG model to load

    The values it returns are:
        - model (torch.nn.Module): the initialised TWIG neural netowrk model
    '''
    print('loading NN')
    n_struct = 23
    n_hps = 9
    n_graph = 1
    assert n_graph == 1, 'If n_graph != 1, parts of load_data must be revised. Search for "n_graph" there'
    if version == 'base':
        model = TWIG_Base(
            n_struct=n_struct,
            n_hps=n_hps,
            n_graph=n_graph
        )
    else:
        assert False, f"Invald NN version given: {version}"
    print("done loading NN")
    return model

def load_dataset(
        dataset_names,
        normalisation,
        rescale_y,
        dataset_to_run_ids,
        testing_percent
    ):
    '''
    load_dataset() KGE results for all given KGs and processes them into a format that can be directly used for training by TWIG.

    The arguments it accepts are:
        - dataset_names (list of str): the names of all KGs TWIG should learn from. SPecifically, it will use this to load results from hyperparamter experiments by KGEs on those KGs, and load those results as data for TWIG to simulate.
        - normalisation (str): the normalisation method to use when loading data. "zscore", "minmax", and "none" are supported.
        - rescale_y (bool): True if the groun-truth data `y` was rescaled onto [0, 1] during data loading, False otherwise
        - dataset_to_run_ids
        - testing_percent (float): the percent of all hyperparameter combinations on each KG to reserve as a hold-out test set. If not given, defaults to 0.1.

    The values it returns are:
        - training_dataloaders (torch.Dataloader): a map for a KG name to the dataloader containing data to use for training TWIG to simulate KGEs on that KG
        - testing_dataloaders (dict of str -> torch.Dataloader): a map for a KG name to the dataloader containing data to use for testing TWIG on simulating KGEs on that KG
        - norm_func_data (dict of str -> any): a dictionary mapping parameters for normalisation (such as mean and standard deviaation) to their values so that a normaliation function can be directly construcgted from these values
    '''
    print('loading dataset')
    dataset_to_run_ids = {
        # larger datasets, more power-law-like structure
        'DBpedia50': ['2.1', '2.2', '2.3', '2.4'],
        'UMLS': ['2.1', '2.2', '2.3', '2.4'],
        'CoDExSmall': ['2.1', '2.2', '2.3', '2.4'],
        'OpenEA': ['2.1', '2.2', '2.3', '2.4'],

        # smaller datasets, generally much more dense
        'Countries': ['2.1', '2.2', '2.3', '2.4'],
        'Nations': ['2.1', '2.2', '2.3', '2.4'],
        'Kinships': ['2.1', '2.2', '2.3', '2.4'],
    }
    training_dataloaders, testing_dataloaders, norm_func_data = do_load(
        dataset_names,
        normalisation=normalisation,
        rescale_y=rescale_y,
        dataset_to_run_ids=dataset_to_run_ids,
        testing_percent=testing_percent
    )
    print("done loading dataset")
    return training_dataloaders, testing_dataloaders, norm_func_data

def train_and_eval(
        model,
        training_dataloaders,
        testing_dataloaders,
        first_epochs,
        second_epochs,
        lr,
        rescale_y,
        verbose,
        model_name_prefix,
        checkpoint_every_n=5
    ):
    '''
    train_and_eval() runs training and evaluation fo TWIG, and reutrns all evaluation results.

    The arguments it accepts are:
        - model (torch.nn.Module): the TWIG model that should be trained
        - training_dataloaders_dict (dict of str -> torch.Dataloader): a dict that maps the name of each KG to the training dataloader for that KG
        - testing_dataloaders_dict (dict of str -> torch.Dataloader): a dict that maps the name of each KG to the testing dataloader for that KG
        - first_epochs (int): the number of epochs to run in phase 1 of training
        - second_epochs (int): the number of epochs to run in phase 2 of training
        - lr (float): the learning rate to use while training
        - rescale_y (bool): True if the groun-truth data `y` was rescaled onto [0, 1] during data loading, False otherwise
        - verbose (bool): whether the batch should output more verbose information
        - model_name_prefix (str): the prefix to use for the model name when saving checkpoints
        - checkpoint_every_n (int): the number of epochs after which a checkpoint should be saved

    The values it returns are:
        - r2_scores (dict of str -> float): A dict mapping the name of each KG to the R2 score calculated between predicted and true MRRs for all hyperparameter settings on that KG
        - test_losses (dict of str -> float): A dict mapping the name of each KG to the test loss TWIG had on that KG
        - mrr_preds_all (dict of str -> float): A dict mapping the name of each KG to all MRR predictions TWIG made for all hyperparameter combinations on that KG
        - mrr_trues_all (dict of str -> float): A dict mapping the name of each KG to all ground truth MRR values for all hyperparameter combinations on that KG
    '''
    print("running training and eval")
    r2_mrrs, test_losses, mrr_preds_all, mrr_trues_all = run_training(
        model=model,
        training_dataloaders_dict=training_dataloaders,
        testing_dataloaders_dict=testing_dataloaders,
        first_epochs=first_epochs,
        second_epochs=second_epochs,
        lr=lr,
        rescale_y=rescale_y,
        verbose=verbose,
        model_name_prefix=model_name_prefix,
        checkpoint_every_n=checkpoint_every_n
    )
    print("done with training and eval")
    return r2_mrrs, test_losses, mrr_preds_all, mrr_trues_all

def main(
        version,
        dataset_names,
        first_epochs,
        second_epochs,
        lr,
        normalisation,
        rescale_y,
        dataset_to_run_ids,
        testing_percent,
        preexisting_model=None
    ):
    '''
    main() runs TWIG. It mostly coordinates other functions in this file, and makes sure that data is proeprly passed between them and reported to the user.

    The arguments it accepts are:
        - see documentation under `if __name__ == "__main__"`
     
    The values it returns are:
        - r2_scores (dict of str -> float): A dict mapping the name of each KG to the R2 score calculated between predicted and true MRRs for all hyperparameter settings on that KG
        - test_losses (dict of str -> float): A dict mapping the name of each KG to the test loss TWIG had on that KG
        - mrr_preds_all (dict of str -> float): A dict mapping the name of each KG to all MRR predictions TWIG made for all hyperparameter combinations on that KG
        - mrr_trues_all (dict of str -> float): A dict mapping the name of each KG to all ground truth MRR values for all hyperparameter combinations on that KG
    '''
    print(f'REC: starting with v{version} and datasets {dataset_names}')
    checkpoint_id = str(int(random.random() * 10**16))
    model_name_prefix = f'chkpt-ID_{checkpoint_id}_v{version}_{"-".join(d for d in dataset_names)}'
    print(f'Using checkpoint_id {checkpoint_id}')

    checkpoint_config_name = os.path.join(checkpoint_dir, f'{model_name_prefix}.pkl')
    with open(checkpoint_config_name, 'wb') as cache:
        to_save = {
            "version": version,
            "dataset_names": dataset_names,
            "first_epochs": first_epochs,
            "second_epochs": second_epochs,
            "lr": lr,
            "normalisation": normalisation,
            "rescale_y": rescale_y,
            "dataset_to_run_ids": dataset_to_run_ids,
            "testing_percent": testing_percent
        }
        pickle.dump(to_save, cache)
    
    if preexisting_model:
        model = preexisting_model
    else:
        model = load_nn(
            version,
            preexisting_model #if None, it will create a new model
        )

    training_dataloaders, testing_dataloaders, norm_func_data = load_dataset(
        dataset_names,
        normalisation=normalisation,
        rescale_y=rescale_y,
        dataset_to_run_ids=dataset_to_run_ids,
        testing_percent=testing_percent
    )
    checkpoint_normfunc_name = os.path.join(checkpoint_dir, f'{model_name_prefix}.normfunc.pkl')
    with open(checkpoint_normfunc_name, 'wb') as cache:
        pickle.dump(norm_func_data, cache)
        print(f'Saved norm funcrtion data to {checkpoint_normfunc_name}')

    r2_mrrs, test_losses, mrr_preds_all, mrr_trues_all = train_and_eval(
        model,
        training_dataloaders,
        testing_dataloaders,
        first_epochs=first_epochs,
        second_epochs=second_epochs,
        lr=lr,
        rescale_y=rescale_y,
        verbose=True,
        model_name_prefix=model_name_prefix,
        checkpoint_every_n=5
    )
    return r2_mrrs, test_losses, mrr_preds_all, mrr_trues_all

if __name__ == '__main__':
    '''
    This section specified the command link API for TWIG. It allows TWIG to be loaded, run, and evaluated from a single command.

    The arguments it accepts are:
        - version: the name of the TWIG model to use
        - dataest_names: "_"-delimited list of datasets to train TWIG on. When more than one is given, a single TWIG model will be created and trained on all of them at once
        - first epochs: the number of epochs to run in phase 1 of training (learning the dist of ranks)
        - first epochs: the number of epochs to run in phase 2 of training (learning to replicate exact MRR values)
        - normalisation: the normalisation method to use when loading data. "zscore", "minmax", and "none" are supported.
        - rescale_y: whether rank values should be resscaled onto [0, 1] when loading ground truth data
        - lr: the learning rate that should be used by TWIG
        - testing_percent: the percent of all hyperparameter combinations on each KG to reserve as a hold-out test set. If not given, defaults to 0.1.
    '''
    version = sys.argv[1]
    dataset_names = sys.argv[2].split('_')
    first_epochs = int(sys.argv[3])
    second_epochs = int(sys.argv[4])
    normalisation = sys.argv[5]
    rescale_y = sys.argv[6] == "1"
    lr = float(sys.argv[7]) #default 5e-3
    if len(sys.argv) > 8:
        testing_percent = float(sys.argv[8])
    else:
        testing_percent = 0.1
    print(f'Using testing ratio: {testing_percent}')
    
    main(
        version=version,
        dataset_names=dataset_names,
        first_epochs=first_epochs,
        second_epochs=second_epochs,
        lr=lr,
        normalisation=normalisation,
        rescale_y=rescale_y,
        testing_percent=testing_percent
    )
