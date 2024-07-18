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
================
Module Functions
================
'''
def load_nn(
        version,
        model=None
    ):
    print('loading NN')
    n_struct = 23
    n_hps = 9
    n_graph = 1
    assert n_graph == 1, 'If n_graph != 1, parts of load_data must be revised. Search for "n_graph" there'
    if version == 'base':
        if model is None:
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
        normalisation='none',
        rescale_y=False,
        dataset_to_run_ids=None,
        dataset_to_training_ids=None,
        dataset_to_testing_ids=None,
        test_mode=None,
        testing_percent=None
    ):
    print('loading dataset')
    training_dataloaders, testing_dataloaders, norm_func_data = do_load(
        dataset_names,
        normalisation=normalisation,
        rescale_y=rescale_y,
        dataset_to_run_ids=dataset_to_run_ids,
        dataset_to_training_ids=dataset_to_training_ids,
        dataset_to_testing_ids=dataset_to_testing_ids,
        test_mode=test_mode,
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
        rescale_y=False,
        verbose=True,
        model_name_prefix='model',
        checkpoint_dir='checkpoints/',
        checkpoint_every_n=5
    ):
    print("running training and eval")
    r2_mrrs, test_losses, mrr_preds_all, mrr_trues_all = run_training(model,
        training_dataloaders,
        testing_dataloaders,
        first_epochs=first_epochs,
        second_epochs=second_epochs,
        lr=lr,
        rescale_y=rescale_y,
        verbose=verbose,
        model_name_prefix=model_name_prefix,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every_n=checkpoint_every_n
    )
    print("done with training and eval")
    return r2_mrrs, test_losses, mrr_preds_all, mrr_trues_all

def main(
        version,
        dataset_names,
        first_epochs,
        second_epochs,
        lr=5e-3,
        normalisation='none',
        rescale_y=False,
        dataset_to_run_ids=None,
        dataset_to_training_ids=None,
        dataset_to_testing_ids=None,
        test_mode=None,
        testing_percent=None,
        preexisting_model=None
    ):
    print(f'REC: starting with v{version} and datasets {dataset_names}')
    checkpoint_dir = 'checkpoints/'
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
            "dataset_to_training_ids": dataset_to_training_ids,
            "dataset_to_testing_ids":dataset_to_testing_ids,
            "test_mode": test_mode,
            "testing_percent": testing_percent
        }
        pickle.dump(to_save, cache)
    
    model = load_nn(
        version,
        preexisting_model #if None, it will create a new model
    )

    training_dataloaders, testing_dataloaders, norm_func_data = load_dataset(
        dataset_names,
        normalisation=normalisation,
        rescale_y=rescale_y,
        dataset_to_run_ids=dataset_to_run_ids,
        dataset_to_training_ids=dataset_to_training_ids,
        dataset_to_testing_ids=dataset_to_testing_ids,
        test_mode=test_mode,
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
        checkpoint_dir=checkpoint_dir,
        checkpoint_every_n=5
    )
    return r2_mrrs, test_losses, mrr_preds_all, mrr_trues_all

if __name__ == '__main__':
    version = sys.argv[1]
    dataset_names = sys.argv[2].split('_')
    first_epochs = int(sys.argv[3])
    second_epochs = int(sys.argv[4])
    normalisation = sys.argv[5]
    rescale_y = sys.argv[6] == "1"
    test_mode = sys.argv[7] #exp, hyp. Hyp means leave hyp combos out for testing. Exp means leave an exp out.
    lr = float(sys.argv[8]) #default 5e-3
    if len(sys.argv) > 9:
        testing_percent = float(sys.argv[9])
    else:
        testing_percent = 0.1
    print(f'Using testing ratio: {testing_percent}')
    
    # hardcoded values
    main(
        version,
        dataset_names,
        first_epochs,
        second_epochs,
        lr=lr,
        normalisation=normalisation,
        rescale_y=rescale_y,
        test_mode=test_mode,
        testing_percent=testing_percent
    )
