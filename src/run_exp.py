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
===============
Reproducibility
===============
'''
torch.manual_seed(17)

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
    if version == 2:
        if model is None:
            model = NeuralNetwork_HPs_v2(
                n_struct=n_struct,
                n_hps=n_hps,
                n_graph=n_graph
            )
        layers_to_freeze = [
            model.linear_struct_1,
            model.linear_struct_2,
            model.linear_hps_1,
            model.linear_integrate_1
        ]
    elif version == 3:
        if model is None:
            model = NeuralNetwork_HPs_v3(
                n_struct=n_struct,
                n_hps=n_hps,
                n_graph=n_graph
            )
        layers_to_freeze = [
            model.linear_struct_1,
            model.linear_struct_2,
            model.linear_hps_1,
            model.linear_integrate_1,
        ]
    else:
        assert False, f"Invald NN version given: {version}"
    print("done loading NN")
    return model, layers_to_freeze

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
        layers_to_freeze,
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
        layers_to_freeze,
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
    
    model, layers_to_freeze = load_nn(
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
        layers_to_freeze,
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

def run_k_fold(version,
            dataset_names,
            first_epochs,
            second_epochs,
            lr,
            normalisation='none',
            rescale_y=False,
            use_full_datasets=False,
            test_mode=None,
            testing_percent=None
        ):
    '''
    Lots of this is hardcoded
    '''
    dataset_to_run_ids = {
        # larger datasets, more power-law-like structure
        'DBpedia50': ['2.1', '2.4'],# '2.2', '2.3', '2.4'],
        'UMLS': ['2.1', '2.4'],# '2.2', '2.3', '2.4'],
        'CoDExSmall': ['2.1', '2.4'],# '2.2', '2.3', '2.4'],
        'OpenEA': ['2.1', '2.4'],# '2.2', '2.3', '2.4'],

        # smaller datasets, generally much more dense
        'Countries': ['2.1', '2.4'],# '2.2', '2.3', '2.4'],
        'Nations': ['2.1', '2.4'],# '2.2', '2.3', '2.4'],
        'Kinships': ['2.1', '2.4'],# '2.2', '2.3', '2.4'],
    }

    if use_full_datasets:
        train_ids = ['2.1', '2.2', '2.3', '2.4']
        test_ids = ['2.1', '2.2', '2.3', '2.4']
    else:
        train_ids = ['2.1']#, '2.2', '2.3']
        test_ids = ['2.4']
    for i, test_dataset in enumerate(dataset_names):
        dataset_to_training_ids = {
            train_dataset:train_ids for train_dataset in set(dataset_names) - {test_dataset}
        }
        dataset_to_testing_ids = {
            train_dataset:test_ids for train_dataset in set(dataset_names)
        }

        for dataset in dataset_names:
            if not dataset in dataset_to_training_ids:
                dataset_to_training_ids[dataset] = []
            if not dataset in dataset_to_testing_ids:
                dataset_to_testing_ids[dataset] = []

        print(f'{"=" * 30} K-FOLD ITERATION {i} {"=" * 30}')
        main(version,
                dataset_names,
                first_epochs,
                second_epochs,
                lr=lr,
                normalisation=normalisation,
                rescale_y=rescale_y,
                dataset_to_run_ids=dataset_to_run_ids,
                dataset_to_training_ids=dataset_to_training_ids,
                dataset_to_testing_ids=dataset_to_testing_ids,
                test_mode=test_mode,
                testing_percent=testing_percent
            )

if __name__ == '__main__':
    version = int(sys.argv[1])
    dataset_names = sys.argv[2].split('_')
    first_epochs = int(sys.argv[3])
    second_epochs = int(sys.argv[4])
    normalisation = sys.argv[5]
    rescale_y = sys.argv[6] == "1"
    test_mode = sys.argv[7] #exp, hyp. Hyp means leave hyp combos out for testing. Exp means leave an exp out.
    
    if len(sys.argv) > 8:
        do_kfold = sys.argv[8] == "1"
    else:
        do_kfold = False

    if len(sys.argv) > 9:
        assert not do_kfold, "testing percent has no meaning when k-fold is being used"
        testing_percent = float(sys.argv[9])
    else:
        testing_percent = 0.1
    print(f'Using testing ratio: {testing_percent}')
    

    # hardcoded values
    lr = 5e-3

    if do_kfold:
        run_k_fold(
            version,
            dataset_names,
            first_epochs,
            second_epochs,
            lr=lr,
            normalisation=normalisation,
            rescale_y=rescale_y,
            test_mode=test_mode,
            testing_percent=None
        )
    else:
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
