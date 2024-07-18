# external imports
import pandas as pd
from utils import gather_data
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import random
from torch.utils.data import TensorDataset, DataLoader
import pickle

'''
==================================================
Feature Counts (Not constant, but will be updated)
==================================================
'''
n_struct = 23
n_hps = 9
n_graph = 1

'''
=========
Functions
=========
'''
def get_adj_data(valid_triples_map):
    ents_to_triples = {} # entity to all relevent data
    for t_idx in valid_triples_map:
        s, p, o = valid_triples_map[t_idx]
        if not s in ents_to_triples:
            ents_to_triples[s] = set()
        if not o in ents_to_triples:
            ents_to_triples[o] = set()
        ents_to_triples[s].add(t_idx)
        ents_to_triples[o].add(t_idx)
    return ents_to_triples

def get_twm_data_augment(
        dataset_name,
        exp_dir,
        exp_id=None,
        incl_hps=True,
        incl_global_struct=False,
        incl_neighbour_structs=False,
        struct_source='train',
        randomise=True
    ):
    
    overall_results, \
        triples_results, \
        grid, \
        valid_triples_map, \
        graph_stats = gather_data(dataset_name, exp_dir)
    ents_to_triples = get_adj_data(valid_triples_map)

    all_data = []
    if exp_id is not None:
        iter_over = [str(exp_id)]
    else:
        iter_over = sorted(int(key) for key in triples_results.keys())
        if randomise:
            random.shuffle(iter_over)
    iter_over = [str(x) for x in iter_over]
    print(f'Loader: Randomise = {randomise}; Using exp_id order, {iter_over}')
    global_struct = {}
    if incl_global_struct:
        max_rank = len(graph_stats['all']['degrees']) # = num nodes
        global_struct["max_rank"] = max_rank

    for exp_id in iter_over:
        hps = grid[exp_id]
        for triple_idx in valid_triples_map:
            s, p, o = valid_triples_map[triple_idx]

            s_deg = graph_stats[struct_source]['degrees'][s] \
                if s in graph_stats[struct_source]['degrees'] else 0
            o_deg = graph_stats[struct_source]['degrees'][o] \
                if o in graph_stats[struct_source]['degrees'] else 0
            p_freq = graph_stats[struct_source]['pred_freqs'][p] \
                if p in graph_stats[struct_source]['pred_freqs'] else 0

            s_p_cofreq = graph_stats[struct_source]['subj_relationship_degrees'][(s,p)] \
                if (s,p) in graph_stats[struct_source]['subj_relationship_degrees'] else 0
            o_p_cofreq = graph_stats[struct_source]['obj_relationship_degrees'][(o,p)] \
                if (o,p) in graph_stats[struct_source]['obj_relationship_degrees'] else 0
            s_o_cofreq = graph_stats[struct_source]['subj_obj_cofreqs'][(s,o)] \
                if (s,o) in graph_stats[struct_source]['subj_obj_cofreqs'] else 0

            head_rank = triples_results[exp_id][triple_idx]['head_rank']
            tail_rank = triples_results[exp_id][triple_idx]['tail_rank']
            
            data = {}
            if incl_global_struct:
                for key in global_struct:
                    data[key] = global_struct[key]
                    
            data['s_deg'] = s_deg
            data['o_deg'] = o_deg
            data['p_freq'] = p_freq

            data['s_p_cofreq'] = s_p_cofreq
            data['o_p_cofreq'] = o_p_cofreq
            data['s_o_cofreq'] = s_o_cofreq

            data['head_rank'] = head_rank
            data['tail_rank'] = tail_rank

            if incl_neighbour_structs:
                target_dict = {'s': s, 'o': o}
                for target_name in target_dict:
                    target = target_dict[target_name]
                    neighbour_nodes = {}
                    neighbour_preds = {}
                    for t_idx in ents_to_triples[target]:
                        t_s, t_p, t_o = valid_triples_map[t_idx]
                        ent = t_s if target != t_s else t_o
                        if not t_p in neighbour_preds:
                            neighbour_preds[t_p] = graph_stats[struct_source]['pred_freqs'][t_p]
                        if not ent in neighbour_nodes:
                            neighbour_nodes[ent] = graph_stats[struct_source]['degrees'][ent]

                    data[f'{target_name} min deg neighbnour'] = np.min(list(neighbour_nodes.values()))
                    data[f'{target_name} max deg neighbnour'] = np.max(list(neighbour_nodes.values()))
                    data[f'{target_name} mean deg neighbnour'] = np.mean(list(neighbour_nodes.values()))
                    data[f'{target_name} num neighbnours'] = len(neighbour_nodes)

                    data[f'{target_name} min freq rel'] = np.min(list(neighbour_preds.values()))
                    data[f'{target_name} max freq rel'] = np.max(list(neighbour_preds.values()))
                    data[f'{target_name} mean freq rel'] = np.mean(list(neighbour_preds.values()))
                    data[f'{target_name} num rels'] = len(neighbour_preds)

            if incl_hps:
                for key in hps:
                    data[key] = hps[key]
            all_data.append(data)

    '''
    We now want to make this to instead of head and tail rank independently,
    we just have one 'rank' column
    '''
    rank_data = []
    for data_dict in all_data:
        # insert rank data in simplified form using a flag
        head_data = {key: data_dict[key] for key in data_dict}
        del head_data['tail_rank']
        rank = head_data['head_rank']
        del head_data['head_rank']
        head_data['rank'] = rank
        head_data['is_head'] = 1

        # insert rank data in simplified form using a flag
        tail_data = {key: data_dict[key] for key in data_dict}
        del tail_data['head_rank']
        rank = tail_data['tail_rank']
        del tail_data['tail_rank']
        tail_data['rank'] = rank
        tail_data['is_head'] = 0

        rank_data.append(head_data)
        rank_data.append(tail_data)

    rank_data_df = pd.DataFrame(rank_data)
    
    # move rank data to just after gobal data
    is_head_col = rank_data_df.pop('is_head')
    rank_data_df.insert(1, 'is_head', is_head_col)

    return rank_data_df

def load_and_prep_run_data(
        dataset_name,
        exp_dir,
        exp_id=None, #if none, get all
        incl_global_struct=True,
        incl_hps=True,
        incl_neighbour_structs=True,
        randomise=True
    ):
    
    assert incl_hps , 'All modern TWIG versions assume HPs should be included'
    categorical_cols = ['loss', 'neg_samp'] if incl_hps else []

    rank_data_df = get_twm_data_augment(dataset_name,
        exp_dir,
        exp_id=exp_id, #if none, get all
        incl_global_struct=incl_global_struct,
        incl_hps=incl_hps,
        incl_neighbour_structs=incl_neighbour_structs,
        randomise=randomise
    )

    y = rank_data_df['rank']
    del rank_data_df['rank']
    X = rank_data_df

    # one-hot code categorical vars: https://www.statology.org/pandas-get-dummies/
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    X["margin"].fillna(0, inplace=True)

    X = torch.tensor(X.to_numpy(dtype=np.float32))
    y = torch.tensor(y.to_numpy(dtype=np.float32))

    return X, y

def load_and_prep_dataset_data(
        dataset_name,
        run_ids,
        try_load=True,
        allow_load_err=True,
        randomise=True,
        exp_id=None
    ):

    save_prefixes = []
    for run_id in run_ids:
        save_prefixes.append(
            (f'data_save/{dataset_name}-{run_id}', run_id)
        )

    dataset_data = {}
    if try_load:
        for save_path, run_id in save_prefixes:
            data_dir = f'{dataset_name}-TWM-run{run_id}'
            print(f'data dir for saving files is: {data_dir}')
            try:
                # we try to load the results of a run
                # DBpedia50-TWM-run2.1 is named "DBpedia50-2.1"
                print(f'Loading the saved dataset with id  {run_id}...')
                print(f'save path prefix is {save_path}')
                if randomise:
                    X = torch.load(save_path + '-rand-X')
                    y = torch.load(save_path + '-rand-y')
                else:
                    X = torch.load(save_path + '-norand-X')
                    y = torch.load(save_path + '-norand-y')
                dataset_data[run_id] = {}
                dataset_data[run_id]['X'] = X
                dataset_data[run_id]['y'] = y
                print('done')
            except:
                if not allow_load_err: raise
                else:
                    # we now load the data, and then save it
                    print('No data to load (or error loading). Manually re-creating dataset...')
                    X, y = load_and_prep_run_data(
                        dataset_name,
                        exp_dir=f'../output/{dataset_name}/{data_dir}',
                        randomise=randomise,
                        exp_id=exp_id
                    )
                    dataset_data[run_id] = {}
                    dataset_data[run_id]['X'] = X
                    dataset_data[run_id]['y'] = y
                    if randomise:
                        torch.save(X, f'{save_path}-rand-X')
                        torch.save(y, f'{save_path}-rand-y')
                    else:
                        torch.save(X, f'{save_path}-norand-X')
                        torch.save(y, f'{save_path}-norand-y')

    return dataset_data

def get_norm_func(
        base_data,
        normalisation='none',
        norm_col_0=True
    ):
    assert normalisation in ('minmax', 'zscore', 'none')
    if normalisation == 'none':
        norm_func_data = {
            'type' : normalisation,
            'params': []
        }
        def norm_func(base_data):
            return
        return norm_func, norm_func_data
    
    elif normalisation == 'minmax':
        running_min = None
        running_max = None
        for dataset_name in base_data:
            dataset_data = base_data[dataset_name]
            for run_id in dataset_data:
                X = dataset_data[run_id]['X']
                if running_min is None:
                    running_min = torch.min(X, dim=0).values
                else:
                    running_min = torch.min(
                        torch.stack(
                            [torch.min(X, dim=0).values, running_min]
                        ),
                        dim=0
                    ).values
                if running_max is None:
                    running_max = torch.max(X, dim=0).values
                else:
                    running_max = torch.max(
                        torch.stack(
                            [torch.max(X, dim=0).values, running_max]
                        ),
                        dim=0
                    ).values

        norm_func_data = {
            'type' : normalisation,
            'params': [running_min, running_max, norm_col_0]
        }
        def norm_func(base_data):
            minmax_norm_func(
                base_data,
                running_min,
                running_max,
                norm_col_0=norm_col_0
            )
        
        return norm_func, norm_func_data

    elif normalisation == 'zscore':
        # running average has been verified to be coreect
        running_avg = None
        num_samples = 0.
        for dataset_name in base_data:
            dataset_data = base_data[dataset_name]
            for run_id in dataset_data:
                X = dataset_data[run_id]['X']
                num_samples += X.shape[0]
                if running_avg is None:
                    running_avg = torch.sum(X, dim=0)
                else:
                    running_avg += torch.sum(X, dim=0)
        running_avg /= num_samples

        # running std has been verified to be coreect
        running_std = None
        for dataset_name in base_data:
            dataset_data = base_data[dataset_name]
            for run_id in dataset_data:
                X = dataset_data[run_id]['X']
                if running_std is None:
                    running_std = torch.sum(
                        (X - running_avg) ** 2,
                        dim=0
                    )
                else:
                    running_std += torch.sum(
                        (X - running_avg) ** 2,
                        dim=0
                    )
        running_std = torch.sqrt(
            (1 / (num_samples - 1)) * running_std
        )

        norm_func_data = {
            'type' : normalisation,
            'params': [running_avg, running_std, norm_col_0]
        }
        def norm_func(base_data):
            zscore_norm_func(
                base_data,
                running_avg,
                running_std,
                norm_col_0=norm_col_0
            )
        
        return norm_func, norm_func_data

def do_rescale_y(base_data):
    for dataset_name in base_data:
        dataset_data = base_data[dataset_name]
        for run_id in dataset_data:
            max_rank = dataset_data[run_id]['X'][0][0]
            y = dataset_data[run_id]['y']
            dataset_data[run_id]['y'] = y / max_rank

def minmax_norm_func(base_data, train_min, train_max, norm_col_0=True):
    for dataset_name in base_data:
        dataset_data = base_data[dataset_name]
        for run_id in dataset_data:
            X = dataset_data[run_id]['X']
            if not norm_col_0:
                X_graph, X_other = X[:, :1], X[:, 1:] # ignore col 0; that is the max rank, and we needs its original value!
                X_other = (X_other - train_min[1:]) / (train_max[1:] - train_min[1:])
                X_norm = torch.concat(
                    [X_graph, X_other],
                    dim=1
                )
            else:
                X_norm = (X - train_min) / (train_max - train_min)

            # if we had nans (i.e. min = max) set them all to 0.5
            X_norm = torch.nan_to_num(X_norm, nan=0.5, posinf=0.5, neginf=0.5) 

            dataset_data[run_id]['X'] = X_norm
            dataset_data[run_id]['y'] = dataset_data[run_id]['y']

def zscore_norm_func(base_data, train_mean, train_std, norm_col_0=True):
    for dataset_name in base_data:
        dataset_data = base_data[dataset_name]
        for run_id in dataset_data:
            X = dataset_data[run_id]['X']
            if not norm_col_0:
                X_graph, X_other = X[:, :1], X[:, 1:] # ignore col 0; that is the max rank, and we needs its original value!
                X_other = (X_other - train_mean[1:]) / train_std[1:]
                X_norm = torch.concat(
                    [X_graph, X_other],
                    dim=1
                )
            else:
                X_norm = (X - train_mean) / train_std

            # if we had nans (i.e. min = max) set them all to 0
            X_norm = torch.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0) 

            dataset_data[run_id]['X'] = X_norm
            dataset_data[run_id]['y'] = dataset_data[run_id]['y']

def twm_load(
        dataset_names,
        normalisation,
        rescale_y,
        dataset_to_run_ids,
        exp_id,
        norm_func_path
    ):
    num_hp_settings = 1215

    # load raw data for each dataset
    base_data = {}
    for dataset_name in dataset_names:
        print(f'Loading data for {dataset_name}')
        # get the data for this dataset
        base_data[dataset_name] = load_and_prep_dataset_data(
            dataset_name,
            run_ids=dataset_to_run_ids[dataset_name],
            randomise=False,
            try_load=True,
            exp_id=exp_id
        )

    # do normalisation
    print(f'Normalising data with strategy {normalisation}... and rescale_y = {rescale_y}', end='')
    if rescale_y:
        do_rescale_y(base_data)
    norm_func = load_norm_func_from_disk(norm_func_path)
    norm_func(base_data)

    # load data into Torch DataLoaers
    twig_data = {}
    for dataset_name in dataset_names:
        # get the batch size for this dataset
        dataset_data = base_data[dataset_name]
        num_datapoints_per_run = dataset_data[f'2.1']['X'].shape[0]
        if exp_id is None:
            dataset_batch_size = num_datapoints_per_run // num_hp_settings
            assert num_datapoints_per_run % num_hp_settings == 0, f"Wrong divisor: should be 0 but is {num_datapoints_per_run % num_hp_settings}"
            assert dataset_batch_size * num_hp_settings == num_datapoints_per_run
        else:
            assert type(exp_id) is int, 'exp must be a single int if it is not None'
            dataset_batch_size = num_datapoints_per_run

        # get training data
        print('run IDs', dataset_to_run_ids[dataset_name])
        data_x = None
        data_y = None
        for run_id in dataset_to_run_ids[dataset_name]:
            if data_x is None and data_y is None:
                data_x = dataset_data[run_id]['X']
                data_y = dataset_data[run_id]['y']
            else:
                data_x = torch.concat(
                    [data_x, dataset_data[run_id]['X']],
                    dim=0
                )
                data_y = torch.concat(
                    [data_y, dataset_data[run_id]['y']],
                    dim=0
                )

        twig_data[dataset_name] = {}

        print(f'configuring batches; using training batch size {dataset_batch_size}')
        training = TensorDataset(data_x, data_y)
        training_dataloader = DataLoader(
            training,
            batch_size=dataset_batch_size
        )
        twig_data[dataset_name] = training_dataloader

    return twig_data

def load_norm_func_from_disk(norm_func_data_path):
    with open(norm_func_data_path, 'rb') as cache:
        print('loading model settings from cache:', norm_func_data_path)
        norm_func_data = pickle.load(cache)

    if norm_func_data['type'] == 'none':
        def norm_func(base_data):
            return
        return norm_func
    elif norm_func_data['type'] == 'minmax':
        def norm_func(base_data):
            minmax_norm_func(
                base_data,
                norm_func_data['params'][0],
                norm_func_data['params'][1],
                norm_col_0=norm_func_data['params'][2]
            )
        return norm_func
    elif norm_func_data['type'] == 'zscore':
        def norm_func(base_data):
            zscore_norm_func(
                base_data,
                norm_func_data['params'][0],
                norm_func_data['params'][1],
                norm_col_0=norm_func_data['params'][2]
            )
        return norm_func
    else:
        assert False, f'Unkown norm func type given: {norm_func_data["type"]}'

def load_and_prep_twig_data(
        datasets,
        normalisation,
        rescale_y,
        dataset_to_run_ids,
        testing_percent,
        try_load,
        exp_id,
        norm_func
    ):   
    '''
    Ok Future Jeffrey, here's the thing. There's this thing called code karma, and it has
    just caught up to you. You really thought you could get away with a hacky solution like
    this forever. Ha. You were wrong.

    See, you don't have access to the exp_id (i.e. 1, 2, ..., 1215) here. So if we want 10% of
    those to be a hold out test set, we have two choices
        1) go through the entire codebase and refactor it to load in a way that is compatible with that
        2) use the fact that they were all entered into the Big Tensor contiguously......

    Basically, right now, base_data is
        base_data[dataset_name][run_id]['X' or 'y'] --> X or y tensor
    
    In other words, we have (for a run_id like 2.1) all 1215 hyperparameter combinations, contiguously
    (meaning that data from hyperparamter exp_id 1215 can be read in a contiguous block). Here, we want
    to always remove the same hyperparameter combination from all rounds, so we make sure to default the
    randomise parameter of the loader function to False.

    This means, however, that we need to mamually randomise the dataset before returning it, as that is
    best prtactice. Once we have it randomised, as can take the last 122 experiments in the Tensor as
    our test set easily as well!

    That just means we need to know the number of rows that encode each exp. Luckily this is constant
    for a constant dataset, so we can just do (num_rows_in_dataset_tensor / num_exps) (num_exps is 1215)
    to get that. Boom! There we go!

    It actually super easy. Ingenious, almost. But if you're reading this again, I bet you're looking for
    functionality I did not directly implement. In specific, I should ccall out that I do not have a map
    back to what exp ids were used in training or testing, since that data onlyt exists way back. But in
    any case: Good luck, Future Jeffrey.

    You wrote this code. Welcome to maintaining it. 
    '''
    num_hp_settings = 1215
    num_exps_to_select = int(testing_percent * num_hp_settings + 0.5)

    print('training TWIG with:')
    print('dataset_to_run_ids')
    print(dataset_to_run_ids)
    print()
    print('num_exps_to_select')
    print(num_exps_to_select)
    print()

    # load raw data for each dataset
    base_data = {}
    for dataset_name in datasets:
        print(f'Loading data for {dataset_name}')
        # get the data for this dataset
        base_data[dataset_name] = load_and_prep_dataset_data(
            dataset_name,
            run_ids=dataset_to_run_ids[dataset_name],
            randomise=False,
            try_load=try_load,
            exp_id=exp_id
        )

    # # randomise block order (we won't let block be broken up!)
    block_randomisation_tensor = torch.randperm(num_hp_settings)
    print('block_randomisation_tensor')
    print([int(x) for x in block_randomisation_tensor])
    print()

    # do a train-test split
    train_dataset_data_raw = {}
    test_dataset_data_raw = {}
    num_rows_per_exp_all = {}
    for dataset_name in base_data:
        print(dataset_name)
        num_rows_per_exp = None
        num_rows_to_select = None
        train_dataset_data_raw[dataset_name] = {}
        test_dataset_data_raw[dataset_name] = {}
        for run_id in base_data[dataset_name]:
            print(run_id)
            X = base_data[dataset_name][run_id]['X']
            y = base_data[dataset_name][run_id]['y']

            if num_rows_per_exp is None:
                assert X.shape[0] % num_hp_settings == 0, 'There should be 1215 exps with an even number of rows in X (and in y)'
                num_rows_per_exp = int(X.shape[0] / num_hp_settings) #num_hp_settings = number of exps
                num_rows_to_select = num_exps_to_select * num_rows_per_exp
                num_rows_per_exp_all[dataset_name] = num_rows_per_exp

                # randomisation step (rm same exp_id from all runs!)
                # that's why block_randomisation_tensor is calculated exactly once
                # internal order of rows in the block is irrelevant but unchanged here
                # we only do this once since its the same idxs for all X's in the same
                # dataset
                randomisation_tensor = []
                for exp_id in block_randomisation_tensor:
                    for i in range(num_rows_per_exp):
                        randomisation_tensor.append(num_rows_per_exp * exp_id + i)
                randomisation_tensor = torch.tensor(randomisation_tensor)
                assert torch.unique(randomisation_tensor).shape[0] == num_hp_settings * num_rows_per_exp, 'There should be no non-unique values in the randomisation tensor!'
            else:
                assert num_rows_per_exp * num_hp_settings == X.shape[0]

            # use the randomisation tensor to randomise the rows of X and y
            X = X[randomisation_tensor]
            y = y[randomisation_tensor]

            # do train / test split for X
            X_train = X[num_rows_to_select:, :]
            y_train = y[num_rows_to_select:]
            train_dataset_data_raw[dataset_name][run_id] = {
                'X' : X_train,
                'y': y_train
            }

            # do train / test split for y
            X_test = X[:num_rows_to_select, :]
            y_test = y[:num_rows_to_select]
            test_dataset_data_raw[dataset_name][run_id] = {
                'X' : X_test,
                'y': y_test
            }

            # some validations
            assert X_train.shape[0] % num_rows_per_exp == 0, X_train.shape
            assert X_test.shape[0] % num_rows_per_exp == 0, X_test.shape
            assert y_train.shape[0] % num_rows_per_exp == 0
            assert y_test.shape[0] % num_rows_per_exp == 0

    # do normalisation (no change if normalisation == 'none')
    if rescale_y:
        do_rescale_y(train_dataset_data_raw)
    if rescale_y:
        do_rescale_y(test_dataset_data_raw)
    if not norm_func: #sometimes one may be given from disk
        norm_func, norm_func_data = get_norm_func(
            train_dataset_data_raw,
            normalisation=normalisation,
            norm_col_0=False
        )
    norm_func(train_dataset_data_raw) #it's in-place!
    norm_func(test_dataset_data_raw)

    train_dataset_data = train_dataset_data_raw
    test_dataset_data = test_dataset_data_raw

    # load data into Torch DataLoaers
    twig_data = {}
    for dataset_name in datasets:
        dataset_batch_size = num_rows_per_exp_all[dataset_name] #since we can only calc loss for one exp (and all its rows) at a time
        # assert dataset_batch_size == 1304 # for UMLS only

        # get training data
        print('training run IDs', train_dataset_data[dataset_name].keys())
        training_data_x = None
        training_data_y = None
        for run_id in train_dataset_data[dataset_name]:
            if training_data_x is None and training_data_y is None:
                training_data_x = train_dataset_data[dataset_name][run_id]['X']
                training_data_y = train_dataset_data[dataset_name][run_id]['y']
            else:
                training_data_x = torch.concat(
                    [training_data_x, train_dataset_data[dataset_name][run_id]['X']],
                    dim=0
                )
                training_data_y = torch.concat(
                    [training_data_y, train_dataset_data[dataset_name][run_id]['y']],
                    dim=0
                )
        print(f'training_data_x shape --> {training_data_x.shape}')
        print(f'training_data_y shape --> {training_data_y.shape}')

        # get testing data
        print('testing run IDs', test_dataset_data[dataset_name].keys())
        testing_data_x = None
        testing_data_y = None
        for run_id in test_dataset_data[dataset_name]:
            if testing_data_x is None and testing_data_y is None:
                testing_data_x = test_dataset_data[dataset_name][run_id]['X']
                testing_data_y = test_dataset_data[dataset_name][run_id]['y']
            else:
                testing_data_x = torch.concat(
                    [testing_data_x, test_dataset_data[dataset_name][run_id]['X']],
                    dim=0
                )
                testing_data_y = torch.concat(
                    [testing_data_y, test_dataset_data[dataset_name][run_id]['y']],
                    dim=0
                )
        print(f'testing_data_x shape --> {testing_data_x.shape}')
        print(f'testing_data_y shape --> {testing_data_y.shape}')

        twig_data[dataset_name] = {}

        if training_data_x is not None and training_data_y is not None:
            print(f'configuring batches; using training batch size {dataset_batch_size}')
            training = TensorDataset(training_data_x, training_data_y)
            training_dataloader = DataLoader(
                training,
                batch_size=dataset_batch_size
            )
            twig_data[dataset_name]['training'] = training_dataloader
        else:
            print(f'No training data has been given to be loaded for {dataset_name}')
            print('please check train and testing data definitions. This is not necessarily an issue')

        if testing_data_x is not None and testing_data_y is not None:
            print(f'configuring batches; using testing batch size {dataset_batch_size}')
            testing = TensorDataset(testing_data_x, testing_data_y)
            testing_dataloader = DataLoader(
                testing,
                batch_size=dataset_batch_size
            )
            twig_data[dataset_name]['testing'] = testing_dataloader
        else:
            print(f'No testing data has been given to be loaded for {dataset_name}')
            print('please check train and testing data definitions. This is not necessarily an issue')

    return twig_data, norm_func_data

def do_load(
        dataset_names,
        normalisation,
        rescale_y,
        dataset_to_run_ids,
        testing_percent
    ):
    '''
    do_load() KGE results for all given KGs and processes them into a format that can be directly used for training by TWIG. It uses random hyperparameter combinations as a hold-out test set.

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
    twig_data, norm_func_data = load_and_prep_twig_data(
        dataset_names,
        normalisation=normalisation,
        rescale_y=rescale_y,
        dataset_to_run_ids=dataset_to_run_ids,
        testing_percent=testing_percent
    )

    training_dataloaders = {}
    for dataset_name in twig_data:
        if 'training' in twig_data[dataset_name]:
            training_dataloaders[dataset_name] = twig_data[dataset_name]['training']
    testing_dataloaders = {}
    for dataset_name in twig_data:
        if 'testing' in twig_data[dataset_name]:
            testing_dataloaders[dataset_name] = twig_data[dataset_name]['testing']

    return training_dataloaders, testing_dataloaders, norm_func_data
