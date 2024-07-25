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
num_hp_settings = 1215

'''
=========
Functions
=========
'''
def get_adj_data(valid_triples_map):
    '''
    get_adj_data() generates a mapping from every entity to all triples that contain it as a subject or object.

    The arguments it accepts are:
        - triples_map (dict int to tuple<int,int,int>): a dict mapping from a triple ID to the IDs of the three elements (subject, predicate, and object) that make up that triple. 

    The values it returns are:
        - ents_to_triples (dict int -> list of tuple<int,int,int>): a dict that maps an entity ID to a list of all triples (expressed as the IDs for the subj, pred, and obj) containing that original entity.
    '''
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

def get_kge_data(
        dataset_name,
        exp_dir,
        struct_source,
        randomise
    ):
    '''
    get_kge_data() creates feature vectors describing link prediction queries (global and local KG structure, and KGE hyperparameters), and maps them all (in a pandas DF) to the rank that the KGE model acheived using the given hyperparameters, on the given KG< for that link prediction query.

    The arguments it accepts are:
        - 

    The values it returns are:
        - 
    '''
    
    overall_results, \
        triples_results, \
        grid, \
        valid_triples_map, \
        graph_stats = gather_data(dataset_name, exp_dir)
    ents_to_triples = get_adj_data(valid_triples_map)

    all_data = []
    iter_over = sorted(int(key) for key in triples_results.keys())
    if randomise:
        random.shuffle(iter_over)
    iter_over = [str(x) for x in iter_over]
    print(f'Loader: Randomise = {randomise}; Using exp_id order, {iter_over}')

    global_struct = {}
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

def load_dataset_data(
        dataset_name,
        run_ids,
        randomise,
        try_load=True,
        allow_load_err=True
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
                    rank_data_df = get_kge_data(
                        dataset_name,
                        exp_dir=f'../output/{dataset_name}/{data_dir}',
                        randomise=randomise
                    )
                    y = rank_data_df['rank']
                    del rank_data_df['rank']
                    X = rank_data_df

                    # one-hot code categorical vars: https://www.statology.org/pandas-get-dummies/
                    categorical_cols = ['loss', 'neg_samp']
                    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
                    X["margin"].fillna(0, inplace=True)

                    dataset_data[run_id] = {
                        'X': torch.tensor(X.to_numpy(dtype=np.float32)),
                        'y': torch.tensor(y.to_numpy(dtype=np.float32))
                    }
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
    '''
    get_norm_func() returns a function that can be called on any input data to normalise it.

    The arguments it accepts are:
        - base_data (dict of str -> str -> str -> torch.Tensor): A dict that maps, in order, a dataset name, a hyperparameter run ID, and a name (X or y) to a tensor containing either feature vector input (X) or output (y). 
        - normalisation (str): the name of the normaalisation method to use. Must be one of ('minmax', 'zscore', 'none')
        - norm_col_0 (bool): whether column 0 should be normalised. As col 0 often represents the max rank of a given dataset, this often should be set to false.

    The values it returns are:
        - norm_func (func): a function that accepts input data and outputs a row-by-row normalised version of the input.
    '''
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
    '''
    do_rescale_y() rescales all y-values onto [0, 1].

    The arguments it accepts are:
        - base_data (dict of str -> str -> str -> torch.Tensor): A dict that maps, in order, a dataset name, a hyperparameter run ID, and a name (X or y) to a tensor containing either feature vector input (X) or output (y). 

    The values it returns are:
        - None (in-placee modification)
    '''
    for dataset_name in base_data:
        dataset_data = base_data[dataset_name]
        for run_id in dataset_data:
            max_rank = dataset_data[run_id]['X'][0][0]
            y = dataset_data[run_id]['y']
            dataset_data[run_id]['y'] = y / max_rank

def minmax_norm_func(base_data, train_min, train_max, norm_col_0):
    '''
    minmax_norm_func() performs min-max normalisation on the given data

    The arguments it accepts are:
        - base_data (dict of str -> str -> str -> torch.Tensor): A dict that maps, in order, a dataset name, a hyperparameter run ID, and a name (X or y) to a tensor containing either feature vector input (X) or output (y). 
        - train_mean (torch.Tensor): A single-row tensor of the mean values observed for each feature
        - train_std (torch.Tensor): A single-row tensor of the standard eviation of the values observed for each feature
        - col_0_already_removed (bool): whether column 0 has been removed from X before passing it to this function or not

    The values it returns are:
        - None (in-place modification is used)

    NOTE:  due to how this function is created, train_mean and train_std will contain "column 0" (the max rank values possible for a given dataset). These should never be used (they are spliced out) since col the max rank column should never be normalisedd if present.
    '''
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

def zscore_norm_func(base_data, train_mean, train_std, norm_col_0):
    '''
    zscore_norm_func() performs z-score normalisation on the given data

    The arguments it accepts are:
        - base_data (dict of str -> str -> str -> torch.Tensor): A dict that maps, in order, a dataset name, a hyperparameter run ID, and a name (X or y) to a tensor containing either feature vector input (X) or output (y). 
        - train_mean (torch.Tensor): A single-row tensor of the mean values observed for each feature
        - train_std (torch.Tensor): A single-row tensor of the standard eviation of the values observed for each feature
        - col_0_already_removed (bool): whether column 0 has been removed from X before passing it to this function or not

    The values it returns are:
        - None (in-place modification is used)

    NOTE:  due to how this function is created, train_mean and train_std will contain "column 0" (the max rank values possible for a given dataset). These should never be used (they are spliced out) since col the max rank column should never be normalisedd if present.
    '''
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

def load_norm_func_from_disk(norm_func_path):
    '''
    load_norm_func_from_disk() loads a normalisation function from data saved to disk.
    
    The arguments it accepts are:
        - norm_func_path (str): the path on disk where normalisation function data was saved, to avoid recomputation
    
    The values it returns are:
        - norm_func (func): a function that can be used for inplace normalisation of TWIG data.
    '''
    with open(norm_func_path, 'rb') as cache:
        print('loading model settings from cache:', norm_func_path)
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

def twm_load(
        dataset_names,
        normalisation,
        rescale_y,
        dataset_to_run_ids,
        norm_func_path
    ):
    '''
    twm_load() manages loading data in the format needed to run TWM. That is to say, all data for all hyperparameter combinations are laoded into a Dataloader (or Dataloaders, if multiple are requested), with no train / test / valid splits.
    
    The arguments it accepts are:
        - dataset_names (list of str): the names of all KG datasets that should be loaded.
        - normalisation (str): the normalisation method to use when loading data. "zscore", "minmax", and "none" are supported.
        - rescale_y (bool): True if the groun-truth data `y` was rescaled onto [0, 1] during data loading, False otherwise
        - dataset_to_run_ids (dict of str -> list<str>): a list that maps each datasest name (in dataset_names) to a list of run_ids for that dataset on disk that should be loaded. These typically take the form 2.x, where x is a single numeric digit
        - norm_func_path (str): the path on disk where normalisation function data was saved, to avoid recomputation
    
    The values it returns are:
        - twig_data (dict of str -> torch.Dataloader): a dict that maps a dataset name to the dataloader for that KG and that split. For example, twig_data['UMLS'] would return a Dataloader containing all TWIG data for the UMLS dataset.
    '''
    num_hp_settings = 1215

    # load raw data for each dataset
    base_data = {}
    for dataset_name in dataset_names:
        print(f'Loading data for {dataset_name}')
        # get the data for this dataset
        base_data[dataset_name] = load_dataset_data(
            dataset_name,
            run_ids=dataset_to_run_ids[dataset_name],
            randomise=False,
            try_load=True,
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
        dataset_batch_size = num_datapoints_per_run // num_hp_settings
        assert num_datapoints_per_run % num_hp_settings == 0, f"Wrong divisor: should be 0 but is {num_datapoints_per_run % num_hp_settings}"
        assert dataset_batch_size * num_hp_settings == num_datapoints_per_run

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

def load_twig_data(
        datasets,
        normalisation,
        rescale_y,
        dataset_to_run_ids,
        testing_percent,
        valid_percent,
        try_load,
        norm_func=None
    ):   
    '''
    load_twig_data() loads all strutrual and hyperparameter data for each triple in the input KG(S). It also manages creating a trai-test split, and returning the loaded data and the normalisation functions used to load it. Much of this is done with calls to other functions in this file.

    The arguments it accepts are:
        - datasets (list of str): the names of all KG datasets that should be loaded.
        - - normalisation (str): the normalisation method to use when loading data. "zscore", "minmax", and "none" are supported.
        - rescale_y (bool): True if the groun-truth data `y` was rescaled onto [0, 1] during data loading, False otherwise
        - dataset_to_run_ids (dict of str -> list<str>): a list that maps each datasest name (in dataset_names) to a list of run_ids for that dataset on dissk that should be loaded. These typically take the form 2.x, where x is a single numeric digit.
        - testing_percent (float): the percent of all hyperparameter combinations on each KG to reserve as a hold-out test set.
        - valid_percent (float): the percent of all hyperparameter combinations on each KG to reserve as a hold-out validation set.
        - try_load (bool): Whether TWIG should try to laod pre-processed data from disk or load everything from scratch.
        - norm_func (func or None): if given, a pre-defined normalisation function (loaded from disk) that should be used. This is useful in when loading a model, so that the same norm func can be used even if data input changes slightly. 

    The Values it returns are:
        - twig_data (dict of str -> str -> torch.Dataloader): a dict that maps a dataset name and a training split to the dataloader for that KG and that split. For example, twig_data['UMLS']['testing'] would return a Dataloader to be used furing the testing phase for the UMLS dataset.
        - norm_func_data (dict of str -> any): a dict with two keys
            - 'type' : mappng to the name of the normalisation technique, Must be zscore, minmax, or none.
            - 'params': the normalisation parameters. For zscore, they are: [running_avg, running_std, norm_col_0]. For minman, they are: [running_min, running_max, norm_col_0]. For none, they are [].

    NOTE: Randomisation by hyperparameter ID is done using the property that all hyperparameter combinations are loaded in the same order, no matter what KG is being loaded. A such, randomising them with the same randomly sampled indexes will result in a random order that is consistent accross all datasets. However, no explicit exp_id -> feature vector exists here.
    '''
    assert testing_percent > 0, f"testing_percent must be > 0 but is {testing_percent}"
    assert valid_percent >= 0, f"testing_percent must be >= = but is {testing_percent}"
    num_test_exps = int(testing_percent * num_hp_settings + 0.5)
    num_valid_exps = int(valid_percent * num_hp_settings + 0.5)

    print('training TWIG with:')
    print('dataset_to_run_ids')
    print(dataset_to_run_ids)
    print()
    print('num test exps:')
    print(num_test_exps)
    print()
    print('num valid exps:')
    print(num_valid_exps)
    print()

    # load raw data for each dataset
    base_data = {}
    for dataset_name in datasets:
        print(f'Loading data for {dataset_name}')
        # get the data for this dataset
        base_data[dataset_name] = load_dataset_data(
            dataset_name,
            run_ids=dataset_to_run_ids[dataset_name],
            randomise=False,
            try_load=try_load,
        )

    # make a sset of random indices so we can randomise block order (we won't let block be broken up!)
    # here, a block refers to an exp_id (i.e. hyperparameter combination 0, 1, 1214) -- the basic unit of TWIG's learning
    block_randomisation_tensor = torch.randperm(num_hp_settings)
    print('block_randomisation_tensor')
    print([int(x) for x in block_randomisation_tensor])
    print()

    # do a train-test-valid split using the block randomisation tensor
    train_dataset_data = {}
    test_dataset_data = {}
    valid_datasest_data = {}
    num_rows_per_exp_all = {}
    for dataset_name in base_data:
        print(dataset_name)
        num_rows_per_exp = None
        num_test_rows = None
        num_valid_rows = None
        train_dataset_data[dataset_name] = {}
        test_dataset_data[dataset_name] = {}
        valid_datasest_data[dataset_name] = {}
        for run_id in base_data[dataset_name]:
            print(run_id)
            X = base_data[dataset_name][run_id]['X']
            y = base_data[dataset_name][run_id]['y']

            if num_rows_per_exp is None:
                assert X.shape[0] % num_hp_settings == 0, 'There should be 1215 exps with an even number of rows in X (and in y)'
                num_rows_per_exp = int(X.shape[0] / num_hp_settings) #num_hp_settings = number of exps
                num_test_rows = num_test_exps * num_rows_per_exp
                num_valid_rows = num_valid_exps * num_rows_per_exp
                num_rows_per_exp_all[dataset_name] = num_rows_per_exp

                # we keep items in a block in numerical order, but randomise the block order
                randomisation_tensor = []
                for exp_id in block_randomisation_tensor:
                    for i in range(num_rows_per_exp):
                        randomisation_tensor.append(num_rows_per_exp * exp_id + i)
                randomisation_tensor = torch.tensor(randomisation_tensor)
                assert torch.unique(randomisation_tensor).shape[0] == num_hp_settings * num_rows_per_exp, 'There should be no non-unique values in the randomisation tensor!'
            else:
                assert num_rows_per_exp * num_hp_settings == X.shape[0]

            # use the randomisation tensor to randomise the blocks of exps in X and y
            X = X[randomisation_tensor]
            y = y[randomisation_tensor]

            # we take X into splits as so: [test][valid][train]
            # do test split
            X_test = X[:num_test_rows, :]
            y_test = y[:num_test_rows]
            test_dataset_data[dataset_name][run_id] = {
                'X' : X_test,
                'y': y_test
            }

            # do valid split, if requested
            if num_valid_exps > 0:
                test_and_valid_end_idx = num_test_rows + num_valid_rows
                X_valid = X[num_test_rows:test_and_valid_end_idx, :]
                y_valid = y[num_test_rows:test_and_valid_end_idx]
                valid_datasest_data[dataset_name][run_id] = {
                    'X' : X_valid,
                    'y': y_valid
                }
            else:
                test_and_valid_end_idx = num_test_rows

            # do train split
            X_train = X[test_and_valid_end_idx:, :]
            y_train = y[test_and_valid_end_idx:]
            train_dataset_data[dataset_name][run_id] = {
                'X' : X_train,
                'y': y_train
            }

            # some validations
            assert X_train.shape[0] % num_rows_per_exp == 0, X_train.shape
            assert X_test.shape[0] % num_rows_per_exp == 0, X_test.shape
            assert X_valid.shape[0] % num_rows_per_exp == 0, X_valid.shape
            assert y_train.shape[0] % num_rows_per_exp == 0
            assert y_test.shape[0] % num_rows_per_exp == 0
            assert y_valid.shape[0] % num_rows_per_exp == 0

    # do normalisation (no change if normalisation == 'none')
    if rescale_y:
        do_rescale_y(train_dataset_data)
    if rescale_y:
        do_rescale_y(test_dataset_data)
    if not norm_func: #sometimes one may be given from disk
        norm_func, norm_func_data = get_norm_func(
            train_dataset_data,
            normalisation=normalisation,
            norm_col_0=False
        )
    norm_func(train_dataset_data) #it's in-place!
    norm_func(test_dataset_data)
    if num_valid_exps > 0:
        norm_func(valid_datasest_data)

    # load data into Torch DataLoaers
    twig_data = {}
    for dataset_name in datasets:
        dataset_batch_size = num_rows_per_exp_all[dataset_name] #since we can only calc loss for one exp (and all its rows) at a time

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

        # get validation data
        if num_valid_exps > 0:
            print('validation run IDs', valid_datasest_data[dataset_name].keys())
            validation_data_x = None
            validation_data_y = None
            for run_id in valid_datasest_data[dataset_name]:
                if validation_data_x is None and validation_data_y is None:
                    validation_data_x = valid_datasest_data[dataset_name][run_id]['X']
                    validation_data_y = valid_datasest_data[dataset_name][run_id]['y']
                else:
                    validation_data_x = torch.concat(
                        [validation_data_x, valid_datasest_data[dataset_name][run_id]['X']],
                        dim=0
                    )
                    validation_data_y = torch.concat(
                        [validation_data_y, valid_datasest_data[dataset_name][run_id]['y']],
                        dim=0
                    )
            print(f'validation_data_x shape --> {validation_data_x.shape}')
            print(f'validation_data_y shape --> {validation_data_y.shape}')

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

        if num_valid_exps > 0:
            if validation_data_x is not None and validation_data_y is not None:
                print(f'configuring batches; using validation batch size {dataset_batch_size}')
                validation = TensorDataset(validation_data_x, validation_data_y)
                validation_dataloader = DataLoader(
                    validation,
                    batch_size=dataset_batch_size
                )
                twig_data[dataset_name]['validation'] = validation_dataloader
            else:
                print(f'No validation data has been given to be loaded for {dataset_name}')
                print('please check train and validation data definitions. This is not necessarily an issue')


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
        - dataset_to_run_ids (dict of str -> list<str>): a list that maps each datasest name (in dataset_names) to a list of run_ids for that dataset on dissk that should be loaded. These typically take the form 2.x, where x is a single numeric digit.
        - testing_percent (float): the percent of all hyperparameter combinations on each KG to reserve as a hold-out test set. If not given, defaults to 0.1.

    The values it returns are:
        - training_dataloaders (torch.Dataloader): a map for a KG name to the dataloader containing data to use for training TWIG to simulate KGEs on that KG
        - testing_dataloaders (dict of str -> torch.Dataloader): a map for a KG name to the dataloader containing data to use for testing TWIG on simulating KGEs on that KG
        - norm_func_data (dict of str -> any): a dictionary mapping parameters for normalisation (such as mean and standard deviaation) to their values so that a normaliation function can be directly construcgted from these values
    '''
    twig_data, norm_func_data = load_twig_data(
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
