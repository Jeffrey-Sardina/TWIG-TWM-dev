from utils import gather_data
import random
import os
import numpy as np
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_canonical_exp_dir(dataset_name, run_id):
    exp_dir = f"../output/{dataset_name}/{dataset_name}-TWM-run{run_id}"
    return exp_dir

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

def load_global_data(graph_stats):
    global_data = {
        'max_rank': len(graph_stats['all']['degrees']) # = num nodes
    }
    return global_data

def load_local_data(triples_map, graph_stats):
    local_data = {}
    ents_to_triples = get_adj_data(triples_map)
    for triple_idx in triples_map:
        s, p, o = triples_map[triple_idx]

        local_data['s_deg'] = graph_stats['train']['degrees'][s]
        local_data['o_deg'] = graph_stats['train']['degrees'][o]
        local_data['p_freq'] = graph_stats['train']['pred_freqs'][p]

        local_data['s_p_cofreq'] = graph_stats['train']['subj_relationship_degrees'][(s,p)] \
            if (s,p) in graph_stats['train']['subj_relationship_degrees'] else 0
        local_data['o_p_cofreq'] = graph_stats['train']['obj_relationship_degrees'][(o,p)] \
            if (o,p) in graph_stats['train']['obj_relationship_degrees'] else 0
        local_data['s_o_cofreq'] = graph_stats['train']['subj_obj_cofreqs'][(s,o)] \
            if (s,o) in graph_stats['train']['subj_obj_cofreqs'] else 0

        target_dict = {'s': s, 'o': o}
        for target_name in target_dict:
            target = target_dict[target_name]
            neighbour_nodes = {}
            neighbour_preds = {}
            for t_idx in ents_to_triples[target]:
                t_s, t_p, t_o = triples_map[t_idx]
                ent = t_s if target != t_s else t_o
                if not t_p in neighbour_preds:
                    neighbour_preds[t_p] = graph_stats['train']['pred_freqs'][t_p]
                if not ent in neighbour_nodes:
                    neighbour_nodes[ent] = graph_stats['train']['degrees'][ent]

            local_data[f'{target_name} min deg neighbnour'] = np.min(list(neighbour_nodes.values()))
            local_data[f'{target_name} max deg neighbnour'] = np.max(list(neighbour_nodes.values()))
            local_data[f'{target_name} mean deg neighbnour'] = np.mean(list(neighbour_nodes.values()))
            local_data[f'{target_name} num neighbnours'] = len(neighbour_nodes)

            local_data[f'{target_name} min freq rel'] = np.min(list(neighbour_preds.values()))
            local_data[f'{target_name} max freq rel'] = np.max(list(neighbour_preds.values()))
            local_data[f'{target_name} mean freq rel'] = np.mean(list(neighbour_preds.values()))
            local_data[f'{target_name} num rels'] = len(neighbour_preds)
    return local_data

def load_hyperparamter_data(grid):
    # replace none with default vals
    for exp_id in grid:
        for hp_name in grid[exp_id]:
            hp_val = grid[exp_id][hp_name]
            if hp_val == None:
                grid[exp_id][hp_name] = 0

    # map names to values
    hp_names_to_vals = {}
    for exp_id in grid:
        hp_names = grid[exp_id]
        for hp_name in hp_names:
            hp_val = hp_names[hp_name]
            if not hp_name in hp_names_to_vals:
                hp_names_to_vals[hp_name] = set()
            hp_names_to_vals[hp_name].add(hp_val)
    
    # sort values
    hp_names_to_vals_sorted = {}
    for hp_name in hp_names_to_vals:
        hp_names_to_vals_sorted[hp_name] = sorted(list(hp_names_to_vals[hp_name]))
    
    # how create a hyperparamter data dict
    hyperparameter_data = {}
    for exp_id in grid:
        hyperparameter_data[exp_id] = {}
        for hp_name in grid[exp_id]:
            hp_val = grid[exp_id][hp_name]
            try:
                # we can can convert to float, no need to encode
                hp_val = float(hp_val)
                hyperparameter_data[exp_id][hp_name] = hp_val
            except:
                # if we cannot, we need to encode as a number
                hp_val_id = hp_names_to_vals_sorted[hp_name].index(hp_val)
                hyperparameter_data[exp_id][hp_name] = hp_val_id
    
    return hyperparameter_data

def load_simulation_dataset(dataset_name, run_id):
    exp_dir = get_canonical_exp_dir(dataset_name, run_id)
    _, rank_data, grid, valid_triples_map, graph_stats = gather_data(dataset_name, exp_dir)
    global_data = load_global_data(graph_stats=graph_stats)
    local_data = load_local_data(
        triples_map=valid_triples_map,
        graph_stats=graph_stats
    )
    hyperparameter_data = load_hyperparamter_data(grid=grid)
    return global_data, local_data, hyperparameter_data, rank_data

def load_simulation_datasets(datasets_to_load):
    global_data = {}
    local_data = {}
    rank_data = {}
    hyperparameter_data = None
    for dataset_name in datasets_to_load:
        global_data[dataset_name] = {}
        local_data[dataset_name] = {}
        rank_data[dataset_name] = {}
        for run_id in datasets_to_load[dataset_name]:
            global_data_kg, local_data_kg, hyperparameter_data_kg, rank_data_kg = load_simulation_dataset(
                dataset_name=dataset_name,
                run_id=run_id
            )
        global_data[dataset_name][run_id] = global_data_kg
        local_data[dataset_name][run_id] = local_data_kg
        rank_data[dataset_name][run_id] = rank_data_kg
        if not hyperparameter_data:
            hyperparameter_data = hyperparameter_data_kg
        else:
            assert hyperparameter_data.keys() == hyperparameter_data_kg.keys()
            for key in hyperparameter_data:
                assert hyperparameter_data[key] == hyperparameter_data_kg[key]
    return global_data, local_data, hyperparameter_data, rank_data

def split_by_hyperparameters(exp_ids, test_ratio, valid_ratio):
    test_num = int(test_ratio * len(exp_ids))
    valid_num = int(valid_ratio * len(exp_ids))

    test_ids = exp_ids[:test_num]
    valid_ids = exp_ids[test_num:valid_num]
    train_ids = exp_ids[(test_num+valid_num):]

    return train_ids, test_ids, valid_ids

def train_test_split(hyperparameter_data, rank_data, test_ratio, valid_ratio):
    train_ids, test_ids, valid_ids = split_by_hyperparameters(
        exp_ids=list(hyperparameter_data.keys()),
        test_ratio=test_ratio,
        valid_ratio=valid_ratio
    )

    # split rank data
    rank_split_data = {}
    for dataset_name in rank_data:
        rank_split_data[dataset_name] = {}
        for run_id in rank_data[dataset_name]:
            rank_split_data[dataset_name][run_id] = {
                'train': {},
                'test': {},
                'valid': {}
            }
            for train_id in train_ids:
                rank_split_data[dataset_name][run_id]['train'][train_id] = rank_data[dataset_name][run_id][train_id]
            for test_id in test_ids:
                rank_split_data[dataset_name][run_id]['test'][test_id] = rank_data[dataset_name][run_id][test_id]
            for valid_id in valid_ids:
                rank_split_data[dataset_name][run_id]['valid'][valid_id] = rank_data[dataset_name][run_id][valid_id]

    # split hyperparamter data
    hyp_split_data = {
        'train': {},
        'test': {},
        'valid': {}
    }
    for train_id in train_ids:
        hyp_split_data['train'][train_id] = hyperparameter_data[train_id]
    for test_id in test_ids:
        hyp_split_data['test'][test_id] = hyperparameter_data[test_id]
    for valid_id in valid_ids:
        hyp_split_data['valid'][valid_id] = hyperparameter_data[valid_id]


    return hyp_split_data, rank_split_data

def get_batch(dataset_name, run_id, exp_id, mode, global_data, local_data, hyp_split_data, rank_split_data):
    global_list = list(global_data[dataset_name][run_id].values())
    local_list = list(local_data[dataset_name][run_id].values())
    hp_list = list(hyp_split_data[mode][exp_id].values())
    input_ft_vec = global_list + local_list + hp_list

    X = []
    y = []
    triples_ids_to_ranks = rank_split_data[dataset_name][run_id][mode][exp_id]
    for triple_id in triples_ids_to_ranks:
        for side in triples_ids_to_ranks[triple_id]:
            X.append(input_ft_vec)
            y.append(triples_ids_to_ranks[triple_id][side])
    X = torch.tensor(X, device=device)
    y = torch.tensor(y, device=device)

    print(X.shape, y.shape)

    return X, y

def do_load(
    datasets_to_load,
    test_ratio,
    valid_ratio,
):
    '''
    do_load() loads all data.

    The arguments it accepts are:
        - datasets_to_load (dict of str -> list<str>): A dict that maps all dataset names to the run IDs of all KGE simulations done on those datasets
        - test_ratio (float): the proportion of hyperparameter combinations to hold out for the test set
        - valid_ratio (float): the proportion of hyperparameter combinations to hold out for the valid set

    The values it returns are:
        - TBD
    '''
    global_data, local_data, hyperparameter_data, rank_data = load_simulation_datasets(
        datasets_to_load=datasets_to_load
    )
    hyp_split_data, rank_split_data = train_test_split(
        hyperparameter_data=hyperparameter_data,
        rank_data=rank_data,
        test_ratio=test_ratio,
        valid_ratio=valid_ratio
    )
    get_batch('UMLS', '2.1', 1214, 'train', global_data, local_data, hyp_split_data, rank_split_data)
