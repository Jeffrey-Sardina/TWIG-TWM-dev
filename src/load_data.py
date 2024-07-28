# internal TWIG imports
from normaliser import Normaliser

# external imports
from utils import gather_data
import random
import os
import numpy as np
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class TWIG_Data:
    def __init__(
            self,
            structs,
            max_ranks,
            hyps,
            head_ranks,
            tail_ranks,
            train_ids,
            test_ids,
            valid_ids,
            normaliser,
            run_ids
        ):
        self.structs = structs
        self.max_ranks=max_ranks
        self.hyps = hyps
        self.head_ranks = head_ranks
        self.tail_ranks = tail_ranks
        self.train_ids = train_ids
        self.test_ids = test_ids
        self.valid_ids = valid_ids
        self.normaliser = normaliser
        self.head_flags = normaliser.head_flags
        self.tail_flags = normaliser.tail_flags
        self.run_ids = run_ids
        self.dataset_names = list(run_ids.keys())
        self.num_struct_fts = structs[self.dataset_names[0]].shape[1] + 1 # all tensors have the same shape. +1 since we add one later to tell which side this is
        self.num_hyp_fts = list(hyps['train'].values())[0].shape[0] # mode = train, exp_id = 0. all tensors have the same shape so which one we access does not matter

    def shuffled_train_iterators(self):
        # shuffle datasets
        dataset_names = [x for x in self.dataset_names]
        random.shuffle(dataset_names)

        # shuffle run ids
        run_ids = {}
        for dataset_name in dataset_names:
            dataeset_run_ids = [x for x in self.run_ids[dataset_name]]
            random.shuffle(dataeset_run_ids)
            run_ids[dataset_name] = dataeset_run_ids

        #shuffle train ids
        train_ids = [x for x in self.train_ids]
        random.shuffle(train_ids)

        # return shuffled values (TWIG_Data instance vars are not changed)
        return dataset_names, run_ids, train_ids

    def get_batch(self, dataset_name, run_id, exp_id, mode):
        struct_tensor = self.structs[dataset_name]
        struct_tensor_heads = torch.concat(
            [self.head_flags[dataset_name], struct_tensor],
            dim=1
        )
        struct_tensor_tails = torch.concat(
            [self.tail_flags[dataset_name], struct_tensor],
            dim=1
        )
        hyps_tensor = self.hyps[mode][exp_id]
        head_rank = self.head_ranks[dataset_name][run_id][mode][exp_id]
        tail_rank = self.tail_ranks[dataset_name][run_id][mode][exp_id]
        return struct_tensor_heads, struct_tensor_tails, hyps_tensor, head_rank, tail_rank

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
        local_data[triple_idx] = {}
        s, p, o = triples_map[triple_idx]

        local_data[triple_idx]['s_deg'] = graph_stats['train']['degrees'][s]
        local_data[triple_idx]['o_deg'] = graph_stats['train']['degrees'][o]
        local_data[triple_idx]['p_freq'] = graph_stats['train']['pred_freqs'][p]

        local_data[triple_idx]['s_p_cofreq'] = graph_stats['train']['subj_relationship_degrees'][(s,p)] \
            if (s,p) in graph_stats['train']['subj_relationship_degrees'] else 0
        local_data[triple_idx]['o_p_cofreq'] = graph_stats['train']['obj_relationship_degrees'][(o,p)] \
            if (o,p) in graph_stats['train']['obj_relationship_degrees'] else 0
        local_data[triple_idx]['s_o_cofreq'] = graph_stats['train']['subj_obj_cofreqs'][(s,o)] \
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

            local_data[triple_idx][f'{target_name} min deg neighbnour'] = np.min(list(neighbour_nodes.values()))
            local_data[triple_idx][f'{target_name} max deg neighbnour'] = np.max(list(neighbour_nodes.values()))
            local_data[triple_idx][f'{target_name} mean deg neighbnour'] = np.mean(list(neighbour_nodes.values()))
            local_data[triple_idx][f'{target_name} num neighbnours'] = len(neighbour_nodes)

            local_data[triple_idx][f'{target_name} min freq rel'] = np.min(list(neighbour_preds.values()))
            local_data[triple_idx][f'{target_name} max freq rel'] = np.max(list(neighbour_preds.values()))
            local_data[triple_idx][f'{target_name} mean freq rel'] = np.mean(list(neighbour_preds.values()))
            local_data[triple_idx][f'{target_name} num rels'] = len(neighbour_preds)
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
        rank_data[dataset_name] = {}
        for run_id in datasets_to_load[dataset_name]:
            global_data_kg, local_data_kg, hyperparameter_data_kg, rank_data_kg = load_simulation_dataset(
                dataset_name=dataset_name,
                run_id=run_id
            )
            rank_data[dataset_name][run_id] = rank_data_kg
        if not dataset_name in global_data:
            global_data[dataset_name] = global_data_kg
        if not dataset_name in local_data:
            local_data[dataset_name] = local_data_kg
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


    return hyp_split_data, rank_split_data, train_ids, test_ids, valid_ids

def to_tensors(global_data, local_data, hyp_split_data, rank_split_data):
    max_ranks = {}
    for dataset_name in global_data:
        global_vec = list(global_data[dataset_name].values())
        assert len(global_vec) == 1, "global data currently only supports max rank"
        max_ranks[dataset_name] = global_vec[0]

    structs = {}
    for dataset_name in local_data:
        structs[dataset_name] = {}
        triple_ids = list(local_data[dataset_name].keys())
        local_vecs = []
        for triple_id in triple_ids:
            local_vec = list(local_data[dataset_name][triple_id].values())
            local_vecs.append(local_vec)
        structs[dataset_name] = torch.tensor(
            local_vecs,
            dtype=torch.float32,
            device=device
        )

    hyps = {}
    for mode in hyp_split_data:
        hyps[mode] = {}
        for exp_id in hyp_split_data[mode]:
            hyps[mode][exp_id] = torch.tensor(
                list(hyp_split_data[mode][exp_id].values()),
                dtype=torch.float32,
                device=device
            )

    head_ranks = {}
    tail_ranks = {}
    for dataset_name in rank_split_data:
        head_ranks[dataset_name] = {}
        tail_ranks[dataset_name] = {}
        for run_id in rank_split_data[dataset_name]:
            head_ranks[dataset_name][run_id] = {}
            tail_ranks[dataset_name][run_id] = {}
            for mode in rank_split_data[dataset_name][run_id]:
                head_ranks[dataset_name][run_id][mode] = {}
                tail_ranks[dataset_name][run_id][mode] = {}
                for exp_id in rank_split_data[dataset_name][run_id][mode]:
                    head_ranks[dataset_name][run_id][mode][exp_id] = []
                    tail_ranks[dataset_name][run_id][mode][exp_id] = []
                    for triple_id in rank_split_data[dataset_name][run_id][mode][exp_id]:
                        head_ranks[dataset_name][run_id][mode][exp_id].append(
                            rank_split_data[dataset_name][run_id][mode][exp_id][triple_id]['head_rank']
                        )
                        tail_ranks[dataset_name][run_id][mode][exp_id].append(
                            rank_split_data[dataset_name][run_id][mode][exp_id][triple_id]['tail_rank']
                        )
                    head_ranks[dataset_name][run_id][mode][exp_id] = torch.tensor(
                        head_ranks[dataset_name][run_id][mode][exp_id],
                        dtype=torch.float32,
                        device=device
                    )
                    tail_ranks[dataset_name][run_id][mode][exp_id] = torch.tensor(
                        tail_ranks[dataset_name][run_id][mode][exp_id],
                        dtype=torch.float32,
                        device=device
                    )

    return structs, max_ranks, hyps, head_ranks, tail_ranks

def _do_load(
    datasets_to_load,
    test_ratio,
    valid_ratio,
    normalisation,
    rescale_ranks
):
    '''
    do_load() loads all data.

    The arguments it accepts are:
        - datasets_to_load (dict of str -> list<str>): A dict that maps all dataset names to the run IDs of all KGE simulations done on those datasets
        - test_ratio (float): the proportion of hyperparameter combinations to hold out for the test set
        - valid_ratio (float): the proportion of hyperparameter combinations to hold out for the valid set
        - normalisation (str): the normalisation to use for input data (not ranks). Options are "zscore", "minmax", and "none"
        - rescale_ranks (bool): whether ranks should re rescaled to have a maximum value of 1.

    The values it returns are:
        - TBD
    '''
    global_data, local_data, hyperparameter_data, rank_data = load_simulation_datasets(
        datasets_to_load=datasets_to_load
    )
    hyp_split_data, rank_split_data, train_ids, test_ids, valid_ids = train_test_split(
        hyperparameter_data=hyperparameter_data,
        rank_data=rank_data,
        test_ratio=test_ratio,
        valid_ratio=valid_ratio
    )
    structs, max_ranks, hyps, head_ranks, tail_ranks = to_tensors(
        global_data=global_data,
        local_data=local_data,
        hyp_split_data=hyp_split_data,
        rank_split_data=rank_split_data
    )
    normaliser = Normaliser(
        method=normalisation,
        rescale_ranks=rescale_ranks,
        structs=structs,
        hyps=hyps,
        max_ranks=max_ranks
    )
    structs, hyps, head_ranks, tail_ranks = normaliser.normalise(
        structs=structs,
        hyps=hyps,
        head_ranks=head_ranks,
        tail_ranks=tail_ranks
    )
    twig_data = TWIG_Data(
        structs=structs,
        max_ranks=max_ranks,
        hyps=hyps,
        head_ranks=head_ranks,
        tail_ranks=tail_ranks,
        train_ids=train_ids,
        test_ids=test_ids,
        valid_ids=valid_ids,
        normaliser=normaliser,
        run_ids=datasets_to_load
    )
    return twig_data
