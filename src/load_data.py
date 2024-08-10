# internal TWIG imports
from normaliser import Normaliser
from trainer import _d_hist

# external imports
from utils import gather_data
import random
import os
import numpy as np
import torch

# Reproducibility
torch.manual_seed(17)
random.seed(17)


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
            run_ids,
            n_bins
        ):
        # constants
        self.HIST_MIN = 0
        self.HIST_MAX = 1
        self.N_BINS=n_bins

        # data vars
        self.max_ranks=max_ranks
        self.hyps = hyps
        self.train_ids = train_ids
        self.test_ids = test_ids
        self.valid_ids = valid_ids
        self.normaliser = normaliser
        self.run_ids = run_ids

        # deleted later
        self.structs = structs
        self.head_ranks = head_ranks
        self.tail_ranks = tail_ranks

        # calculated vars
        self.dataset_names = list(run_ids.keys())
        self.num_struct_fts = structs[self.dataset_names[0]].shape[1] + 1 # all tensors have the same shape. +1 since we add one later to tell which side this is
        self.num_hyp_fts = list(hyps['train'].values())[0].shape[0] # mode = train, exp_id = 0. all tensors have the same shape so which one we access does not matter
        self.struct_data, self.rank_lists, self.mrrs, self.rank_dists = self.precalc_train_data()

    def precalc_train_data(self):
        # precalc struct data
        struct_data = {}
        for dataset_name in self.dataset_names:
            struct_tensor = self.structs[dataset_name]
            struct_tensor_heads = torch.concat(
                [self.normaliser.head_flags[dataset_name], struct_tensor],
                dim=1
            )
            struct_tensor_tails = torch.concat(
                [self.normaliser.tail_flags[dataset_name], struct_tensor],
                dim=1
            )
            struct_tensor = torch.concat(
                [struct_tensor_heads, struct_tensor_tails],
                dim=0
            )
            struct_data[dataset_name] = struct_tensor

        # precalc rank data
        rank_lists = {}
        mrrs = {}
        rank_dists = {}
        for dataset_name in self.dataset_names:
            rank_lists[dataset_name] = {}
            mrrs[dataset_name] = {}
            rank_dists[dataset_name] = {}
            for run_id in self.head_ranks[dataset_name]:
                rank_lists[dataset_name][run_id] = {}
                mrrs[dataset_name][run_id] = {}
                rank_dists[dataset_name][run_id] = {}
                for mode in self.head_ranks[dataset_name][run_id]:
                    rank_lists[dataset_name][run_id][mode] = {}
                    mrrs[dataset_name][run_id][mode] = {}
                    rank_dists[dataset_name][run_id][mode] = {}
                    for exp_id in self.head_ranks[dataset_name][run_id][mode]:
                        head_rank = self.head_ranks[dataset_name][run_id][mode][exp_id]
                        tail_rank = self.tail_ranks[dataset_name][run_id][mode][exp_id]
                        rank_list = torch.concat(
                            [head_rank, tail_rank],
                            dim=0
                        )
                        rank_lists[dataset_name][run_id][mode][exp_id] = rank_list
                        mrrs[dataset_name][run_id][mode][exp_id] = torch.mean(1 / (rank_list * self.max_ranks[dataset_name]))
                        rank_dists[dataset_name][run_id][mode][exp_id] = _d_hist(
                            X=rank_list,
                            n_bins=self.N_BINS,
                            min_val=self.HIST_MIN,
                            max_val=self.HIST_MAX
                        )

        # delete vars we no longer need
        del self.structs
        del self.head_ranks
        del self.tail_ranks

        return struct_data, rank_lists, mrrs, rank_dists

    def get_train_epoch(self, shuffle):
        epoch_data = []
        for dataset_name in self.dataset_names:
            for run_id in self.run_ids[dataset_name]:
                for exp_id in self.train_ids:
                    epoch_data.append((dataset_name, run_id, exp_id))
        if shuffle:
            random.shuffle(epoch_data)
        return epoch_data
    
    def get_eval_epoch(self, mode, dataset_name):
        if mode == 'train':
            exp_ids = self.train_ids
            print('WARNING: Eval running on the TRAIN set. This WILL have overfitting and SHOULD NOT be cited!')
        elif mode == 'test':
            exp_ids = self.test_ids
        elif mode == 'valid':
            exp_ids = self.valid_ids
        else:
            assert False, f"Invalid mode for evaluation: {mode}"

        epoch_data = []
        for run_id in self.run_ids[dataset_name]:
            for exp_id in exp_ids:
                epoch_data.append((dataset_name, run_id, exp_id))

        return epoch_data
    
    def get_batch(self, dataset_name, run_id, exp_id, mode):
        # get struct data
        struct_tensor = self.struct_data[dataset_name]

        # get hyps data
        hyps_tensor = self.hyps[mode][exp_id]
        hyps_tensor = hyps_tensor.repeat(struct_tensor.shape[0], 1)
        
        # get precalc'd rank data
        mrr_true = self.mrrs[dataset_name][run_id][mode][exp_id]
        rank_dists_true = self.rank_dists[dataset_name][run_id][mode][exp_id]
        
        return struct_tensor, hyps_tensor, mrr_true, rank_dists_true

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
                # # if we cannot, we need to encode as a number
                # hp_val_id = hp_names_to_vals_sorted[hp_name].index(hp_val)
                # hyperparameter_data[exp_id][hp_name] = hp_val_id
                
                # if we cannot, one-hot code as a categorical variable
                onehot_len = len(hp_names_to_vals_sorted[hp_name]) - 1
                hp_val_id = hp_names_to_vals_sorted[hp_name].index(hp_val)
                bin_str = np.binary_repr(hp_val_id) #get binary val (one-hot proto-vector)
                bin_str = bin_str.rjust(onehot_len, '0') # pad with 0s to constant length (full one-hot vector)
                for oneshot_idx in range(onehot_len):
                    oneshot_name = hp_name + str(oneshot_idx)
                    hyperparameter_data[exp_id][oneshot_name] = int(bin_str[oneshot_idx])
    
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

def load_simulation_datasets(datasets_to_load, do_print):
    global_data = {}
    local_data = {}
    rank_data = {}
    hyperparameter_data = None
    for dataset_name in datasets_to_load:
        if do_print:
            print(f'Loading {dataset_name}...')
        rank_data[dataset_name] = {}
        for run_id in datasets_to_load[dataset_name]:
            if do_print:
                print(f'- loading run {run_id}...')
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
    random_ids = [x for x in exp_ids]
    random.shuffle(random_ids)
    test_num = int(test_ratio * len(random_ids))
    valid_num = int(valid_ratio * len(random_ids))

    test_ids = random_ids[:test_num]
    valid_ids = random_ids[test_num:valid_num]
    train_ids = random_ids[(test_num+valid_num):]

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
    # make sure we iterate over everything in the same order, even if dicts have different orders
    dataset_names_canonical = sorted(list(global_data.keys()))
    triple_ids_canonical = {}
    for dataset_name in dataset_names_canonical:
        triple_ids_canonical[dataset_name] = sorted(list(local_data[dataset_name].keys()))
    modes_canonical = ['test', 'train', 'valid']
    exp_ids_canonical = {}
    for mode in modes_canonical:
        exp_ids_canonical[mode] = sorted(list(hyp_split_data[mode]))
    run_ids_canonical = {}
    for dataset_name in dataset_names_canonical:
        run_ids_canonical[dataset_name] = sorted(list(rank_split_data[dataset_name]))

    # get global max trank data for each dataset
    max_ranks = {}
    for dataset_name in dataset_names_canonical:
        global_vec = list(global_data[dataset_name].values())
        assert len(global_vec) == 1, "global data currently only supports max rank"
        max_ranks[dataset_name] = global_vec[0]

    # get localised struct data for each dataset
    structs = {}
    for dataset_name in dataset_names_canonical:
        structs[dataset_name] = {}
        local_vecs = []
        for triple_id in triple_ids_canonical[dataset_name]:
            local_vec = list(local_data[dataset_name][triple_id].values())
            local_vecs.append(local_vec)
        structs[dataset_name] = torch.tensor(
            local_vecs,
            dtype=torch.float32,
            device=device
        )

    # get hyperparameter datat for each experiment (exp_id)
    hyps = {}
    for mode in modes_canonical:
        hyps[mode] = {}
        for exp_id in exp_ids_canonical[mode]:
            hyps[mode][exp_id] = torch.tensor(
                list(hyp_split_data[mode][exp_id].values()),
                dtype=torch.float32,
                device=device
            )

    # get rank data for all dataset, run_id, exp_id combinations in all training modes
    head_ranks = {}
    tail_ranks = {}
    for dataset_name in dataset_names_canonical:
        head_ranks[dataset_name] = {}
        tail_ranks[dataset_name] = {}
        for run_id in run_ids_canonical[dataset_name]:
            head_ranks[dataset_name][run_id] = {}
            tail_ranks[dataset_name][run_id] = {}
            for mode in modes_canonical:
                head_ranks[dataset_name][run_id][mode] = {}
                tail_ranks[dataset_name][run_id][mode] = {}
                for exp_id in exp_ids_canonical[mode]:
                    head_ranks[dataset_name][run_id][mode][exp_id] = []
                    tail_ranks[dataset_name][run_id][mode][exp_id] = []
                    for triple_id in triple_ids_canonical[dataset_name]:
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
    n_bins,
    do_print
):
    '''
    do_load() loads all data.

    The arguments it accepts are:
        - datasets_to_load (dict of str -> list<str>): A dict that maps all dataset names to the run IDs of all KGE simulations done on those datasets
        - test_ratio (float): the proportion of hyperparameter combinations to hold out for the test set
        - valid_ratio (float): the proportion of hyperparameter combinations to hold out for the valid set
        - normalisation (str): the normalisation to use for input data (not ranks). Options are "zscore", "minmax", and "none"

    The values it returns are:
        - twig_data (TWIG_Data): a TWIG_Data object containing all data needed to load and run batches for TWIG.
    '''
    if do_print:
        print('Loading datasets')
    global_data, local_data, hyperparameter_data, rank_data = load_simulation_datasets(
        datasets_to_load=datasets_to_load,
        do_print=do_print
    )
    
    if do_print:
        print('Creating the train-test split')
    hyp_split_data, rank_split_data, train_ids, test_ids, valid_ids = train_test_split(
        hyperparameter_data=hyperparameter_data,
        rank_data=rank_data,
        test_ratio=test_ratio,
        valid_ratio=valid_ratio
    )
    if do_print:
        print('using splits:')
        print(f'test_ids ({len(test_ids)}): {test_ids}')
        print(f'valid_ids ({len(valid_ids)}): {valid_ids}')
        print(f'train_ids ({len(train_ids)}): {train_ids}')

    if do_print:
        print('Converting data to tensors')
    structs, max_ranks, hyps, head_ranks, tail_ranks = to_tensors(
        global_data=global_data,
        local_data=local_data,
        hyp_split_data=hyp_split_data,
        rank_split_data=rank_split_data
    )

    if do_print:
        print('Normalising data')
    normaliser = Normaliser(
        method=normalisation,
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

    if do_print:
        print('Finalising data preprocessing')
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
        run_ids=datasets_to_load,
        n_bins=n_bins
    )
    return twig_data
