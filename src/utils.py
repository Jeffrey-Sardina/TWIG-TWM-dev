import glob
import os
import ast
from pykeen import datasets
import numpy as np
from pykeen.triples import TriplesFactory

class Custom_Dataset():
    '''
    For loading a custom dataset with the same API as a PyKEEN dataset
    '''
    def __init__(self, factory_dict):
        self.factory_dict = factory_dict

'''
================
Helper Functions
================
'''
def get_triples(dataset):
    train_triples = []
    test_triples = []
    valid_triples = []

    for s, p, o in dataset.factory_dict['training'].mapped_triples:
        s, p, o = int(s), int(p), int(o)
        train_triples.append((s, p, o))
    for s, p, o in dataset.factory_dict['testing'].mapped_triples:
        s, p, o = int(s), int(p), int(o)
        test_triples.append((s, p, o))
    for s, p, o in dataset.factory_dict['validation'].mapped_triples:
        s, p, o = int(s), int(p), int(o)
        valid_triples.append((s, p, o))
    all_triples = train_triples + test_triples + valid_triples

    return {
        'all': all_triples,
        'train': train_triples,
        'test': test_triples,
        'valid': valid_triples
    }

def get_count_data(triples):
    degrees = {}
    pred_freqs = {}
    for s, p, o in triples:
        if not s in degrees:
            degrees[s] = 0
        if not o in degrees:
            degrees[o] = 0
        if not p in pred_freqs:
            pred_freqs[p] = 0
        degrees[s] += 1
        degrees[o] += 1
        pred_freqs[p] += 1
    return dict(sorted(degrees.items())), dict(sorted(pred_freqs.items()))

def get_relationship_degrees(triples):
    subj_relationship_degrees = {}
    obj_relationship_degrees = {}
    total_relationship_degrees = {}
    for s, p, o in triples:
        if not (s, p) in subj_relationship_degrees:
            subj_relationship_degrees[(s, p)] = 0
        if not (o, p) in obj_relationship_degrees:
            obj_relationship_degrees[(o, p)] = 0
        if not (s, p) in total_relationship_degrees:
            total_relationship_degrees[(s, p)] = 0
        if not (o, p) in total_relationship_degrees:
            total_relationship_degrees[(o, p)] = 0

        subj_relationship_degrees[(s, p)] += 1
        obj_relationship_degrees[(o, p)] += 1
        total_relationship_degrees[(s, p)] += 1
        total_relationship_degrees[(o, p)] += 1

    return dict(sorted(subj_relationship_degrees.items())), \
        dict(sorted(obj_relationship_degrees.items())), \
        dict(sorted(total_relationship_degrees.items()))

def get_subj_obj_cofreqs(triples):
    subj_obj_cofreqs = {}
    for s, p, o in triples:
        if not (s, o) in subj_obj_cofreqs:
            subj_obj_cofreqs[(s, o)] = 0
        subj_obj_cofreqs[(s, o)] += 1
    return dict(sorted(subj_obj_cofreqs.items()))

def get_percentiles(degrees):
    percentiles = {}
    degrees_list = [degrees[key] for key in degrees]
    percentiles_wanted = [0, 1, 5, 10, 20, 25, 30, 33, 40, 50, 60, 67, 76, 75, 80, 90, 95, 99, 100]
    for percentile in percentiles_wanted:
        percentiles[percentile] = np.percentile(degrees_list, percentile)
    return percentiles

def calc_triples_stats(triples):
    degrees, pred_freqs = get_count_data(triples)
    subj_relationship_degrees, \
        obj_relationship_degrees, \
        total_relationship_degrees = get_relationship_degrees(triples)
    subj_obj_cofreqs = get_subj_obj_cofreqs(triples)
    percentiles = get_percentiles(degrees)
    pred_percentiles = get_percentiles(pred_freqs)
    subj_rel_degree_percentiles = get_percentiles(subj_relationship_degrees)
    obj_rel_degree_percentiles = get_percentiles(obj_relationship_degrees)
    total_rel_degree_percentiles = get_percentiles(total_relationship_degrees)
    subj_obj_cofreqs_percentiles = get_percentiles(subj_obj_cofreqs)
    return degrees, \
                pred_freqs, \
                subj_relationship_degrees, \
                obj_relationship_degrees, \
                total_relationship_degrees, \
                subj_obj_cofreqs, \
                percentiles, \
                pred_percentiles, \
                subj_rel_degree_percentiles, \
                obj_rel_degree_percentiles, \
                total_rel_degree_percentiles, \
                subj_obj_cofreqs_percentiles

def write_stats_data(triples_set_name, triples_data_dict):
    degrees = triples_data_dict['degrees']
    pred_freqs = triples_data_dict['pred_freqs']
    subj_relationship_degrees = triples_data_dict['subj_relationship_degrees']
    obj_relationship_degrees = triples_data_dict['obj_relationship_degrees']
    total_relationship_degrees = triples_data_dict['total_relationship_degrees']
    subj_obj_cofreqs = triples_data_dict['subj_obj_cofreqs']
    percentiles = triples_data_dict['percentiles']
    pred_percentiles = triples_data_dict['pred_percentiles']
    subj_rel_degree_percentiles = triples_data_dict['subj_rel_degree_percentiles']
    obj_rel_degree_percentiles = triples_data_dict['obj_rel_degree_percentiles']
    total_rel_degree_percentiles = triples_data_dict['total_rel_degree_percentiles']
    subj_obj_cofreqs_percentiles = triples_data_dict['subj_obj_cofreqs_percentiles']

    print(f'{"="*10} Stats for the {triples_set_name} triples set {"="*10}')
    print(f'{"="*5} Degree percentiles {"="*5}')
    for percentile in percentiles:
        print(f'{percentile}%:\t{percentiles[percentile]}')
    print()

    print(f'{"="*5} Predicate freq percentiles {"="*5}')
    for percentile in pred_percentiles:
        print(f'{percentile}%:\t{pred_percentiles[percentile]}')
    print()

    print(f'{"="*5} (Subj,rel) degree percentiles {"="*5}')
    for percentile in subj_rel_degree_percentiles:
        print(f'{percentile}%:\t{subj_rel_degree_percentiles[percentile]}')
    print()

    print(f'{"="*5} (Obj,rel) degree percentiles {"="*5}')
    for percentile in obj_rel_degree_percentiles:
        print(f'{percentile}%:\t{obj_rel_degree_percentiles[percentile]}')
    print()

    print(f'{"="*5} (Node,rel) degree percentiles {"="*5}')
    for percentile in total_rel_degree_percentiles:
        print(f'{percentile}%:\t{total_rel_degree_percentiles[percentile]}')
    print()

    print(f'{"="*5} (Subj,obj) co-frequency percentiles {"="*5}')
    for percentile in subj_obj_cofreqs_percentiles:
        print(f'{percentile}%:\t{subj_obj_cofreqs_percentiles[percentile]}')
    print()

    print(f'{"="*5} Node : degree {"="*5}')
    for node in degrees:
        print(f'{node}:\t{degrees[node]}')
    print()
    
    print(f'{"="*5} Pred : freq {"="*5}')
    for pred in pred_freqs:
        print(f'{pred}:\t{pred_freqs[pred]}')
    print()

    print(f'{"="*5} (Subj,rel) degrees : freq {"="*5}')
    for (node, rel) in subj_relationship_degrees:
        print(f'{(node, rel)}:\t{subj_relationship_degrees[(node, rel)]}')
    print()

    print(f'{"="*5} (Obj,rel) degrees : freq {"="*5}')
    for (node, rel) in obj_relationship_degrees:
        print(f'{(node, rel)}:\t{obj_relationship_degrees[(node, rel)]}')
    print()

    print(f'{"="*5} (Node,rel) degrees : freq {"="*5}')
    for (node, rel) in total_relationship_degrees:
        print(f'{(node, rel)}:\t{total_relationship_degrees[(node, rel)]}')
    print()

    print(f'{"="*5} (Subj,obj) degrees : freq {"="*5}')
    for (subj, obj) in subj_obj_cofreqs:
        print(f'{(subj, obj)}:\t{subj_obj_cofreqs[(subj, obj)]}')
    print()
    print()

def calc_graph_stats(triples_dicts, do_print=True):
    all_triples_struct_data = {}
    for name in triples_dicts:
        triples = triples_dicts[name]
        degrees, \
            pred_freqs, \
            subj_relationship_degrees, \
            obj_relationship_degrees, \
            total_relationship_degrees, \
            subj_obj_cofreqs, \
            percentiles, \
            pred_percentiles, \
            subj_rel_degree_percentiles, \
            obj_rel_degree_percentiles, \
            total_rel_degree_percentiles, \
            subj_obj_cofreqs_percentiles = calc_triples_stats(triples)
        
        all_triples_struct_data[name] = {
            'degrees': degrees,
            'pred_freqs': pred_freqs,
            'subj_relationship_degrees': subj_relationship_degrees,
            'obj_relationship_degrees': obj_relationship_degrees,
            'total_relationship_degrees': total_relationship_degrees,
            'subj_obj_cofreqs': subj_obj_cofreqs,
            'percentiles': percentiles,
            'pred_percentiles': pred_percentiles,
            'subj_rel_degree_percentiles': subj_rel_degree_percentiles,
            'obj_rel_degree_percentiles': obj_rel_degree_percentiles,
            'total_rel_degree_percentiles': total_rel_degree_percentiles,
            'subj_obj_cofreqs_percentiles': subj_obj_cofreqs_percentiles,
        }
        if do_print:
            write_stats_data(name, all_triples_struct_data[name])
    return all_triples_struct_data

def get_results_dicts(exp_dir):
    results_file = glob.glob(os.path.join(exp_dir, '*.res'))
    assert len(results_file) == 1, f"more than 1 results file found in {exp_dir}, unsure which to use"
    results_file = results_file[0]

    with open(results_file, 'r') as res:
        curr_exp = 0
        curr_section = None
        overall_results = {}
        triples_results = {}
        for line in res:
            if 'End of exp ' in line:
                curr_exp = line.strip().replace('End of exp ', '')
                overall_results[curr_exp] = {}
                triples_results[curr_exp] = {}
            elif 'MR = ' in line:
                mr = float(line.strip().replace('MR = ', ''))
                overall_results[curr_exp]['mr'] = mr
            elif 'MRR = ' in line:
                mrr = float(line.strip().replace('MRR = ', ''))
                overall_results[curr_exp]['mrr'] = mrr
            elif 'Hits@(1,3,5,10) = ' in line:
                data = line.strip().replace('Hits@(1,3,5,10) = ', '')
                data = data.replace('(', '').replace(')', '')
                h1, h3, h5, h10 = data.split(',')
                overall_results[curr_exp]['h1'] = float(h1)
                overall_results[curr_exp]['h3'] = float(h3)
                overall_results[curr_exp]['h5'] = float(h5)
                overall_results[curr_exp]['h10'] = float(h10)
            elif 'Head ranks: (idx, rank)' in line:
                curr_section = 'HEAD'
            elif 'Tail ranks: (idx, rank)' in line:
                curr_section = 'TAIL'
            elif line.strip() == '':
                curr_section = None

            if ' --> ' in line:
                idx, rank = line.strip().split(' --> ')
                idx = int(idx)
                rank = float(rank)
                if not idx in triples_results[curr_exp]:
                    triples_results[curr_exp][idx] = {}
                if curr_section == 'HEAD':
                    triples_results[curr_exp][idx]['head_rank'] = rank
                elif curr_section == 'TAIL':
                    triples_results[curr_exp][idx]['tail_rank'] = rank
                else:
                    assert False, 'This should be impossible'

    triples_results = dict(sorted(triples_results.items())) # NEW -- for use with TWM. Should not affect anything else
    return overall_results, triples_results

def get_grid_dict(exp_dir):
    grid_file = glob.glob(os.path.join(exp_dir, '*.grid'))[0]
    grid = {}
    with open(grid_file, 'r') as inp:
        for line in inp:
            curr_exp, hpo_dict_str = line.strip().split(' --> ')
            hpo_dict = ast.literal_eval(hpo_dict_str)
            grid[curr_exp] = hpo_dict
    return grid

def get_triples_by_idx(triples_dicts, triples_set):
    idx_to_triples = {}
    for idx, triple in enumerate(triples_dicts[triples_set]):
        idx_to_triples[idx] = triple
    return idx_to_triples

def load_custom_dataset(dataset_name, verbose=True):
    '''
    For loading a dataset from disk.
    Outputs data in the same format as pykeen.datasets.get_dataset()
   
    Data must be in 3-col TSV format (tab-separated values)
    https://pykeen.readthedocs.io/en/stable/byo/data.html
    '''
    train_path = f'custom_datasets/{dataset_name}.train'
    test_path = f'custom_datasets/{dataset_name}.test'
    valid_path = f'custom_datasets/{dataset_name}.valid'
    factory_dict = {
        'training': TriplesFactory.from_path(train_path),
        'testing': TriplesFactory.from_path(test_path),
        'validation': TriplesFactory.from_path(valid_path)
    }
    dataset = Custom_Dataset(factory_dict)
    return dataset

def gather_data(dataset_name, exp_dir):
    '''
    overall_results is
        exp_id : {
            'mr': MR,
            'mrr': MRR,
            'hk': Hits@k (for k=1,3,5,10)
        }

    grid is
        exp id: hyperparameter setting dict

    valid_triples_map is
        triples_idx : (s, p, o)

    triples_results is
        triples_idx : {
            'head': <head rank>,
            'tail': <tail rank> 
        }

    graph_stats is
        all / train / test / valid : {
            'degrees': dict(degrees),
            'pred_freqs': dict(pred_freqs),
            'subj / obj / total _relationship_degrees': dict(relationship_degrees),
            'percentiles': dict(percentiles),
            'subj / obj / total _rel_degree_percentiles': dict(rel_degree_percentiles)
        }

        each dict maps the node / pred ID to its degree / freq. In the case of
        relationship degrees, this maps (s, p) --> freq or (o, p) --> freq
        percentile dicts contain percentiles for each; percentiles used are:
        {0, 1, 5, 10, 20, 25, 30, 33, 40, 50, 60, 67, 76, 75, 80, 90, 95, 99, 100}
    '''
    assert dataset_name in exp_dir, 'if this is False, you either have violated naming conventions, or tried to load incompatible data'
    try:
        dataset = datasets.get_dataset(dataset=dataset_name)
    except:
        dataset = load_custom_dataset(dataset_name)
    triples_dicts = get_triples(dataset)

    overall_results, triples_results = get_results_dicts(exp_dir)
    grid = get_grid_dict(exp_dir)
    valid_triples_map = get_triples_by_idx(triples_dicts, 'valid') #we did eval on valid, not test
    graph_stats = calc_graph_stats(triples_dicts, do_print=False)
    return overall_results, \
        triples_results, \
        grid, \
        valid_triples_map, \
        graph_stats

def print_graph_stats(dataset_name, write_obj=None):
    try:
        dataset = datasets.get_dataset(dataset=dataset_name)
    except:
        dataset = load_custom_dataset(dataset_name)

    triples_dicts = get_triples(dataset)
    if write_obj:
        return calc_graph_stats(triples_dicts, do_print=True, write_obj=write_obj)
    else:
        return calc_graph_stats(triples_dicts, do_print=True)
