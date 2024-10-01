# TWIG imports
from load_data import load_simulation_dataset, get_canonical_exp_dir
from utils import gather_data

# external imports
import sys
import numpy as np
from scipy import stats

def get_mrr(rank_data, hyp_id):
    ranks = []
    for triple_id in rank_data[hyp_id]:
        ranks.append(rank_data[hyp_id][triple_id]['head_rank'])
        ranks.append(rank_data[hyp_id][triple_id]['tail_rank'])
    ranks = np.array(ranks)
    mrr = np.mean(1 / ranks)
    return mrr

def get_best_by_mrr(rank_data):
    best_mrr = 0
    best_hyp_id = None
    for hyp_id in rank_data:
        mrr = get_mrr(rank_data, hyp_id)
        if mrr > best_mrr:
            best_mrr = mrr
            best_hyp_id = hyp_id
    return best_mrr, best_hyp_id

def is_same_hyps_except(hyps_1, hyps_2, hyp_names, hyp_ft):
    for hyp_name in hyp_names:
        if hyp_name != hyp_ft:
            if hyps_1[hyp_name] != hyps_2[hyp_name]:
                if hyp_ft == 'loss' and hyp_name == 'margin':
                    pass # allow margin, which is loss, specific, to vary if loss varies
                else:
                    return False, None # they differ in smth other than the target hyperparam
    return True, hyps_2[hyp_ft]

def get_mrrs_without(rank_data, ref_hyp_id, hyp_ft, grid):
    '''
    All hyps the same excet for the blacklisted one (hyp_ft)
    That one can be the same -- we get all 3 versions
    '''
    mrrs = {}
    hyp_names = list(grid[ref_hyp_id].keys()) # ['loss', 'neg_samp', 'lr', 'reg_coeff', 'npp', 'margin', 'dim']
    for hyp_id in rank_data:
        match, other_hyp_val = is_same_hyps_except(grid[ref_hyp_id], grid[hyp_id], hyp_names, hyp_ft)
        if match:
            mrrs[other_hyp_val] = get_mrr(rank_data, hyp_id)
    return mrrs

def get_struct_correls(local_data, rank_data, hyp_id):
    best_ranks = rank_data[hyp_id]
    correlators = {}
    for triple_id in local_data:
        head_rank = best_ranks[triple_id]['head_rank']
        tail_rank = best_ranks[triple_id]['tail_rank']
        struct = local_data[triple_id]
        for struct_ft in struct:
            if not struct_ft in correlators:
                correlators[struct_ft] = {}
            if not 'structs' in correlators[struct_ft]:
                correlators[struct_ft]['structs'] = []
            if not 'ranks' in correlators[struct_ft]:
                correlators[struct_ft]['ranks'] = []
            correlators[struct_ft]['structs'].append(struct[struct_ft])
            correlators[struct_ft]['structs'].append(struct[struct_ft])
            correlators[struct_ft]['ranks'].append(head_rank)
            correlators[struct_ft]['ranks'].append(tail_rank)
    
    correls = {}
    for struct_ft in correlators:
        r = stats.pearsonr(
            x=correlators[struct_ft]['structs'],
            y=correlators[struct_ft]['ranks']
        ).statistic
        r2 = r ** 2
        correls[struct_ft] = {
            'r': r,
            'r2': r2,
            'structs': correlators[struct_ft]['structs'],
            'ranks': correlators[struct_ft]['ranks'],
        }
    return correls

def get_mrrs_by_struct_cuttoff(rank_data, local_data, struct_ft):
    struct_ft_vals = []
    for triple_id in local_data:
        struct_ft_vals.append(local_data[triple_id][struct_ft])
    struct_ft_median = np.percentile(struct_ft_vals, 0.5)

    rank_data_below_median = {}
    rank_data_above_median = {}
    for hyp_id in rank_data:
        rank_data_below_median[hyp_id] = {}
        rank_data_above_median[hyp_id] = {}
        for triple_id in rank_data[hyp_id]:
            if local_data[triple_id][struct_ft] <= struct_ft_median:
                rank_data_below_median[hyp_id][triple_id] = {
                    'head_rank': rank_data[hyp_id][triple_id]["head_rank"],
                    'tail_rank': rank_data[hyp_id][triple_id]["tail_rank"]
                }
            if local_data[triple_id][struct_ft] > struct_ft_median:
                rank_data_above_median[hyp_id][triple_id] = {
                    'head_rank': rank_data[hyp_id][triple_id]["head_rank"],
                    'tail_rank': rank_data[hyp_id][triple_id]["tail_rank"]
                }
    
    best_mrr_below, best_hyp_id_below = get_best_by_mrr(rank_data_below_median)
    best_mrr_above, best_hyp_id_above = get_best_by_mrr(rank_data_above_median)
    if best_hyp_id_above == None:
        no_upper = True
    else:
        no_upper = False

    return struct_ft_vals, best_mrr_below, best_hyp_id_below, best_mrr_above, best_hyp_id_above, no_upper
    
def main(dataset, kgem, run_id):
    exp_dir = get_canonical_exp_dir(
        dataset_name=dataset,
        model_name=kgem,
        run_id=run_id
    )
    _, _, grid, _, _ = gather_data(dataset, exp_dir)
    _, local_data, _, rank_data = load_simulation_dataset(
        dataset_name=dataset,
        model_name=kgem,
        run_id=run_id
    )
    # print(local_data[1]['s_deg']) # triple ids -> struct ft names -> value
    # print(rank_data[1][1].keys()) # hyp ids -> triple ids -> head_rank, tail_rank
    best_mrr, best_hyp_id = get_best_by_mrr(rank_data)
    
    print('STRUCT CORRELATION ANAYSIS')
    correls = get_struct_correls(local_data, rank_data, best_hyp_id)
    for struct_ft in correls:
        print(struct_ft)
        print(f'\tr : {correls[struct_ft]["r"]}')
        print(f'\tr2 : {correls[struct_ft]["r2"]}')
        print()

    print()
    print('HYP EFFECT ANAYSIS')
    hyp_names = list(grid[1].keys())
    mrr_hyp_ablations = {}
    for hyp_name in hyp_names:
        mrr_hyp_ablations[hyp_name] = get_mrrs_without(
            rank_data,
            best_hyp_id,
            hyp_name,
            grid
        )
    for hyp_name in mrr_hyp_ablations:
        print(hyp_name, "single-element alternates -> mrr:")
        for alt in mrr_hyp_ablations[hyp_name]:
            print('\t', alt, '->', mrr_hyp_ablations[hyp_name][alt])
        print()
    
    print()
    print('STRUCTURAL CUTTOFF ANALYSIS')
    print('Overall')
    print('\tbest mrr:', best_mrr)
    print('\tbest hyp id:', best_hyp_id)
    print('\tbest hyperparameters:', grid[best_hyp_id])
    print()
    for struct_ft in correls:
            struct_ft_vals, best_mrr_below, best_hyp_id_below, best_mrr_above, best_hyp_id_above, no_upper = get_mrrs_by_struct_cuttoff(
                rank_data,
                local_data,
                struct_ft
            )
            struct_ft_min = np.min(struct_ft_vals)
            struct_ft_median = np.median(struct_ft_vals)
            struct_ft_max = np.max(struct_ft_vals)
            print(f'For {struct_ft} < median ({struct_ft_min, struct_ft_median, struct_ft_max})')
            print('\tbest mrr:', best_mrr_below)
            print('\tbest hyp id:', best_hyp_id_below)
            print('\tbest hyperparameters:', grid[best_hyp_id_below])
            print()
            if not no_upper:
                print(f'For {struct_ft} > median ({struct_ft_min, struct_ft_median, struct_ft_max})')
                print('\tbest mrr:', best_mrr_above)
                print('\tbest hyp id:', best_hyp_id_above)
                print('\tbest hyperparameters:', grid[best_hyp_id_above])
            else:
                print(f'For {struct_ft} > median ({struct_ft_min, struct_ft_median, struct_ft_max})')
                print('\tmedian = max, so no upper-set exists')
                print()
    
if __name__ == '__main__':
    dataset = sys.argv[1]
    kgem = sys.argv[2]
    run_id = sys.argv[3]
    main(dataset, kgem, run_id)
