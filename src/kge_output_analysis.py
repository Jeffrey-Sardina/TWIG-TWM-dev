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
                    pass # allow margin, which is loss-specific, to vary if loss varies
                else:
                    return False # they differ in smth other than the target hyperparam
        else:
            if hyps_1[hyp_name] == hyps_2[hyp_name]:
                return False # they are the same in teh val we want to be different
    return True

def get_mrrs_without(rank_data, ref_hyp_id, hyp_ft, grid):
    '''
    All hyps the same excet for the blacklisted one (hyp_ft)
    That one can be the same -- we get all 3 versions
    '''
    hyp_names = list(grid[ref_hyp_id].keys()) # ['loss', 'neg_samp', 'lr', 'reg_coeff', 'npp', 'margin', 'dim']
    for hyp_id in rank_data:
        match = is_same_hyps_except(grid[ref_hyp_id], grid[hyp_id], hyp_names, hyp_ft)
        if match:
            return hyp_id, get_mrr(rank_data, hyp_id)
    assert False

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
    struct_ft_median = np.median(struct_ft_vals)

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

def get_canonical_str(inp_str):
    inp_str = inp_str.replace('MarginRankingLoss', "MRL")
    inp_str = inp_str.replace('BCEWithLogitsLoss', "BCE")
    inp_str = inp_str.replace('CrossEntropyLoss', "CE")
    inp_str = inp_str.replace('BernoulliNegativeSampler', "Bernoulli")
    inp_str = inp_str.replace('BasicNegativeSampler', "Basic")
    inp_str = inp_str.replace('PseudoTypedNegativeSampler', "Pseudo")
    inp_str = inp_str.replace('loss', "Loss")
    inp_str = inp_str.replace('neg_samp', "N. Samp")
    inp_str = inp_str.replace('lr', "LR")
    inp_str = inp_str.replace('npp', "NPP")
    inp_str = inp_str.replace('margin', "Margin")
    inp_str = inp_str.replace('dim', "Dim")
    inp_str = inp_str.replace('reg_coeff', "Reg")
    inp_str = inp_str.replace('s_deg', "s deg")
    inp_str = inp_str.replace('o_deg', "o deg")
    inp_str = inp_str.replace('p_freq', "p freq")
    inp_str = inp_str.replace('s_p_cofreq', "s p cofreq")
    inp_str = inp_str.replace('o_p_cofreq', "o p cofreq")
    inp_str = inp_str.replace('s_o_cofreq', "s o cofreq")
    inp_str = inp_str.replace('neighbnours', "s o cofreq")
    inp_str = inp_str.replace('neighbnour', "nbr")
    inp_str = inp_str.replace('neighbnours', "nbrs")
    inp_str = inp_str.replace('<=', "$\\leq$")
    inp_str = inp_str.replace('>', "$>$")
    return inp_str
    
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
    print("\\textbf{Struct Ft}\t\\textbf{Correlation to Rank(r)}")
    for struct_ft in correls:
        r = round(correls[struct_ft]['r'], 2)
        r_color_indicator = int(abs(r * 100))
        print(f"{get_canonical_str(struct_ft)}\t\\cellcolor{{blue!{r_color_indicator}}}{r}")

    print()
    print('HYP EFFECT ANAYSIS')
    hyp_names = list(grid[1].keys())
    print(get_canonical_str('\t'.join(f"\\textbf{{{x}}}" for x in hyp_names)),"MRR",sep='\t')
    hyps_best = grid[best_hyp_id]
    print(get_canonical_str('\t'.join(f"{x}" for x in hyps_best.values())), f"{round(best_mrr, 2)}", sep='\t')
    for hyp_name in hyp_names:
        if hyps_best['margin'] == None and hyp_name == 'margin':
            continue # we can't find an alt to margin if margin is None, since not all losses use margin
        hyp_id, mrr = get_mrrs_without(
            rank_data,
            best_hyp_id,
            hyp_name,
            grid
        )
        local_hyps = grid[hyp_id]
        output_strs = []
        for hyp_name_inner in local_hyps:
            if hyp_name_inner == hyp_name:
                output_strs.append(f"\\textbf{{{local_hyps[hyp_name_inner]}}}")
            else:
                output_strs.append(f"{local_hyps[hyp_name_inner]}")
        print(get_canonical_str('\t'.join(output_strs)), round(mrr, 2), sep='\t')
    
    print()
    print('STRUCTURAL CUTTOFF ANALYSIS')
    print("\\textbf{Mode}",get_canonical_str('\t'.join(f"\\textbf{{{x}}}" for x in hyp_names)),"MRR",sep='\t')
    print('Overall', get_canonical_str('\t'.join(str(x) for x in hyps_best.values())), round(best_mrr, 2), sep='\t')
    for i, struct_ft in enumerate(correls):
        color_row = i % 2 == 0
        struct_ft_vals, best_mrr_below, best_hyp_id_below, best_mrr_above, best_hyp_id_above, no_upper = get_mrrs_by_struct_cuttoff(
            rank_data,
            local_data,
            struct_ft
        )
        struct_ft_median = round(np.median(struct_ft_vals), 1)
        if not no_upper:
            output_strs = []
            for hyp_name_inner in grid[best_hyp_id_below]:
                hyp_val_inner = grid[best_hyp_id_below][hyp_name_inner]
                if hyp_val_inner != hyps_best[hyp_name_inner]:
                    output_strs.append(f"\\textbf{{{hyp_val_inner}}}")
                else:
                    output_strs.append(f"{hyp_val_inner}")
            if color_row:
                print("\\rowcolor{lightgray} " + get_canonical_str(f"{struct_ft} <= {struct_ft_median}"), get_canonical_str('\t'.join(output_strs)), round(best_mrr_below, 2), sep='\t')
            else:
                print(get_canonical_str(f"{struct_ft} <= {struct_ft_median}"), get_canonical_str('\t'.join(output_strs)), round(best_mrr_below, 2), sep='\t')
            output_strs = []
            for hyp_name_inner in grid[best_hyp_id_above]:
                hyp_val_inner = grid[best_hyp_id_above][hyp_name_inner]
                if hyp_val_inner != hyps_best[hyp_name_inner]:
                    output_strs.append(f"\\textbf{{{hyp_val_inner}}}")
                else:
                    output_strs.append(f"{hyp_val_inner}")
            if color_row:
                print("\\rowcolor{lightgray} " + get_canonical_str(f"{struct_ft} > {struct_ft_median}"), get_canonical_str('\t'.join(output_strs)), round(best_mrr_above, 2), sep='\t')
            else:
                print(get_canonical_str(f"{struct_ft} > {struct_ft_median}"), get_canonical_str('\t'.join(output_strs)), round(best_mrr_above, 2), sep='\t')
        else:
            print(f"{struct_ft}", "N/A", sep='\t')
    
if __name__ == '__main__':
    dataset = sys.argv[1]
    kgem = sys.argv[2]
    run_id = sys.argv[3]
    main(dataset, kgem, run_id)
