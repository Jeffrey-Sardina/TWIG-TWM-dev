import sys
from sklearn.metrics import r2_score
from scipy import stats


def load_mrrs(
        data_file,
        pred_conditions=lambda x : True,
        true_conditions=lambda x : True
    ):
    pred_mrrs = []
    true_mrrs = []
    with open(data_file, 'r') as inp:
        reading_preds = False
        reading_trues = False
        for i, line in enumerate(inp):
            if reading_preds == True or reading_trues == True:
                assert not (reading_preds and reading_trues), f'ERR: {reading_preds}, {reading_trues}'
            line = line.strip()

            if "Predicted MRRs" in line:
                reading_preds = True
                reading_trues = False
            if reading_preds:
                try:
                    val = float(line)
                    assert val > 0 and val <= 1, f'bounds check for MRR values failed; mrr = {val} on line {i}'
                    pred_mrrs.append(val)
                except:
                    # first two are strings for the file format, this err is ok
                    pass
            
            if "True MRRs" in line:
                reading_preds = False
                reading_trues = True
            if reading_trues:
                try:
                    val = float(line)
                    assert val > 0 and val <= 1, f'bounds check for MRR values failed; mrr = {val} on line {i}'
                    true_mrrs.append(val)
                except:
                    # first two are strings for the file format, this err is ok
                    pass

    assert len(pred_mrrs) == len(true_mrrs), f'both should have equal length! {len(pred_mrrs)} =/=  {len(true_mrrs)}'
    pred_mrrs_filtered = []
    true_mrrs_filtered = []
    for i in range(len(pred_mrrs)):
        pred_mrr = pred_mrrs[i]
        true_mrr = true_mrrs[i]
        should_accept = pred_conditions(pred_mrr) and true_conditions(true_mrr)
        if should_accept:
            pred_mrrs_filtered.append(pred_mrr)
            true_mrrs_filtered.append(true_mrr)
    
    assert len(pred_mrrs_filtered) == len(true_mrrs_filtered), f'both should have equal length after filtering! {len(pred_mrrs_filtered)} =/=  {len(true_mrrs_filtered)}'
    return pred_mrrs_filtered, true_mrrs_filtered

def get_corr_stats(preds, trues):
    r2 = r2_score(trues, preds)
    pearson_r = stats.pearsonr(preds, trues).statistic
    spearman_r = stats.spearmanr(preds, trues).statistic
    return r2, pearson_r, spearman_r

def main(data_file, min_true_val):
    true_conditions = lambda x : x >= min_true_val
    pred_mrrs, true_mrrs = load_mrrs(
        data_file,
        true_conditions=true_conditions
    )
    r2, pearson_r, spearman_r = get_corr_stats(pred_mrrs, true_mrrs)
    print(f'stats for file {data_file}')
    print(f'R2: {r2}')
    print(f'Pearson r: {pearson_r}')
    print(f'Spearman r: {spearman_r}')

if __name__ == '__main__':
    data_file = sys.argv[1] #../TWIG/rec_v3_CoDExSmall-UMLS-DBpedia50-OpenEA-Kinships_norm-zscore_e5-e10_tag-TWM_run_hyp.log
    min_true_val = float(sys.argv[2])
    main(data_file, min_true_val)
