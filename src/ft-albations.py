from twig_twm import ablation_job

def run_ft_ablation(blacklist):
    return ablation_job(
        data_to_load = {
            "ComplEx": "UMLS"
        },
        test_ratio=0.1,
        valid_ratio=0.0,
        normalisation=['zscore'],
        n_bins=[30],
        model_or_version=['base'],
        model_kwargs=[None],
        optimizer=['adam'],
        optimizer_args=[
            {'lr': 5e-3},
        ],
        epochs=[
            [5, 10]
        ],
        mrr_loss_coeffs=[
            [0, 10]
        ],
        rank_dist_loss_coeffs=[
            [1, 1]
        ],
        rescale_mrr_loss=[True, False],
        rescale_rank_dist_loss=[True, False],
        ft_blacklist=[
            blacklist
        ],
        verbose=True,
        tag=f'Ablation-job-blacklist-{"_".join(blacklist)}',
        ablation_metric='r2_mrr', #r_mrr, r2_mrr, spearmanr_mrr@5, 10, 50, 100, or All
        ablation_type=None, #full or rand, if given (default full)
        timeout=-1, #seconds
        max_iterations=-1,
        train_and_eval_after=False,
        train_and_eval_args={
            'epochs': [5, 10],
            'verbose': True,
            'tag': 'Post-Ablation-Train-job'
        },
        print_exp_meta=True
    )

def main():
    for ft in [
        "loss",
        "neg_samp",
        "lr",
        "reg_coeff",
        "npp",
        "margin",
        "dim",

        "s_deg",
        "o_deg",
        "p_freq",
        "s_p_cofreq",
        "o_p_cofreq",
        "s_o_cofreq",

        "s min deg neighbnour",
        "o min deg neighbnour",
        "s max deg neighbnour",
        "o max deg neighbnour",
        "s mean deg neighbnour",
        "o mean deg neighbnour",
        "s num neighbnours",
        "o num neighbnours",

        "s min freq rel",
        "o min freq rel",
        "s max freq rel",
        "o max freq rel",
        "s mean freq rel",
        "o mean freq rel",
        "s num rels",
        "o num rels",
    ]:
        blacklist = set(ft)
        run_ft_ablation(blacklist)

if __name__ == '__main__':
    main()
