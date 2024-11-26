from twig_twm import ablation_job
import sys

def run_ft_ablation(data_to_load, ft_blacklist, ft_name):
    return ablation_job(
        data_to_load=data_to_load, 
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
        rescale_mrr_loss=[False],
        rescale_rank_dist_loss=[False],
        ft_blacklist=ft_blacklist,
        verbose=True,
        tag=f'Ablation-job-blacklist-{"_".join(ft_name)}',
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
    kgem = sys.argv[1]
    do_indiv = sys.argv[2] == '1'
    do_aggr = sys.argv[3] == '1'

    for kgem in [kgem]:
        data_to_load = {
            kgem: {
                "CoDExSmall": ["2.1"],
                "DBpedia50": ["2.1"],
                "Kinships": ["2.1"],
                "OpenEA": ["2.1"],
                "UMLS": ["2.1"]
            }
        }
        if do_indiv:
            for ft_name in [
                ("loss",),
                ("neg_samp",),
                ("lr",),
                ("reg_coeff",),
                ("npp",),
                ("margin",),
                ("dim",),

                ("s_deg",),
                ("o_deg",),
                ("p_freq",),
                ("s_p_cofreq",),
                ("o_p_cofreq",),
                ("s_o_cofreq",),

                ("s min deg neighbnour", "o min deg neighbnour"),
                ("s max deg neighbnour" ,"o max deg neighbnour"),
                ("s mean deg neighbnour", "o mean deg neighbnour"),
                ("s num neighbnours", "o num neighbnours"),

                ("s min freq rel", "o min freq rel"),
                ("s max freq rel", "o max freq rel"),
                ("s mean freq rel", "o mean freq rel"),
                ("s num rels", "o num rels"),
            ]:
                ft_blacklist = [
                    set(ft_name)
                ]
                run_ft_ablation(data_to_load, ft_blacklist, ft_name=ft_name)

        if do_aggr:
            ft_blacklist = [
                set((
                    "s_deg", "o_deg", "p_freq", "s_p_cofreq", "o_p_cofreq", "s_o_cofreq"
                ))
            ]
            run_ft_ablation(data_to_load, ft_blacklist, ft_name="fine-grained")

            ft_blacklist = [
                set((
                    "s min deg neighbnour", "o min deg neighbnour", "s max deg neighbnour" ,"o max deg neighbnour", \
                    "s mean deg neighbnour", "o mean deg neighbnour", "s num neighbnours", "o num neighbnours", \
                    "s min freq rel", "o min freq rel", \
                    "s max freq rel", "o max freq rel", \
                    "s mean freq rel", "o mean freq rel", \
                    "s num rels", "o num rels"
                ))
            ]
            run_ft_ablation(data_to_load, ft_blacklist, ft_name="coarse-grained")

if __name__ == '__main__':
    main()
