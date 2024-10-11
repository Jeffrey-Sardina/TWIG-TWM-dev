from twig_twm import do_job, finetune_job

# datasets = {
#     "UMLS": ["2.1"]
# }
# data_lo_load = {
#     "TransE": datasets,
# }
# do_job(
#     data_lo_load,
#     epochs=[10, 0],
#     mrr_loss_coeffs=[10, 0],
#     rank_dist_loss_coeffs=[1, 0],
#     rescale_mrr_loss=False,
#     rescale_rank_dist_loss=False,
#     test_ratio=0.1,
#     tag=f"test-kgems"
# )

# finetune_job(
#     data_lo_load,
#     model_save_path="checkpoints/chkpt-ID_2531937323629027_tag_test-kgems_e10-e0.pt",
#     model_config_path="checkpoints/chkpt-ID_2531937323629027_tag_test-kgems.pkl",
#     epochs=[1, 0],
# )

for kgem in ['ComplEx', 'DistMult', 'TransE']:
    # all
    datasets = {
        "UMLS": ["2.1"],
        "CoDExSmall": ["2.1"],
        "DBpedia50": ["2.1"],
        "Kinships": ["2.1"],
        "OpenEA": ["2.1"],
    }
    data_lo_load = {
        kgem: datasets
    }
    do_job(
        data_lo_load,
        epochs=[10, 0],
        mrr_loss_coeffs=[10, 0],
        rank_dist_loss_coeffs=[1, 0],
        rescale_mrr_loss=False,
        rescale_rank_dist_loss=False,
        test_ratio=0.1,
        tag=f"{kgem}-all"
    )

    # minus 1
    for kg_omit in datasets:
        datasets_omit = set(datasets.keys()) - {kg_omit}
        data_lo_load = {
            kgem: datasets_omit
        }
        do_job(
            epochs=[10, 0],
            mrr_loss_coeffs=[10, 0],
            rank_dist_loss_coeffs=[1, 0],
            rescale_mrr_loss=False,
            rescale_rank_dist_loss=False,
            test_ratio=0.1,
            tag=f"{kgem}-omit-{kg_omit}"
        )

for dataset in ['UMLS', 'CoDExSmall', 'DBpedia50', 'Kinships', 'OpenEA']:
    # all
    datasets = {
        dataset: ['2.1']
    }
    data_lo_load = {
        'ComplEx': datasets,
        'DistMult': datasets,
        'TransE': datasets
    }
    do_job(
        data_lo_load,
        epochs=[10, 0],
        mrr_loss_coeffs=[10, 0],
        rank_dist_loss_coeffs=[1, 0],
        rescale_mrr_loss=False,
        rescale_rank_dist_loss=False,
        test_ratio=0.1,
        tag=f"{dataset}-all"
    )

    # minus 1
    for kgem_omit in data_lo_load:
        kgem_omit = set(data_lo_load.keys()) - {kgem_omit}
        data_lo_load = {
            kgem: datasets for kgem in kgem_omit
        }
        do_job(
            epochs=[10, 0],
            mrr_loss_coeffs=[10, 0],
            rank_dist_loss_coeffs=[1, 0],
            rescale_mrr_loss=False,
            rescale_rank_dist_loss=False,
            test_ratio=0.1,
            tag=f"{dataset}-omit-{kgem_omit}"
        )
    
datasets = {
    "UMLS": ["2.1"],
    "CoDExSmall": ["2.1"],
    "DBpedia50": ["2.1"],
    "Kinships": ["2.1"],
    "OpenEA": ["2.1"]
}
data_lo_load = {
    "ComplEx": datasets,
    "DistMult": datasets,
    "TransE": datasets,
}
do_job(
    data_lo_load,
    epochs=[10, 0],
    mrr_loss_coeffs=[10, 0],
    rank_dist_loss_coeffs=[1, 0],
    rescale_mrr_loss=False,
    rescale_rank_dist_loss=False,
    test_ratio=0.1,
    tag=f"test-kgems"
)