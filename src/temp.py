from twig_twm import do_job, finetune_job

# for kgem in ['ComplEx', 'DistMult', 'TransE']:
#     # all
#     datasets = {
#         "UMLS": ["2.1"],
#         "CoDExSmall": ["2.1"],
#         "DBpedia50": ["2.1"],
#         "Kinships": ["2.1"],
#         "OpenEA": ["2.1"],
#     }
#     data_lo_load = {
#         kgem: datasets
#     }
#     # do_job(
#     #     data_lo_load,
#     #     epochs=[10, 0],
#     #     mrr_loss_coeffs=[10, 0],
#     #     rank_dist_loss_coeffs=[1, 0],
#     #     rescale_mrr_loss=False,
#     #     rescale_rank_dist_loss=False,
#     #     test_ratio=0.1,
#     #     tag=f"{kgem}-all"
#     # )

#     # minus 1
#     for kg_omit in datasets:
#         datasets_omit = {dataset:datasets[dataset] for dataset in set(datasets.keys()) - {kg_omit}}
#         data_lo_load = {
#             kgem: datasets_omit
#         }
#         do_job(
#             data_lo_load,
#             epochs=[10, 0],
#             mrr_loss_coeffs=[10, 0],
#             rank_dist_loss_coeffs=[1, 0],
#             rescale_mrr_loss=False,
#             rescale_rank_dist_loss=False,
#             test_ratio=0.1,
#             tag=f"{kgem}-omit-{kg_omit}"
#         )

# for dataset in ['UMLS', 'CoDExSmall', 'DBpedia50', 'Kinships', 'OpenEA']:
#     # all
#     datasets = {
#         dataset: ['2.1']
#     }
#     data_to_load = {
#         'ComplEx': datasets,
#         'DistMult': datasets,
#         'TransE': datasets
#     }
#     do_job(
#         data_to_load,
#         epochs=[10, 0],
#         mrr_loss_coeffs=[10, 0],
#         rank_dist_loss_coeffs=[1, 0],
#         rescale_mrr_loss=False,
#         rescale_rank_dist_loss=False,
#         test_ratio=0.1,
#         tag=f"{dataset}-all"
#     )

#     # minus 1
#     for kgem_omit in data_to_load:
#         kgems_omit = {kgem:data_to_load[kgem] for kgem in set(data_to_load.keys()) - {kgem_omit}}
#         data_to_load_local = {
#             kgem: datasets for kgem in kgems_omit
#         }
#         do_job(
#             data_to_load_local,
#             epochs=[10, 0],
#             mrr_loss_coeffs=[10, 0],
#             rank_dist_loss_coeffs=[1, 0],
#             rescale_mrr_loss=False,
#             rescale_rank_dist_loss=False,
#             test_ratio=0.1,
#             tag=f"{dataset}-omit-{kgem_omit}"
#         )
    
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
    epochs=[1, 0],
    mrr_loss_coeffs=[10, 0],
    rank_dist_loss_coeffs=[1, 0],
    rescale_mrr_loss=False,
    rescale_rank_dist_loss=False,
    test_ratio=0.1,
    tag=f"test-kgems"
)
