from twig_twm import do_job

# do_job({"UMLS": ["2.1"]}, kge_model_name='ComplEx', rescale_mrr_loss=True, rescale_rank_dist_loss=True)
# do_job({"Kinships": ["2.1"]}, kge_model_name='ComplEx', rescale_mrr_loss=True, rescale_rank_dist_loss=True)
# do_job({"CoDExSmall": ["2.1"]}, kge_model_name='ComplEx', rescale_mrr_loss=True, rescale_rank_dist_loss=True)
# do_job({"DBpedia50": ["2.1"]}, kge_model_name='ComplEx', rescale_mrr_loss=True, rescale_rank_dist_loss=True)
# do_job({"OpenEA": ["2.1"]}, kge_model_name='ComplEx', rescale_mrr_loss=True, rescale_rank_dist_loss=True)

do_job(
    {
        "UMLS": ["2.1"],
        "Kinships": ["2.1"],
        "CoDExSmall": ["2.1"],
        "DBpedia50": ["2.1"],
        "OpenEA": ["2.1"],
    },
    kge_model_name='ComplEx',
    epochs=[15, 100],
    rescale_mrr_loss=False,
    rescale_rank_dist_loss=False
)
