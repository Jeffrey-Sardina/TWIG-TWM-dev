from twig_twm import do_job
import sys


def cross_kgem():
    for dataset in ['CoDExSmall', 'DBpedia50', 'Kinships', 'OpenEA']: #'UMLS', 
        # all
        datasets = {
            dataset: ['2.1']
        }
        data_to_load = {
            'ComplEx': datasets,
            'DistMult': datasets,
            'TransE': datasets
        }
        do_job(
            data_to_load,
            epochs=N_EPOCHS,
            mrr_loss_coeffs=[10, 0],
            rank_dist_loss_coeffs=[1, 0],
            rescale_mrr_loss=False,
            rescale_rank_dist_loss=False,
            test_ratio=0.1,
            tag=f"{dataset}-all"
        )

def kgem_omit():
    for dataset in ['UMLS', 'CoDExSmall', 'DBpedia50', 'Kinships', 'OpenEA']:
        datasets = {
            dataset: ['2.1']
        }
        data_to_load = {
            'ComplEx': datasets,
            'DistMult': datasets,
            'TransE': datasets
        }
        for kgem_omit in data_to_load:
            kgems_omit = {kgem:data_to_load[kgem] for kgem in set(data_to_load.keys()) - {kgem_omit}}
            data_to_load_local = {
                kgem: datasets for kgem in kgems_omit
            }
            do_job(
                data_to_load_local,
                epochs=N_EPOCHS,
                mrr_loss_coeffs=[10, 0],
                rank_dist_loss_coeffs=[1, 0],
                rescale_mrr_loss=False,
                rescale_rank_dist_loss=False,
                test_ratio=0.1,
                tag=f"{dataset}-omit-{kgem_omit}"
            )

def cross_kg():
    for kgem in ['ComplEx', 'DistMult', 'TransE']:
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
            epochs=N_EPOCHS,
            mrr_loss_coeffs=[10, 0],
            rank_dist_loss_coeffs=[1, 0],
            rescale_mrr_loss=False,
            rescale_rank_dist_loss=False,
            test_ratio=0.1,
            tag=f"{kgem}-all"
        )

def kg_omit():
    datasets = {
        "UMLS": ["2.1"],
        "CoDExSmall": ["2.1"],
        "DBpedia50": ["2.1"],
        "Kinships": ["2.1"],
        "OpenEA": ["2.1"],
    }
    for kgem in ['ComplEx', 'DistMult', 'TransE']:
        # minus 1
        for kg_omit in datasets:
            datasets_omit = {dataset:datasets[dataset] for dataset in set(datasets.keys()) - {kg_omit}}
            data_lo_load = {
                kgem: datasets_omit
            }
            do_job(
                data_lo_load,
                epochs=N_EPOCHS,
                mrr_loss_coeffs=[10, 0],
                rank_dist_loss_coeffs=[1, 0],
                rescale_mrr_loss=False,
                rescale_rank_dist_loss=False,
                test_ratio=0.1,
                tag=f"{kgem}-omit-{kg_omit}"
            )

def all_combos():
    datasets = {
        "UMLS": ["2.1"],
        "CoDExSmall": ["2.1"],
        "DBpedia50": ["2.1"],
        "Kinships": ["2.1"],
        "OpenEA": ["2.1"],
    }
    data_lo_load = {
        "ComplEx": datasets,
        "DistMult": datasets,
        "TransE": datasets
    }
    do_job(
        data_lo_load,
        epochs=N_EPOCHS,
        mrr_loss_coeffs=[10, 0],
        rank_dist_loss_coeffs=[1, 0],
        rescale_mrr_loss=False,
        rescale_rank_dist_loss=False,
        test_ratio=0.1,
        tag=f"all-comboes-at-once"
    )

if __name__ == '__main__':
    N_EPOCHS = [100, 0]
    procedure = int(sys.argv[1])
    if procedure == 1:
        cross_kg()
    if procedure == 2:
        kg_omit()
    elif procedure == 3:
        cross_kgem()
    elif procedure == 4:
        kgem_omit()
    elif procedure == 5:
        all_combos()
    else:
        assert False
