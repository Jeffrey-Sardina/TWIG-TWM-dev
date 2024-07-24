# external imports
from pykeen.pipeline import pipeline
import sys
import os
import glob
from torch.multiprocessing import Process
import random
from pykeen.evaluation import RankBasedEvaluator
from utils.utils import load_custom_dataset, Custom_Dataset
from pykeen import datasets

'''
inverse_harmonic_mean_rank is MRR!
'''

GRID_SIZE = 1215

def get_hp_grid():
    losses = [
        'MarginRankingLoss',
        'BCEWithLogitsLoss',
        'CrossEntropyLoss'
    ]

    negative_samplers = [
        'BasicNegativeSampler',
        'BernoulliNegativeSampler',
        'PseudoTypedNegativeSampler'
    ]

    learning_rates = [
        1e-2, 1e-4, 1e-6
    ]

    regulariser_coefficients = [
        1e-2, 1e-4, 1e-6
    ]

    negs_per_pos = [
        5, 25, 125
    ]

    margins = [
        0.5, 1, 2
    ]

    dimensions = [
        50, 100, 250
    ]

    grid = []
    for loss in losses:
        for neg_samp in negative_samplers:
            for lr in learning_rates:
                for reg_coeff in regulariser_coefficients:
                    for npp in negs_per_pos:
                        for dim in dimensions:
                            if loss == 'MarginRankingLoss':
                                for margin in margins:
                                    hps = {}
                                    hps['loss'] = loss
                                    hps['neg_samp'] = neg_samp
                                    hps['lr'] = lr
                                    hps['reg_coeff'] = reg_coeff
                                    hps['npp'] = npp
                                    hps['margin'] = margin
                                    hps['dim'] = dim
                                    grid.append(hps)
                            else:
                                hps = {}
                                hps['loss'] = loss
                                hps['neg_samp'] = neg_samp
                                hps['lr'] = lr
                                hps['reg_coeff'] = reg_coeff
                                hps['npp'] = npp
                                hps['margin'] = None
                                hps['dim'] = dim
                                grid.append(hps)
    return [(run_id,hps) for run_id,hps in enumerate(grid)]

def run_exp(hps, exp_dir, dataset, model, seed=None, use_testing_data=False):
    evaluator = RankBasedEvaluator(clear_on_finalize=False)

    if not 'epochs' in hps:
        hps['epochs'] = 100
    if not 'lr_scheduler' in hps:
        hps['lr_scheduler'] = None
    if seed is None:
        seed = int(random.random() * 1e9)
    print(f'Using seed {seed}', file=sys.stderr)

    if type(dataset) is Custom_Dataset:
        print(f'Loading custom datast for use in pipeline', file=sys.stderr)
        pipeline_result = pipeline(
            training=dataset.factory_dict['training'],
            testing=dataset.factory_dict['testing'],
            validation=dataset.factory_dict['validation'],
            model=model,
            epochs=hps['epochs'],
            evaluator=evaluator,

            dimensions=hps['dim'],

            training_loop='sLCWA',
            negative_sampler=hps['neg_samp'],
            negative_sampler_kwargs={
                'num_negs_per_pos':hps['npp']
            },

            loss=hps['loss'],
            loss_kwargs={
                'margin':hps['margin']
            } if hps['loss'] == 'MarginRankingLoss' else None,

            optimizer='Adam',
            optimizer_kwargs={
                'lr': hps['lr']
            },
            lr_scheduler=hps['lr_scheduler'],

            regularizer='LpRegularizer',
            regularizer_kwargs={
                'p':3,
                'weight':hps['reg_coeff']
            },

            use_testing_data=use_testing_data, #use validation set
            random_seed=seed,
            evaluation_fallback=True, #recover from eval errors
            use_tqdm=False
        )
    else:
        pipeline_result = pipeline(
            dataset=dataset,
            model=model,
            epochs=hps['epochs'],
            evaluator=evaluator,

            dimensions=hps['dim'],

            training_loop='sLCWA',
            negative_sampler=hps['neg_samp'],
            negative_sampler_kwargs={
                'num_negs_per_pos':hps['npp']
            },

            loss=hps['loss'],
            loss_kwargs={
                'margin':hps['margin']
            } if hps['loss'] == 'MarginRankingLoss' else None,

            optimizer='Adam',
            optimizer_kwargs={
                'lr': hps['lr']
            },
            lr_scheduler=hps['lr_scheduler'],

            regularizer='LpRegularizer',
            regularizer_kwargs={
                'p':3,
                'weight':hps['reg_coeff']
            },

            use_testing_data=use_testing_data, #use validation set
            random_seed=seed,
            evaluation_fallback=True, #recover from eval errors
            use_tqdm=False
        )
    pipeline_result.save_to_directory(exp_dir)
    return pipeline_result, evaluator

def run_exps(num_processes, grid, out_dir, dataset, model, seed):
    if num_processes > 1:
        grid_subset_size = int(len(grid) / num_processes) #had 0.5 + 
        grid_subsets = []
        next_init_idx = 0
        end = False
        while not end:
            # I think this will work when the divisor is not a factor of len(grid)
            next_end_idx = next_init_idx+grid_subset_size
            if next_end_idx + grid_subset_size > len(grid):
                end = True
                grid_subsets.append(
                    grid[next_init_idx:]
                )
            else:
                grid_subsets.append(
                    grid[next_init_idx : next_end_idx]
                )
            next_init_idx = next_end_idx

        total_len = 0
        for grid_subset in grid_subsets:
            total_len += len(grid_subset)
        assert total_len == len(grid)
        assert len(grid_subsets) == num_processes, f'{len(grid_subsets)} vs {num_processes}'

        for grid_subset in grid_subsets:
            Process(target=run_block, args=(grid_subset, out_dir, dataset, seed)).start()

def run_block(grid, out_dir, dataset, model, seed):
    for run_id, hps in grid:
        exp_dir = os.path.join(out_dir, str(run_id))

        # determine if this exp was already done (and skip it)
        # of if we should do it (i.e. continue where we left off)
        try:
            os.makedirs(exp_dir)
        except: # i.e. if the dir already exists
            if len(glob.glob(f'{exp_dir}/*')) > 0: # if its data was written already
                print(f'Skipping exp {run_id} -- already completed')
                continue

        print(f'Starting exp with run_id {run_id}')
        pipeline_result, evaluator = run_exp(hps, exp_dir, dataset, model, seed)
        print(f'Finished exp with run_id {run_id}')

        mr = pipeline_result.get_metric('mr')
        mrr = pipeline_result.get_metric('mrr')
        h1 = pipeline_result.get_metric('Hits@1')
        h3 = pipeline_result.get_metric('Hits@3')
        h5 = pipeline_result.get_metric('Hits@5')
        h10 = pipeline_result.get_metric('Hits@10')

        results_str = f'End of exp {run_id} \n{"="*100}\n'

        results_str += f"Head ranks: (idx, rank) \n"
        last_batch_end = 0
        for batch in range(len(evaluator.ranks[('head', 'realistic')])):
            for idx, rank in enumerate(evaluator.ranks[('head', 'realistic')][batch]):
                true_triple_index = idx + last_batch_end
                results_str +=f'{true_triple_index} --> {rank}\n'
            last_batch_end += idx + 1 # +1 since each batch starts at 0. If we ended at 1023, however, we should start at one after that -- 1024. So we add one

        results_str += f"\nTail ranks: (idx, rank) \n"
        last_batch_end = 0
        for batch in range(len(evaluator.ranks[('head', 'realistic')])):
            for idx, rank in enumerate(evaluator.ranks[('tail', 'realistic')][batch]):
                true_triple_index = idx + last_batch_end
                results_str +=f'{true_triple_index} --> {rank}\n'
            last_batch_end += idx + 1 # +1 since each batch starts at 0. If we ended at 1023, however, we should start at one after that -- 1024. So we add one
        results_str += f'\nMR = {mr} \nMRR = {mrr} \nHits@(1,3,5,10) = {h1, h3, h5, h10}\n'
        results_str += f'{"="*100}\n'

        print(results_str)
        evaluator.clear()

def write_id_to_hps(grid, out_file):
    with open(out_file, 'w') as out:
        for run_id, hps in grid:
            print(f'{run_id} --> {hps}', file=out)

def main(out_file, out_dir, num_processes, dataset, model, seed, grid_override_idxs=None):
    grid = get_hp_grid()
    write_id_to_hps(grid, out_file)
    if grid_override_idxs:
        sub_grid = []
        for idx in grid_override_idxs:
            sub_grid.append(grid[idx])
        grid = sub_grid
    if num_processes == 1:
        run_block(grid, out_dir, dataset, model, seed)
    else:
        run_exps(num_processes, grid, out_dir, dataset, model, seed)

if __name__ == '__main__':
    out_file = sys.argv[1]
    out_dir = sys.argv[2]
    num_processes = int(sys.argv[3])
    dataset_name = sys.argv[4]
    model = sys.argv[5]
    seed = sys.argv[6]
    if len(sys.argv) > 7:
        # if two numbers are given, take everything in that range (inclusive on both ends)
        if len(sys.argv) > 8:
            start_exp_num = int(sys.argv[7])
            end_exp_num = int(sys.argv[8])
            grid_override_idxs = [x for x in range(start_exp_num, end_exp_num+1)]

        # if only one number is given (not a list separated by dashes),
        # choose a random subset with that size. If one number is given with a dash,
        # that will NOT run this section
        elif not '-' in sys.argv[7]:
            num_random_to_get = int(sys.argv[7])
            grid_override_idxs = set()
            while len(grid_override_idxs) < num_random_to_get:
                idx = random.randint(0, GRID_SIZE-1) #both are inclusive
                grid_override_idxs.add(idx)
        
        # if a list separated by dashes (even with just one number) is given, run those
        # hyperparameter combinations only
        else:
            grid_override_idxs_str = sys.argv[7].split('-')
            grid_override_idxs = [int(x) for x in grid_override_idxs_str]
    else:
        grid_override_idxs = None
    if seed == 'None':
        # use different seeds for every different run
        seed = None
    else:
        seed = int(seed)

    # dataset loading -- allow for custom
    try:
        dataset = datasets.get_dataset(dataset=dataset_name)
    except:
        dataset = load_custom_dataset(dataset_name)  

    main(out_file, out_dir, num_processes, dataset, model, seed, grid_override_idxs=grid_override_idxs)
