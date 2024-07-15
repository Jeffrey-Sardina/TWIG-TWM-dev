# twig-twm imports
from KGE_pipeline import get_hp_grid, write_id_to_hps

# external imports
import sys
import os
import glob
from torch.multiprocessing import Process
import random
from pykeen.evaluation import RankBasedEvaluator
import torch
from pykeen.datasets import get_dataset

def run_exps(num_processes, grid, out_dir, dataset, seed):
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

def run_block(grid, out_dir, dataset, seed):
    for run_id, hps in grid:
        exp_dir = os.path.join(out_dir, str(run_id))

        if len(glob.glob(f'{exp_dir}/*')) == 0: # if its data was written already
            assert False, f"model to eval on does not exist; {exp_dir}"

        print(f'Starting exp with run_id {run_id}')
        pipeline_result, evaluator = run_eval(hps, exp_dir, dataset, seed)
        print(f'Finished exp with run_id {run_id}')

        mr = pipeline_result.get_metric('mr')
        mrr = pipeline_result.get_metric('mrr')
        h1 = pipeline_result.get_metric('Hits@1')
        h3 = pipeline_result.get_metric('Hits@3')
        h5 = pipeline_result.get_metric('Hits@5')
        h10 = pipeline_result.get_metric('Hits@10')

        '''Note: see split_list_in_batches_iter https://pykeen.readthedocs.io/en/stable/_modules/pykeen/utils.html#split_list_in_batches_iter
        batches do conserve index order, so the opeations used here are valid'''

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

def run_eval(hps, exp_dir, dataset, seed=None):
    evaluator = RankBasedEvaluator(clear_on_finalize=False)

    if not 'lr_scheduler' in hps:
        hps['lr_scheduler'] = None
    if seed is None:
        seed = int(random.random() * 1e9)
    print(f'Using seed {seed}', file=sys.stderr)

    print("-----", exp_dir, file=sys.stderr)
    loaded_model = torch.load(os.path.join(exp_dir, 'trained_model.pkl'))
    dataset = get_dataset(dataset=dataset)
    mapped_triples = dataset.validation.mapped_triples

    results = evaluator.evaluate(
        model=loaded_model,
        mapped_triples=mapped_triples,
        additional_filter_triples=[
            dataset.training.mapped_triples,
            dataset.validation.mapped_triples,
        ],
    )
    return results, evaluator

def get_finished_ids(res_file):
    finished_ids = set()
    with open(res_file, 'r') as inp:
        for line in inp:
            if "End of exp" in line:
                exp_id = int(line.strip().replace("End of exp", ""))
                finished_ids.add(exp_id)
    return finished_ids

def main(out_file, out_dir, num_processes, dataset, seed, sub_grid_range=None, finished_ids=None):
    grid = get_hp_grid()
    write_id_to_hps(grid, out_file)
    if sub_grid_range:
        assert finished_ids is None, "cannot set both sub_grid_range and finished_ids"
        start_exp_num = sub_grid_range[0]
        end_exp_num = sub_grid_range[1]
        grid = grid[start_exp_num:end_exp_num+1]
    if finished_ids:
        assert sub_grid_range is None, "cannot set both finished_ids and sub_grid_range"
        sub_grid = []
        for run_id, hps in grid:
            if run_id not in finished_ids:
                sub_grid.append(run_id, hps)
        grid = sub_grid
    if num_processes == 1:
        run_block(grid, out_dir, dataset, seed)
    else:
        run_exps(num_processes, grid, out_dir, dataset, seed)

if __name__ == '__main__':
    out_file = sys.argv[1]
    out_dir = sys.argv[2]
    num_processes = int(sys.argv[3])
    dataset = sys.argv[4]
    seed = sys.argv[5]
    sub_grid_range = None
    finished_ids = None
    if len(sys.argv) > 7:
        # run a range of exps only
        start_exp_num = int(sys.argv[6])
        end_exp_num = int(sys.argv[7])
        sub_grid_range = (start_exp_num, end_exp_num)
    elif len(sys.argv) > 6:
        # run all that have not been eval'd already only
        already_done_list = sys.argv[6]
        finished_ids = get_finished_ids(already_done_list)
    if seed == 'None':
        # use different seeds for every different run
        seed = None
    else:
        seed = int(seed)
    main(out_file, out_dir, num_processes, dataset, seed, sub_grid_range=sub_grid_range, finished_ids=finished_ids)

