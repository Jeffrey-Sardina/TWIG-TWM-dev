# twig-twm imports
from KGE_pipeline import get_hp_grid, run_exp

# external imports
import sys

def get_hps_from_id(exp_id):
    grid = get_hp_grid()
    for grid_id, hps in grid:
        if grid_id == exp_id:
            return hps
    assert False, f"could not find exp_id {exp_id} in hyperparameter grid"

def main(hyp_id, model, dataset, output_dir):
    hps = get_hps_from_id(hyp_id)
    hps['epochs'] = 1000
    pipeline_result, _ = run_exp(
        hps,
        output_dir,
        dataset,
        model,
        use_testing_data=True
    )

    mr = pipeline_result.get_metric('mr')
    mrr = pipeline_result.get_metric('mrr')
    h1 = pipeline_result.get_metric('Hits@1')
    h3 = pipeline_result.get_metric('Hits@3')
    h5 = pipeline_result.get_metric('Hits@5')
    h10 = pipeline_result.get_metric('Hits@10')

    print(f'MR: {mr}')
    print(f'MRR: {mrr}')
    print(f'H@1: {h1}')
    print(f'H@3: {h3}')
    print(f'H@5: {h5}')
    print(f'H@10: {h10}')

if __name__ == '__main__':
    hyp_id = int(sys.argv[1])
    model = sys.argv[2]
    dataset = sys.argv[3]
    output_dir = sys.argv[4]
    main(hyp_id, model, dataset, output_dir)
