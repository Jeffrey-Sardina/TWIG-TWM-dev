# twig-twm imports
from run_exp import main

# external imports
import pickle
import torch
import sys
import json

def load_and_run_from_chkpt(
        torch_checkpont_path,
        model_config_path,
        model_config_override_path
    ):
    # load original config
    with open(model_config_path, 'rb') as cache:
        print('loading model settings from cache:', model_config_path)
        model_config = pickle.load(cache)

    # NOTE: you may want to override datasets to test on new datasets in the
    # few- or zero- shot setting
    with open(model_config_override_path) as inp:
        model_config_override = json.load(inp)
    for key in model_config_override:
        print(f'overriding original values for {key}. Was {model_config[key]}, now is {model_config_override[key]}')
        model_config[key] = model_config_override[key]

    if not "first_epochs" in model_config_override or not "second_epochs" in model_config_override:
        assert False, "A new number of first and second epochs, at least, must be given in the override config"
    print(f'It will be trained for first epochs {model_config["first_epochs"]}  and second epochs {model_config["second_epochs"]} more epochs now.')
    print('If you are loading a modle that was itself loaded from checkpoint, the number of epochs listed as already completed above will be incorrect')
    print('until I get around to fxing this, you will have to recursively go through logs to find the total number of epochs run')

    # load the checkpointed model
    print('loadng TWIG model from disk at:', torch_checkpont_path)
    model = torch.load(torch_checkpont_path)

    # run checkpointed model with new config
    print(f'the full config being used is: {model_config}')
    main(
        model_config['version'],
        model_config['dataset_names'],
        model_config['first_epochs'],
        model_config['second_epochs'],
        model_config['lr'],
        model_config['normalisation'],
        model_config['rescale_y'],
        model_config['dataset_to_run_ids'],
        model_config['dataset_to_training_ids'],
        model_config['dataset_to_testing_ids'],
        model_config['test_mode'],
        model_config['testing_percent'],
        preexisting_model=model
    )

if __name__ == '__main__':
    torch_checkpont_path = sys.argv[1]
    model_config_path = sys.argv[2]
    model_config_override_path = sys.argv[3] #can be "None"
    load_and_run_from_chkpt(
        torch_checkpont_path,
        model_config_path,
        model_config_override_path
    )
    