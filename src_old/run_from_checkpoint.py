# twig-twm imports
from run_exp import main

# external imports
import pickle
import torch
import sys
import json

def load_model_config(model_config_path):
    '''
    load_model_config() loads a saved model config from disk.

    The arguments it accepts are:
        - model_config_path (str): the path to the model configuration file on disk

    The values it returns are;
        - model_config (dict of str -> any): a dictionary mapping each element of the model config, by name, to its value
    '''
    with open(model_config_path, 'rb') as cache:
        print('loading model settings from cache:', model_config_path)
        model_config = pickle.load(cache)
    return model_config

def load_override_config(model_config_override):
    '''
    load_override_config() loads settings to override (replacing the original model config).

    The arguments it accepts are:
        - model_config_path (str or dict): the path to the override configuration file on disk (if a str) or the override dict itself (if a dict)

    The values it returns are;
        - model_config_override (dict of str -> any): a dictionary mapping each element of the model config that should be overwritten, by name, to its value
    '''
    if type(model_config_override) is str:
        with open(model_config_override) as inp:
            model_config_override = json.load(inp)
    if not "first_epochs" in model_config_override or not "second_epochs" in model_config_override:
        assert False, "A new number of first and second epochs, at least, must be given in the override config"
    return model_config_override

def apply_override(model_config, model_config_override):
    '''
    apply_override() applied a given override to the base model config, and returns the final result

    The arguments it accepts are:
        - model_config (dict of str -> any): a dictionary mapping each element of the model config, by name, to its value
        - model_config_override (dict of str -> any): a dictionary mapping each element of the model config that should be overwritten, by name, to its value

    The values it returns are:
        - model_config (dict of str -> any): a dictionary containing the model config with all overrides applied
    '''
    for key in model_config_override:
        print(f'overriding original values for {key}. Was {model_config[key]}, now is {model_config_override[key]}')
        model_config[key] = model_config_override[key]
    return model_config

def load_chkpt(
        torch_checkpont_path,
        model_config_path,
        model_config_override
    ):
    '''
    load_chkpt() loads a checkpoint, TWIG hyperparameters / settings, and overrides to those settings

    The arguments it accepts are:
        - torch_checkpont_path (str): the path to the checkpoint file written by torch that should be loaded. The default (every 5 epochs) checkpoints written by TWIG-I are located at `./checkpoints//[checkpoint_id]_e[num_epochs].pt`
        - model_config_path (str): the path to the saved faile containing a serialisation of all command-line arguments given to the original model for training (this means that when you load a chekcpoint, you can use the same hyperparameters, datasets, settings etc as specifiied in this fuile without any further effort). By default, TWIG-I will write this to `./checkpoints/[checkpoint_id].pkl`
        - model_config_override_path (str or dict): the path to a custom-user made override file to specify new hyperparameters, datasets, etc to be used with the loaded model or a pre-laoded dict containing that information. For example, you can use a saved TWIG-I checkpoint as a pretrained model to then fine-tune on a new dataset, or specify more epochs to run to continue training. NOTE: This file MUST be created as a .json file and MUST contain, at a minimum, the line `"eppochs": X` to specify hw many more epochs to run. TWIG-I does not currently know how many epochs a model was run for, so if you want to finish training after a crash, for example, you need to manually tell if how many more epochs it needs to do.

    The values it returns are:
        - model (torch.nn.Module): the loaded TWIG model to use
        - model_config (dict of str -> any): the training config to use to train the model
    '''
    # load the checkpointed model
    print('loadng TWIG model from disk at:', torch_checkpont_path)
    model = torch.load(torch_checkpont_path)

    # load config with override
    model_config = apply_override(
        model_config = load_model_config(model_config_path),
        model_config_override = load_override_config(model_config_override)
    )

    return model, model_config


def run_from_chkpt(
        model,
        model_config
    ):
    '''
    run_from_chkpt() uses the given model and model configuration to run more training for TWIG (i.e. finetuning, or continuation) from where training last stopped. All training is done by handing this off to `run_exp.py` to avoid code duplication -- this function just gets the data in the fight order and format.

    The arguments it accepts are:
        - model (torch.nn.Module): the loaded TWIG model to use
        - model_config (dict of str -> any): the training config to use to train the model

    The values it returns are:
        - results (dict): the results output from training and evaluation. TODO: add more detail here
    '''
    print(f'It will be trained for first epochs {model_config["first_epochs"]}  and second epochs {model_config["second_epochs"]} more epochs now.')
    print('If you are loading a modle that was itself loaded from checkpoint, the number of epochs listed as already completed above will be incorrect')
    print('until I get around to fxing this, you will have to recursively go through logs to find the total number of epochs run')

    # load the checkpointed model
    print('loadng TWIG model from disk at:', torch_checkpont_path)
    model = torch.load(torch_checkpont_path)

    # run checkpointed model with new config
    print(f'the full config being used is: {model_config}')
    results = main(
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
    return results

if __name__ == '__main__':
    torch_checkpont_path = sys.argv[1]
    model_config_path = sys.argv[2]
    model_config_override = sys.argv[3] #can be "None"
    model, model_config = load_chkpt(
        torch_checkpont_path=torch_checkpont_path,
        model_config_path=model_config_path,
        model_config_override=model_config_override
    )
    run_from_chkpt(model, model_config)
    