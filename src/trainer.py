# external imports
import torch
from torch import nn
from torcheval.metrics.functional import r2_score
import torch.nn.functional as F
import os
from scipy import stats

'''
====================
Constant Definitions
====================
'''
device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint_dir='checkpoints/'

def d_hist(X, n_bins, min_val, max_val):
    '''
    d_hhist() implements a differentiable "soft histogram" based on differentiable counting. The key insight here (thanks Alok) is that sigmoid can approximate the counting function by assigning 0 (not present) or 1 (present). Since it actually assigns s full continuum, it can be differentiated -- at the cost of exactitide. Nevertheless, these counts are used to buuild a histogram representing the distribution of ranks for a given set of hyperparameters, and in practive this is the "secret sauce" that makes TWIG work.

    The arguments it accepts are:
        - X (torch.Tensor): a tensor containing a single column of all ranks, either those predicted by TWIG or those observed in the ground truth
        - n_bins (int): the number of bins to use in the constructed histogram
        - min_val (float): the minimum value that should be present in the histogram.
        - max_val (float): the maximum value that should be present in the histogram.

    NOTE: This function is finicky. It tends to finick. If you change your normalisation (or stop normalising), or change this or that or the other things and suddently your loss never updates, it's probably this function turning all the derivaties to 0. The hyperparameters in the function (sharpness, n_bins) are quite arbitrary. But I can say that higher sharpness is actually bad for performance form some limited testing -- it ends up zeroing a lot of derivaties in backprop. n_bins I am less sure of but 30 seems to work work empirically, so I never changed it.

    The values it returns are:
        - freqs (torch.Tensor): the bin frequencies of the histogram constructed from the input ranks list.
    '''
    # n_elems = torch.prod(torch.tensor(X.shape))
    bins = torch.linspace(start=min_val, end=max_val, steps=n_bins+1)[1:]
    freqs = torch.zeros(size=(n_bins,)).to(device)
    last_val = None
    sharpness = 1
    for i, curr_val in enumerate(bins):
        if i == 0:
            count = F.sigmoid(sharpness * (curr_val - X))
        elif i == len(bins) - 1:
            count = F.sigmoid(sharpness * (X - last_val))
        else:
            count = F.sigmoid(sharpness * (X - last_val)) \
                * F.sigmoid(sharpness * (curr_val - X))
        count = torch.sum(count)
        freqs[i] += (count + 1) #new; +1 to avoid 0s as this will be logged
        last_val = curr_val
    freqs = freqs / torch.sum(freqs) # put on [0, 1]
    return freqs

def do_batch(
        model,
        mrr_loss,
        unordered_rank_list_loss,
        X,
        y,
        batch,
        num_total_batches,
        alpha,
        gamma,
        n_bins,
        rescale_y,
        verbose=False
    ):
    '''
    do_batch() runs a single batch for TWIG. Note that, as outlined in do_load, batches are extremely constrained. That is to say, every batch **must** represent the full ranked list output of evlaution of a KGE model on its test / valid set. This is because TWIG needs to predict notn nonly ranks, but also overall MRR (from those ranks). And it can only get to MRR if it has the full list from which MRR is calculated in its current formulation. As such, batch size is not, and cannot be, configurable in the current formulation. Note that this also moeans **batch size will be variable** when TWIG is trained on multiple KGs, as it depends on the size of each KG's valid / test set.

    As such, do_batch() can be better said to run a batch of all link prediction queries in a KG test / valid set, compare predicted ranks and MRR to ground truth ranks and MRR, and to use these to calculate loss. This loss is then backpropogated as normal, etc.

    The arguments is accepts are:
        - model (torch.nn.Module): the TWIG model that is being trained
        - mrr_loss (func): the loss function that computes a loss vavlue for predicted MRR values vsv true MRR values
        - unordered_rank_list_loss (func): the loss function that computes a loss for the predicted vs the ground truth distribution of ranks
        - X (torch.Tensor): input feature vectors for TWIG
        - y (torch.Tensor): output vector (of ranks) that TWIG is attempting to replicate
        - batch (int): the current batch number, used for printing stats
        - num_total_batches (int): the total number of batches that will be run per epoch, used for printing stats
        - alpha (float): the coefficient to MRR loss. Should be 0 in phase 1 of training.
        - gamma (float) the coefficient to unordered_rank_list_loss. Should always have a positive value regardless of the training phase.
        - n_bins (int): the number of bins to use when creating the histoggram to represent the distribtuion of ranks (predicted or ground truth).
        - rescale_y (bool): True if the groun-truth data `y` was rescaled onto [0, 1] during data loading, False otherwise
        - verbose (bool): whether the batch should output more verbose information

    The values it returns are:
        - loss (float): the computed value of the loss functions, aggregated into a single score
        - mrr_pred (float): the MRR predicted by TWIG for this batch
        - mrr_true (float): the ground truth MRR for this batch
    '''
    X = X.to(device)
    R_true = y.to(device)
    max_rank = X[0][0]

    # get ground truth data
    if rescale_y:
        mrr_true = torch.mean(1 / (R_true * max_rank))
    else:
        mrr_true = torch.mean(1 / R_true)

    # get predicted data
    R_pred = model(X)
    mrr_pred = torch.mean(1 / (1 + R_pred * (max_rank - 1))) # mrr from ranks on range [0,max]

    # get dists
    min_val = float(torch.min(R_true))
    max_val = float(torch.max(R_true))
    
    R_true_dist = d_hist(
        X=R_true,
        n_bins=n_bins,
        min_val=min_val,
        max_val=max_val
    )
    R_pred_dist = d_hist(
        X=R_pred,
        n_bins=n_bins,
        min_val=min_val,
        max_val=max_val
    )

    # compute loss
    mrrl = mrr_loss(mrr_pred, mrr_true)
    urll = unordered_rank_list_loss(R_pred_dist.log(), R_true_dist) #https://discuss.pytorch.org/t/kl-divergence-produces-negative-values/16791/5
    loss = alpha * mrrl + gamma * urll

    if batch % 500 == 0 and verbose:
        print(f"batch {batch} / {num_total_batches} mrrl: {alpha * mrrl.item()}; urll: {gamma * urll.item()}; ")
        print('\trank: pred, true, means', torch.mean(R_pred).item(), torch.mean(R_true).item())
        print('\trank: pred, true, stds', torch.std(R_pred).item(), torch.std(R_true).item())
        print('\tpred, true, mrr', mrr_pred.item(), mrr_true.item())
        print()

    return loss, mrr_pred, mrr_true

def train_epoch(
        dataloaders,
        model,
        mrr_loss,
        unordered_rank_list_loss,
        optimizer,
        alpha,
        gamma,
        rescale_y,
        n_bins,
        verbose=False
    ):
    '''
    train_epoch() trains TWIG for a single epoch. While this sounds simple, it actually has  abit of data-juggling to do to manage different dataloaders for different KGs. It runs one batch on each KG until all batches have been run, rather than all batches of one KG and then all of the next, to ensure that new KG's done 'erase' or 'unlearn' what was lerned from previous KGs. Besides that, it's stanard "hey bro let's train for an epoch" code.

    The arguments it accepts are:
        - dataloaders (list of torch.Dataloader): a list of all dataloaders for all KGs that TWIG is training on
        - model (torch.nn.Module): the TWIG model that is being trained
        - mrr_loss (func): the loss function that computes a loss vavlue for predicted MRR values vsv true MRR values
        - unordered_rank_list_loss (func): the loss function that computes a loss for the predicted vs the ground truth distribution of ranks
        - optimizer (torch.optim,Optimizer): the instantiated optimizer that should be used for backpropagation
        - alpha (float): the coefficient to MRR loss. Should be 0 in phase 1 of training.
        - gamma (float) the coefficient to unordered_rank_list_loss. Should always have a positive value regardless of the training phase.
        - rescale_y (bool): True if the groun-truth data `y` was rescaled onto [0, 1] during data loading, False otherwise
        - n_bins (int): the number of bins to use when creating the histoggram to represent the distribtuion of ranks (predicted or ground truth).
        - verbose (bool): whether the batch should output more verbose information

    The values it returns are:
        - None
    '''
    dataloader_iterators = [
        iter(dataloader) for dataloader in dataloaders
    ]
    num_batches_by_loader = [
        len(dataloader) for dataloader in dataloaders
    ]
    num_batches = num_batches_by_loader[0] #all the same for now

    num_total_batches = 0
    for num in num_batches_by_loader:
        num_total_batches += num_batches
        assert num == num_batches

    batch = -1
    for _ in range(num_batches):
        for it in dataloader_iterators:
            batch += 1
            X, y = next(it)

            loss, _, _ = do_batch(
                model=model,
                mrr_loss=mrr_loss,
                unordered_rank_list_loss=unordered_rank_list_loss,
                X=X,
                y=y,
                batch=batch,
                num_total_batches=num_total_batches,
                alpha=alpha,
                gamma=gamma,
                n_bins=n_bins,
                rescale_y=rescale_y,
                verbose=verbose
            )
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

def test(
        dataloader,
        dataloader_name,
        model,
        mrr_loss,
        unordered_rank_list_loss,
        alpha,
        gamma,
        rescale_y,
        n_bins,
        verbose=False
    ):
    '''
    test() runs testing on a trained TWIG model. This is done in terms of how well TWIG predicts overall MRR for each given KG. In essence, TWIG predicts the MRR that the KGE model it is simulating would precit on each hyperparameter setting for the given KG. From there, it calcualted the R2 value between this and the groud truth MRR value, and this R2 score is returned as the main evaluation statistic.

    The arguments it accepts are:
        - dataloader (torch.Dataloader): the dataloader containing hold-out test data for TWIG
        - dataloader_name (str): the name of the KG from which this datalaoder was creaated
        - model (torch.nn.Module): the TWIG model that is being trained
        - mrr_loss (func): the loss function that computes a loss vavlue for predicted MRR values vsv true MRR values
        - unordered_rank_list_loss (func): the loss function that computes a loss for the predicted vs the ground truth distribution of ranks
        - alpha (float): the coefficient to MRR loss. Should be 0 in phase 1 of training.
        - gamma (float) the coefficient to unordered_rank_list_loss. Should always have a positive value regardless of the training phase.
        - rescale_y (bool): True if the groun-truth data `y` was rescaled onto [0, 1] during data loading, False otherwise
        - n_bins (int): the number of bins to use when creating the histoggram to represent the distribtuion of ranks (predicted or ground truth).
        - verbose (bool): whether the batch should output more verbose information

    The values it returns are:
        - r2_mrr (float): The R2 score calculated between predicted and true MRRs for all hyperparameter settings on on the given KG
        - test_loss (float): the loss value from testing
        - mrr_preds (list of float): a list of all MRR values predicted by TWIG
        - mrr_trues (list of float): a list of all ground-truth MRR values
    '''
    model.eval()
    test_loss = 0
    mrr_preds = []
    mrr_trues = []
    num_total_batches = len(dataloader)
    with torch.no_grad():
        batch = -1
        for X, y in dataloader:
            batch += 1

            if batch % 500 == 0 and verbose:
                print(f'Testing: batch {batch} / {num_total_batches}')
            
            loss, mrr_pred, mrr_true = do_batch(
                model=model,
                mrr_loss=mrr_loss,
                unordered_rank_list_loss=unordered_rank_list_loss,
                X=X,
                y=y,
                batch=batch,
                num_total_batches=num_total_batches,
                alpha=alpha,
                gamma=gamma,
                n_bins=n_bins,
                rescale_y=rescale_y,
                verbose=verbose
            )
            
            test_loss += loss.item()
            mrr_preds.append(float(mrr_pred))
            mrr_trues.append(float(mrr_true))

    # validations and data collection
    assert len(mrr_preds) > 1, "TWIG should be running inference for multiple runs, not just one, here"
    # spearman_r = stats.spearmanr(mrr_preds, mrr_trues)
    r2_mrr = r2_score(
        torch.tensor(mrr_preds),
        torch.tensor(mrr_trues),
    )
    test_loss /= num_total_batches  

    # data output
    print()
    print()
    print(f'Testing data for dataloader(s) {dataloader_name}')
    print("=" * 42)
    print()
    print("Predicted MRRs")
    print('-' * 42)
    for x in mrr_preds:
        print(x)
    print()

    print("True MRRs")
    print('-' * 42)
    for x in mrr_trues:
        print(x)
    print()

    print(f'r_mrr = {torch.corrcoef(torch.tensor([mrr_preds, mrr_trues]))}')
    print(f'r2_mrr = {r2_mrr}')
    # print(f'spearman_r = {spearman_r.statistic}; p = {spearman_r.pvalue}')
    print(f"test_loss: {test_loss}")

    return r2_mrr, test_loss, mrr_preds, mrr_trues

def run_training(
        model,
        training_dataloaders_dict,
        testing_dataloaders_dict,
        first_epochs,
        second_epochs,
        lr,
        rescale_y,
        verbose,
        model_name_prefix,
        checkpoint_every_n
    ):
    '''
    run_training() runs training and evaluation fo TWIG, and reutrns all evaluation results.

    The arguments it accepts are:
        - model (torch.nn.Module): the TWIG model that should be trained
        - training_dataloaders_dict (dict of str -> torch.Dataloader): a dict that maps the name of each KG to the training dataloader for that KG
        - testing_dataloaders_dict (dict of str -> torch.Dataloader): a dict that maps the name of each KG to the testing dataloader for that KG
        - first_epochs (int): the number of epochs to run in phase 1 of training
        - second_epochs (int): the number of epochs to run in phase 2 of training
        - lr (float): the learning rate to use while training
        - rescale_y (bool): True if the groun-truth data `y` was rescaled onto [0, 1] during data loading, False otherwise
        - verbose (bool): whether the batch should output more verbose information
        - model_name_prefix (str): the prefix to use for the model name when saving checkpoints
        - checkpoint_every_n (int): the number of epochs after which a checkpoint should be saved

    The values it returns are:
        - r2_scores (dict of str -> float): A dict mapping the name of each KG to the R2 score calculated between predicted and true MRRs for all hyperparameter settings on that KG
        - test_losses (dict of str -> float): A dict mapping the name of each KG to the test loss TWIG had on that KG
        - mrr_preds_all (dict of str -> float): A dict mapping the name of each KG to all MRR predictions TWIG made for all hyperparameter combinations on that KG
        - mrr_trues_all (dict of str -> float): A dict mapping the name of each KG to all ground truth MRR values for all hyperparameter combinations on that KG
    '''
    model.to(device)
    mrr_loss = nn.MSELoss()
    unordered_rank_list_loss = nn.KLDivLoss(reduction='batchmean')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(model)

    # We don't need dataset data for training so just make a list of those
    # We'll want the testing ones as a list for data validation as well
    training_dataloaders_list = list(training_dataloaders_dict.values())

    # Quick valiations
    for dataset_name in training_dataloaders_dict:
        num_batches = len(training_dataloaders_dict[dataset_name])
        if num_batches % 1215 != 0:
            print(f'IMPORTANT WARNING :: all dataloaders should have a multiple of 1215 batches, but I calculated {num_batches} for {dataset_name}. THIS IS ONLY VALID IF YOU ARE USING THE "HYP" TESTING MODE. If you are not, this is a critical error and means that data was not loaded properly.')
        print(f'validated training data for {dataset_name}')
    for dataset_name in testing_dataloaders_dict:
        num_batches = len(testing_dataloaders_dict[dataset_name])
        if num_batches % 1215 != 0:
            print(f'IMPORTANT WARNING :: all dataloaders should have a multiple of 1215 batches, but I calculated {num_batches} for {dataset_name}. THIS IS ONLY VALID IF YOU ARE USING THE "HYP" TESTING MODE. If you are not, this is a critical error and means that data was not loaded properly.')
        print(f'validated training data for {dataset_name}')

    # Training
    model.train()
    print(f'REC: Training with epochs in stages 1: {first_epochs} and 2: {second_epochs}')

    alpha = 0
    gamma = 1
    for layer in model.layers_to_freeze:
        layer.requires_grad_ = True
    for t in range(first_epochs):
        print(f"Epoch {t+1} -- ", end='')
        train_epoch(
            dataloaders=training_dataloaders_list,
            model=model,
            mrr_loss=mrr_loss,
            unordered_rank_list_loss=unordered_rank_list_loss,
            optimizer=optimizer, 
            alpha=alpha,
            gamma=gamma,
            rescale_y=rescale_y,
            verbose=verbose
        )
        if (t+1) % checkpoint_every_n == 0:
            print(f'Saving checkpoint at [1] epoch {t+1}')
            state_data = f'e{t+1}-e0'
            torch.save(
                model,
                    os.path.join(
                        checkpoint_dir,
                        f'{model_name_prefix}_{state_data}.pt'
                    )
                )
    print("Done Training (dist)!")

    alpha = 10
    gamma = 1
    for layer in model.layers_to_freeze:
        layer.requires_grad_ = False
    for t in range(second_epochs):
        print(f"Epoch {t+1} -- ", end='')
        train_epoch(
            dataloaders=training_dataloaders_list,
            model=model,
            mrr_loss=mrr_loss,
            unordered_rank_list_loss=unordered_rank_list_loss,
            optimizer=optimizer, 
            alpha=alpha,
            gamma=gamma,
            rescale_y=rescale_y,
            verbose=verbose
        )
        if (t+1) % checkpoint_every_n == 0:
            print(f'Saving checkpoint at [2] epoch {t+1}')
            state_data = f'e{first_epochs}-e{t+1}'
            torch.save(
                model,
                    os.path.join(
                        checkpoint_dir,
                        f'{model_name_prefix}_{state_data}.pt'
                    )
                )
    print("Done Training (mrr)!")

    # Testing
    # we do it for each DL since we want to do each dataset testing separately for now
    r2_scores = {}
    test_losses = {}
    mrr_preds_all = {}
    mrr_trues_all = {}
    for dataset_name in testing_dataloaders_dict:
        testing_dataloader = testing_dataloaders_dict[dataset_name]
        model.eval()
        print(f'REC: Testing model with dataloader {dataset_name}')
        r2_mrr, test_loss, mrr_preds, mrr_trues = test(
            dataloader=testing_dataloader,
            dataloader_name=dataset_name,                                     
            model=model,
            mrr_loss=mrr_loss,
            unordered_rank_list_loss=unordered_rank_list_loss,
            alpha=alpha,
            gamma=gamma,
            rescale_y=rescale_y,
            verbose=verbose
        )
        print("Done Testing!")

        r2_scores[dataset_name] = r2_mrr
        test_losses[dataset_name] = test_loss
        mrr_preds_all[dataset_name] = mrr_preds
        mrr_trues_all[dataset_name] = mrr_trues


    return r2_scores, test_losses, mrr_preds_all, mrr_trues_all
