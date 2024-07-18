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

'''
================
Helper Functions
================
'''
def d_hist(X, n_bins, min_val, max_val):
    '''
    I think if X and curr_val are too far appart, the dertivative becomes 0
    We need to fix this somehow!

    Sharpness? Noramliation? I like normalisaation
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
        # freqs[i] += (count + 1) / (n_elems + n_bins) # +1, +n_bins since if a count is 0, we need it to be 1 instead
        freqs[i] += (count + 1) #new; +1 to avoid 0s as this will be logged
        last_val = curr_val
    freqs = freqs / torch.sum(freqs) #new
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
    Purpose-built for TWIGv3, not tested yet
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
    if rescale_y:
        mrr_pred = torch.mean(1 / (1 + R_pred * (max_rank - 1))) # mrr from ranks on range [0,max]
    else:
        assert False, "this code might be wrong"
        R_pred = 1 + R_pred * (max_rank - 1) # get into range [0,max]
        mrr_pred = torch.mean(1 / (R_pred)) # mrr from ranks on range [0,max]

    # get dists
    min_val = float(torch.min(R_true))
    max_val = float(torch.max(R_true))
    
    R_true_dist = d_hist(
        R_true,
        n_bins=n_bins,
        min_val=min_val,
        max_val=max_val
    )
    R_pred_dist = d_hist(
        R_pred,
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

def train_epoch(dataloaders,
            model,
            mrr_loss,
            unordered_rank_list_loss,
            optimizer,
            alpha=1,
            gamma=1,
            rescale_y=False,
            n_bins=30,
            verbose=False):

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
    # for it in dataloader_iterators:
    #     for _ in range(num_batches):
            batch += 1
            X, y = next(it)

            loss, _, _ = do_batch(model,
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
                verbose=verbose)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

def test(
        dataloaders,
        dataloader_name,
        model,
        mrr_loss,
        unordered_rank_list_loss,
        alpha=1,
        gamma=1,
        rescale_y=False,
        n_bins=30,
        verbose=False
    ):
    model.eval()
    test_loss = 0
    mrr_preds = []
    mrr_trues = []

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

    with torch.no_grad():
        batch = -1
        for _ in range(num_batches):
            for it in dataloader_iterators:
                batch += 1
                X, y = next(it)

                if batch % 500 == 0 and verbose:
                    print(f'Testing: batch {batch} / {num_total_batches}')
                
                loss, mrr_pred, mrr_true = do_batch(model,
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
                    verbose=False)
                
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
        first_epochs = 30,
        second_epochs = 60,
        lr=5e-3,
        rescale_y=False,
        verbose=True,
        model_name_prefix="model",
        checkpoint_dir='checkpoints/',
        checkpoint_every_n=5
    ):
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
        train_epoch(training_dataloaders_list,
                model,
                mrr_loss,
                unordered_rank_list_loss,
                optimizer, 
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
    '''
    TODO: come back to this later.
    it seems I had this reversed...and yet it still works amazingly. Interesting
    '''
    for layer in model.layers_to_freeze:
        layer.requires_grad_ = False
    for t in range(second_epochs):
        print(f"Epoch {t+1} -- ", end='')
        train_epoch(training_dataloaders_list,
                model,
                mrr_loss,
                unordered_rank_list_loss,
                optimizer, 
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
        r2_mrr, test_loss, mrr_preds, mrr_trues = test([testing_dataloader],
            dataset_name,                                     
            model,
            mrr_loss,
            unordered_rank_list_loss,
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
