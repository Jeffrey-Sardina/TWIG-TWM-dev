'''
Precompute struct half each batch, then use it again
should result in a 2x speed boost
'''

# external imports
import torch
from torch import nn
from torcheval.metrics.functional import r2_score
import torch.nn.functional as F
import os
import time
import random

# Reproducibility
torch.manual_seed(17)
random.seed(17)

'''
====================
Constant Definitions
====================
'''
device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint_dir='checkpoints/'

def _d_hist(X, n_bins, min_val, max_val):
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
    bins = torch.linspace(start=min_val, end=max_val, steps=n_bins+1)[1:]
    freqs = torch.zeros(size=(n_bins,)).to(device)
    last_val = None
    sharpness = 3
    for i, curr_val in enumerate(bins):
        if i == 0:
            # count (X < min bucket val)
            count = F.sigmoid(sharpness * (curr_val - X))
        elif i == len(bins) - 1:
            # count (X > max bucket val)
            count = F.sigmoid(sharpness * (X - last_val))
        else:
            # count (X > bucket left and X < bucket right)
            count = F.sigmoid(sharpness * (X - last_val)) \
                * F.sigmoid(sharpness * (curr_val - X))
        count = torch.sum(count)
        freqs[i] += (count + 1) #new; +1 to avoid 0s as this will be logged
        last_val = curr_val
    freqs = freqs / torch.sum(freqs) # put on [0, 1]
    return freqs

def _plot_hist(dist1, dist2, n_bins):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(
        1, 2,
        sharex=True,
        sharey=True,
        tight_layout=True
    )
    axs[0].hist(dist1, bins=n_bins)
    axs[1].hist(dist2, bins=n_bins)
    plt.show()

def _do_batch(
        model,
        mrr_loss,
        mrr_loss_coeff,
        rank_dist_loss,
        rank_dist_loss_coeff,
        hist_min_val,
        hist_max_val,
        n_bins,
        struct_tensor,
        max_rank_possible,
        hyps_tensor,
        mrr_true,
        rank_dist_true,
        do_print
    ):
    '''
    ideas:
    if mrr_true < 0.1:
        mrr_true = torch.tensor(0.05, dtype=torch.float32, device=device)
    loss for std in ranks (expected vs obtained)
    '''
    if mrr_true < 0.1:
        modified_mrr_true = torch.tensor(0.05, dtype=torch.float32, device=device)
    else:
        modified_mrr_true = mrr_true

    # get predicted data
    rank_list_pred = model(struct_tensor, hyps_tensor)
    rank_dist_pred = _d_hist(
        X=rank_list_pred,
        n_bins=n_bins,
        min_val=hist_min_val,
        max_val=hist_max_val
    )
    mrr_pred = torch.mean(1 / (1 + rank_list_pred * (max_rank_possible - 1)))
    
    # compute loss
    mrrl_multiplier = (10 / ((1 - mrr_true) ** 2)) ** 2
    if mrr_loss_coeff > 0:
        # good mrrs are higher -- we want to penalise missing those ones more
        mrrl = mrrl_multiplier * mrr_loss_coeff * mrr_loss(mrr_pred, modified_mrr_true)
    else:
        mrrl = torch.tensor(0.0, dtype=torch.float32, device=device)

    # "good" dists have a lot of low ranks. We want to penalise not being able to match those ones more
    rdl_multiplier = (10 / (1 - torch.sum(rank_dist_true[:n_bins//10])) ** 2) ** 2
    rdl = rdl_multiplier * rank_dist_loss_coeff * rank_dist_loss(rank_dist_pred.log(), rank_dist_true) #https://discuss.pytorch.org/t/kl-divergence-produces-negative-values/16791/5
    loss = mrrl + rdl

    # print state
    if do_print:
        pred_mean = str(round(float(torch.mean(rank_list_pred)), 3)).ljust(5, '0')
        pred_std = str(round(float(torch.std(rank_list_pred)), 3)).ljust(5, '0')
        mrr_pred_str = str(round(mrr_pred.item(), 3)).ljust(5, '0')
        mrr_true_str = str(round(mrr_true.item(), 3)).ljust(5, '0')
        print(f'rank avg (pred): {pred_mean} +- {pred_std}')
        print(f'mrr vals (pred, true): {mrr_pred_str}, {mrr_true_str}')
        print(float(mrrl_multiplier), float(rdl_multiplier))
        
    return loss, mrrl, rdl, mrr_pred, mrr_true    

def _train_epoch(
        model,
        twig_data,
        mrr_loss,
        mrr_loss_coeff,
        rank_dist_loss,
        rank_dist_loss_coeff,
        optimizer,
        do_print
    ):
    mode = 'train'
    print_batch_on = 500
    epoch_start_time = time.time()
    superbatch = 1
    superloss = 0
    # 1: 35s / batch. 5: 28s/b. 7: 30s/b. 50: 32s/b.
    # 1: r2 = 0.93/0.98. 5: 0.97. 7: 0.95. 50: 0.93.
    # tested on UMLS (2 runs of 1215 exps) with epochs = [2,3]
    epoch_batches = twig_data.get_train_epoch(shuffle=False)
    for batch_num, batch_data in enumerate(epoch_batches):
        # load batch data
        dataset_name, run_id, exp_id = batch_data
        struct_tensor, hyps_tensor, mrr_true, rank_dist_true = twig_data.get_batch(
            dataset_name=dataset_name,
            run_id=run_id,
            exp_id=exp_id,
            mode=mode
        )

        # run batch
        if batch_num % print_batch_on == 0:
            print(f'running batch: {batch_num} / {len(epoch_batches)} and superbatch({superbatch}); data from {dataset_name}, run {run_id}, exp {exp_id}')
        loss, mrrl, rdl, _, _ = _do_batch(
            model=model,
            mrr_loss=mrr_loss,
            rank_dist_loss=rank_dist_loss,
            mrr_loss_coeff=mrr_loss_coeff,
            rank_dist_loss_coeff=rank_dist_loss_coeff,
            hist_min_val=twig_data.HIST_MIN,
            hist_max_val=twig_data.HIST_MAX,
            n_bins=twig_data.N_BINS,
            struct_tensor=struct_tensor,
            max_rank_possible=twig_data.max_ranks[dataset_name],
            hyps_tensor=hyps_tensor,
            mrr_true=mrr_true,
            rank_dist_true=rank_dist_true,
            do_print=do_print and batch_num % print_batch_on == 0
        )
        if do_print and batch_num % print_batch_on == 0:
            print(f'batch losses (mrrl, rdl): {round(float(mrrl), 10)}, {round(float(rdl), 10)}')
            print()

        # backprop
        superloss += loss
        if (batch_num + 1) % superbatch == 0 or (batch_num + 1) == len(epoch_batches):
            superloss.backward()
            optimizer.step()
            optimizer.zero_grad()
            superloss = 0

    # print epoch results
    epoch_end_time = time.time()
    print('Epoch over!')
    print(f'epoch time: {round(epoch_end_time - epoch_start_time, 3)}')
    print()

def _eval(
        model,
        twig_data,
        dataset_name,
        mrr_loss,
        mrr_loss_coeff,
        rank_dist_loss,
        rank_dist_loss_coeff,
        mode,
        do_print
):
    model.eval()
    test_loss = 0
    mrr_preds = []
    mrr_trues = []
    batch_num = 0
    print_batch_on = 500
    print(f'Running eval on the {mode} set')
    test_start_time = time.time()
    with torch.no_grad():
        epoch_batches = twig_data.get_eval_epoch(
            mode=mode,
            dataset_name=dataset_name
        )
        for dataset_name, run_id, exp_id in epoch_batches:
            if do_print and batch_num % print_batch_on == 0:
                print(f'running batch: {batch_num}')

            struct_tensor, hyps_tensor, mrr_true, rank_dist_true = twig_data.get_batch(
                dataset_name=dataset_name,
                run_id=run_id,
                exp_id=exp_id,
                mode=mode
            )
            loss, _, _, mrr_pred, mrr_true = _do_batch(
                model=model,
                mrr_loss=mrr_loss,
                rank_dist_loss=rank_dist_loss,
                mrr_loss_coeff=mrr_loss_coeff,
                rank_dist_loss_coeff=rank_dist_loss_coeff,
                hist_min_val=twig_data.HIST_MIN,
                hist_max_val=twig_data.HIST_MAX,
                n_bins=twig_data.N_BINS,
                struct_tensor=struct_tensor,
                max_rank_possible=twig_data.max_ranks[dataset_name],
                hyps_tensor=hyps_tensor,
                mrr_true=mrr_true,
                rank_dist_true=rank_dist_true,
                do_print=do_print and batch_num % print_batch_on == 0
            )
            if do_print and batch_num % print_batch_on == 0:
                print()

            test_loss += loss.item()
            mrr_preds.append(float(mrr_pred))
            mrr_trues.append(float(mrr_true))
            batch_num += 1

    # data output
    test_end_time = time.time()
    if do_print:
        print(f'Evaluation for {dataset_name} on the {mode} set')
        print("=" * 42)
        print_mrr_predictions(mrr_preds=mrr_preds, mrr_trues=mrr_trues)
        print("=" * 42)
    r2_mrr, r_mrr, spearman_mrrs = get_correlation_info(
        mrr_preds=mrr_preds,
        mrr_trues=mrr_trues,
        do_print=do_print
    )
    if do_print:
        print("=" * 42)
        print(f'test time: {round(test_end_time - test_start_time, 3)}')

    return r2_mrr, r_mrr, spearman_mrrs, test_loss, mrr_preds, mrr_trues

def print_mrr_predictions(mrr_preds, mrr_trues):
    idx_sort_by_true = sorted( # https://stackoverflow.com/questions/7851077/how-to-return-index-of-a-sorted-list
        range(len(mrr_trues)),
        key=lambda k: mrr_trues[k]
    )
    mrr_preds_sorted = list(sorted(mrr_preds))
    print("(Sorted by True MRR values)")
    print('i_pred \t i_true \t Pred MRR \t True MRR \t Change Flag')
    for i, idx in enumerate(idx_sort_by_true):
        pred_idx = str(mrr_preds_sorted.index(mrr_preds[idx])).rjust(5, ' ')
        true_idx = str(i).rjust(5, ' ')
        pred_val = str(round(mrr_preds[idx], 5)).ljust(7, '0')
        true_val = str(round(mrr_trues[idx], 5)).ljust(7, '0')
        if abs(mrr_preds[idx] - mrr_trues[idx]) < 0.03:
            change = '~...'
        elif abs(mrr_preds[idx] - mrr_trues[idx]) < 0.1:
            change = 'miss'
        else:
            change = 'MISS'
        print(f"{pred_idx} \t {true_idx} \t {pred_val} \t {true_val} \t {change}")

def get_correlation_info(mrr_preds, mrr_trues, do_print):
    r2_mrr = r2_score(
        torch.tensor(mrr_preds),
        torch.tensor(mrr_trues),
    )
    r_mrr = torch.corrcoef(torch.tensor([mrr_preds, mrr_trues]))[0][1]
    spearman_mrrs = {}
    for k in [5, 10, 50, 100]:
        spearman_mrrs[k] = _calc_spearman(mrr_preds=mrr_preds, mrr_trues=mrr_trues, k=k)
    spearman_mrrs["All"] = _calc_spearman(mrr_preds=mrr_preds, mrr_trues=mrr_trues, k=-1)
    if do_print:
        print(f'r_mrr = {r_mrr}')
        print(f'r2_mrr = {r2_mrr}')
        for k in [5, 10, 50, 100]:
            print(f'spearmanr_mrr@{k} = {spearman_mrrs[k]}')
        print(f'spearmanr_mrr@All = {spearman_mrrs["All"]}')
    return r2_mrr, r_mrr, spearman_mrrs

def _calc_spearman(mrr_preds, mrr_trues, k):
    mrr_preds = list(sorted(mrr_preds))
    mrr_trues = list(sorted(mrr_trues))
    if k == -1:
        spearman = torch.corrcoef(torch.tensor([mrr_preds, mrr_trues]))[0][1]
    else:
        spearman = torch.corrcoef(torch.tensor([mrr_preds[-k:], mrr_trues[-k:]]))[0][1]
    return spearman

def _train_and_eval(
        model,
        twig_data,
        mrr_loss_coeffs,
        rank_dist_loss_coeffs,
        epochs,
        optimizer,
        model_name_prefix,
        checkpoint_every_n,
        do_print
):
    model.to(device)
    mrr_loss = nn.MSELoss()
    rank_dist_loss = nn.KLDivLoss(reduction='batchmean')
    print(model)

    model.train()
    print(f'Training with epochs in stages 1: {epochs[0]} and 2: {epochs[1]}')

    # Training phase 0 and 1
    for phase in (0, 1):
        for layer in model.layers_to_freeze[phase]:
            layer.requires_grad_ = False
        mrr_loss_coeff = mrr_loss_coeffs[phase]
        rank_dist_loss_coeff = rank_dist_loss_coeffs[phase]
        for t in range(epochs[phase]):
            print(f"Epoch {t+1} -- ")
            _train_epoch(
                model=model,
                twig_data=twig_data,
                mrr_loss=mrr_loss,
                mrr_loss_coeff=mrr_loss_coeff,
                rank_dist_loss=rank_dist_loss,
                rank_dist_loss_coeff=rank_dist_loss_coeff,
                optimizer=optimizer,
                do_print=do_print
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
        print("Done training phase: ", phase)

    # Testing
    model.eval()
    r2_scores = {}
    test_losses = {}
    mrr_preds_all = {}
    mrr_trues_all = {}
    for dataset_name in twig_data.dataset_names:
        print(f'Testing model with dataset {dataset_name}')
        r2_mrr, r_mrr, spearman_mrrs, test_loss, mrr_preds, mrr_trues =_eval(
            model,
            twig_data,
            dataset_name,
            mrr_loss,
            mrr_loss_coeff,
            rank_dist_loss,
            rank_dist_loss_coeff,
            mode='test',
            do_print=do_print
        )
        print(f"Done Testing dataset {dataset_name}")

        r2_scores[dataset_name] = r2_mrr
        test_losses[dataset_name] = test_loss
        mrr_preds_all[dataset_name] = mrr_preds
        mrr_trues_all[dataset_name] = mrr_trues

    return r2_scores, test_losses, mrr_preds_all, mrr_trues_all
