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

def _do_batch(
        model,
        mrr_loss,
        mrr_loss_coeff,
        rank_dist_loss,
        rank_dist_loss_coeff,
        n_bins,
        struct_tensor_heads,
        struct_tensor_tails,
        max_rank_possible,
        hyps_tensor,
        head_rank,
        tail_rank,
        ranks_are_rescaled,
        batch_num
):  
    # get ground truth data
    rank_list_true = torch.concat(
        [head_rank, tail_rank],
        dim=0
    )
    if rank_dist_loss_coeff:
        min_rank_observed = float(torch.min(rank_list_true))
        max_rank_observed = float(torch.max(rank_list_true))
        rank_dist_true = _d_hist(
            X=rank_list_true,
            n_bins=n_bins,
            min_val=min_rank_observed,
            max_val=max_rank_observed
        )
    if mrr_loss_coeff:
        if ranks_are_rescaled:
            mrr_true = mrr_true = torch.mean(1 / (rank_list_true * max_rank_possible))
        else:
            mrr_true = torch.mean(1 / rank_list_true)
    else:
        mrr_true = None

    # get predicted data
    ranks_head_pred = model(struct_tensor_heads, hyps_tensor)
    ranks_tail_pred = model(struct_tensor_tails, hyps_tensor)
    rank_list_pred = torch.concat(
        [ranks_head_pred, ranks_tail_pred],
        dim=1
    )
    if rank_dist_loss_coeff:
        rank_dist_pred = _d_hist(
            X=rank_list_pred,
            n_bins=n_bins,
            min_val=min_rank_observed,
            max_val=max_rank_observed
        )
    if mrr_loss_coeff:
        mrr_pred = torch.mean(1 / (1 + rank_list_pred * (max_rank_possible - 1)))
    else:
        mrr_pred = None

    # compute the loss values
    if not mrr_loss_coeff:
        # use only rdl (i.e. in phase 1 of training)
        loss = rank_dist_loss_coeff * rank_dist_loss(rank_dist_pred.log(), rank_dist_true) #https://discuss.pytorch.org/t/kl-divergence-produces-negative-values/16791/5
        rdl = loss.item()
        mrrl = 0
    elif not rank_dist_loss_coeff:
        loss = mrrl = mrr_loss_coeff * mrr_loss(mrr_pred, mrr_true)
        rdl = 0
        mrrl = loss.item()
    else:
        # compute mrr loss
        mrrl = mrr_loss_coeff * mrr_loss(mrr_pred, mrr_true)
        rdl = rank_dist_loss_coeff * rank_dist_loss(rank_dist_pred.log(), rank_dist_true) #https://discuss.pytorch.org/t/kl-divergence-produces-negative-values/16791/5
        loss = mrrl + rdl
        rdl = rdl.item()
        mrrl = mrrl.item()

    return loss, mrrl, rdl, mrr_pred, mrr_true    

def _train_epoch(
        model,
        twig_data,
        mrr_loss,
        mrr_loss_coeff,
        rank_dist_loss,
        rank_dist_loss_coeff,
        n_bins,
        optimizer
):
    mode = 'train'
    batch_num = 0
    loss_sum = 0
    mrrl_sum = 0
    rdl_sum = 0
    print_batch_on = 100
    epoch_start_time = time.time()
    for dataset_name in twig_data.dataset_names:
        for run_id in twig_data.head_ranks[dataset_name]:
            for exp_id in twig_data.head_ranks[dataset_name][run_id][mode]:
                batch_start_time = time.time()

                # run batch
                struct_tensor_heads, struct_tensor_tails, hyps_tensor, head_rank, tail_rank = twig_data.get_batch(
                    dataset_name=dataset_name,
                    run_id=run_id,
                    exp_id=exp_id,
                    mode=mode
                )
                loss, mrrl, rdl, _, _ = _do_batch(
                    model=model,
                    mrr_loss=mrr_loss,
                    rank_dist_loss=rank_dist_loss,
                    mrr_loss_coeff=mrr_loss_coeff,
                    rank_dist_loss_coeff=rank_dist_loss_coeff,
                    n_bins=n_bins,
                    struct_tensor_heads=struct_tensor_heads,
                    struct_tensor_tails=struct_tensor_tails,
                    max_rank_possible=twig_data.max_ranks[dataset_name],
                    hyps_tensor=hyps_tensor,
                    head_rank=head_rank,
                    tail_rank=tail_rank,
                    ranks_are_rescaled=twig_data.normaliser.rescale_ranks,
                    batch_num=batch_num
                )

                # collect data
                loss_sum += loss.item()
                mrrl_sum += mrrl
                rdl_sum += rdl
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                batch_num += 1

                # print results
                if batch_num % print_batch_on == 0:
                    batch_end_time = time.time()
                    print(f'running bactch: {batch_num}')
                    print(f'\tloss coeffs [mrrl, rdl]: {[mrr_loss_coeff, rank_dist_loss_coeff]}')
                    loss_data = f"\t\tloss: {round(float(loss_sum / print_batch_on), 10)}\n"
                    loss_data += f"\t\tmrrl: {round(float(mrrl_sum / print_batch_on), 10)}\n"
                    loss_data += f"\t\trdl: {round(float(rdl_sum / print_batch_on), 10)}"
                    print(f'\tloss values:\n{loss_data}')
                    print(f'\tbatch time: {round(batch_end_time - batch_start_time, 3)}')
                    print()
                    loss_sum = 0
                    mrrl_sum = 0
                    rdl_sum = 0
    
    # print epoch results
    epoch_end_time = time.time()
    print('Epoch over!')
    print(f'\tepoch time: {round(epoch_end_time - epoch_start_time, 3)}')
    print()

def _eval(
        model,
        twig_data,
        dataset_name,
        mrr_loss,
        mrr_loss_coeff,
        rank_dist_loss,
        rank_dist_loss_coeff,
        n_bins,
        mode
):
    model.eval()
    test_loss = 0
    mrr_preds = []
    mrr_trues = []
    batch_num = 0
    print(f'Running eval on the {mode} set')
    test_start_time = time.time()
    with torch.no_grad():
        for run_id in twig_data.head_ranks[dataset_name]:
            for exp_id in twig_data.head_ranks[dataset_name][run_id][mode]:
                struct_tensor_heads, struct_tensor_tails, hyps_tensor, head_rank, tail_rank = twig_data.get_batch(
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
                    n_bins=n_bins,
                    struct_tensor_heads=struct_tensor_heads,
                    struct_tensor_tails=struct_tensor_tails,
                    max_rank_possible=twig_data.max_ranks[dataset_name],
                    hyps_tensor=hyps_tensor,
                    head_rank=head_rank,
                    tail_rank=tail_rank,
                    ranks_are_rescaled=twig_data.normaliser.rescale_ranks,
                    batch_num=batch_num
                )
                test_loss += loss.item()
                mrr_preds.append(float(mrr_pred))
                mrr_trues.append(float(mrr_true))
                batch_num += 1

    # validations and data collection
    print(mrr_preds)
    assert len(mrr_preds) > 1, f"TWIG should be running inference for multiple runs, not just one, here"
    r2_mrr = r2_score(
        torch.tensor(mrr_preds),
        torch.tensor(mrr_trues),
    )
    test_end_time = time.time()

    # data output
    print(f'Testing data for dataloader(s) {dataset_name}')
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
    print(f"test_loss: {test_loss}")
    print(f'\ttest time: {round(test_end_time - test_start_time, 3)}')

    return r2_mrr, test_loss, mrr_preds, mrr_trues

def _train_and_eval(
        model,
        twig_data,
        mrr_loss_coeffs,
        rank_dist_loss_coeffs,
        epochs,
        n_bins,
        optimizer,
        model_name_prefix,
        checkpoint_every_n
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
                n_bins=n_bins,
                optimizer=optimizer
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
        r2_mrr, test_loss, mrr_preds, mrr_trues =_eval(
            model,
            twig_data,
            dataset_name,
            mrr_loss,
            mrr_loss_coeff,
            rank_dist_loss,
            rank_dist_loss_coeff,
            n_bins,
            mode='test'
        )
        print(f"Done Testing dataset {dataset_name}")

        r2_scores[dataset_name] = r2_mrr
        test_losses[dataset_name] = test_loss
        mrr_preds_all[dataset_name] = mrr_preds
        mrr_trues_all[dataset_name] = mrr_trues

    return r2_scores, test_losses, mrr_preds_all, mrr_trues_all
