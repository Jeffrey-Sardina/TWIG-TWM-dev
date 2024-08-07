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
    bins = torch.linspace(start=min_val, end=max_val, steps=n_bins+1)[1:]
    freqs = torch.zeros(size=(n_bins,)).to(device)
    last_val = None
    sharpness = 1
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
        rank_list_true,
        mrr_true,
        rank_dist_true,
        do_print
    ):  
    '''
    Optimisations:
        precalc rank_dist_true and mrr_true
        put rank list in [s..., o...] order in the data loader
        should remove a third of current batch time on average
    '''
    # get ground truth data
    # rank_dist_true = _d_hist(
    #     X=rank_list_true,
    #     n_bins=n_bins,
    #     min_val=hist_min_val,
    #     max_val=hist_max_val
    # )
    # mrr_true = torch.mean(1 / (rank_list_true * max_rank_possible))

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
    if mrr_loss_coeff > 0:
        mrrl = mrr_loss_coeff * mrr_loss(mrr_pred, mrr_true)
    else:
        mrrl = torch.tensor(0.0)
    rdl = rank_dist_loss_coeff * rank_dist_loss(rank_dist_pred.log(), rank_dist_true) #https://discuss.pytorch.org/t/kl-divergence-produces-negative-values/16791/5
    loss = mrrl + rdl

    # print state
    if do_print:
        pred_mean = str(round(float(torch.mean(rank_list_pred)), 3)).ljust(5, '0')
        true_mean = str(round(float(torch.mean(rank_list_true)), 3)).ljust(5, '0')
        pred_std = str(round(float(torch.std(rank_list_pred)), 3)).ljust(5, '0')
        true_std = str(round(float(torch.std(rank_list_true)), 3)).ljust(5, '0')
        mrr_pred_str = str(round(mrr_pred.item(), 3)).ljust(5, '0')
        mrr_true_str = str(round(mrr_true.item(), 3)).ljust(5, '0')
        print(f'rank avg (pred, true): {pred_mean}, {true_mean}')
        print(f'rank std (pred, true): {pred_std}, {true_std}')
        print(f'mrr vals (pred, true): {mrr_pred_str}, {mrr_true_str}')
        # _plot_hist(
        #     dist1=rank_list_true.cpu(),
        #     dist2=rank_list_pred.detach().cpu(),
        #     n_bins=n_bins
        # )

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
    batch_num = 0
    mrrl_sum = 0
    rdl_sum = 0
    print_batch_on = 500
    epoch_start_time = time.time()
    epoch_batches = twig_data.get_train_epoch(shuffle=False)
    for dataset_name, run_id, exp_id in epoch_batches:
        # print state
        if batch_num % print_batch_on == 0:
            print(f'running batch: {batch_num}')
            
        # load batch data
        struct_tensor, hyps_tensor, rank_list_true, mrr_true, rank_dist_true = twig_data.get_batch(
            dataset_name=dataset_name,
            run_id=run_id,
            exp_id=exp_id,
            mode=mode
        )

        # run batch
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
            rank_list_true=rank_list_true,
            mrr_true=mrr_true,
            rank_dist_true=rank_dist_true,
            do_print=do_print and batch_num % print_batch_on == 0
        )

        # collect data
        mrrl_sum += mrrl.item()
        rdl_sum += rdl.item()

        # backprop
        loss.backward()
        # if do_print and batch_num % print_batch_on == 0:
        #     if batch_num > 0:
        #         print((model.linear_struct_1.weight.grad))
        #         print((model.linear_struct_2.weight.grad))
        #         print((model.linear_hps_1.weight.grad))
        #         print((model.linear_integrate_1.weight.grad))
        #         print((model.linear_final.weight.grad))
        optimizer.step()
        optimizer.zero_grad()

        # print results
        if do_print and batch_num % print_batch_on == 0:
            divisor = print_batch_on if batch_num > 0 else 1
            mrrl_print = round(float(mrrl / divisor), 10)
            rdl_print = round(float(rdl / divisor), 10)
            print(f'losses (mrrl, rdl): {mrrl_print}, {rdl_print}')
            print()
        batch_num += 1

    # print epoch results
    epoch_end_time = time.time()
    print('Epoch over!')
    print(f'epoch time: {round(epoch_end_time - epoch_start_time, 3)}')
    loss_data = f"\tmrrl: {round(float(mrrl_sum / batch_num), 10)}\n"
    loss_data += f"\trdl: {round(float(rdl_sum / batch_num), 10)}"
    print(f'loss values:\n{loss_data}')
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

            struct_tensor, hyps_tensor, rank_list_true, mrr_true, rank_dist_true = twig_data.get_batch(
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
                rank_list_true=rank_list_true,
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

    # validations and data collection
    assert len(mrr_preds) > 1, f"TWIG should be running inference for multiple runs, not just one, here"
    r2_mrr = r2_score(
        torch.tensor(mrr_preds),
        torch.tensor(mrr_trues),
    )
    test_end_time = time.time()

    # data output
    print(f'Evaluation for {dataset_name} on the {mode} set')
    print("=" * 42)
    print()
    print("Predicted MRRs \t True MRRs")
    print('-' * 42)
    for i in range(len(mrr_preds)):
        print(f"{mrr_preds[i]} \t {mrr_trues[i]}")
    print()
    print(f'r_mrr = {torch.corrcoef(torch.tensor([mrr_preds, mrr_trues]))}')
    print(f'r2_mrr = {r2_mrr}')
    print(f"average test loss: {test_loss / batch_num}")
    print(f'\ttest time: {round(test_end_time - test_start_time, 3)}')

    return r2_mrr, test_loss, mrr_preds, mrr_trues

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
        r2_mrr, test_loss, mrr_preds, mrr_trues =_eval(
            model,
            twig_data,
            dataset_name,
            mrr_loss,
            mrr_loss_coeff,
            rank_dist_loss,
            rank_dist_loss_coeff,
            mode='test', # TODO -- train does not learn either
            do_print=do_print
        )
        print(f"Done Testing dataset {dataset_name}")

        r2_scores[dataset_name] = r2_mrr
        test_losses[dataset_name] = test_loss
        mrr_preds_all[dataset_name] = mrr_preds
        mrr_trues_all[dataset_name] = mrr_trues

    return r2_scores, test_losses, mrr_preds_all, mrr_trues_all
