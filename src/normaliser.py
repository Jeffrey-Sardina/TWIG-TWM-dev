import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Normaliser:
    def __init__(self, method, structs, hyps, max_ranks):
        self.valid_norm_methods = ('zscore', 'minmax', 'none')
        assert method in self.valid_norm_methods, f"normalisation method must be one of {self.valid_norm_methods} but is {method}"
        self.method = method
        self.max_ranks = max_ranks

        # calc num lp quries we are simulatuig per dataset. 2*structs since structs holds the number of triples
        self.head_flags = {}
        self.tail_flags = {}
        for dataset_name in structs:
            num_triples = structs[dataset_name].shape[0]
            self.head_flags[dataset_name], self.tail_flags[dataset_name] = self._get_lp_side_flags(
                num_triples=num_triples
            )

        if method == 'zscore':
            # get the average of all structural data values
            struct_avg = None
            num_samples = 0.
            for dataset_name in structs:
                struct_data = structs[dataset_name]
                num_samples += struct_data.shape[0]
                if struct_avg is None:
                    struct_avg = torch.sum(struct_data, dim=0)
                else:
                    struct_avg += torch.sum(struct_data, dim=0)
            struct_avg /= num_samples

            # get the standard deviation of all structural data values
            struct_std = None
            for dataset_name in structs:
                struct_data = structs[dataset_name]
                if struct_std is None:
                    struct_std = torch.sum(
                        (struct_data - struct_avg) ** 2,
                        dim=0
                    )
                else:
                    struct_std += torch.sum(
                        (struct_data - struct_avg) ** 2,
                        dim=0
                    )
            struct_std = torch.sqrt(
                (1 / (num_samples - 1)) * struct_std
            )

            # get the average of all hyperparameter data values
            hyp_avg = None
            num_samples = 0
            for exp_id in hyps['train']:
                hyp_data = hyps['train'][exp_id]
                num_samples += 1 # hyp_data.shape[0]
                if hyp_avg is None:
                    hyp_avg = hyp_data
                else:
                    hyp_avg += hyp_data
            hyp_avg /= num_samples

            # get the standard deviation of all hyperparameter data values
            hyp_std = None
            for exp_id in hyps['train']:
                hyp_data = hyps['train'][exp_id]
                if hyp_std is None:
                    hyp_std = (hyp_data - hyp_avg) ** 2
                else:
                    hyp_std += (hyp_data - hyp_avg) ** 2
            hyp_std = torch.sqrt(
                (1 / (num_samples - 1)) * hyp_std
            )

            # save the resultant values for use in normalisation
            self.hyp_avg = hyp_avg
            self.hyp_std = hyp_std
            self.struct_avg = struct_avg
            self.struct_std = struct_std
        elif method == 'minmax':
            # get the min and max of all structural values
            struct_min = None
            struct_max = None
            for dataset_name in structs:
                struct_data = structs[dataset_name]
                if struct_min is None:
                    struct_min = torch.min(struct_data, dim=0).values
                else:
                    struct_min = torch.min(
                        torch.stack(
                            [torch.min(struct_data, dim=0).values, struct_min]
                        ),
                        dim=0
                    ).values
                if struct_max is None:
                    struct_max = torch.max(struct_data, dim=0).values
                else:
                    struct_max = torch.max(
                        torch.stack(
                            [torch.max(struct_data, dim=0).values, struct_max]
                        ),
                        dim=0
                    ).values

            # get the min and max of allo hyperparameters values
            hyp_min = None
            hyp_max = None
            for exp_id in hyps['train']:
                hyp_data = hyps['train'][exp_id]
                if hyp_min is None:
                    hyp_min = hyp_data
                else:
                    hyp_min = torch.min(
                        torch.stack(
                            [hyp_data, hyp_min]
                        ),
                        dim=0
                    ).values
                if hyp_max is None:
                    hyp_max = hyp_data
                else:
                    hyp_max = torch.max(
                        torch.stack(
                            [hyp_data, hyp_max]
                        ),
                        dim=0
                    ).values

            self.struct_min = struct_min
            self.struct_max = struct_max
            self.hyp_min = hyp_min
            self.hyp_max = hyp_max
            print(struct_min.shape)
            print(struct_max.shape)
            print(hyp_min.shape)
            print(hyp_max.shape)
        elif method == 'none':
            pass
        else:
            assert False, f"normalisation method must be one of {self.valid_norm_methods} but is {method}"

    def _zscore_norm_func(self, structs, hyps):
        structs_norm = {}
        for dataset_name in structs:
            struct_data = structs[dataset_name]
            struct_norm = (struct_data - self.struct_avg) / self.struct_std
            struct_norm = torch.nan_to_num(struct_norm, nan=0.0, posinf=0.0, neginf=0.0) # if we had nans (i.e. min = max) set them all to 0 (average)
            structs_norm[dataset_name] = struct_norm

        hyps_norm = {}
        for mode in hyps:
            hyps_norm[mode] = {}
            for exp_id in hyps[mode]:
                hyp_norm = (hyps[mode][exp_id] - self.hyp_avg) / self.hyp_std
                hyp_norm = torch.nan_to_num(hyp_norm, nan=0.0, posinf=0.0, neginf=0.0) # if we had nans (i.e. min = max) set them all to 0 (average)
                hyps_norm[mode][exp_id] = hyp_norm
        
        return structs_norm, hyps_norm

    def _minmax_score_func(self, structs, hyps):
        structs_norm = {}
        for dataset_name in structs:
            struct_data = structs[dataset_name]
            struct_norm = (struct_data - self.struct_min) / (self.struct_max - self.struct_min)
            struct_norm = torch.nan_to_num(struct_norm, nan=0.0, posinf=0.0, neginf=0.0) # if we had nans (i.e. min = max) set them all to 0 (average)
            structs_norm[dataset_name] = struct_norm

        hyps_norm = {}
        for mode in hyps:
            hyps_norm[mode] = {}
            for exp_id in hyps[mode]:
                hyp_norm = (hyps[mode][exp_id] - self.hyp_min) / (self.hyp_max - self.hyp_min)
                hyp_norm = torch.nan_to_num(hyp_norm, nan=0.0, posinf=0.0, neginf=0.0) # if we had nans (i.e. min = max) set them all to 0 (average)
                hyps_norm[mode][exp_id] = hyp_norm
            
        return structs_norm, hyps_norm

    def _do_rescale_ranks(self, ranks):
        rescaled_ranks = {}
        for dataset_name in ranks:
            rescaled_ranks[dataset_name] = {}
            for run_id in ranks[dataset_name]:
                rescaled_ranks[dataset_name][run_id] = {}
                for mode in ranks[dataset_name][run_id]:
                    rescaled_ranks[dataset_name][run_id][mode] = {}
                    for exp_id in ranks[dataset_name][run_id][mode]:
                        rescaled_rank = ranks[dataset_name][run_id][mode][exp_id] / self.max_ranks[dataset_name]
                        rescaled_ranks[dataset_name][run_id][mode][exp_id] = rescaled_rank
        return rescaled_ranks

    def normalise(self, structs, hyps, head_ranks, tail_ranks):
        if self.method == 'zscore':
            structs_norm, hyps_norm = self._zscore_norm_func(
                structs=structs,
                hyps=hyps
            )
        elif self.method == 'minmax':
            structs_norm, hyps_norm = self._minmax_score_func(
                structs=structs,
                hyps=hyps
            )
        elif self.method == 'none':
            structs_norm, hyps_norm = structs, hyps
        else:
            assert False, f"normalisation method must be one of {self.valid_norm_methods} but is {self.method}"

        head_ranks_norm = self._do_rescale_ranks(head_ranks)
        tail_ranks_norm = self._do_rescale_ranks(tail_ranks)
        
        return structs_norm, hyps_norm, head_ranks_norm, tail_ranks_norm

    def _get_lp_side_flags(self, num_triples):
        if self.method == 'zscore':
            flag_mean = torch.tensor([0.5], dtype=torch.float32, device=device) # mean if half are 1s and half are 0s
            flag_std = torch.tensor([0.5002501876563868], dtype=torch.float32, device=device) # stdev if half are 1s and half are 0s
            flag_true = (1 - flag_mean) / flag_std
            flag_false = (0 - flag_mean) / flag_std
        elif self.method == 'minmax':
            flag_true = torch.tensor([1], dtype=torch.float32, device=device)
            flag_false = torch.tensor([0], dtype=torch.float32, device=device)
        elif self.method == 'none':
            flag_true = torch.tensor([1], dtype=torch.float32, device=device)
            flag_false = torch.tensor([0], dtype=torch.float32, device=device)
        else:
            assert False, f"normalisation method must be one of {self.valid_norm_methods} but is {self.method}"
        flag_true = flag_true.repeat(num_triples).unsqueeze(1)
        flag_false = flag_false.repeat(num_triples).unsqueeze(1)
        return flag_true, flag_false
