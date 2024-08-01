class Early_Stopper:
    def __init__(self, start_epoch, patience, mode, precision, higher_is_better):
        '''
        __init__() creates the Early_Stopper object. This class serves to provide an simple way to define an early-stopping protocol for TWIG.

        NOTE that epochs are 1-indexed, so the first epoch is epoch 1.

        The arguments it accepts are:
            - start_epoch (int): the number of epochs to wait before early stopping metrics will be considered. This means that prevous MRR values are not collected before this value, and as such at they cannot contribute to early stopping.
            - patience (int): the number of epochs to wait after seeing what would otherwise be a stop-condition before the early stopping is actually applied. This is inclusive: so if it is 10, and early stopping is done very 5 epochs, it will make a choice based on the current eval and the one previous validation cycle. If it is 15, in the above sample, it will make a choice based on the last 2 validation cycles, etc.
            - mode (str): the mode of early stopping to use. Options are as follows:
                - "on-falter" -- trigger early stopping the first time a validation result does not get better than a previous result
                - "never" -- never do early stopping
            - precision (int): the number of decimal points to consider when testing to change in performance.
            - higher_is_better (bool): whether the metric that the early stopper is used is expected to increase during learning

        The values it returns are:
            - None
        '''
        self.start_epoch = start_epoch
        self.patience = patience
        self.mode = mode
        self.precision = precision
        self.higher_is_better = higher_is_better
        self.validation_results = []
        self.valid_modes = ("on-falter", "never")

        # input validation
        assert type(start_epoch) == int and start_epoch >= 0
        assert type(patience) == int
        assert type(precision) == int and precision >= 1
        assert self.mode in self.valid_modes, f"Unknown mode: {self.mode}. Mode must be one of: {self.valid_modes}"

    def assess_validation_result(self, curr_epoch, curr_performance):
        '''
        assess_validation_result() examines the current validation result and returns a bool that thells TWIG whether it should trigger early stopping or not.

        The arguments is accepts are:
            - epoch_num (int): the current epoch
            - curr_performance (float): the performance metric (i.e. loss, R2, etc) value acheived on the validation round for the given (current) epoch

        The values it returns are:
            - should_stop (bool): True if TWIG-I should trigger early stopping, False otherwise
        '''
        should_stop = False
        if self.mode == "never":
            return should_stop
        
        curr_performance = int(round((float(curr_performance) * 10  ** self.precision), 0))
        if len(self.validation_results) > 0:
            if self.mode == "on-falter":
                if self.is_faltering(curr_epoch, curr_performance):
                    should_stop = True
            else:
                assert False, f"Unknown mode: {self.mode}. Mode must be one of: {self.valid_modes}"
        if curr_epoch > self.start_epoch:
            self.validation_results.append((curr_epoch, curr_performance))
        return should_stop

    def is_faltering(self, curr_epoch, curr_performance):
        '''
        is_faltering is a helper function that returs true is a model is faltering (based on the arguments the Early Stopper was given) and False otherwise

        The arguments it accepts are:
            - curr_epoch (int): the current epoch the model is at
            - curr_performance (float): Performance value (i.e. loss, R2, etc) the model acheived on the validation set this epoch

        The values it returns are:
            - faltering (Bool): Whether the model is faltering in learning or not
        '''
        if not self.higher_is_better:
            curr_performance = -curr_performance
        faltering = True
        for epoch, val in reversed(self.validation_results):
            if curr_epoch - epoch <= self.patience:
                if val < curr_performance:
                    faltering = False
            else:
                break
        return faltering
