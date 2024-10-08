# external imports
import torch
from torch import nn
import torch.nn.functional as F

'''
==========================
Neural Network Definitions
==========================
'''
class TWIG_Base(nn.Module):
    '''
    TWIG_Base is the base TWIG model that has been (so far) most heavily tested and allows for running TWIG on multiple KGs at the same time. It is currently the only version of TWIG included in the library.
    '''
    def __init__(self, n_struct, n_hps):
        super().__init__()
        self.n_struct = n_struct
        self.n_hps = n_hps

        # struct parts are from the version with no hps included
        # we now want to cinclude hps, hwoever
        self.linear_struct_1 = nn.Linear(
            in_features=n_struct,
            out_features=10
        )
        self.relu_1 = nn.ReLU()
        
        self.linear_struct_2 = nn.Linear(
            in_features=10,
            out_features=10
        )
        self.relu_2 = nn.ReLU()

        self.linear_hps_1 = nn.Linear(
            in_features=n_hps,
            out_features=6
        )
        self.relu_3 = nn.ReLU()

        self.linear_integrate_1 = nn.Linear(
            in_features=6 + 10,
            out_features=8
        )
        self.relu_4 = nn.ReLU()

        self.linear_final = nn.Linear(
            in_features=8,
            out_features=1
        )
        self.sigmoid_final = nn.Sigmoid()

        self.layers_to_freeze = [None, None]
        self.layers_to_freeze[0] = [] # phase 0
        self.layers_to_freeze[1] = [  # phase 1
            self.linear_struct_1,
            self.linear_struct_2,
            self.linear_hps_1,
            self.linear_integrate_1,
        ]

    def forward(self, X_struct, X_hps):
        X_struct = self.linear_struct_1(X_struct)
        X_struct = self.relu_1(X_struct)

        X_struct = self.linear_struct_2(X_struct)
        X_struct = self.relu_2(X_struct)

        X_hps = self.linear_hps_1(X_hps)
        X_hps = self.relu_3(X_hps)

        # X_hps = X_hps.repeat(X_struct.shape[0], 1)
        X = self.linear_integrate_1(
            torch.concat(
                [X_struct, X_hps],
                dim=1
            )
        )
        X = self.relu_4(X)

        R_pred = self.linear_final(X)
        R_pred = self.sigmoid_final(R_pred)

        return R_pred
    
class TWIG_Large(nn.Module):
    def __init__(self, n_struct, n_hps):
        super().__init__()
        self.n_struct = n_struct
        self.n_hps = n_hps

        # struct parts are from the version with no hps included
        # we now want to cinclude hps, hwoever
        self.linear_struct_1 = nn.Linear(
            in_features=n_struct,
            out_features=10
        )
        self.relu_1 = nn.ReLU()
        
        self.linear_struct_2 = nn.Linear(
            in_features=10,
            out_features=10
        )
        self.relu_2 = nn.ReLU()

        self.linear_hps_1 = nn.Linear(
            in_features=n_hps,
            out_features=6
        )
        self.relu_3 = nn.ReLU()

        self.linear_integrate_1 = nn.Linear(
            in_features=6 + 10,
            out_features=8
        )
        self.relu_4 = nn.ReLU()

        self.linear_final_1 = nn.Linear(
            in_features=8,
            out_features=8
        )
        self.relu_5 = nn.ReLU()

        self.linear_final_2 = nn.Linear(
            in_features=8,
            out_features=1
        )
        self.sigmoid_final = nn.Sigmoid()

        self.layers_to_freeze = [None, None]
        self.layers_to_freeze[0] = [] # phase 0
        self.layers_to_freeze[1] = [  # phase 1
            self.linear_struct_1,
            self.linear_struct_2,
            self.linear_hps_1,
            self.linear_integrate_1,
        ]

    def forward(self, X_struct, X_hps):
        X_struct = self.linear_struct_1(X_struct)
        X_struct = self.relu_1(X_struct)

        X_struct = self.linear_struct_2(X_struct)
        X_struct = self.relu_2(X_struct)

        X_hps = self.linear_hps_1(X_hps)
        X_hps = self.relu_3(X_hps)

        # X_hps = X_hps.repeat(X_struct.shape[0], 1)
        X = self.linear_integrate_1(
            torch.concat(
                [X_struct, X_hps],
                dim=1
            )
        )
        X = self.relu_4(X)

        X = self.linear_final_1(X)
        X = self.relu_5(X)

        R_pred = self.linear_final_2(X)
        R_pred = self.sigmoid_final(R_pred)

        return R_pred

class TWIG_Small(nn.Module):
    def __init__(self, n_struct, n_hps):
        super().__init__()
        self.n_struct = n_struct
        self.n_hps = n_hps

        # struct parts are from the version with no hps included
        # we now want to cinclude hps, hwoever
        self.linear_struct_1 = nn.Linear(
            in_features=n_struct,
            out_features=10
        )
        self.relu_1 = nn.ReLU()

        self.linear_hps_1 = nn.Linear(
            in_features=n_hps,
            out_features=6
        )
        self.relu_3 = nn.ReLU()

        self.linear_integrate_1 = nn.Linear(
            in_features=6 + 10,
            out_features=8
        )
        self.relu_4 = nn.ReLU()

        self.linear_final = nn.Linear(
            in_features=8,
            out_features=1
        )
        self.sigmoid_final = nn.Sigmoid()

        self.layers_to_freeze = [None, None]
        self.layers_to_freeze[0] = [] # phase 0
        self.layers_to_freeze[1] = [  # phase 1
            self.linear_struct_1,
            self.linear_hps_1,
            self.linear_integrate_1,
        ]

    def forward(self, X_struct, X_hps):
        X_struct = self.linear_struct_1(X_struct)
        X_struct = self.relu_1(X_struct)

        X_hps = self.linear_hps_1(X_hps)
        X_hps = self.relu_3(X_hps)

        # X_hps = X_hps.repeat(X_struct.shape[0], 1)
        X = self.linear_integrate_1(
            torch.concat(
                [X_struct, X_hps],
                dim=1
            )
        )
        X = self.relu_4(X)

        R_pred = self.linear_final(X)
        R_pred = self.sigmoid_final(R_pred)

        return R_pred

class TWIG_Tiny(nn.Module):
    def __init__(self, n_struct, n_hps):
        super().__init__()
        self.n_struct = n_struct
        self.n_hps = n_hps

        # struct parts are from the version with no hps included
        # we now want to cinclude hps, hwoever
        self.linear_struct_1 = nn.Linear(
            in_features=n_struct,
            out_features=10
        )
        self.relu_1 = nn.ReLU()

        self.linear_hps_1 = nn.Linear(
            in_features=n_hps,
            out_features=6
        )
        self.relu_3 = nn.ReLU()

        self.linear_integrate_1 = nn.Linear(
            in_features=6 + 10,
            out_features=1
        )
        self.sigmoid_final = nn.Sigmoid()

        self.layers_to_freeze = [None, None]
        self.layers_to_freeze[0] = [] # phase 0
        self.layers_to_freeze[1] = [  # phase 1
            self.linear_struct_1,
            self.linear_hps_1,
        ]

    def forward(self, X_struct, X_hps):
        X_struct = self.linear_struct_1(X_struct)
        X_struct = self.relu_1(X_struct)

        X_hps = self.linear_hps_1(X_hps)
        X_hps = self.relu_3(X_hps)

        # X_hps = X_hps.repeat(X_struct.shape[0], 1)
        X = self.linear_integrate_1(
            torch.concat(
                [X_struct, X_hps],
                dim=1
            )
        )
        R_pred = self.sigmoid_final(X)

        return R_pred
