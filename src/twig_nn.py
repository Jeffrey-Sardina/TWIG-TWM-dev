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
    def __init__(self, n_struct, n_hps, n_graph):
        '''
        __init__() initialises the TWIG_Base neural network.

        The arguments it accepts are:
            - n_struct (int): the number of local structural features (around a given triple)
            - n_hps (int): the number of hyperparameter features expected in the input
            - n_graph (int): the number of global, graph-level features present in the input.

        NOTE: see the documentation on forward() for the expected order of, and constraints on, these inputs.
        NOTE: it is EXTREMELY IMPORTANT that the list `self.layers_to_freeze` be set in TWIG_Base and in any other TWIG NNs that you may create. This is because TWIG needs to know which layers of the NN should be frozen in the second phase of 2-phase training. This should typically ber all by the last 1 or 2 layers. Without this, TWIG typically cannot learn effectively.

        The values is returns are:
            - None
        '''
        super().__init__()
        self.n_struct = n_struct
        self.n_hps = n_hps
        self.n_graph = n_graph

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

        self.layers_to_freeze = [
            self.linear_struct_1,
            self.linear_struct_2,
            self.linear_hps_1,
            self.linear_integrate_1,
        ]

    def forward(self, X):
        '''
        forward() runs the foward pass of the TWIG neural network. It uses structural and hyperparameter feature inputs to predict the rank that a link predictor (using those hyperparamters) would assign to the given input link predction query.

        The arguments it accepts are:
            - X (torch,Tensor): a tensor containing feature vectors as rows. Each feature vector represents a single link prediction query. It does this by encoding:
                    - the structure around the triple (s, p, o)
                    - which side of the triple (s or o) is being corrupted in the query
                    - the hyperparameters used to learn link prediction for this triple (tehse may vary, as TWIG-I runs on sets of many hyperparameter combinations from various KGs)
                    - global graph stats that should be taken into account
                - NOTE that these have a *very* strict order. The first element in the graph must be the global graph stats. There are `self.n_graph` of these -- currently only one feature is supported here, and that feature is not using in the NN. Instead, it represent the maximum rank that a link prediction query can be assigned, and is used later in processing the output of the NN. This is also why all output is passed through a sigmooid layer -- so they are on [0:1] before this multiply. After this must come ``self.n_struct`-many local structural features around a triple (including a flag to indicate which side of the triple is being currupted) and then finally `self.n_hps`-many features describing the hyperparameters used to create the link predictor that assigned a certain rank to this link prediction query. Since TWIG wants to predict that rank using this input, the hyperparameters are needed to tell is which settings TWIG is currently simulating.
        
        The values is returns are:
            - R_pred (torch.Tensor): all predicted ranks, in order, for each input link prediction query.
        '''
        X_graph, X_struct_and_hps = X[:, :self.n_graph], X[:, self.n_graph:]
        X_struct, X_hps = X_struct_and_hps[:, :self.n_struct], X_struct_and_hps[:, self.n_struct:]

        X_struct = self.linear_struct_1(X_struct)
        X_struct = self.relu_1(X_struct)

        X_struct = self.linear_struct_2(X_struct)
        X_struct = self.relu_2(X_struct)

        X_hps = self.linear_hps_1(X_hps)
        X_hps = self.relu_3(X_hps)

        X = self.linear_integrate_1(
            torch.concat(
                [X_struct, X_hps],
                dim=1
            ),
        )
        X = self.relu_4(X)

        R_pred = self.linear_final(X)
        R_pred = self.sigmoid_final(R_pred)

        return R_pred
