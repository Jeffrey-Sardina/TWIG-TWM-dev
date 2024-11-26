#!/bin/bash

# python -u ft-albations.py TransE > transe-aggr.table
# python -u ft-albations.py DistMult > distmult-aggr.table

# python -u ft-albations-cross-kg.py ComplEx > complex-aggr-crosskg-indiv.table 1 0
# python -u ft-albations-cross-kg.py DistMult > distmult-aggr-crosskg.table 0 1
# python -u ft-albations-cross-kg.py DistMult > distmult-aggr-crosskg-indiv.table 1 0
# python -u ft-albations-cross-kg.py TransE > transe-aggr-crosskg.table 0 1

python -u ft-albations-cross-kg.py TransE > transe-aggr-crosskg-indiv.table 1 0
