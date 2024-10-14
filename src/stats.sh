# !/bin/bash

for kg in umls codexsmall dbpedia50 kinships openea
do
    for kgem in complex transe distmult
    do
        python -u kge_output_analysis.py $kg $kgem 2.1 > results_save/kgem_structural_stats/$kg-$kgem.stats.log
        exit
    done
done
