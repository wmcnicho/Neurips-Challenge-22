#!/bin/bash

for n in {420..650..15};
do 
    # echo $(($n))
    python3 construct_solution.py -f final_10_27_20_25_29_final_stretch_batch_64_epoch_50_embed_300 -L $n &> tau_$n.log 
done
