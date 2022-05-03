#!/bin/bash
#./train_search.sh train_search_concate_sum_five_rd.py arch_1

COUNTER=0
#while :
while [ $COUNTER -lt 10 ]
do
    python3 ./$1 --arch_file $2
    python3 ./train.py --arch_file $2.pl
    let COUNTER+=1
done
