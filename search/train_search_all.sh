#!/bin/bash

pushd GDAS
./train_search.sh train_search_gdas.py svhn &> gdas_svhn.log
./train_search.sh train_search_gdas.py cifar100 &> gdas_cifar100.log
./train_search.sh train_search_gdas.py cifar10 &> gdas_cifar10.log
popd

pushd drnas
./train_search.sh train_search_concate_sum_five_dr.py cifar100 &> drnas_cifar100.log
./train_search.sh train_search_concate_sum_five_dr.py svhn &> drnas_svhn.log
./train_search.sh train_search_concate_sum_five_dr.py cifar10 &> drnas_cifar10.log
popd 

pushd random
./train_search.sh train_search_concate_sum_five_rd.py cifar100 &> random_cifar100.log
./train_search.sh train_search_concate_sum_five_rd.py svhn &> random_svhn.log
./train_search.sh train_search_concate_sum_five_rd.py cifar10 &> random_cifar10.log
popd 

#pushd gaea
#./train_search.sh train_search_concate_sum_five_gaea.py svhn &> gaea_silevel_svhn.log
#./train_search.sh train_search_concate_sum_five_gaea_bilevel.py svhn &> gaea_bilevel_svhn.log
#popd

#./train_search.sh train_search_concate_sum_five_mixlevel.py svhn &> darts_mixlevel_svhn.log
#./train_search.sh train_search_concate_sum_five_v1.py svhn &> darts_bilevel_svhn.log

#pushd sp-darts
#./train_search.sh train_search_concate_sum_five_sp.py svhn &> spdarts_svhn.log
#popd

#pushd PC-DARTS
#./train_search.sh train_search_concate_sum_five_pc.py svhn &> pcdarts_svhn.log
#popd
