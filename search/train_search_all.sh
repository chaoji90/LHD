#!/bin/bash

pushd GDAS
./train_search.sh train_search_gdas.py svhn &> gdas_svhn.log
./train_search.sh train_search_gdas.py cifar100 &> gdas_cifar100.log
./train_search.sh train_search_gdas.py cifar10 &> gdas_cifar10.log
popd

pushd DrNAS
./train_search.sh train_search_drnas.py cifar100 &> drnas_cifar100.log
./train_search.sh train_search_drnas.py svhn &> drnas_svhn.log
./train_search.sh train_search_drnas.py cifar10 &> drnas_cifar10.log
popd 

pushd Random
./train_search.sh train_search_rd.py cifar100 &> random_cifar100.log
./train_search.sh train_search_rd.py svhn &> random_svhn.log
./train_search.sh train_search_rd.py cifar10 &> random_cifar10.log
popd 

pushd DARTS
./train_search.sh train_search_darts.py cifar100 &> darts_cifar100.log
./train_search.sh train_search_darts.py svhn &> darts_svhn.log
./train_search.sh train_search_darts.py cifar10 &> darts_cifar10.log
popd 

pushd GAEA
./train_search.sh train_search_gaea_erm.py cifar100 &> gaea_erm_cifar100.log
./train_search.sh train_search_gaea_erm.py svhn &> gaea_erm_svhn.log
./train_search.sh train_search_gaea_erm.py cifar10 &> gaea_erm_cifar10.log
./train_search.sh train_search_gaea_bilevel.py cifar100 &> gaea_bilevel_cifar100.log
./train_search.sh train_search_gaea_bilevel.py svhn &> gaea_bilevel_svhn.log
./train_search.sh train_search_gaea_bilevel.py cifar10 &> gaea_bilevel_cifar10.log
popd 

pushd DARTS-
./train_search.sh train_search_darts_minus.py cifar100 &> darts_minus_cifar100.log
./train_search.sh train_search_darts_minus.py svhn &> darts_minus_svhn.log
./train_search.sh train_search_darts_minus.py cifar10 &> darts_minus_cifar10.log
popd 

pushd PC-DARTS
./train_search.sh train_search_pcdarts.py cifar100 &> pcdarts_cifar100.log
./train_search.sh train_search_pcdarts.py svhn &> pcdarts_svhn.log
./train_search.sh train_search_pcdarts.py cifar10 &> pcdarts_cifar10.log
popd 

pushd SP-DARTS
./train_search.sh train_search_spdarts.py cifar100 &> spdarts_cifar100.log
./train_search.sh train_search_spdarts.py svhn &> spdarts_svhn.log
./train_search.sh train_search_spdarts.py cifar10 &> spdarts_cifar10.log
popd 

pushd MiLeNAS
./train_search.sh train_search_milenas.py cifar100 &> milenas_cifar100.log
./train_search.sh train_search_milenas.py svhn &> milenas_svhn.log
./train_search.sh train_search_milenas.py cifar10 &> milenas_cifar10.log
popd 
