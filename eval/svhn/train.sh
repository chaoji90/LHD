#!/bin/bash

./train_ori.sh $1 &> $2"_ori.log"
./train_1M.sh $1 &> $2"_1M.log"
./train_3op.sh $1 &> $2"_3op.log"
./train_4out.sh $1 &> $2"_4out.log"


