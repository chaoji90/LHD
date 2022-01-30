#!/bin/bash

function read_dir(){
for file in `ls $1` #注意此处这是两个反引号，表示运行系统命令
do
 #if [ -d $1"/"$file ] #注意此处之间一定要加上空格，否则会报错
 #then
 #read_dir $1"/"$file
 #else
 if [ "${file##*.}" == "pl" ]       #  this is the snag
 then
     echo $1"/"$file #在此处处理文件即可
     python train.py --arch_file $1"/"$file
 fi
 #fi
done
} 
#读取第一个参数
read_dir $1

#python train.py --arch_file 20211004-093326-concate-sum-darts.pl --node_threshold 2 --threshold_element_wise 0.2
