#!/bin/bash

GRAINSET_DIR='/home/sergei/Downloads/GrainSetData/wheat'


cd $GRAINSET_DIR
mkdir val
cd val
for k in 0_NOR '1_F&S' 2_SD 3_MY 4_AP 5_BN 6_BP 7_IM
do
    mkdir $k
done

