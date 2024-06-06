#!/bin/bash

#S_ALG=('SOURCE_ALG_OUR')
#C_ALG=('CANDIDATE_ALG_OUR')
#U_ALG=('UE_LONGEST')
#Max_ACC=('4')

# Define the arrays
SS_ALG=('SS_LONGEST')
C_ALG=('C_OUR')
SD_ALG=('SD_OUR')
Max_ACC=('4')

#SD_ALG=('SD_LONGEST')
#C_ALG=('C_RANDOM')
#SS_ALG=('SS_LONGEST')
#Max_ACC=('8' '12')

for ss in "${SS_ALG[@]}"; do
  for c in "${C_ALG[@]}"; do
    for sd in "${SD_ALG[@]}"; do
      for a in "${Max_ACC[@]}"; do
        python3 main.py $sd $c $ss $a 1> ./result/${sd}${c}${ss}${a}logs.txt
      done
    done
  done
done

zip -r result.zip result/*
