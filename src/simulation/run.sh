#!/bin/bash

# Define the arrays
SS_ALG=('SS_LONGEST' 'SS_RANDOM')
C_ALG=('C_EARLIEST' 'C_RANDOM')
SD_ALG=('SD_LONGEST' 'SD_EARLIEST' 'SD_RANDOM')
Max_ACC=('56')

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
