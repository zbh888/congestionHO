#!/bin/bash

# Define the arrays
#S_ALG=('S_LONGEST' 'S_EARLIEST' 'S_RANDOM')
#C_ALG=('C_EARLIEST' 'C_RANDOM')
#U_ALG=('UE_LONGEST' 'UE_RANDOM')
#Max_ACC=('3')

S_ALG=('S_LONGEST')
C_ALG=('C_EARLIEST' 'C_RANDOM')
U_ALG=('UE_LONGEST')
Max_ACC=('3')

# Loop through each element in S_ALG
for s in "${S_ALG[@]}"; do
  # Loop through each element in C_ALG
  for c in "${C_ALG[@]}"; do
    for u in "${U_ALG[@]}"; do
      for a in "${Max_ACC[@]}"; do
        python3 main.py $s $c $u $a 1> ./result/${s}${c}${u}${a}logs.txt
      done
    done
  done
done

zip -r result.zip result/*
