#!/bin/bash

# Define the arrays
#S_ALG=('S_LONGEST' 'S_EARLIEST' 'S_RANDOM')
#C_ALG=('C_EARLIEST' 'C_RANDOM')

S_ALG=('S_LONGEST')
C_ALG=('C_EARLIEST' 'C_RANDOM')

# Loop through each element in S_ALG
for s in "${S_ALG[@]}"; do
  # Loop through each element in C_ALG
  for c in "${C_ALG[@]}"; do
    # Print the combination
    python3 main.py $s $c 1> ./result/${s}${c}logs.txt
  done
done
