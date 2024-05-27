#!/bin/bash

cd result
shopt -s extglob
rm -rf !(result.ipynb|README.md)
cd ..
