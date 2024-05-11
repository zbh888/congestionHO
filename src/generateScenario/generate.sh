#!/bin/bash
g++ -std=c++11 -o runcpp generateScenario.cpp -Iinclude
./runcpp
python3 load_save.py
rm *bin
rm runcpp
