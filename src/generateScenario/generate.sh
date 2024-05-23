#!/bin/bash
g++ -std=c++11 -o runcpp generateScenario.cpp -Iinclude
./runcpp
python3 load_save.py
python3 draw_scenario.py
rm *bin
rm duration.txt
rm runcpp
