#!/bin/bash

start_time=$(date +%s)

g++ -std=c++11 -o runcpp generateScenario.cpp -Iinclude
./runcpp
python3 -u load_save.py
python3 draw_scenario.py
rm *bin
rm duration.txt
rm runcpp

end_time=$(date +%s)

# Calculate and display total time
total_time=$((end_time - start_time))
echo "Total time (in seconds): $total_time" > total_time.txt
