#!/bin/bash

# Record start time
start_time=$(date +%s)

# Compile the C++ code
g++ -std=c++11 -o runcpp generateScenario.cpp -Iinclude

# Measure memory usage while running the C++ executable
/usr/bin/time -v ./runcpp 2> runcpp_memory_usage.txt

# Measure memory usage while running the first Python script
/usr/bin/time -v python3 load_save.py 2> load_save_memory_usage.txt

# Measure memory usage while running the second Python script
/usr/bin/time -v python3 draw_scenario.py 2> draw_scenario_memory_usage.txt

# Clean up generated files
rm *bin
rm duration.txt
rm runcpp

# Record end time
end_time=$(date +%s)

# Calculate and display total time
total_time=$((end_time - start_time))
echo "Total time (in seconds): $total_time" > total_time.txt
