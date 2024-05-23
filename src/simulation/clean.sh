#!/bin/bash

# Loop through all items in the current directory
for item in *; do
  # Check if the item is a directory
  if [ -d "$item" ]; then
    # Remove the directory and its contents
    rm -rf "$item"
    echo "Removed directory: $item"
  fi
done
