#!/bin/bash

cd result
shopt -s extglob
rm -rf !(result.ipynb|README.md)
cd ..

read -p "Do you want to remove the zip file? (y/n): " response

case "$response" in
    yes|y|Y|Yes|YES)
        echo "Removed zip file"
        rm result.zip
        # Add the commands to proceed here
        ;;
    no|n|N|No|NO)
        echo "Did not remove zip file"
        ;;
    *)
        echo "Invalid response. Please enter yes or no."
        ;;
esac
