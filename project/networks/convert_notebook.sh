#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <file_name>"
    exit 1
fi

# Assign the provided argument to a variable
input_file="$1"

# Strip the extension from the provided filename
notebook="${input_file%.*}"

# Run the operations
python3 clean.py "$notebook".ipynb && \
jupyter nbconvert --to script "$notebook".ipynb && \
mv "$notebook".txt "$notebook".py

