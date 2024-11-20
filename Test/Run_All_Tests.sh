#!/bin/bash

# Iterate over all files that match the pattern test_*.py
for file in test_*.py; do
    # Check if any files match the pattern
    if [[ -f "$file" ]]; then
        echo "Running $file..."
        python "$file"
    else
        echo "No files matching 'test_*.py' found."
        exit 1
    fi
done
