#!/bin/bash
# Script to rename the output files in the wind fork directories.
# It replaces 'periodic_shoals36' with 'sweep_<dir_name>' for all files.

# List of wind fork directories
DIRS=(
    "wind_m0p05"
    "wind_m0p03"
    "wind_m0p01"
    "wind_0p01"
    "wind_0p03"
    "wind_0p05"
)

# The original string in the filenames to be replaced
OLD_STR="periodic_shoals36"

for dir in "${DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "Processing directory: $dir"
        NEW_STR="sweep_${dir}"
        
        # Loop through all files in the directory
        for filepath in "$dir"/*; do
            # Check if it's a file and the filename contains the OLD_STR
            if [ -f "$filepath" ] && [[ "$filepath" == *"$OLD_STR"* ]]; then
                # Extract the filename without the directory path
                filename=$(basename "$filepath")
                
                # Create the new filename by replacing the string
                new_filename="${filename//$OLD_STR/$NEW_STR}"
                
                # Execute the rename
                mv "$filepath" "$dir/$new_filename"
                echo "  Renamed: $filename -> $new_filename"
            fi
        done
    else
        echo "Directory '$dir' not found. Skipping..."
    fi
done

echo "Finished renaming files!"
