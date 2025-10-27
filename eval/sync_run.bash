#!/bin/bash

 # Check if substring argument provided
 if [ $# -eq 0 ]; then
     echo "Error: No substring provided"
     echo "Usage: $0 SUBSTRING1 [SUBSTRING2 ...]"
     exit 1
 fi

 # Store the starting directory
 START_DIR=$(pwd)

 # Check if wandb directory exists
 if [ ! -d "wandb" ]; then
     echo "Error: wandb/ directory not found"
     exit 1
 fi

 # Loop through each substring argument
 for SUBSTRING in "$@"; do
     echo ""
     echo "=== Searching for folders with substring: $SUBSTRING ==="

     # Find folders with substring in wandb directory only
     for dir in wandb/*"${SUBSTRING}"*/; do
         if [ -d "$dir" ]; then
             echo "Processing: $dir"

             if wandb sync "$dir"; then
                 echo "✓ Successfully synced: $dir"
             else
                 echo "✗ Failed to sync: $dir"
             fi
         fi
     done
 done

 echo ""
 echo "Sync process completed"
