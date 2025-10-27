#!/bin/bash

 # Check if substring argument provided
 if [ $# -eq 0 ]; then
     echo "Error: No substring provided"
     echo "Usage: $0 SUBSTRING1 [SUBSTRING2 ...]"
     exit 1
 fi

 # Loop through each substring argument
 for SUBSTRING in "$@"; do
     echo ""
     echo "=== Searching for folders with substring: $SUBSTRING ==="

     # Find folders with substring in current directory only
     for dir in *"${SUBSTRING}"*/; do
         if [ -d "$dir" ]; then
             echo "Processing: $dir"
             cd "$dir" || continue

             if wandb sync "run-${SUBSTRING}.wandb"; then
                 echo "✓ Successfully synced: $dir"
             else
                 echo "✗ Failed to sync: $dir"
             fi

             cd ..
         fi
     done
 done

 echo ""
 echo "Sync process completed"
