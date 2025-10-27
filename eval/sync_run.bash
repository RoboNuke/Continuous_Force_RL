 #!/bin/bash

  # Check if substring argument provided
  if [ $# -eq 0 ]; then
      echo "Error: No substring provided"
      echo "Usage: $0 SUBSTRING"
      exit 1
  fi

  SUBSTRING="$1"

  # Find folders with substring in current directory only
  for dir in *"${SUBSTRING}"*/; do
      if [ -d "$dir" ]; then
          echo "Processing: $dir"
          cd "$dir" || continue

          if wandb sync "run-*${SUBSTRING}*.wandb"; then
              echo "✓ Successfully synced: $dir"
          else
              echo "✗ Failed to sync: $dir"
          fi

          cd ..
      fi
  done

  echo "Sync process completed"