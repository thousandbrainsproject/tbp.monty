#!/bin/bash

# Script to run disk_learning_2d experiment with multiple zoom and resolution combinations

BASE_DIR="/Users/hlee/tbp/feat.2d_sensor"
EXPERIMENT="experiment=2d_sm/learning/disk_learning_2d"

# Arrays of values to test
ZOOMS=(10 20 30)
# ZOOMS=(10 20 30)  # Commented out - already ran experiments for zoom 20 and 30
RESOLUTIONS=(64)
BLURS=(5.0)  # Only running blur 3.0 and 4.0 for zoom 10 (already ran 1.0, 2.0, 5.0)

# Export PYTHONPATH
export PYTHONPATH="${BASE_DIR}/src:${PYTHONPATH}"

# Counter for tracking progress
total_runs=$((${#ZOOMS[@]} * ${#RESOLUTIONS[@]} * ${#BLURS[@]}))
current_run=0

echo "Starting experiments: ${total_runs} total runs"
echo "Zoom values: ${ZOOMS[@]}"
echo "Resolution values: ${RESOLUTIONS[@]}"
echo "Blur values: ${BLURS[@]}"
echo ""

# Loop through all combinations
for zoom in "${ZOOMS[@]}"; do
    for resolution in "${RESOLUTIONS[@]}"; do
        for blur in "${BLURS[@]}"; do
            current_run=$((current_run + 1))
            echo "=========================================="
            echo "Run ${current_run}/${total_runs}: zoom=${zoom}, resolution=${resolution}, blur=${blur}"
            echo "=========================================="
            
            cd "${BASE_DIR}"
            python run.py ${EXPERIMENT} config.zoom=${zoom} config.resolution=${resolution} config.blur_sigma=${blur}
            
            # Check if the command succeeded
            if [ $? -eq 0 ]; then
                echo "✓ Completed: zoom=${zoom}, resolution=${resolution}, blur=${blur}"
            else
                echo "✗ Failed: zoom=${zoom}, resolution=${resolution}, blur=${blur}"
            fi
            echo ""
        done
    done
done

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
