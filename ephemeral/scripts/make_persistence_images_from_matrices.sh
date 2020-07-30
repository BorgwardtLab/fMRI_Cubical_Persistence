#!/usr/bin/env zsh
#
# Calculates persistence images for a different set of experimental
# scenarios and stores them in an output file. This script uses the
# persistence diagrams created from matrices. 

function calculate_persistence_images {
  DIRECTORY=$1
  OUTPUT=$2
  SIGMA=$3
  RESOLUTION=$4

  poetry run python ../calculate_embeddings_persistence_images.py -d 0           \
                                                                  -s $SIGMA      \
                                                                  -r $RESOLUTION \
                                                                  -i $DIRECTORY  \
                                                                  -o $OUTPUT
}


ROOT="../../results/baseline_autocorrelation"

for SIGMA in "1.0" "0.1"; do
  for RESOLUTION in "10" "20"; do
    for EXPERIMENT in "brainmask" "occipitalmask" "brainmask_normalised" "occipitalmask_normalised"; do
      echo "Running conversion for $EXPERIMENT..."
      INPUT="$ROOT/$EXPERIMENT/persistence_diagrams" 
      OUTPUT="../../results/persistence_images_from_matrices/${EXPERIMENT}_sigma${SIGMA}_r${RESOLUTION}.json"
      calculate_persistence_images $INPUT $OUTPUT $SIGMA $RESOLUTION
    done
    wait
  done
done
