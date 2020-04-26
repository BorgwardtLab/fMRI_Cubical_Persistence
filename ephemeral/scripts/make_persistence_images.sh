#!/usr/bin/env zsh
#
# Calculates persistence images for a different set experimental
# scenarios and stores them in an output file. This script is an
# almost insulting simple one; it will overwrite everything, and
# spawn multiple jobs. All the 'intelligence' lies in the script
# it calls.

function calculate_persistence_images {
  DIRECTORY=$1
  OUTPUT=$2
  SIGMA=$3
  RESOLUTION=$4

  poetry run python ../calculate_embeddings_persistence_images.py -s $SIGMA      \
                                                                  -r $RESOLUTION \
                                                                  -i $DIRECTORY  \
                                                                  -o $OUTPUT &
}


ROOT="/links/groups/borgwardt/Projects/Ephemeral/Partly Cloudy/tda"

for SIGMA in "1.0" "0.1"; do
  for RESOLUTION in "10" "20"; do
    for EXPERIMENT in "brainmask" "occipitalmask" "brainmask_normalised" "occipitalmask_normalised"; do
      echo "Running conversion for $EXPERIMENT..."
      INPUT="$ROOT/$EXPERIMENT/persistence_diagrams" 
      OUTPUT="../../results/persistence_images/${EXPERIMENT}_sigma${SIGMA}_r${RESOLUTION}.json"
      calculate_persistence_images $INPUT $OUTPUT $SIGMA $RESOLUTION
    done
    wait
  done
done
