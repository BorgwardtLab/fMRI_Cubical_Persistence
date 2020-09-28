#!/usr/bin/env zsh
#
# Calculates summary statistics of persistence diagrams to summarise the
# time series of one patient.

function calculate_summary_statistics {
  DIRECTORY=$1
  EXPERIMENT=$2
  DIMENSION=$3

  echo $EXPERIMENT

  poetry run python ../summary_statistics_persistence_diagrams.py -d $DIMENSION -i $DIRECTORY -s total_persistence infinity_norm -p 1 2 -o ../../results/summary_statistics/$EXPERIMENT.json
}

ROOT="/links/groups/borgwardt/Projects/Ephemeral/Partly Cloudy/tda"

for EXPERIMENT in "brainmask" "occipitalmask" "brainmask_normalised" "occipitalmask_normalised"; do
  DIRECTORY="$ROOT/$EXPERIMENT/persistence_diagrams"
  calculate_summary_statistics $DIRECTORY $EXPERIMENT 2
done

# Need to use dimension zero here because it is the only dimension
# guaranteed to exist.
for EXPERIMENT in "brainmask_parcellated" "occipitalmask_parcellated"; do
  DIRECTORY="$ROOT/$EXPERIMENT/persistence_diagrams"
  calculate_summary_statistics $DIRECTORY $EXPERIMENT 0
done
