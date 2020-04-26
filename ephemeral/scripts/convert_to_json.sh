#!/bin/sh
#
# Performs batch conversion of all existing data sets: all of the
# persistence diagrams are converted to JSON files. Each existing
# file will remain untouched.

for EXPERIMENT in "brainmask" "occipitalmask" "brainmask_normalised" "occipitalmask_normalised"; do
  echo "Experiment $EXPERIMENT..."
  find /links/groups/borgwardt/Projects/Ephemeral/Partly\ Cloudy/tda/$EXPERIMENT/persistence_diagrams/ -name "*.bin" -print0 | xargs -0 -P 24 poetry run python ../dipha2json.py
done
