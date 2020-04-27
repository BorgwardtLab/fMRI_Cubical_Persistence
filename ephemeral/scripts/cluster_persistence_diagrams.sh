#!/usr/bin/env zsh
#
# Clusters persistence diagrams according to various metrics and
# prefixes. This is required for assessing the stability of each
# representation.

function cluster_persistence_diagrams {
  DIRECTORY=$1
  EXPERIMENT=$2

  echo $EXPERIMENT

  # Infinity norm
	poetry run python ../cluster_persistence_diagrams.py -e $EXPERIMENT -m infinity_norm -p 1 $DIRECTORY 
	poetry run python ../cluster_persistence_diagrams.py -e $EXPERIMENT -m infinity_norm -p 2 $DIRECTORY 

	# Total persistence
	poetry run python ../cluster_persistence_diagrams.py -e $EXPERIMENT -p 1 $DIRECTORY 
	poetry run python ../cluster_persistence_diagrams.py -e $EXPERIMENT -p 2 $DIRECTORY
}

ROOT="/links/groups/borgwardt/Projects/Ephemeral/Partly Cloudy/tda"

for EXPERIMENT in "brainmask" "occipitalmask" "brainmasked_normalised" "occipitalmask_normalised"; do
  DIRECTORY="$ROOT/$EXPERIMENT/persistence_diagrams"
  cluster_persistence_diagrams $DIRECTORY $EXPERIMENT
done
