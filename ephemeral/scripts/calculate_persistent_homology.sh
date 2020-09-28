#!/usr/bin/env zsh
#
# Main script for performing persistent homology calculations using
# DIPHA.

# Main directory for the `dipha` executable. This is required in order
# to calculate persistent homology of each input file. The result will
# be a binary file containing all persistence diagrams.
#
# TODO:
#   - make configurable
#   - add to path?
DIPHA_BIN=$HOME/Projects/dipha/build/dipha

# Main function for calculating persistent homology of a data set. This
# requires only a root directory for the conversion; output files shall
# be created automatically.
function calculate_persistent_homology {
  # Number of jobs to run in parallel. This is a poor-man's version of
  # a proper scheduler.
  N=8
  I=0
  (
  for FILE in $1/*.bin; do
    ((I=I%N)); ((I++ == 0)) && wait

    mkdir -p $1/persistence_diagrams/

    OUTPUT_FILE=$1/persistence_diagrams/${FILE:t}

    ## Do not overwrite any files; this is just waiting for problems to
    ## happen.
    if [ ! -f ${OUTPUT_FILE} ]; then
      mpiexec -n 4 $DIPHA_BIN $FILE $OUTPUT_FILE &
    else
      echo "Refusing to overwrite ${OUTPUT_FILE}. Continuing..."
    fi
  done
  )
}

ROOT="/links/groups/borgwardt/Projects/Ephemeral/Partly Cloudy/tda"

calculate_persistent_homology "$ROOT/brainmask"
calculate_persistent_homology "$ROOT/occipitalmask"
calculate_persistent_homology "$ROOT/xormask"

calculate_persistent_homology "$ROOT/brainmask_normalised"
calculate_persistent_homology "$ROOT/occipitalmask_normalised"
