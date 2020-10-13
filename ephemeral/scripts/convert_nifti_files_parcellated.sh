#!/usr/bin/env zsh
#
# Performs a conversion of a NIfTI data set to a set of binary
# matrices/volumes for subsequent topology-based processing. A
# single run of the script should be sufficient. However, this
# script ensures that no data will be overwritten.
#
# In contrast to other scripts, this one uses *parcellated*
# inputs.

# Main function for performing the conversion of a data set. Gets a root
# directory (containing a `preproc` folder) and the name of mask to use.
function convert_data_set {
  NIFTI_FILES=($1/preproc/*.nii.gz)

  for NIFTI_FILE in $NIFTI_FILES; do
    # Remove both extensions; this is easier to do in two lines than in
    # a single command :)
    STEM=${NIFTI_FILE:t}
    STEM=${STEM:r}
    STEM=${STEM:r}

    echo "Processing $STEM..."

    # Use new suffix to ensure that we do not overwrite old results.
    OUTPUT=$1/tda/$2_parcellated_large

    poetry run python ../nifti2matrix_parcellated.py --image $NIFTI_FILE                                                                     \
                                                     --parcellated ../../results/parcellated_data/shaefer_masked_subject_data_shifted_$2.npy \
                                                     --map ../../results/parcellated_data/schaefer_100_3mm_array.npy                         \
                                                     --output $OUTPUT
  done
}

ROOT="/links/groups/borgwardt/Projects/Ephemeral/Partly Cloudy"

convert_data_set "$ROOT" "brainmask"
convert_data_set "$ROOT" "occipitalmask"
convert_data_set "$ROOT" "xormask"
