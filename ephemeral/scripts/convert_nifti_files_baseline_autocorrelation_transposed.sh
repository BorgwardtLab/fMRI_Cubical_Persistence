#!/usr/bin/env zsh
#
# Performs a conversion of a NIfTI data set to a set of binary
# matrices based on autocorrelation. This constitutes a simple
# baseline method that we have to compare to.
#
# In this examples, volumes will be transposed.

ROOT="/links/groups/borgwardt/Projects/Ephemeral/Partly Cloudy"
OUTPUT="../../results/baseline_autocorrelation"

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

    # This assumes that brain masks follow the same naming scheme as the
    # NIfTI files. We ensure this by collating the data accordingly.
    BRAINMASK="${STEM//preproc/$2}.nii.gz"
    BRAINMASK=$1/$2/$BRAINMASK

    echo "Processing $STEM..."

    poetry run python ../baseline_autocorrelation.py --image $NIFTI_FILE \
                                                     --mask $BRAINMASK   \
                                                     --output $OUTPUT/$2 \
                                                     --transpose
  done
}

convert_data_set "$ROOT" "brainmask"
convert_data_set "$ROOT" "occipitalmask"
convert_data_set "$ROOT" "xormask"
