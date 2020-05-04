#!/usr/bin/env zsh


for ENCODER in 'pca' 'mds' 'phate' 'm-phate'; do
  for DIMENSION in 2 3; do
    for GLOBAL in "" "-g"; do
      for FILE in "../../results/persistence_images/brainmask_sigma1.0_r20.json" "../../results/persistence_images/occipitalmask_sigma1.0_r20.json"; do
        echo "encoder = $ENCODER, dimension = $dimension, file = $FILE"
        poetry run python ../visualise_embeddings.py -d $DIMENSION -e $ENCODER $GLOBAL $FILE 
      done
    done
  done
done
