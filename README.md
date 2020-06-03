# Ephemeral: fMRI data analysis

## Age prediction experiment

To run the calculations, call the `predict_age.py` script on different
input data sets. The script is sufficiently smart to automatically use
the appropriate fitting method based on the input data:

    # For the baseline-tt and baseline-pp experiments, respectively. For
    # both experiments, other brain masks are also available.
    python predict_age.py ../results/baseline_autocorrelation/brainmask/*.npz
    python predict_age.py ../results/baseline_autocorrelation/occipitalmask/*.npz
    python predict_age.py ../results/baseline_autocorrelation/xormask/*.npz
    python predict_age.py ../results/baseline_autocorrelation_parcellated/brainmask/*.npz
    python predict_age.py ../results/baseline_autocorrelation_parcellated/occipitalmask/*.npz
    python predict_age.py ../results/baseline_autocorrelation_parcellated/brainmask/*.npz

    # For the topological summary statistics. Again, they also work for
    # different brain masks (not showing all of them here). Please note
    # that `total_persistence_p1` is equivalent to the `p1_norm`, as we
    # describe it in the paper (the terminology in the paper was chosen
    # because it is more consistent).
    python predict_age.py -s 'infinity_norm_p1' ../results/summary_statistics/brainmask.json
    python predict_age.py -s 'total_persistence_p1' ../results/summary_statistics/brainmask.json

## Variability analysis: Showing the event histograms

    python peri_histogram_analysis.py ../results/across_cohort_variability/brainmask_sigma1.0_r20.csv
    python peri_histogram_analysis.py ../results/across_cohort_variability/occipitalmask_sigma1.0_r20.csv
    python peri_histogram_analysis.py ../results/across_cohort_variability/xormask_sigma1.0_20.csv

## Variability analysis: Running the bootstrap experiments

    python peri_histogram_bootstrap_analysis.py ../results/across_cohort_variability/brainmask_sigma1.0_r20.csv
    python peri_histogram_bootstrap_analysis.py ../results/across_cohort_variability/occipitalmask_sigma1.0_r20.csv
    python peri_histogram_bootstrap_analysis.py ../results/across_cohort_variability/xormask_sigma1.0_20.csv

## Embeddings

It is possible to recreate the embeddings that are stored in the
`results` folder by calling:

    python embed_baseline_autocorrelation.py ../results/baseline_autocorrelation_parcellated/*.npz

Alternative `baseline_autocorrelation` matrices are also permissible.
