# Ephemeral: fMRI data analysis

## Age prediction experiment

To run the calculations, call the `predict_age.py` script on different
input data sets. The script is sufficiently smart to automatically use
the appropriate fitting method based on the input data:

    # For the baseline-tt and baseline-vv experiments, respectively. For
    # the baseline-tt experiment, other brain masks are also available.
    python predict_age.py ../results/baseline_autocorrelation/brainmask/*.npz
    python predict_age.py ../results/baseline_autocorrelation/occipitalmask/*.npz
    python predict_age.py ../results/baseline_autocorrelation/xormask/*.npz
    python predict_age.py ../results/baseline_autocorrelation_parcellated/*.npz

## Variability analysis: Showing the event histograms

    python peri_histogram_analysis.py ../results/across_cohort_variability/brainmask_sigma1.0_r20.csv
    python peri_histogram_analysis.py ../results/across_cohort_variability/occipitalmask_sigma1.0_r20.csv
    python peri_histogram_analysis.py ../results/across_cohort_variability/xormask_sigma1.0_20.csv

## Variability analysis: Running the bootstrap experiments

    python peri_histogram_bootstrap_analysis.py ../results/across_cohort_variability/brainmask_sigma1.0_r20.csv
    python peri_histogram_bootstrap_analysis.py ../results/across_cohort_variability/occipitalmask_sigma1.0_r20.csv
    python peri_histogram_bootstrap_analysis.py ../results/across_cohort_variability/xormask_sigma1.0_20.csv
