# Results

## Preliminary Classifier Comparison

Six classifiers were trained and compared using repeated cross-validation on an initial feature set (200 recordings): Logistic Regression, k-Nearest Neighbors, Decision Tree, SVM with an RBF kernel, Random Forest, and Gradient Boosting.

| Model | Mean Accuracy | Mean F1 | Mean AUC |
|---|---|---|---|
| Logistic Regression | 96.0% | 94.1% | 0.9928 |
| k-NN | 97.0% | 95.3% | 0.9905 |
| Decision Tree | 97.3% | 96.0% | 0.9833 |
| SVM-RBF | 95.7% | 93.9% | 0.9659 |
| Random Forest | 98.0% | 97.0% | 0.9980 |
| Gradient Boosting | 97.0% | 95.5% | 0.9880 |

Random Forest achieved the highest mean accuracy (98.0%), F1 score (97.0%), and AUC (0.998), and was therefore selected for the main feature-set comparison (see below).

## Global vs. Temporal Feature Comparison (Paired Repeated Cross-Validation)

Using the full 500-recording dataset (400 non-seizure, 100 seizure), a paired repeated stratified 5-fold cross-validation (10 repeats, 50 total folds) was run with a Random Forest (200 trees) to test whether adding temporal spectral-dynamics features improves on global features alone. The same fold partitions were used across all four feature sets to allow a matched, paired comparison.

| Feature Set | Accuracy | F1 | AUC |
|---|---|---|---|
| Global | 98.34% ± 1.15% | 95.83% ± 2.91% | 0.99863 ± 0.00208 |
| Global + MeanTransition | 98.50% ± 1.00% | 96.21% ± 2.52% | 0.99845 ± 0.00260 |
| Global + MaxTransition | 98.18% ± 1.19% | 95.40% ± 3.04% | 0.99797 ± 0.00277 |
| Global + All Temporal | 98.26% ± 1.23% | 95.61% ± 3.08% | 0.99733 ± 0.00435 |

Paired Wilcoxon signed-rank tests on the matched folds (each temporal-augmented set vs. Global) gave:

| Comparison | ΔAccuracy (pp) | p (Acc.) | ΔF1 (pp) | p (F1) | ΔAUC | p (AUC) |
|---|---|---|---|---|---|---|
| + MeanTransition | +0.16 | 0.074 | +0.39 | 0.236 | −0.00018 | 0.272 |
| + MaxTransition | −0.16 | 0.074 | −0.43 | 0.013 | −0.00066 | <0.0001 |
| + All Temporal | −0.08 | 0.463 | −0.22 | 0.149 | −0.00130 | <0.001 |

None of the temporal-augmented feature sets produced a statistically significant improvement in accuracy or F1 over the Global set. MaxTransition and All Temporal were associated with small but statistically detectable *decreases* in AUC.

*(Insert Figures 1–6 here — representative signals, global/temporal feature distributions, model comparison, RF feature importance, and CV robustness.)*

Separately, Random Forest feature-importance rankings placed Theta_MeanTransition and Theta_MaxTransition among the top predictors, indicating that individual temporal features do carry class-discriminative information even though the full temporal feature block did not improve validated performance.

# Discussion

The results indicate that global spectral features (relative band power, spectral centroid/entropy, dominant frequency, amplitude statistics) already capture most of the information needed to separate seizure from non-seizure EEG segments in this dataset, achieving ~98% accuracy and AUC above 0.998 on their own. Adding temporal spectral-dynamics features (mean transition, max transition, drift) did not yield a statistically supported improvement in classification performance, and in two of the three augmented sets was associated with a small, statistically significant *decrease* in AUC.

This is not a contradiction of the feature-importance results. A feature can be individually informative to a tree-based ensemble (i.e., useful when combined with others in a split) without its addition changing overall generalization performance, particularly when the information it carries substantially overlaps with what the global features already provide. It is also possible that the small performance decreases associated with MaxTransition and the full temporal set reflect increased dimensionality and finite-sample variance (15 additional, more noise-prone features) rather than those features being uninformative in principle.

Several limitations should be stated explicitly:

- **Dataset characteristics.** The Bonn EEG dataset is a well-known benchmark whose seizure and non-seizure segments are drawn from different subjects and recording conditions, making the classes more separable than would be expected in a real clinical seizure-detection setting (e.g., intra-patient discrimination). The ~98% accuracy obtained here is consistent with, not exceptional relative to, prior work on this dataset, and should not be interpreted as evidence of clinical-grade performance.
- **Statistical caveat on the paired tests.** The 50 "folds" used in the Wilcoxon signed-rank tests come from 10 repeats of 5-fold cross-validation on the same 500 recordings, so the folds are not fully independent observations — individual recordings reappear across repeats. The reported p-values should therefore be treated as exploratory, paired evidence of a consistent direction of effect, not as rigorous inferential statistics with a true 50-observation sample size.
- **No independent external validation.** All evaluation was performed via cross-validation within a single dataset; no separate dataset (e.g., a different EEG recording set or patient population) was used to test generalization.
- **No hyperparameter tuning.** The Random Forest was run with a fixed configuration (200 trees, default splitting criteria) rather than a tuned configuration, so reported performance reflects one reasonable but unoptimized model, not necessarily the ceiling achievable with these features.
- **Moderate class imbalance.** The 400:100 non-seizure:seizure ratio was accounted for by reporting precision, recall, F1, and AUC alongside accuracy, but was not addressed through resampling or class weighting; this is unlikely to materially change the paired comparison since the same imbalance affects every feature set equally.

# Conclusion

This project developed and evaluated a leakage-free, cross-validated machine learning pipeline for classifying seizure vs. non-seizure EEG recordings from the Bonn dataset, comparing six classifiers before focusing on Random Forest for a controlled ablation study of global versus temporal spectral-dynamics features. The main finding is a nuanced, statistically grounded negative result: temporal transition and drift features, while individually ranked as important by the Random Forest, do not provide a statistically supported improvement over global spectral features alone under paired repeated cross-validation, and in two configurations were associated with a small decrease in AUC.

These results should be understood as a benchmark classification study on a well-characterized public EEG dataset, not as a clinically validated seizure detector or a seizure-forecasting system. The dataset's between-subject class separability, combined with the exploratory nature of the paired significance tests, means the appropriate claim is that global spectral features are sufficient for high performance on this benchmark, and that the temporal features tested here do not add meaningful value on top of them — a useful, well-supported conclusion for guiding feature selection in future work on this or similar EEG classification tasks.
