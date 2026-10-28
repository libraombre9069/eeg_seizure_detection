# EEG Seizure Classification (Bonn Dataset)

A machine-learning pipeline that classifies seizure vs. non-seizure EEG recordings using global spectral features and temporal spectral-dynamics features, with a paired repeated-CV ablation study testing whether the temporal features add value.

## Contents
- `index.html` — interactive project story/website (open directly in a browser, or serve with GitHub Pages).
- `step11_figures.m` — MATLAB script generating the final report figures.
- `results_discussion_conclusion_draft.md` — draft Results/Discussion/Conclusion write-up.
- `*.m` — full MATLAB pipeline (feature extraction, model comparison, statistical analysis, ablation study).
- `*.csv` — extracted features and result tables produced by the pipeline.

## Headline result
Global spectral features alone reach ~98.4% accuracy / 0.999 AUC (Random Forest, 5-fold CV). Adding temporal spectral-dynamics features (mean transition, max transition, drift) does **not** produce a statistically significant improvement (paired Wilcoxon signed-rank test, 10×5 repeated CV) — see `step10_paired_cv_comparison.csv`.

## Publishing the site
1. Push this repo to GitHub.
2. Settings → Pages → deploy from the branch containing `index.html`.
3. Your story page will be live at `https://<username>.github.io/<repo>/`.

## Caveats
This is a benchmark classification study on a public dataset (Bonn), not a clinically validated seizure detector — see the Limitations section on the site.
