%% ============================================================
% STEP 11 — FINAL FIGURES
%
% EEG Project
%
% Produces the six figures for the final report:
%   1. Representative EEG signals — non-seizure vs seizure
%   2. Global feature distributions
%   3. Temporal feature distributions
%   4. Model performance comparison (Global vs temporal-augmented sets)
%   5. Combined RF feature importance (Global + All Temporal)
%   6. Repeated-CV robustness (fold-level spread)
%
% Requires (same folder):
%   nonseizure_vs_seizure_features.csv
%   temporal_nonseizure_vs_seizure_features.csv
%   step10_paired_cv_fold_results.csv
%
% Figure 1 needs raw signal data that isn't part of the feature
% CSVs — see the note in that section and edit as needed.
% =============================================================

clear; clc; close all;

%% ------------------------------------------------------------
% 1. REPRESENTATIVE EEG SIGNALS — NON-SEIZURE VS SEIZURE
% ---------------------------------------------------------------
% EDIT THIS SECTION: point it at wherever your raw signal segments
% live (e.g. a .mat file with one non-seizure and one seizure trace).
% This is the only figure that can't be built from the feature CSVs.

Fs = 173.61;  % Bonn dataset sampling rate — adjust if different

% Example (uncomment and edit once you have the file/variable names):
% load('representative_signals.mat');   % expects nonseizure_signal, seizure_signal

if exist('nonseizure_signal','var') && exist('seizure_signal','var')
    t = (0:length(nonseizure_signal)-1) / Fs;

    figure('Name','Representative EEG Signals');
    subplot(2,1,1);
    plot(t, nonseizure_signal, 'b');
    title('Representative Non-Seizure EEG');
    xlabel('Time (s)'); ylabel('Amplitude');

    subplot(2,1,2);
    plot(t, seizure_signal, 'r');
    title('Representative Seizure EEG');
    xlabel('Time (s)'); ylabel('Amplitude');

    saveas(gcf, 'fig1_representative_signals.png');
else
    warning(['Skipping Figure 1: load your raw signal data and set ' ...
             'nonseizure_signal / seizure_signal before running this section.']);
end

%% ------------------------------------------------------------
% 2. GLOBAL FEATURE DISTRIBUTIONS
% ---------------------------------------------------------------

G = readtable('nonseizure_vs_seizure_features.csv');
global_names = G.Properties.VariableNames(1:10);

figure('Name','Global Feature Distributions');
for i = 1:10
    subplot(2,5,i);
    boxplot(G{:,i}, G.Class, 'Labels', {'Non-seizure','Seizure'});
    title(global_names{i}, 'Interpreter','none');
end
sgtitle('Global Feature Distributions by Class');
saveas(gcf, 'fig2_global_feature_distributions.png');

%% ------------------------------------------------------------
% 3. TEMPORAL FEATURE DISTRIBUTIONS
% ---------------------------------------------------------------

T = readtable('temporal_nonseizure_vs_seizure_features.csv');
temporal_names = T.Properties.VariableNames(1:15);

figure('Name','Temporal Feature Distributions');
for i = 1:15
    subplot(3,5,i);
    boxplot(T{:,i}, G.Class, 'Labels', {'Non-seizure','Seizure'});
    title(temporal_names{i}, 'Interpreter','none');
end
sgtitle('Temporal Feature Distributions by Class');
saveas(gcf, 'fig3_temporal_feature_distributions.png');

%% ------------------------------------------------------------
% 4. MODEL PERFORMANCE COMPARISON
% ---------------------------------------------------------------

FoldResults = readtable('step10_paired_cv_fold_results.csv');
set_names = {'Global','Global + MeanTransition','Global + MaxTransition','Global + All Temporal'};

acc_means = zeros(1,4); acc_stds = zeros(1,4);
f1_means  = zeros(1,4); f1_stds  = zeros(1,4);
auc_means = zeros(1,4); auc_stds = zeros(1,4);

for s = 1:4
    acc_means(s) = 100*mean(FoldResults.(sprintf('Accuracy_%d',s)));
    acc_stds(s)  = 100*std(FoldResults.(sprintf('Accuracy_%d',s)));
    f1_means(s)  = 100*mean(FoldResults.(sprintf('F1_%d',s)));
    f1_stds(s)   = 100*std(FoldResults.(sprintf('F1_%d',s)));
    auc_means(s) = mean(FoldResults.(sprintf('AUC_%d',s)));
    auc_stds(s)  = std(FoldResults.(sprintf('AUC_%d',s)));
end

figure('Name','Model Performance Comparison');

subplot(1,3,1);
bar(acc_means); hold on;
errorbar(1:4, acc_means, acc_stds, 'k.', 'LineWidth',1);
set(gca,'XTickLabel',set_names,'XTickLabelRotation',30);
ylabel('Accuracy (%)'); title('Accuracy'); ylim([90 100]);

subplot(1,3,2);
bar(f1_means); hold on;
errorbar(1:4, f1_means, f1_stds, 'k.', 'LineWidth',1);
set(gca,'XTickLabel',set_names,'XTickLabelRotation',30);
ylabel('F1 (%)'); title('F1 Score'); ylim([85 100]);

subplot(1,3,3);
bar(auc_means); hold on;
errorbar(1:4, auc_means, auc_stds, 'k.', 'LineWidth',1);
set(gca,'XTickLabel',set_names,'XTickLabelRotation',30);
ylabel('AUC'); title('AUC'); ylim([0.99 1]);

sgtitle('Repeated-CV Performance: Global vs Temporal-Augmented Feature Sets');
saveas(gcf, 'fig4_model_comparison.png');

%% ------------------------------------------------------------
% 5. COMBINED RF FEATURE IMPORTANCE (GLOBAL + ALL TEMPORAL)
% ---------------------------------------------------------------

Y = G.Class;
X_global   = G{:,1:10};
X_temporal = T{:,1:15};
X_combined = [X_global X_temporal];
combined_names = [global_names, temporal_names];

rng(42);
finalModel = TreeBagger(200, X_combined, Y, ...
    'Method','classification', ...
    'OOBPredictorImportance','on');

importance = finalModel.OOBPermutedPredictorDeltaError;
[sorted_importance, idx] = sort(importance, 'descend');

figure('Name','RF Feature Importance');
barh(sorted_importance);
set(gca, 'YTick', 1:length(idx), 'YTickLabel', combined_names(idx), 'YDir','reverse');
xlabel('OOB Permuted Predictor Importance');
title('Random Forest Feature Importance (Global + All Temporal)');
saveas(gcf, 'fig5_feature_importance.png');

%% ------------------------------------------------------------
% 6. REPEATED-CV ROBUSTNESS
% ---------------------------------------------------------------

acc_matrix = zeros(height(FoldResults), 4);
for s = 1:4
    acc_matrix(:,s) = 100*FoldResults.(sprintf('Accuracy_%d',s));
end

figure('Name','Repeated-CV Robustness');
boxplot(acc_matrix, 'Labels', set_names);
xtickangle(30);
ylabel('Accuracy (%)');
title('Fold-Level Accuracy Distribution Across 50 Repeated-CV Folds');
saveas(gcf, 'fig6_cv_robustness.png');

fprintf('\nAll six figures saved as PNG files in the current folder.\n');
