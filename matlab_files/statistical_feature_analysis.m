%% statistical_feature_analysis.m
% Statistical analysis of global EEG features
% Interictal (F + N) vs Ictal (S)
%
% Input:
%   interictal_vs_ictal_features.csv
%   rf_feature_importance.csv
%
% Outputs:
%   statistical_feature_analysis.csv
%   Boxplots for all 10 features
%   Cohen's d plot
%   -log10(p-value) plot

clear;
clc;
close all;

fprintf('========================================\n');
fprintf('STATISTICAL FEATURE ANALYSIS\n');
fprintf('Interictal (F+N) vs Ictal (S)\n');
fprintf('========================================\n\n');


%% ============================================================
% 1. LOAD FEATURE DATA
% =============================================================

T = readtable('interictal_vs_ictal_features.csv');

% Extract feature names
feature_names = T.Properties.VariableNames(1:end-1);

% Extract feature matrix
X = T{:,1:end-1};

% Extract class labels
y = T.Class;

% Class definitions
% 0 = Interictal (F + N)
% 1 = Ictal (S)

interictal_idx = (y == 0);
ictal_idx      = (y == 1);

X_interictal = X(interictal_idx,:);
X_ictal      = X(ictal_idx,:);

fprintf('Total recordings : %d\n', size(X,1));
fprintf('Interictal       : %d\n', sum(interictal_idx));
fprintf('Ictal            : %d\n', sum(ictal_idx));
fprintf('Features         : %d\n\n', size(X,2));


%% ============================================================
% 2. INITIALIZE RESULT ARRAYS
% =============================================================

n_features = length(feature_names);

Mean_Interictal   = zeros(n_features,1);
Std_Interictal    = zeros(n_features,1);
Median_Interictal = zeros(n_features,1);

Mean_Ictal        = zeros(n_features,1);
Std_Ictal         = zeros(n_features,1);
Median_Ictal      = zeros(n_features,1);

Cohens_d          = zeros(n_features,1);

P_Value            = zeros(n_features,1);


%% ============================================================
% 3. CALCULATE STATISTICS
% =============================================================

for i = 1:n_features

    % Current feature
    interictal = X_interictal(:,i);
    ictal      = X_ictal(:,i);

    % Remove NaN values if any
    interictal = interictal(~isnan(interictal));
    ictal      = ictal(~isnan(ictal));

    % ---------------------------------------------------------
    % Descriptive statistics
    % ---------------------------------------------------------

    Mean_Interictal(i)   = mean(interictal);
    Std_Interictal(i)    = std(interictal);
    Median_Interictal(i) = median(interictal);

    Mean_Ictal(i)        = mean(ictal);
    Std_Ictal(i)         = std(ictal);
    Median_Ictal(i)      = median(ictal);


    % ---------------------------------------------------------
    % Cohen's d
    %
    % d = (mean1 - mean2) / pooled standard deviation
    % ---------------------------------------------------------

    n1 = length(interictal);
    n2 = length(ictal);

    pooled_std = sqrt( ...
        ((n1-1)*std(interictal)^2 + ...
         (n2-1)*std(ictal)^2) / ...
        (n1+n2-2));

    if pooled_std ~= 0
        Cohens_d(i) = ...
            (mean(ictal) - mean(interictal)) / pooled_std;
    else
        Cohens_d(i) = NaN;
    end


    % ---------------------------------------------------------
    % Mann-Whitney U / Wilcoxon rank-sum test
    %
    % Non-parametric test of distributional difference
    % ---------------------------------------------------------

    P_Value(i) = ranksum(interictal, ictal);

end


%% ============================================================
% 4. FDR CORRECTION
% =============================================================

% Benjamini-Hochberg False Discovery Rate correction
%
% MATLAB's mafdr requires Bioinformatics Toolbox.
% To avoid requiring that toolbox, the correction is implemented
% manually here.

[sorted_p, sort_idx] = sort(P_Value);

m = n_features;

adjusted_sorted = sorted_p .* m ./ (1:m)';

% Enforce monotonicity from largest rank backwards
for i = m-1:-1:1
    adjusted_sorted(i) = min(adjusted_sorted(i), ...
                              adjusted_sorted(i+1));
end

FDR_P_Value = zeros(n_features,1);

FDR_P_Value(sort_idx) = min(adjusted_sorted,1);


%% ============================================================
% 5. SIGNIFICANCE LABEL
% =============================================================

Significant = FDR_P_Value < 0.05;


%% ============================================================
% 6. LOAD RANDOM FOREST IMPORTANCE
% =============================================================

RF_Importance = NaN(n_features,1);

if isfile('rf_feature_importance.csv')

    RF_Table = readtable('rf_feature_importance.csv');

    for i = 1:n_features

        idx = strcmp(RF_Table.Feature, feature_names{i});

        if any(idx)
            RF_Importance(i) = ...
                RF_Table.Importance(find(idx,1));
        end

    end

    fprintf('Random Forest importance loaded.\n\n');

else

    fprintf('WARNING: rf_feature_importance.csv not found.\n');
    fprintf('RF importance will be left as NaN.\n\n');

end


%% ============================================================
% 7. CREATE RESULTS TABLE
% =============================================================

Results = table( ...
    feature_names(:), ...
    Mean_Interictal, ...
    Std_Interictal, ...
    Median_Interictal, ...
    Mean_Ictal, ...
    Std_Ictal, ...
    Median_Ictal, ...
    Cohens_d, ...
    P_Value, ...
    FDR_P_Value, ...
    Significant, ...
    RF_Importance, ...
    'VariableNames', { ...
    'Feature', ...
    'Mean_Interictal', ...
    'Std_Interictal', ...
    'Median_Interictal', ...
    'Mean_Ictal', ...
    'Std_Ictal', ...
    'Median_Ictal', ...
    'Cohens_d', ...
    'P_Value', ...
    'FDR_P_Value', ...
    'Significant_FDR_0_05', ...
    'RF_Importance'});


%% ============================================================
% 8. SORT BY ABSOLUTE COHEN'S d
% =============================================================

Results.Abs_Cohens_d = abs(Results.Cohens_d);

Results = sortrows( ...
    Results, ...
    'Abs_Cohens_d', ...
    'descend');


%% ============================================================
% 9. DISPLAY RESULTS
% =============================================================

fprintf('========================================\n');
fprintf('STATISTICAL RESULTS\n');
fprintf('========================================\n\n');

disp(Results);


%% ============================================================
% 10. SAVE RESULTS
% =============================================================

writetable(Results, ...
    'statistical_feature_analysis.csv');

fprintf('\nResults saved as:\n');
fprintf('statistical_feature_analysis.csv\n');


%% ============================================================
% 11. BOXPLOTS FOR ALL FEATURES
% =============================================================

fprintf('\nGenerating boxplots...\n');

for i = 1:n_features

    figure;

    data = [ ...
        X_interictal(:,i); ...
        X_ictal(:,i)];

    group = [ ...
        repmat({'Interictal'}, size(X_interictal,1), 1); ...
        repmat({'Ictal'}, size(X_ictal,1), 1)];

    boxplot(data, group);

    xlabel('Class');
    ylabel(feature_names{i});

    title(sprintf( ...
        '%s: Interictal vs Ictal', ...
        feature_names{i}), ...
        'Interpreter','none');

    grid on;

end


%% ============================================================
% 12. COHEN'S d PLOT
% =============================================================

% Sort features by absolute effect size
[~, idx] = sort(abs(Cohens_d), 'descend');

figure;

bar(Cohens_d(idx));

set(gca, ...
    'XTick', 1:n_features, ...
    'XTickLabel', feature_names(idx), ...
    'XTickLabelRotation', 45);

xlabel('Feature');
ylabel("Cohen's d");

title("Effect Size: Interictal vs Ictal");

grid on;


%% ============================================================
% 13. -log10(P-VALUE) PLOT
% =============================================================

% Avoid log10(0)
safe_p = max(FDR_P_Value, realmin);

neglog_p = -log10(safe_p);

[~, idx_p] = sort(neglog_p, 'descend');

figure;

bar(neglog_p(idx_p));

hold on;

% FDR = 0.05 threshold
threshold = -log10(0.05);

yline(threshold, '--', ...
    'FDR = 0.05');

hold off;

set(gca, ...
    'XTick', 1:n_features, ...
    'XTickLabel', feature_names(idx_p), ...
    'XTickLabelRotation', 45);

xlabel('Feature');
ylabel('-log_{10}(FDR-adjusted p-value)');

title('Statistical Significance of EEG Features');

grid on;


%% ============================================================
% 14. RF IMPORTANCE VS EFFECT SIZE
% =============================================================

if any(~isnan(RF_Importance))

    figure;

    scatter( ...
        abs(Cohens_d), ...
        RF_Importance, ...
        70, ...
        'filled');

    xlabel("|Cohen's d|");
    ylabel('RF OOB Permutation Importance');

    title('Effect Size vs Random Forest Feature Importance');

    grid on;

    % Add feature labels
    for i = 1:n_features

        text( ...
            abs(Cohens_d(i)), ...
            RF_Importance(i), ...
            ['  ' feature_names{i}], ...
            'Interpreter','none');

    end

end


%% ============================================================
% 15. FINAL SUMMARY
% =============================================================

fprintf('\n========================================\n');
fprintf('SUMMARY\n');
fprintf('========================================\n');

fprintf('\nFeatures significant after FDR correction:\n');

for i = 1:height(Results)

    if Results.Significant_FDR_0_05(i)

        fprintf('%-20s  FDR p = %.4g\n', ...
            Results.Feature{i}, ...
            Results.FDR_P_Value(i));

    end

end

fprintf('\n========================================\n');
fprintf('STATISTICAL ANALYSIS COMPLETE\n');
fprintf('========================================\n');