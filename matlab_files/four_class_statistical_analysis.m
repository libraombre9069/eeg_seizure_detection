%% four_class_statistical_analysis.m
% Statistical analysis of F, N, O, S EEG recordings
%
% F = Interictal
% N = Interictal
% O = Healthy, eyes closed
% S = Ictal / seizure

clear;
clc;
close all;

fprintf('========================================\n');
fprintf('F / N / O / S STATISTICAL ANALYSIS\n');
fprintf('========================================\n\n');


%% ============================================================
% 1. LOAD DATA
% =============================================================

T = readtable('f_n_o_s_features.csv');

feature_names = T.Properties.VariableNames(1:end-1);
X = T{:,1:end-1};
Group = string(T.Group);

groups = ["F","N","O","S"];

n_features = length(feature_names);


%% ============================================================
% 2. CHECK DATA
% =============================================================

fprintf('Total recordings: %d\n',height(T));

for g = 1:length(groups)
    fprintf('%s: %d recordings\n', ...
        groups(g), sum(Group == groups(g)));
end

fprintf('Features: %d\n\n',n_features);


%% ============================================================
% 3. DESCRIPTIVE STATISTICS
% =============================================================

Mean_F = zeros(n_features,1);
Mean_N = zeros(n_features,1);
Mean_O = zeros(n_features,1);
Mean_S = zeros(n_features,1);

Std_F = zeros(n_features,1);
Std_N = zeros(n_features,1);
Std_O = zeros(n_features,1);
Std_S = zeros(n_features,1);

Median_F = zeros(n_features,1);
Median_N = zeros(n_features,1);
Median_O = zeros(n_features,1);
Median_S = zeros(n_features,1);


for i = 1:n_features

    F = X(Group=="F",i);
    N = X(Group=="N",i);
    O = X(Group=="O",i);
    S = X(Group=="S",i);

    Mean_F(i) = mean(F);
    Mean_N(i) = mean(N);
    Mean_O(i) = mean(O);
    Mean_S(i) = mean(S);

    Std_F(i) = std(F);
    Std_N(i) = std(N);
    Std_O(i) = std(O);
    Std_S(i) = std(S);

    Median_F(i) = median(F);
    Median_N(i) = median(N);
    Median_O(i) = median(O);
    Median_S(i) = median(S);

end


%% ============================================================
% 4. KRUSKAL-WALLIS TEST
%
% Tests whether at least one of the four groups differs.
% =============================================================

P_KW = zeros(n_features,1);

for i = 1:n_features

    data = X(:,i);

    P_KW(i) = kruskalwallis( ...
        data, ...
        Group, ...
        'off');

end


%% ============================================================
% 5. FDR CORRECTION
% =============================================================

[sorted_p, sort_idx] = sort(P_KW);

m = n_features;

adjusted_sorted = sorted_p .* m ./ (1:m)';

for i = m-1:-1:1
    adjusted_sorted(i) = min( ...
        adjusted_sorted(i), ...
        adjusted_sorted(i+1));
end

FDR_P = zeros(n_features,1);

FDR_P(sort_idx) = min(adjusted_sorted,1);

Significant = FDR_P < 0.05;


%% ============================================================
% 6. CREATE RESULTS TABLE
% =============================================================

Results = table( ...
    feature_names(:), ...
    Mean_F, Std_F, Median_F, ...
    Mean_N, Std_N, Median_N, ...
    Mean_O, Std_O, Median_O, ...
    Mean_S, Std_S, Median_S, ...
    P_KW, ...
    FDR_P, ...
    Significant, ...
    'VariableNames',{ ...
    'Feature', ...
    'Mean_F','Std_F','Median_F', ...
    'Mean_N','Std_N','Median_N', ...
    'Mean_O','Std_O','Median_O', ...
    'Mean_S','Std_S','Median_S', ...
    'KruskalWallis_P', ...
    'FDR_P', ...
    'Significant_FDR_0_05'});


%% ============================================================
% 7. DISPLAY RESULTS
% =============================================================

fprintf('\n========================================\n');
fprintf('KRUSKAL-WALLIS RESULTS\n');
fprintf('========================================\n\n');

disp(Results);


%% ============================================================
% 8. SAVE RESULTS
% =============================================================

writetable( ...
    Results, ...
    'f_n_o_s_statistical_analysis.csv');

fprintf('\nSaved:\n');
fprintf('f_n_o_s_statistical_analysis.csv\n');


%% ============================================================
% 9. BOXPLOTS
% =============================================================

fprintf('\nGenerating boxplots...\n');

for i = 1:n_features

    figure;

    boxplot( ...
        X(:,i), ...
        categorical(Group,groups));

    xlabel('Recording group');
    ylabel(feature_names{i});

    title(sprintf( ...
        '%s: F vs N vs O vs S', ...
        feature_names{i}), ...
        'Interpreter','none');

    grid on;

end


%% ============================================================
% 10. GROUP MEAN PLOT FOR EACH FEATURE
% =============================================================

for i = 1:n_features

    figure;

    means = [ ...
        Mean_F(i), ...
        Mean_N(i), ...
        Mean_O(i), ...
        Mean_S(i)];

    bar(means);

    set(gca, ...
        'XTick',1:4, ...
        'XTickLabel',groups);

    xlabel('Recording group');
    ylabel(feature_names{i});

    title(sprintf( ...
        'Mean %s across F / N / O / S', ...
        feature_names{i}), ...
        'Interpreter','none');

    grid on;

end


%% ============================================================
% 11. P-VALUE PLOT
% =============================================================

figure;

neglog_p = -log10(max(FDR_P,realmin));

bar(neglog_p);

hold on;

yline( ...
    -log10(0.05), ...
    '--', ...
    'FDR = 0.05');

hold off;

set(gca, ...
    'XTick',1:n_features, ...
    'XTickLabel',feature_names, ...
    'XTickLabelRotation',45);

xlabel('Feature');
ylabel('-log_{10}(FDR-adjusted p-value)');

title('F / N / O / S Statistical Significance');

grid on;


%% ============================================================
% 12. SUMMARY
% =============================================================

fprintf('\n========================================\n');
fprintf('SIGNIFICANT FEATURES\n');
fprintf('========================================\n\n');

for i = 1:n_features

    if Significant(i)

        fprintf('%-20s  FDR p = %.4g\n', ...
            feature_names{i}, ...
            FDR_P(i));

    end

end


fprintf('\n========================================\n');
fprintf('ANALYSIS COMPLETE\n');
fprintf('========================================\n');