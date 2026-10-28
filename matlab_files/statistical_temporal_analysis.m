clear;
clc;

fprintf('Loading temporal feature dataset...\n');

T = readtable('temporal_nonseizure_vs_seizure_features.csv');

% ------------------------------------------------------------
% Separate features and labels
% ------------------------------------------------------------

X = T{:,1:15};
Y = T.Label;

% ------------------------------------------------------------
% Feature names
% ------------------------------------------------------------

featureNames = {
    'Delta_MeanTransition'
    'Delta_MaxTransition'
    'Delta_Drift'
    'Theta_MeanTransition'
    'Theta_MaxTransition'
    'Theta_Drift'
    'Alpha_MeanTransition'
    'Alpha_MaxTransition'
    'Alpha_Drift'
    'Beta_MeanTransition'
    'Beta_MaxTransition'
    'Beta_Drift'
    'Gamma_MeanTransition'
    'Gamma_MaxTransition'
    'Gamma_Drift'
};

% ------------------------------------------------------------
% Separate groups
% ------------------------------------------------------------

X_non = X(Y == 0,:);
X_seiz = X(Y == 1,:);

fprintf('\nData summary:\n');
fprintf('Non-seizure recordings : %d\n', size(X_non,1));
fprintf('Seizure recordings     : %d\n', size(X_seiz,1));
fprintf('Temporal features      : %d\n', size(X,2));

% ------------------------------------------------------------
% Statistical analysis
% ------------------------------------------------------------

nFeatures = size(X,2);

MeanNon = zeros(nFeatures,1);
MeanSeiz = zeros(nFeatures,1);

CohenD = zeros(nFeatures,1);

PValue = zeros(nFeatures,1);

% ------------------------------------------------------------
% Cohen's d
% ------------------------------------------------------------

for i = 1:nFeatures

    x1 = X_non(:,i);
    x2 = X_seiz(:,i);

    MeanNon(i) = mean(x1);
    MeanSeiz(i) = mean(x2);

    % Pooled standard deviation
    n1 = length(x1);
    n2 = length(x2);

    s1 = std(x1);
    s2 = std(x2);

    pooledStd = sqrt( ...
        ((n1-1)*s1^2 + (n2-1)*s2^2) ...
        / (n1+n2-2) );

    CohenD(i) = (MeanSeiz(i) - MeanNon(i)) / pooledStd;

    % Mann-Whitney U / Wilcoxon rank-sum
    PValue(i) = ranksum(x1,x2);

end

% ------------------------------------------------------------
% Benjamini-Hochberg FDR correction
% ------------------------------------------------------------

[sortedP, sortIdx] = sort(PValue);

m = nFeatures;

FDR_sorted = sortedP .* m ./ (1:m)';

% Enforce monotonicity
for i = m-1:-1:1
    FDR_sorted(i) = min(FDR_sorted(i),FDR_sorted(i+1));
end

FDR_PValue = zeros(m,1);
FDR_PValue(sortIdx) = min(FDR_sorted,1);

% ------------------------------------------------------------
% Load RF importance from Step 6
% ------------------------------------------------------------

fprintf('\nLoading Random Forest importance...\n');

RF = readtable('rf_combined_25_feature_importance.csv');

% Match temporal feature names
RFImportance = zeros(nFeatures,1);

for i = 1:nFeatures

    idx = find(strcmp(RF.Feature, featureNames{i}));

    if ~isempty(idx)
        RFImportance(i) = RF.Importance(idx(1));
    else
        RFImportance(i) = NaN;
    end

end

% ------------------------------------------------------------
% Create result table
% ------------------------------------------------------------

Results = table( ...
    featureNames, ...
    MeanNon, ...
    MeanSeiz, ...
    CohenD, ...
    PValue, ...
    FDR_PValue, ...
    RFImportance, ...
    'VariableNames', { ...
    'Feature', ...
    'Mean_NonSeizure', ...
    'Mean_Seizure', ...
    'CohensD', ...
    'PValue', ...
    'FDR_PValue', ...
    'RF_Importance'});

% ------------------------------------------------------------
% Sort by absolute Cohen's d
% ------------------------------------------------------------

[~,idx] = sort(abs(Results.CohensD),'descend');

Results = Results(idx,:);

% ------------------------------------------------------------
% Display
% ------------------------------------------------------------

fprintf('\n');
fprintf('============================================================\n');
fprintf('TEMPORAL FEATURE STATISTICAL ANALYSIS\n');
fprintf('============================================================\n');

disp(Results);

% ------------------------------------------------------------
% Save
% ------------------------------------------------------------

writetable(Results,'statistical_temporal_analysis.csv');

fprintf('\nResults saved to:\n');
fprintf('statistical_temporal_analysis.csv\n');

% ------------------------------------------------------------
% Significant features
% ------------------------------------------------------------

fprintf('\n============================================================\n');
fprintf('SIGNIFICANCE SUMMARY\n');
fprintf('============================================================\n');

sig = Results.FDR_PValue < 0.05;

fprintf('Significant temporal features after FDR: %d / %d\n', ...
    sum(sig), nFeatures);

fprintf('\n');

for i = 1:height(Results)

    if Results.FDR_PValue(i) < 0.05

        fprintf('%-30s  Cohen d = %8.4f  FDR p = %.4e\n', ...
            Results.Feature{i}, ...
            Results.CohensD(i), ...
            Results.FDR_PValue(i));

    end

end

% ------------------------------------------------------------
% Plot Cohen's d
% ------------------------------------------------------------

figure;

bar(Results.CohensD);

xticks(1:nFeatures);
xticklabels(Results.Feature);
xtickangle(45);

ylabel('Cohen''s d');
title('Effect Size of Temporal Features');

grid on;

saveas(gcf,'temporal_cohens_d.png');

% ------------------------------------------------------------
% Plot RF importance
% ------------------------------------------------------------

figure;

bar(Results.RF_Importance);

xticks(1:nFeatures);
xticklabels(Results.Feature);
xtickangle(45);

ylabel('Random Forest Importance');
title('Random Forest Importance of Temporal Features');

grid on;

saveas(gcf,'temporal_rf_importance.png');

fprintf('\nPlots saved:\n');
fprintf('temporal_cohens_d.png\n');
fprintf('temporal_rf_importance.png\n');

fprintf('\n============================================================\n');
fprintf('STEP 7 COMPLETE\n');
fprintf('============================================================\n');