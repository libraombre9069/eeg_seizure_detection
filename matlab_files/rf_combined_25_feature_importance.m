%% RF FEATURE IMPORTANCE - COMBINED 25 FEATURES
%
% 10 Global features + 15 Temporal features
%
% Input:
%   nonseizure_vs_seizure_features.csv
%   temporal_nonseizure_vs_seizure_features.csv
%
% Output:
%   rf_combined_25_feature_importance.csv
%   rf_combined_25_feature_importance.png
%
% The original CSV files are NOT modified.

clear;
clc;
close all;

%% ============================================================
% 1. SETTINGS
% =============================================================

rng(42);

globalFile   = 'nonseizure_vs_seizure_features.csv';
temporalFile = 'temporal_nonseizure_vs_seizure_features.csv';

outputCSV = 'rf_combined_25_feature_importance.csv';
outputFig = 'rf_combined_25_feature_importance.png';


%% ============================================================
% 2. LOAD DATA
% =============================================================

fprintf('Loading global feature dataset...\n');

G = readtable(globalFile);

fprintf('Loading temporal feature dataset...\n');

T = readtable(temporalFile);


%% ============================================================
% 3. EXTRACT FEATURES
% ============================================================

% First 10 columns of global dataset
X_global = G{:,1:10};

% Class labels
Y = G.Class;


% First 15 columns of temporal dataset
X_temporal = T{:,1:15};


%% ============================================================
% 4. CHECK DATA
% ============================================================

fprintf('\nChecking data...\n');

fprintf('Global recordings   : %d\n', size(X_global,1));
fprintf('Global features     : %d\n', size(X_global,2));

fprintf('Temporal recordings : %d\n', size(X_temporal,1));
fprintf('Temporal features   : %d\n', size(X_temporal,2));

fprintf('Total recordings    : %d\n', length(Y));

fprintf('Non-seizure         : %d\n', sum(Y == 0));
fprintf('Seizure             : %d\n', sum(Y == 1));


if size(X_global,1) ~= size(X_temporal,1)
    error('Global and temporal datasets have different numbers of recordings.');
end


if any(Y ~= T.Label)
    error('Class labels do not match between the two CSV files.');
end


%% ============================================================
% 5. COMBINE 10 GLOBAL + 15 TEMPORAL FEATURES
% ============================================================

X = [
    X_global ...
    X_temporal
    ];


fprintf('\nCombined feature matrix:\n');

fprintf('Recordings : %d\n', size(X,1));
fprintf('Features   : %d\n', size(X,2));


%% ============================================================
% 6. FEATURE NAMES
% ============================================================

featureNames = {
    'Std'
    'PeakToPeak'
    'RelativeDelta'
    'RelativeTheta'
    'RelativeAlpha'
    'RelativeBeta'
    'RelativeGamma'
    'DominantFrequency'
    'SpectralCentroid'
    'SpectralEntropy'

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


%% ============================================================
% 7. VERIFY FEATURE COUNT
% ============================================================

if size(X,2) ~= length(featureNames)
    error('Number of feature names does not match number of features.');
end


fprintf('\nAll 25 features:\n');

for i = 1:length(featureNames)

    fprintf('%2d. %s\n', ...
        i, featureNames{i});

end


%% ============================================================
% 8. TRAIN RANDOM FOREST
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('TRAINING RANDOM FOREST\n');
fprintf('============================================================\n');

numTrees = 200;

RF = TreeBagger( ...
    numTrees, ...
    X, ...
    Y, ...
    'Method','classification', ...
    'OOBPrediction','on', ...
    'OOBPredictorImportance','on');


fprintf('\nRandom Forest trained.\n');
fprintf('Number of trees: %d\n', numTrees);


%% ============================================================
% 9. EXTRACT OOB PERMUTATION IMPORTANCE
% ============================================================

importance = RF.OOBPermutedPredictorDeltaError;


%% ============================================================
% 10. CREATE IMPORTANCE TABLE
% ============================================================

importanceTable = table( ...
    featureNames, ...
    importance(:), ...
    'VariableNames', ...
    {'Feature','Importance'});


%% ============================================================
% 11. SORT BY IMPORTANCE
% ============================================================

importanceTable = sortrows( ...
    importanceTable, ...
    'Importance', ...
    'descend');


%% ============================================================
% 12. ADD RANK
% ============================================================

importanceTable.Rank = ...
    (1:height(importanceTable))';


% Reorder columns
importanceTable = importanceTable(:, ...
    {'Rank','Feature','Importance'});


%% ============================================================
% 13. DISPLAY RESULTS
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('25-FEATURE RANDOM FOREST IMPORTANCE\n');
fprintf('============================================================\n');

disp(importanceTable);


%% ============================================================
% 14. SAVE RESULTS
% ============================================================

writetable( ...
    importanceTable, ...
    outputCSV);


fprintf('\nImportance results saved to:\n');
fprintf('%s\n', outputCSV);


%% ============================================================
% 15. DISPLAY TOP 10 FEATURES
% ============================================================

fprintf('\n');
fprintf('TOP 10 FEATURES:\n');
fprintf('------------------------------------------------------------\n');

topN = min(10,height(importanceTable));

for i = 1:topN

    fprintf( ...
        '%2d. %-30s %.6f\n', ...
        importanceTable.Rank(i), ...
        importanceTable.Feature{i}, ...
        importanceTable.Importance(i));

end


%% ============================================================
% 16. PLOT FEATURE IMPORTANCE
% ============================================================

figure;

barh(importanceTable.Importance);

set(gca, ...
    'YTick',1:height(importanceTable), ...
    'YTickLabel',importanceTable.Feature, ...
    'YDir','reverse');

xlabel('OOB Permuted Predictor Delta Error');

ylabel('Feature');

title('Random Forest Importance - 25 Combined Features');

grid on;


%% ============================================================
% 17. SAVE FIGURE
% ============================================================

saveas(gcf,outputFig);

fprintf('\nFeature-importance plot saved to:\n');
fprintf('%s\n', outputFig);


%% ============================================================
% 18. END
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('STEP 6 COMPLETE\n');
fprintf('============================================================\n');
