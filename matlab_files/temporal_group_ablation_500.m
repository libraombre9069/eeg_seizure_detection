%% TEMPORAL GROUP ABLATION - 500 EEG RECORDINGS
% Tests whether different groups of temporal features add information
% beyond the 10 conventional global EEG features.
%
% Input files:
%   1. nonseizure_vs_seizure_features.csv
%   2. temporal_nonseizure_vs_seizure_features.csv
%
% IMPORTANT:
%   - Input CSV files are NOT modified.
%   - Results are saved separately as:
%       temporal_group_ablation_500_results.csv
%
% Feature groups:
%   Global
%   Global + MeanTransition
%   Global + MaxTransition
%   Global + Drift
%   Global + All Temporal
%
% Models:
%   Random Forest
%   Bagged Trees
%
% Validation:
%   5-fold cross-validation
%   Training-fold-only normalization
%   rng(42)

clear;
clc;
close all;

%% ============================================================
% 1. SETTINGS
% =============================================================

rng(42);

K = 5;

globalFile  = 'nonseizure_vs_seizure_features.csv';
temporalFile = 'temporal_nonseizure_vs_seizure_features.csv';

outputFile = 'temporal_group_ablation_500_results.csv';


%% ============================================================
% 2. LOAD DATA
% =============================================================

fprintf('Loading global feature dataset...\n');

G = readtable(globalFile);

fprintf('Loading temporal feature dataset...\n');

T = readtable(temporalFile);


%% ============================================================
% 3. DISPLAY COLUMN NAMES
% =============================================================

fprintf('\nGlobal dataset variables:\n');
disp(G.Properties.VariableNames');

fprintf('\nTemporal dataset variables:\n');
disp(T.Properties.VariableNames');


%% ============================================================
% 4. EXTRACT FEATURES AND LABELS
% ============================================================

% Global CSV structure:
%
% First 10 columns:
%   Std
%   PeakToPeak
%   RelativeDelta
%   RelativeTheta
%   RelativeAlpha
%   RelativeBeta
%   RelativeGamma
%   DominantFrequency
%   SpectralCentroid
%   SpectralEntropy
%
% Then:
%   Group
%   Class

X_global = G{:,1:10};

Y_global = G.Class;


% Temporal CSV structure:
%
% First 15 columns:
%
% Delta_MeanTransition
% Delta_MaxTransition
% Delta_Drift
%
% Theta_MeanTransition
% Theta_MaxTransition
% Theta_Drift
%
% Alpha_MeanTransition
% Alpha_MaxTransition
% Alpha_Drift
%
% Beta_MeanTransition
% Beta_MaxTransition
% Beta_Drift
%
% Gamma_MeanTransition
% Gamma_MaxTransition
% Gamma_Drift

X_temporal = T{:,1:15};

Y_temporal = T.Label;


%% ============================================================
% 5. CHECK DATA CONSISTENCY
% =============================================================

fprintf('\nChecking dataset dimensions...\n');

fprintf('Global recordings   : %d\n', size(X_global,1));
fprintf('Global features     : %d\n', size(X_global,2));

fprintf('Temporal recordings : %d\n', size(X_temporal,1));
fprintf('Temporal features   : %d\n', size(X_temporal,2));


if size(X_global,1) ~= size(X_temporal,1)
    error('Global and temporal datasets contain different numbers of recordings.');
end


if any(Y_global ~= Y_temporal)
    error('Class labels do not match between global and temporal datasets.');
end


Y = Y_global;

fprintf('\nTotal recordings: %d\n', length(Y));

fprintf('Non-seizure recordings: %d\n', sum(Y == 0));
fprintf('Seizure recordings    : %d\n', sum(Y == 1));


%% ============================================================
% 6. DEFINE TEMPORAL FEATURE GROUPS
% =============================================================

% Temporal feature ordering:
%
% 1  Delta_MeanTransition
% 2  Delta_MaxTransition
% 3  Delta_Drift
%
% 4  Theta_MeanTransition
% 5  Theta_MaxTransition
% 6  Theta_Drift
%
% 7  Alpha_MeanTransition
% 8  Alpha_MaxTransition
% 9  Alpha_Drift
%
% 10 Beta_MeanTransition
% 11 Beta_MaxTransition
% 12 Beta_Drift
%
% 13 Gamma_MeanTransition
% 14 Gamma_MaxTransition
% 15 Gamma_Drift


meanTransitionIdx = [1 4 7 10 13];

maxTransitionIdx  = [2 5 8 11 14];

driftIdx          = [3 6 9 12 15];


%% ============================================================
% 7. CREATE FEATURE SETS
% =============================================================

featureSetNames = {
    'Global'
    'Global + MeanTransition'
    'Global + MaxTransition'
    'Global + Drift'
    'Global + All Temporal'
    };


X_sets = cell(5,1);

% ------------------------------------------------------------
% Feature Set 1: Global only
% ------------------------------------------------------------

X_sets{1} = X_global;


% ------------------------------------------------------------
% Feature Set 2: Global + MeanTransition
% ------------------------------------------------------------

X_sets{2} = [
    X_global ...
    X_temporal(:,meanTransitionIdx)
    ];


% ------------------------------------------------------------
% Feature Set 3: Global + MaxTransition
% ------------------------------------------------------------

X_sets{3} = [
    X_global ...
    X_temporal(:,maxTransitionIdx)
    ];


% ------------------------------------------------------------
% Feature Set 4: Global + Drift
% ------------------------------------------------------------

X_sets{4} = [
    X_global ...
    X_temporal(:,driftIdx)
    ];


% ------------------------------------------------------------
% Feature Set 5: Global + All Temporal
% ------------------------------------------------------------

X_sets{5} = [
    X_global ...
    X_temporal
    ];


%% ============================================================
% 8. DISPLAY FEATURE COUNTS
% =============================================================

fprintf('\nFeature-set sizes:\n');

for i = 1:length(X_sets)

    fprintf('%-28s : %d features\n', ...
        featureSetNames{i}, size(X_sets{i},2));

end


%% ============================================================
% 9. CREATE 5-FOLD CROSS-VALIDATION PARTITION
% =============================================================

cv = cvpartition(Y,'KFold',K);


%% ============================================================
% 10. RESULT STORAGE
% =============================================================

results = table();


%% ============================================================
% 11. LOOP THROUGH FEATURE SETS
% =============================================================

for fs = 1:length(X_sets)

    fprintf('\n');
    fprintf('============================================================\n');
    fprintf('FEATURE SET: %s\n', featureSetNames{fs});
    fprintf('============================================================\n');

    X = X_sets{fs};


    %% --------------------------------------------------------
    % Store fold-level metrics
    % ---------------------------------------------------------

    rfAccuracy = zeros(K,1);
    rfPrecision = zeros(K,1);
    rfRecall = zeros(K,1);
    rfF1 = zeros(K,1);
    rfAUC = zeros(K,1);

    bagAccuracy = zeros(K,1);
    bagPrecision = zeros(K,1);
    bagRecall = zeros(K,1);
    bagF1 = zeros(K,1);
    bagAUC = zeros(K,1);


    %% ========================================================
    % 12. CROSS-VALIDATION
    % ========================================================

    for fold = 1:K

        fprintf('\nFold %d/%d\n', fold, K);


        %% ----------------------------------------------------
        % TRAIN / TEST INDICES
        % -----------------------------------------------------

        trainIdx = training(cv,fold);
        testIdx  = test(cv,fold);

        X_train = X(trainIdx,:);
        X_test  = X(testIdx,:);

        Y_train = Y(trainIdx);
        Y_test  = Y(testIdx);


        %% ----------------------------------------------------
        % TRAINING-FOLD-ONLY NORMALIZATION
        % -----------------------------------------------------

        mu = mean(X_train,1);

        sigma = std(X_train,0,1);

        % Prevent division by zero for constant features
        sigma(sigma == 0) = 1;


        X_train_norm = ...
            (X_train - mu) ./ sigma;

        X_test_norm = ...
            (X_test - mu) ./ sigma;


        %% ====================================================
        % RANDOM FOREST
        % ====================================================

        fprintf('  Training Random Forest...\n');

        RF = TreeBagger( ...
            200, ...
            X_train_norm, ...
            Y_train, ...
            'Method','classification', ...
            'OOBPrediction','off');


        [predRF,scoreRF] = predict(RF,X_test_norm);


        % TreeBagger returns labels as cell strings
        predRF = str2double(predRF);


        %% ----------------------------------------------------
        % RF CONFUSION MATRIX
        % -----------------------------------------------------

        TP = sum((predRF == 1) & (Y_test == 1));

        TN = sum((predRF == 0) & (Y_test == 0));

        FP = sum((predRF == 1) & (Y_test == 0));

        FN = sum((predRF == 0) & (Y_test == 1));


        %% ----------------------------------------------------
        % RF METRICS
        % -----------------------------------------------------

        accuracy = (TP + TN) / ...
            (TP + TN + FP + FN);


        if (TP + FP) == 0
            precision = 0;
        else
            precision = TP / (TP + FP);
        end


        if (TP + FN) == 0
            recall = 0;
        else
            recall = TP / (TP + FN);
        end


        if (precision + recall) == 0
            F1 = 0;
        else
            F1 = 2 * precision * recall / ...
                (precision + recall);
        end


        %% ----------------------------------------------------
        % RF ROC-AUC
        % -----------------------------------------------------

        % TreeBagger score columns correspond to classes.
        % Find the column corresponding to class 1.

        classNamesRF = RF.ClassNames;

        if iscell(classNamesRF)
            classNamesRF = str2double(classNamesRF);
        end

        seizureColumn = find(classNamesRF == 1);


        if isempty(seizureColumn)

            AUC = NaN;

        else

            [~,~,~,AUC] = perfcurve( ...
                Y_test, ...
                scoreRF(:,seizureColumn), ...
                1);

        end


        %% ----------------------------------------------------
        % STORE RF FOLD RESULTS
        % -----------------------------------------------------

        rfAccuracy(fold) = accuracy;

        rfPrecision(fold) = precision;

        rfRecall(fold) = recall;

        rfF1(fold) = F1;

        rfAUC(fold) = AUC;


        %% ====================================================
        % BAGGED TREES
        % ====================================================

        fprintf('  Training Bagged Trees...\n');

        Bag = fitcensemble( ...
            X_train_norm, ...
            Y_train, ...
            'Method','Bag', ...
            'NumLearningCycles',100);


        [predBag,scoreBag] = predict(Bag,X_test_norm);


        %% ----------------------------------------------------
        % BAGGED CONFUSION MATRIX
        % -----------------------------------------------------

        TP = sum((predBag == 1) & (Y_test == 1));

        TN = sum((predBag == 0) & (Y_test == 0));

        FP = sum((predBag == 1) & (Y_test == 0));

        FN = sum((predBag == 0) & (Y_test == 1));


        %% ----------------------------------------------------
        % BAGGED METRICS
        % -----------------------------------------------------

        accuracy = (TP + TN) / ...
            (TP + TN + FP + FN);


        if (TP + FP) == 0
            precision = 0;
        else
            precision = TP / (TP + FP);
        end


        if (TP + FN) == 0
            recall = 0;
        else
            recall = TP / (TP + FN);
        end


        if (precision + recall) == 0
            F1 = 0;
        else
            F1 = 2 * precision * recall / ...
                (precision + recall);
        end


        %% ----------------------------------------------------
        % BAGGED ROC-AUC
        % -----------------------------------------------------

        classNamesBag = Bag.ClassNames;

        if iscell(classNamesBag)
            classNamesBag = str2double(classNamesBag);
        end

        seizureColumn = find(classNamesBag == 1);


        if isempty(seizureColumn)

            AUC = NaN;

        else

            [~,~,~,AUC] = perfcurve( ...
                Y_test, ...
                scoreBag(:,seizureColumn), ...
                1);

        end


        %% ----------------------------------------------------
        % STORE BAGGED FOLD RESULTS
        % -----------------------------------------------------

        bagAccuracy(fold) = accuracy;

        bagPrecision(fold) = precision;

        bagRecall(fold) = recall;

        bagF1(fold) = F1;

        bagAUC(fold) = AUC;


        %% ----------------------------------------------------
        % DISPLAY FOLD RESULTS
        % -----------------------------------------------------

        fprintf('    RF     Accuracy = %.2f%% | F1 = %.2f%% | AUC = %.4f\n', ...
            rfAccuracy(fold)*100, ...
            rfF1(fold)*100, ...
            rfAUC(fold));


        fprintf('    Bagged Accuracy = %.2f%% | F1 = %.2f%% | AUC = %.4f\n', ...
            bagAccuracy(fold)*100, ...
            bagF1(fold)*100, ...
            bagAUC(fold));

    end


    %% ========================================================
    % 13. CALCULATE MEANS AND STANDARD DEVIATIONS
    % ========================================================

    %% Random Forest

    row = table( ...
        string(featureSetNames{fs}), ...
        "Random Forest", ...
        size(X,2), ...
        mean(rfAccuracy)*100, ...
        std(rfAccuracy)*100, ...
        mean(rfPrecision)*100, ...
        std(rfPrecision)*100, ...
        mean(rfRecall)*100, ...
        std(rfRecall)*100, ...
        mean(rfF1)*100, ...
        std(rfF1)*100, ...
        mean(rfAUC), ...
        std(rfAUC), ...
        'VariableNames', { ...
        'FeatureSet', ...
        'Model', ...
        'NumFeatures', ...
        'AccuracyMean', ...
        'AccuracyStd', ...
        'PrecisionMean', ...
        'PrecisionStd', ...
        'RecallMean', ...
        'RecallStd', ...
        'F1Mean', ...
        'F1Std', ...
        'AUCMean', ...
        'AUCStd'});


    results = [results; row];


    %% Bagged Trees

    row = table( ...
        string(featureSetNames{fs}), ...
        "Bagged Trees", ...
        size(X,2), ...
        mean(bagAccuracy)*100, ...
        std(bagAccuracy)*100, ...
        mean(bagPrecision)*100, ...
        std(bagPrecision)*100, ...
        mean(bagRecall)*100, ...
        std(bagRecall)*100, ...
        mean(bagF1)*100, ...
        std(bagF1)*100, ...
        mean(bagAUC), ...
        std(bagAUC), ...
        'VariableNames', { ...
        'FeatureSet', ...
        'Model', ...
        'NumFeatures', ...
        'AccuracyMean', ...
        'AccuracyStd', ...
        'PrecisionMean', ...
        'PrecisionStd', ...
        'RecallMean', ...
        'RecallStd', ...
        'F1Mean', ...
        'F1Std', ...
        'AUCMean', ...
        'AUCStd'});


    results = [results; row];


    %% ========================================================
    % 14. DISPLAY SUMMARY FOR CURRENT FEATURE SET
    % ========================================================

    fprintf('\nSummary for %s:\n', featureSetNames{fs});

    fprintf('Random Forest:\n');

    fprintf('  Accuracy : %.2f +/- %.2f %%\n', ...
        mean(rfAccuracy)*100, ...
        std(rfAccuracy)*100);

    fprintf('  Precision: %.2f +/- %.2f %%\n', ...
        mean(rfPrecision)*100, ...
        std(rfPrecision)*100);

    fprintf('  Recall   : %.2f +/- %.2f %%\n', ...
        mean(rfRecall)*100, ...
        std(rfRecall)*100);

    fprintf('  F1       : %.2f +/- %.2f %%\n', ...
        mean(rfF1)*100, ...
        std(rfF1)*100);

    fprintf('  AUC      : %.4f +/- %.4f\n', ...
        mean(rfAUC), ...
        std(rfAUC));


    fprintf('\nBagged Trees:\n');

    fprintf('  Accuracy : %.2f +/- %.2f %%\n', ...
        mean(bagAccuracy)*100, ...
        std(bagAccuracy)*100);

    fprintf('  Precision: %.2f +/- %.2f %%\n', ...
        mean(bagPrecision)*100, ...
        std(bagPrecision)*100);

    fprintf('  Recall   : %.2f +/- %.2f %%\n', ...
        mean(bagRecall)*100, ...
        std(bagRecall)*100);

    fprintf('  F1       : %.2f +/- %.2f %%\n', ...
        mean(bagF1)*100, ...
        std(bagF1)*100);

    fprintf('  AUC      : %.4f +/- %.4f\n', ...
        mean(bagAUC), ...
        std(bagAUC));

end


%% ============================================================
% 15. SAVE RESULTS
% =============================================================

writetable(results,outputFile);

fprintf('\n');
fprintf('============================================================\n');
fprintf('ABLATION COMPLETE\n');
fprintf('============================================================\n');

fprintf('\nResults saved to:\n');
fprintf('%s\n',outputFile);


%% ============================================================
% 16. DISPLAY FINAL COMPARISON TABLE
% =============================================================

fprintf('\nFINAL RESULTS:\n\n');

disp(results);


%% ============================================================
% 17. OPTIONAL: DISPLAY COMPACT TABLE
% ============================================================

fprintf('\nCompact comparison:\n\n');

compactResults = results(:, ...
    {'FeatureSet','Model','NumFeatures', ...
     'AccuracyMean','AccuracyStd', ...
     'F1Mean','F1Std', ...
     'AUCMean','AUCStd'});

disp(compactResults);


%% ============================================================
% END
% ============================================================