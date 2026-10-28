clear;
clc;

fprintf('============================================================\n');
fprintf('STEP 9: REPEATED CROSS-VALIDATION ROBUSTNESS\n');
fprintf('============================================================\n');

%% ------------------------------------------------------------
% SETTINGS
% -------------------------------------------------------------

rng(42);

numRepeats = 10;
numFolds = 5;
numTrees = 200;

%% ------------------------------------------------------------
% LOAD DATA
% -------------------------------------------------------------

fprintf('\nLoading global feature dataset...\n');

G = readtable('nonseizure_vs_seizure_features.csv');

fprintf('Loading temporal feature dataset...\n');

T = readtable('temporal_nonseizure_vs_seizure_features.csv');

%% ------------------------------------------------------------
% DATA
% -------------------------------------------------------------

X_global = G{:,1:10};
X_temporal = T{:,1:15};

Y = G.Class;

%% ------------------------------------------------------------
% CHECK DATA
% -------------------------------------------------------------

if height(G) ~= height(T)
    error('Global and temporal datasets have different numbers of recordings.');
end

if any(Y ~= T.Label)
    error('Global and temporal labels do not match.');
end

fprintf('\nDataset:\n');
fprintf('Recordings       : %d\n',length(Y));
fprintf('Global features  : %d\n',size(X_global,2));
fprintf('Temporal features: %d\n',size(X_temporal,2));

fprintf('Non-seizure      : %d\n',sum(Y==0));
fprintf('Seizure          : %d\n',sum(Y==1));

%% ------------------------------------------------------------
% FEATURE SETS
% -------------------------------------------------------------

FeatureSetNames = {
    'Global'
    'Global + MeanTransition'
    'Global + MaxTransition'
    'Global + All Temporal'
};

X_sets = {
    X_global
    [X_global X_temporal(:,[1 4 7 10 13])]
    [X_global X_temporal(:,[2 5 8 11 14])]
    [X_global X_temporal]
};

fprintf('\nFeature sets:\n');

for i = 1:length(X_sets)
    fprintf('%-28s : %d features\n', ...
        FeatureSetNames{i},size(X_sets{i},2));
end

%% ------------------------------------------------------------
% RESULT STORAGE
% -------------------------------------------------------------

numSets = length(X_sets);

allResults = [];

%% ============================================================
% REPEATED CROSS-VALIDATION
% =============================================================

for s = 1:numSets

    X = X_sets{s};

    fprintf('\n');
    fprintf('============================================================\n');
    fprintf('FEATURE SET: %s\n',FeatureSetNames{s});
    fprintf('Features: %d\n',size(X,2));
    fprintf('============================================================\n');

    accuracy = zeros(numRepeats,numFolds);
    precision = zeros(numRepeats,numFolds);
    recall = zeros(numRepeats,numFolds);
    f1 = zeros(numRepeats,numFolds);
    auc = zeros(numRepeats,numFolds);

    for r = 1:numRepeats

        % Different random seed for each repetition
        rng(1000 + r);

        % Stratified 5-fold partition
        cv = cvpartition(Y,'KFold',numFolds);

        for k = 1:numFolds

            trainIdx = training(cv,k);
            testIdx = test(cv,k);

            Xtrain = X(trainIdx,:);
            Xtest = X(testIdx,:);

            Ytrain = Y(trainIdx);
            Ytest = Y(testIdx);

            %% ------------------------------------------------
            % TRAINING-ONLY NORMALIZATION
            % -------------------------------------------------

            mu = mean(Xtrain,1);
            sigma = std(Xtrain,0,1);

            sigma(sigma == 0) = 1;

            Xtrain_norm = (Xtrain - mu) ./ sigma;
            Xtest_norm = (Xtest - mu) ./ sigma;

            %% ------------------------------------------------
            % RANDOM FOREST
            % -------------------------------------------------

            RF = TreeBagger( ...
                numTrees, ...
                Xtrain_norm, ...
                Ytrain, ...
                'Method','classification', ...
                'MinLeafSize',5, ...
                'NumPredictorsToSample','all');

            %% ------------------------------------------------
            % PREDICTION
            % -------------------------------------------------

            [predictedLabels, scores] = predict(RF,Xtest_norm);

            predictedLabels = str2double(predictedLabels);

            %% ------------------------------------------------
            % CONFUSION MATRIX
            % -------------------------------------------------

            TP = sum((Ytest == 1) & (predictedLabels == 1));
            TN = sum((Ytest == 0) & (predictedLabels == 0));
            FP = sum((Ytest == 0) & (predictedLabels == 1));
            FN = sum((Ytest == 1) & (predictedLabels == 0));

            %% ------------------------------------------------
            % METRICS
            % -------------------------------------------------

            accuracy(r,k) = ...
                (TP + TN) / length(Ytest);

            if (TP + FP) > 0
                precision(r,k) = TP / (TP + FP);
            else
                precision(r,k) = 0;
            end

            if (TP + FN) > 0
                recall(r,k) = TP / (TP + FN);
            else
                recall(r,k) = 0;
            end

            if (precision(r,k) + recall(r,k)) > 0
                f1(r,k) = ...
                    2 * precision(r,k) * recall(r,k) / ...
                    (precision(r,k) + recall(r,k));
            else
                f1(r,k) = 0;
            end

            %% ------------------------------------------------
            % ROC AUC
            % -------------------------------------------------

            % TreeBagger normally returns scores for classes
            % in the order shown in RF.ClassNames.
            classNames = RF.ClassNames;

            positiveClassIdx = find( ...
                str2double(classNames) == 1);

            positiveScores = scores(:,positiveClassIdx);

            try
                [~,~,~,auc(r,k)] = perfcurve( ...
                    Ytest, ...
                    positiveScores, ...
                    1);
            catch
                auc(r,k) = NaN;
            end

        end

        fprintf('Repeat %2d/%d complete\n',r,numRepeats);

    end

    %% --------------------------------------------------------
    % STORE RESULTS
    % ---------------------------------------------------------

    resultRow = table( ...
        string(FeatureSetNames{s}), ...
        size(X,2), ...
        mean(accuracy(:))*100, ...
        std(accuracy(:))*100, ...
        mean(precision(:))*100, ...
        std(precision(:))*100, ...
        mean(recall(:))*100, ...
        std(recall(:))*100, ...
        mean(f1(:))*100, ...
        std(f1(:))*100, ...
        mean(auc(:),'omitnan'), ...
        std(auc(:),'omitnan'), ...
        'VariableNames', { ...
        'FeatureSet', ...
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

    allResults = [allResults; resultRow];

end

%% ============================================================
% FINAL RESULTS
% =============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('REPEATED CV RESULTS\n');
fprintf('============================================================\n');

disp(allResults);

%% ------------------------------------------------------------
% SAVE
% -------------------------------------------------------------

writetable(allResults, ...
    'repeated_cv_temporal_robustness.csv');

fprintf('\nResults saved to:\n');
fprintf('repeated_cv_temporal_robustness.csv\n');

%% ============================================================
% COMPARISON WITH GLOBAL
% =============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf('CHANGE RELATIVE TO GLOBAL FEATURES\n');
fprintf('============================================================\n');

baseAccuracy = allResults.AccuracyMean(1);
baseF1 = allResults.F1Mean(1);
baseAUC = allResults.AUCMean(1);

for i = 2:height(allResults)

    fprintf('\n%s\n',allResults.FeatureSet(i));

    fprintf('Accuracy change : %+0.4f percentage points\n', ...
        allResults.AccuracyMean(i) - baseAccuracy);

    fprintf('F1 change       : %+0.4f percentage points\n', ...
        allResults.F1Mean(i) - baseF1);

    fprintf('AUC change      : %+0.6f\n', ...
        allResults.AUCMean(i) - baseAUC);

end

%% ============================================================
% PLOTS
% =============================================================

%% Accuracy

figure;

bar(allResults.AccuracyMean);

xticks(1:height(allResults));
xticklabels(allResults.FeatureSet);
xtickangle(30);

ylabel('Accuracy (%)');
title('Repeated-CV Accuracy');

grid on;

saveas(gcf,'repeated_cv_accuracy.png');

%% F1

figure;

bar(allResults.F1Mean);

xticks(1:height(allResults));
xticklabels(allResults.FeatureSet);
xtickangle(30);

ylabel('F1 Score (%)');
title('Repeated-CV F1 Score');

grid on;

saveas(gcf,'repeated_cv_f1.png');

%% AUC

figure;

bar(allResults.AUCMean);

xticks(1:height(allResults));
xticklabels(allResults.FeatureSet);
xtickangle(30);

ylabel('ROC-AUC');
title('Repeated-CV ROC-AUC');

grid on;

saveas(gcf,'repeated_cv_auc.png');

fprintf('\nPlots saved.\n');

fprintf('\n');
fprintf('============================================================\n');
fprintf('STEP 9 COMPLETE\n');
fprintf('============================================================\n');
