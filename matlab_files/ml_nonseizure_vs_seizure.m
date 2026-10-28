%% ========================================================
% NON-SEIZURE vs SEIZURE EEG CLASSIFICATION
%
% Non-seizure:
%   Z = Healthy, eyes open
%   O = Healthy, eyes closed
%   F = Interictal
%   N = Interictal
%
% Seizure:
%   S = Ictal
%
% Total:
%   Non-seizure = 400
%   Seizure     = 100
%   Total       = 500
%
% Features:
%   10 global EEG features
%
% Models:
%   1. Logistic Regression
%   2. k-NN
%   3. Decision Tree
%   4. SVM-RBF
%   5. Random Forest
%   6. Bagged Trees
%
% Validation:
%   Stratified 5-fold cross-validation
%   Training-fold-only normalization
%
% Metrics:
%   Accuracy
%   Precision
%   Recall
%   F1
%   ROC-AUC
%
% Outputs:
%   nonseizure_vs_seizure_features.csv
%   nonseizure_vs_seizure_results.csv
%   nonseizure_vs_seizure_oof_predictions.csv
%
% ========================================================

clear;
clc;
close all;

rng(42);

fprintf('\n');
fprintf('========================================\n');
fprintf('NON-SEIZURE vs SEIZURE EEG CLASSIFICATION\n');
fprintf('========================================\n\n');


%% ========================================================
% SETTINGS
% ========================================================

Fs = 173.61;

% Folder names
folders = {'non_seizure','O','F','N','seizure'};

% Class labels
% Z/O/F/N -> 0 = Non-seizure
% S       -> 1 = Seizure
classes = [0 0 0 0 1];

folder_names = {'Z','O','F','N','S'};


%% ========================================================
% FEATURE NAMES
% ========================================================

feature_names = { ...
    'Std', ...
    'PeakToPeak', ...
    'RelativeDelta', ...
    'RelativeTheta', ...
    'RelativeAlpha', ...
    'RelativeBeta', ...
    'RelativeGamma', ...
    'DominantFrequency', ...
    'SpectralCentroid', ...
    'SpectralEntropy'};


%% ========================================================
% FEATURE EXTRACTION
% ========================================================

X = [];
y = [];
Group = strings(0,1);

fprintf('========================================\n');
fprintf('FEATURE EXTRACTION\n');
fprintf('========================================\n\n');

for f = 1:length(folders)

    current_folder = folders{f};

    % Look for uppercase TXT
    files = dir(fullfile(current_folder,'*.TXT'));

    % If none found, look for lowercase txt
    if isempty(files)
        files = dir(fullfile(current_folder,'*.txt'));
    end

    fprintf('%s folder: %d files found\n', ...
        folder_names{f}, length(files));

    if isempty(files)

        warning('No TXT files found in folder: %s', ...
            current_folder);

        continue;

    end


    %% Process every recording

    for k = 1:length(files)

        filepath = fullfile( ...
            files(k).folder, ...
            files(k).name);

        fprintf('Processing %s/%s\n', ...
            folder_names{f}, ...
            files(k).name);

        % Existing feature extraction function
        features = extract_features(filepath, Fs);

        X = [X; features];

        y = [y; classes(f)];

        Group = [Group; ...
            string(folder_names{f})];

    end

end


%% ========================================================
% CHECK DATA
% ========================================================

fprintf('\n');
fprintf('========================================\n');
fprintf('DATASET SUMMARY\n');
fprintf('========================================\n\n');

fprintf('Total recordings: %d\n', size(X,1));
fprintf('Total features: %d\n\n', size(X,2));

fprintf('Class distribution:\n');

fprintf('Non-seizure: %d\n', sum(y == 0));
fprintf('Seizure:     %d\n', sum(y == 1));


%% ========================================================
% SAVE FEATURE DATASET
% ========================================================

T = array2table( ...
    X, ...
    'VariableNames', feature_names);

T.Group = Group;
T.Class = y;

writetable( ...
    T, ...
    'nonseizure_vs_seizure_features.csv');

fprintf('\nFeature dataset saved as:\n');
fprintf('nonseizure_vs_seizure_features.csv\n');


%% ========================================================
% 5-FOLD STRATIFIED CROSS-VALIDATION
% ========================================================

fprintf('\n');
fprintf('========================================\n');
fprintf('5-FOLD CROSS-VALIDATION\n');
fprintf('========================================\n\n');

Y = categorical(y, [0 1], ...
    {'NonSeizure','Seizure'});

cv = cvpartition(Y,'KFold',5);

num_folds = cv.NumTestSets;
num_models = 6;


%% ========================================================
% MODEL NAMES
% ========================================================

model_names = { ...
    'Logistic Regression', ...
    'k-NN', ...
    'Decision Tree', ...
    'SVM-RBF', ...
    'Random Forest', ...
    'Bagged Trees'};


%% ========================================================
% STORAGE
% ========================================================

Accuracy  = zeros(num_folds,num_models);
Precision = zeros(num_folds,num_models);
Recall    = zeros(num_folds,num_models);
F1        = zeros(num_folds,num_models);
AUC       = zeros(num_folds,num_models);


% Out-of-fold predictions
OOF_Predictions = categorical( ...
    strings(size(X,1),num_models));

% Out-of-fold scores
OOF_Scores = NaN(size(X,1),num_models);


%% ========================================================
% CROSS-VALIDATION LOOP
% ========================================================

for fold = 1:num_folds

    fprintf('\n');
    fprintf('Processing Fold %d / %d...\n', ...
        fold,num_folds);


    %% ----------------------------------------------------
    % TRAIN / TEST INDICES
    % ----------------------------------------------------

    train_idx = training(cv,fold);
    test_idx  = test(cv,fold);

    Xtrain = X(train_idx,:);
    Xtest  = X(test_idx,:);

    ytrain = Y(train_idx);
    ytest  = Y(test_idx);


    %% ----------------------------------------------------
    % TRAINING-ONLY NORMALIZATION
    % ----------------------------------------------------

    mu = mean(Xtrain,1);

    sigma = std(Xtrain,0,1);

    % Prevent division by zero
    sigma(sigma == 0) = 1;

    Xtrain_norm = ...
        (Xtrain - mu) ./ sigma;

    Xtest_norm = ...
        (Xtest - mu) ./ sigma;


    %% ====================================================
    % MODEL 1: LOGISTIC REGRESSION
    % ====================================================

    fprintf('  Logistic Regression...\n');

    t = templateLinear( ...
        'Learner','logistic');

    Mdl = fitcecoc( ...
        Xtrain_norm, ...
        ytrain, ...
        'Learners',t, ...
        'Coding','onevsall');

    [pred,score] = predict( ...
        Mdl, ...
        Xtest_norm);

    [Accuracy(fold,1), ...
     Precision(fold,1), ...
     Recall(fold,1), ...
     F1(fold,1)] = ...
        classification_metrics(ytest,pred);

    AUC(fold,1) = calculate_auc( ...
        ytest, ...
        score, ...
        Mdl.ClassNames);

    OOF_Predictions(test_idx,1) = pred;

    % Find score corresponding to Seizure class
    seizure_idx = ...
        find(Mdl.ClassNames == categorical("Seizure"));

    OOF_Scores(test_idx,1) = ...
        score(:,seizure_idx);


    %% ====================================================
    % MODEL 2: k-NN
    % ====================================================

    fprintf('  k-NN...\n');

    Mdl = fitcknn( ...
        Xtrain_norm, ...
        ytrain, ...
        'NumNeighbors',5, ...
        'Distance','euclidean');

    [pred,score] = predict( ...
        Mdl, ...
        Xtest_norm);

    [Accuracy(fold,2), ...
     Precision(fold,2), ...
     Recall(fold,2), ...
     F1(fold,2)] = ...
        classification_metrics(ytest,pred);

    AUC(fold,2) = calculate_auc( ...
        ytest, ...
        score, ...
        Mdl.ClassNames);

    OOF_Predictions(test_idx,2) = pred;

    seizure_idx = ...
        find(Mdl.ClassNames == categorical("Seizure"));

    OOF_Scores(test_idx,2) = ...
        score(:,seizure_idx);


    %% ====================================================
    % MODEL 3: DECISION TREE
    % ====================================================

    fprintf('  Decision Tree...\n');

    Mdl = fitctree( ...
        Xtrain_norm, ...
        ytrain, ...
        'MaxNumSplits',20);

    [pred,score] = predict( ...
        Mdl, ...
        Xtest_norm);

    [Accuracy(fold,3), ...
     Precision(fold,3), ...
     Recall(fold,3), ...
     F1(fold,3)] = ...
        classification_metrics(ytest,pred);

    AUC(fold,3) = calculate_auc( ...
        ytest, ...
        score, ...
        Mdl.ClassNames);

    OOF_Predictions(test_idx,3) = pred;

    seizure_idx = ...
        find(Mdl.ClassNames == categorical("Seizure"));

    OOF_Scores(test_idx,3) = ...
        score(:,seizure_idx);


    %% ====================================================
    % MODEL 4: SVM-RBF
    % ====================================================

    fprintf('  SVM-RBF...\n');

    t = templateSVM( ...
        'KernelFunction','rbf', ...
        'KernelScale','auto', ...
        'Standardize',false);

    Mdl = fitcecoc( ...
        Xtrain_norm, ...
        ytrain, ...
        'Learners',t, ...
        'Coding','onevsone');

    [pred,score] = predict( ...
        Mdl, ...
        Xtest_norm);

    [Accuracy(fold,4), ...
     Precision(fold,4), ...
     Recall(fold,4), ...
     F1(fold,4)] = ...
        classification_metrics(ytest,pred);

    AUC(fold,4) = calculate_auc( ...
        ytest, ...
        score, ...
        Mdl.ClassNames);

    OOF_Predictions(test_idx,4) = pred;

    % ECOC score is not necessarily a calibrated probability.
    % Still useful for ranking observations for ROC-AUC.
    seizure_idx = ...
        find(Mdl.ClassNames == categorical("Seizure"));

    OOF_Scores(test_idx,4) = ...
        score(:,seizure_idx);


    %% ====================================================
    % MODEL 5: RANDOM FOREST
    % ====================================================

    fprintf('  Random Forest...\n');

    Mdl = TreeBagger( ...
        200, ...
        Xtrain_norm, ...
        ytrain, ...
        'Method','classification', ...
        'OOBPrediction','off');

    [pred_cell,score] = predict( ...
        Mdl, ...
        Xtest_norm);

    pred = categorical( ...
        pred_cell, ...
        {'NonSeizure','Seizure'});

    [Accuracy(fold,5), ...
     Precision(fold,5), ...
     Recall(fold,5), ...
     F1(fold,5)] = ...
        classification_metrics(ytest,pred);

    % TreeBagger score columns follow ClassNames
    seizure_idx = ...
        find(strcmp(Mdl.ClassNames,'Seizure'));

    OOF_Scores(test_idx,5) = ...
        score(:,seizure_idx);

    AUC(fold,5) = calculate_auc_from_binary( ...
        ytest, ...
        OOF_Scores(test_idx,5));

    OOF_Predictions(test_idx,5) = pred;


    %% ====================================================
    % MODEL 6: BAGGED TREES
    % ====================================================

    fprintf('  Bagged Trees...\n');

    template = templateTree( ...
        'MaxNumSplits',20);

    Mdl = fitcensemble( ...
        Xtrain_norm, ...
        ytrain, ...
        'Method','Bag', ...
        'NumLearningCycles',100, ...
        'Learners',template);

    [pred,score] = predict( ...
        Mdl, ...
        Xtest_norm);

    [Accuracy(fold,6), ...
     Precision(fold,6), ...
     Recall(fold,6), ...
     F1(fold,6)] = ...
        classification_metrics(ytest,pred);

    AUC(fold,6) = calculate_auc( ...
        ytest, ...
        score, ...
        Mdl.ClassNames);

    OOF_Predictions(test_idx,6) = pred;

    seizure_idx = ...
        find(Mdl.ClassNames == categorical("Seizure"));

    OOF_Scores(test_idx,6) = ...
        score(:,seizure_idx);

end


%% ========================================================
% DISPLAY RESULTS
% ========================================================

fprintf('\n');
fprintf('========================================\n');
fprintf('FINAL 5-FOLD RESULTS\n');
fprintf('========================================\n\n');

for m = 1:num_models

    fprintf('%s\n',model_names{m});

    fprintf('Accuracy : %.2f +/- %.2f %%\n', ...
        mean(Accuracy(:,m))*100, ...
        std(Accuracy(:,m))*100);

    fprintf('Precision: %.2f +/- %.2f %%\n', ...
        mean(Precision(:,m))*100, ...
        std(Precision(:,m))*100);

    fprintf('Recall   : %.2f +/- %.2f %%\n', ...
        mean(Recall(:,m))*100, ...
        std(Recall(:,m))*100);

    fprintf('F1 Score : %.2f +/- %.2f %%\n', ...
        mean(F1(:,m))*100, ...
        std(F1(:,m))*100);

    fprintf('ROC-AUC  : %.4f +/- %.4f\n\n', ...
        mean(AUC(:,m)), ...
        std(AUC(:,m)));

end


%% ========================================================
% RESULTS TABLE
% ========================================================

Results = table( ...
    model_names', ...
    mean(Accuracy,1)'*100, ...
    std(Accuracy,0,1)'*100, ...
    mean(Precision,1)'*100, ...
    std(Precision,0,1)'*100, ...
    mean(Recall,1)'*100, ...
    std(Recall,0,1)'*100, ...
    mean(F1,1)'*100, ...
    std(F1,0,1)'*100, ...
    mean(AUC,1)', ...
    std(AUC,0,1)', ...
    'VariableNames', { ...
    'Model', ...
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


%% ========================================================
% SAVE RESULTS
% ========================================================

writetable( ...
    Results, ...
    'nonseizure_vs_seizure_results.csv');

fprintf('Results saved as:\n');
fprintf('nonseizure_vs_seizure_results.csv\n');


%% ========================================================
% SAVE OUT-OF-FOLD PREDICTIONS
% ========================================================

OOF_Table = table( ...
    Group, ...
    Y, ...
    OOF_Predictions(:,1), ...
    OOF_Predictions(:,2), ...
    OOF_Predictions(:,3), ...
    OOF_Predictions(:,4), ...
    OOF_Predictions(:,5), ...
    OOF_Predictions(:,6), ...
    'VariableNames', { ...
    'RecordingGroup', ...
    'ActualClass', ...
    'LogisticRegression', ...
    'kNN', ...
    'DecisionTree', ...
    'SVM_RBF', ...
    'RandomForest', ...
    'BaggedTrees'});

writetable( ...
    OOF_Table, ...
    'nonseizure_vs_seizure_oof_predictions.csv');

fprintf('\nOut-of-fold predictions saved as:\n');
fprintf('nonseizure_vs_seizure_oof_predictions.csv\n');


%% ========================================================
% OUT-OF-FOLD CONFUSION MATRICES
% ========================================================

fprintf('\n');
fprintf('========================================\n');
fprintf('OUT-OF-FOLD CONFUSION MATRICES\n');
fprintf('========================================\n');

for m = 1:num_models

    actual = Y;

    predicted = OOF_Predictions(:,m);

    figure;

    confusionchart( ...
        actual, ...
        predicted);

    title([ ...
        'Non-seizure vs Seizure - ', ...
        model_names{m}]);

end


%% ========================================================
% OVERALL OOF ROC CURVES
% ========================================================

figure;
hold on;

for m = 1:num_models

    valid = ~isnan(OOF_Scores(:,m));

    [Xroc,Yroc,~,AUC_oof] = perfcurve( ...
        Y(valid), ...
        OOF_Scores(valid,m), ...
        'Seizure');

    plot( ...
        Xroc, ...
        Yroc, ...
        'LineWidth',1.5);

    fprintf( ...
        'Overall OOF AUC - %s: %.4f\n', ...
        model_names{m}, ...
        AUC_oof);

end

xlabel('False Positive Rate');
ylabel('True Positive Rate');

title('Out-of-Fold ROC Curves');

legend(model_names, ...
    'Location','southeast');

grid on;
hold off;


%% ========================================================
% LOCAL FUNCTION:
% CLASSIFICATION METRICS
% ========================================================

function [accuracy,macro_precision,macro_recall,macro_f1] = ...
    classification_metrics(y_true,y_pred)

    classes = categories(y_true);

    num_classes = length(classes);

    precision = zeros(num_classes,1);
    recall = zeros(num_classes,1);
    f1 = zeros(num_classes,1);

    accuracy = mean(y_true == y_pred);

    for i = 1:num_classes

        current_class = categorical( ...
            classes(i), ...
            classes);

        TP = sum( ...
            (y_true == current_class) & ...
            (y_pred == current_class));

        FP = sum( ...
            (y_true ~= current_class) & ...
            (y_pred == current_class));

        FN = sum( ...
            (y_true == current_class) & ...
            (y_pred ~= current_class));


        if (TP + FP) > 0
            precision(i) = ...
                TP / (TP + FP);
        else
            precision(i) = 0;
        end


        if (TP + FN) > 0
            recall(i) = ...
                TP / (TP + FN);
        else
            recall(i) = 0;
        end


        if (precision(i) + recall(i)) > 0

            f1(i) = ...
                2 * precision(i) * recall(i) / ...
                (precision(i) + recall(i));

        else

            f1(i) = 0;

        end

    end


    macro_precision = mean(precision);
    macro_recall = mean(recall);
    macro_f1 = mean(f1);

end


%% ========================================================
% LOCAL FUNCTION:
% AUC
% ========================================================

function auc = calculate_auc( ...
    y_true,score,class_names)

    seizure_idx = ...
        find(class_names == categorical("Seizure"));

    if isempty(seizure_idx)

        auc = NaN;
        return;

    end

    seizure_score = ...
        score(:,seizure_idx);

    auc = calculate_auc_from_binary( ...
        y_true, ...
        seizure_score);

end


%% ========================================================
% LOCAL FUNCTION:
% BINARY AUC
% ========================================================

function auc = calculate_auc_from_binary( ...
    y_true,seizure_score)

    try

        [~,~,~,auc] = perfcurve( ...
            y_true, ...
            seizure_score, ...
            'Seizure');

    catch

        auc = NaN;

    end

end