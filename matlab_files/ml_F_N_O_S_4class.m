%% ========================================================
% 4-CLASS EEG CLASSIFICATION
% F vs N vs O vs S
%
% F = Interictal
% N = Interictal
% O = Healthy, eyes closed
% S = Ictal / seizure
%
% Models:
% 1. Logistic Regression
% 2. k-NN
% 3. Decision Tree
% 4. SVM-RBF
% 5. Random Forest
% 6. Bagged Trees
%
% Validation:
% 5-fold stratified cross-validation
% Training-fold-only normalization
%
% Metrics:
% Accuracy
% Macro Precision
% Macro Recall
% Macro F1
%
% Output:
% F_N_O_S_4class_results.csv
% F_N_O_S_4class_oof_predictions.csv
% ========================================================

clear;
clc;
close all;

rng(42);

fprintf('\n========================================\n');
fprintf('4-CLASS EEG CLASSIFICATION\n');
fprintf('F vs N vs O vs S\n');
fprintf('========================================\n\n');


%% ========================================================
% LOAD DATA
% ========================================================

data = readtable('f_n_o_s_features.csv');

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

X = data{:, feature_names};

% Group labels
Group = categorical(data.Group);

fprintf('Total recordings: %d\n', size(X,1));
fprintf('Total features: %d\n\n', size(X,2));

fprintf('Class distribution:\n');

categories_list = categories(Group);

for i = 1:length(categories_list)
    fprintf('%s: %d\n', ...
        categories_list{i}, ...
        sum(Group == categories_list{i}));
end

fprintf('\n');


%% ========================================================
% 5-FOLD CROSS VALIDATION
% ========================================================

cv = cvpartition(Group, 'KFold', 5);

num_folds = cv.NumTestSets;
num_models = 6;

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

Accuracy  = zeros(num_folds, num_models);
Precision = zeros(num_folds, num_models);
Recall    = zeros(num_folds, num_models);
F1        = zeros(num_folds, num_models);

% Out-of-fold predictions
OOF_Predictions = strings(length(Group), num_models);

% Convert actual labels to strings
Actual_Labels = string(Group);


%% ========================================================
% CROSS-VALIDATION LOOP
% ========================================================

for fold = 1:num_folds

    fprintf('Processing Fold %d / %d...\n', fold, num_folds);

    train_idx = training(cv, fold);
    test_idx  = test(cv, fold);

    Xtrain = X(train_idx,:);
    Xtest  = X(test_idx,:);

    ytrain = Group(train_idx);
    ytest  = Group(test_idx);


    %% ====================================================
    % TRAINING-ONLY NORMALIZATION
    % ====================================================

    mu = mean(Xtrain, 1);
    sigma = std(Xtrain, 0, 1);

    % Avoid division by zero
    sigma(sigma == 0) = 1;

    Xtrain_norm = (Xtrain - mu) ./ sigma;
    Xtest_norm  = (Xtest  - mu) ./ sigma;


    %% ====================================================
    % MODEL 1: LOGISTIC REGRESSION
    % Multiclass ECOC
    % ====================================================

    fprintf('  Logistic Regression...\n');

    t = templateLinear( ...
        'Learner','logistic');

    Mdl = fitcecoc( ...
        Xtrain_norm, ...
        ytrain, ...
        'Learners',t, ...
        'Coding','onevsall');

    pred = predict(Mdl, Xtest_norm);

    [Accuracy(fold,1), ...
     Precision(fold,1), ...
     Recall(fold,1), ...
     F1(fold,1)] = classification_metrics( ...
        ytest, pred);

    OOF_Predictions(test_idx,1) = string(pred);


    %% ====================================================
    % MODEL 2: k-NN
    % ====================================================

    fprintf('  k-NN...\n');

    Mdl = fitcknn( ...
        Xtrain_norm, ...
        ytrain, ...
        'NumNeighbors',5, ...
        'Distance','euclidean');

    pred = predict(Mdl, Xtest_norm);

    [Accuracy(fold,2), ...
     Precision(fold,2), ...
     Recall(fold,2), ...
     F1(fold,2)] = classification_metrics( ...
        ytest, pred);

    OOF_Predictions(test_idx,2) = string(pred);


    %% ====================================================
    % MODEL 3: DECISION TREE
    % ====================================================

    fprintf('  Decision Tree...\n');

    Mdl = fitctree( ...
        Xtrain_norm, ...
        ytrain, ...
        'MaxNumSplits',20);

    pred = predict(Mdl, Xtest_norm);

    [Accuracy(fold,3), ...
     Precision(fold,3), ...
     Recall(fold,3), ...
     F1(fold,3)] = classification_metrics( ...
        ytest, pred);

    OOF_Predictions(test_idx,3) = string(pred);


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

    pred = predict(Mdl, Xtest_norm);

    [Accuracy(fold,4), ...
     Precision(fold,4), ...
     Recall(fold,4), ...
     F1(fold,4)] = classification_metrics( ...
        ytest, pred);

    OOF_Predictions(test_idx,4) = string(pred);


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

    pred = predict(Mdl, Xtest_norm);

    % TreeBagger returns cell array of character labels
    pred = categorical(pred);

    [Accuracy(fold,5), ...
     Precision(fold,5), ...
     Recall(fold,5), ...
     F1(fold,5)] = classification_metrics( ...
        ytest, pred);

    OOF_Predictions(test_idx,5) = string(pred);


    %% ====================================================
    % MODEL 6: BAGGED TREES
    % Multiclass-compatible ensemble
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

    pred = predict(Mdl, Xtest_norm);

    [Accuracy(fold,6), ...
     Precision(fold,6), ...
     Recall(fold,6), ...
     F1(fold,6)] = classification_metrics( ...
        ytest, pred);

    OOF_Predictions(test_idx,6) = string(pred);

end


%% ========================================================
% DISPLAY RESULTS
% ========================================================

fprintf('\n');
fprintf('========================================\n');
fprintf('5-FOLD CROSS-VALIDATION RESULTS\n');
fprintf('========================================\n\n');

for m = 1:num_models

    fprintf('%s\n', model_names{m});

    fprintf('Accuracy : %.2f +/- %.2f %%\n', ...
        mean(Accuracy(:,m))*100, ...
        std(Accuracy(:,m))*100);

    fprintf('Precision: %.2f +/- %.2f %%\n', ...
        mean(Precision(:,m))*100, ...
        std(Precision(:,m))*100);

    fprintf('Recall   : %.2f +/- %.2f %%\n', ...
        mean(Recall(:,m))*100, ...
        std(Recall(:,m))*100);

    fprintf('F1 Score : %.2f +/- %.2f %%\n\n', ...
        mean(F1(:,m))*100, ...
        std(F1(:,m))*100);

end


%% ========================================================
% CREATE RESULTS TABLE
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
    'VariableNames', { ...
    'Model', ...
    'AccuracyMean', ...
    'AccuracyStd', ...
    'PrecisionMean', ...
    'PrecisionStd', ...
    'RecallMean', ...
    'RecallStd', ...
    'F1Mean', ...
    'F1Std'});


%% ========================================================
% SAVE RESULTS
% ========================================================

writetable( ...
    Results, ...
    'F_N_O_S_4class_results.csv');

fprintf('Results saved as:\n');
fprintf('F_N_O_S_4class_results.csv\n\n');


%% ========================================================
% SAVE OUT-OF-FOLD PREDICTIONS
% ========================================================

OOF_Table = table( ...
    Actual_Labels, ...
    OOF_Predictions(:,1), ...
    OOF_Predictions(:,2), ...
    OOF_Predictions(:,3), ...
    OOF_Predictions(:,4), ...
    OOF_Predictions(:,5), ...
    OOF_Predictions(:,6), ...
    'VariableNames', { ...
    'Actual', ...
    'LogisticRegression', ...
    'kNN', ...
    'DecisionTree', ...
    'SVM_RBF', ...
    'RandomForest', ...
    'BaggedTrees'});

writetable( ...
    OOF_Table, ...
    'F_N_O_S_4class_oof_predictions.csv');

fprintf('Out-of-fold predictions saved as:\n');
fprintf('F_N_O_S_4class_oof_predictions.csv\n\n');


%% ========================================================
% OUT-OF-FOLD CONFUSION MATRICES
% ========================================================

fprintf('========================================\n');
fprintf('OUT-OF-FOLD CONFUSION MATRICES\n');
fprintf('========================================\n');

for m = 1:num_models

    actual = categorical(OOF_Table.Actual);
    predicted = categorical(OOF_Table{:,m+1});

    figure;

    confusionchart(actual, predicted);

    title(['4-Class EEG - ', model_names{m}]);

end


%% ========================================================
% LOCAL FUNCTION
% ========================================================

function [accuracy, macro_precision, macro_recall, macro_f1] = ...
    classification_metrics(y_true, y_pred)

    classes = categories(y_true);

    num_classes = length(classes);

    precision = zeros(num_classes,1);
    recall = zeros(num_classes,1);
    f1 = zeros(num_classes,1);

    % Overall accuracy
    accuracy = mean(y_true == y_pred);

    for i = 1:num_classes

        current_class = categorical( ...
            classes(i), ...
            classes);

        TP = sum((y_true == current_class) & ...
                 (y_pred == current_class));

        FP = sum((y_true ~= current_class) & ...
                 (y_pred == current_class));

        FN = sum((y_true == current_class) & ...
                 (y_pred ~= current_class));

        if (TP + FP) > 0
            precision(i) = TP / (TP + FP);
        else
            precision(i) = 0;
        end

        if (TP + FN) > 0
            recall(i) = TP / (TP + FN);
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

    % Macro-averaging
    macro_precision = mean(precision);
    macro_recall    = mean(recall);
    macro_f1        = mean(f1);

end