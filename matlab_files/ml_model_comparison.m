%% ============================================================
% ML MODEL COMPARISON
% Bonn EEG Dataset: Z vs S
%
% Models:
% 1. Logistic Regression
% 2. k-NN
% 3. Decision Tree
% 4. SVM-RBF
% 5. Random Forest
% 6. Gradient Boosting
%
% Evaluation:
% 5-fold cross-validation
% Accuracy, Precision, Recall, F1, ROC-AUC
%
% Normalization is performed INSIDE each training fold.
%% ============================================================

clear;
clc;
close all;

rng(42);

%% ============================================================
% 1. LOAD DATA
% =============================================================

data = readtable('eeg_features.csv');

fprintf('===============================================\n');
fprintf('ML MODEL COMPARISON\n');
fprintf('===============================================\n');

disp('Columns in dataset:');
disp(data.Properties.VariableNames');

%% ============================================================
% 2. SELECT FEATURES
% =============================================================

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

X = zeros(height(data), length(feature_names));

for i = 1:length(feature_names)
    X(:,i) = data.(feature_names{i});
end

%% ============================================================
% 3. CLASS LABEL
% =============================================================

if iscell(data.Class)

    y = zeros(height(data),1);

    for i = 1:height(data)

        if strcmpi(data.Class{i},'Seizure')
            y(i) = 1;
        else
            y(i) = 0;
        end

    end

elseif isstring(data.Class)

    y = double(strcmpi(data.Class,'Seizure'));

elseif iscategorical(data.Class)

    y = double(strcmpi(string(data.Class),'Seizure'));

else

    y = data.Class;

end

fprintf('\nDataset:\n');
fprintf('Total samples : %d\n',length(y));
fprintf('Non-seizure   : %d\n',sum(y==0));
fprintf('Seizure       : %d\n',sum(y==1));

%% ============================================================
% 4. 5-FOLD CROSS VALIDATION
% =============================================================

cv = cvpartition(y,'KFold',5);

num_folds = cv.NumTestSets;

%% ============================================================
% 5. MODEL NAMES
% =============================================================

model_names = { ...
    'Logistic Regression', ...
    'k-NN', ...
    'Decision Tree', ...
    'SVM-RBF', ...
    'Random Forest', ...
    'Gradient Boosting'};

num_models = length(model_names);

%% ============================================================
% 6. STORAGE
% =============================================================

accuracy  = zeros(num_folds,num_models);
precision = zeros(num_folds,num_models);
recall    = zeros(num_folds,num_models);
f1        = zeros(num_folds,num_models);
auc       = zeros(num_folds,num_models);

%% ============================================================
% 7. CROSS-VALIDATION
% =============================================================

for fold = 1:num_folds

    fprintf('\n');
    fprintf('===============================================\n');
    fprintf('FOLD %d / %d\n',fold,num_folds);
    fprintf('===============================================\n');

    %% --------------------------------------------------------
    % TRAIN / TEST SPLIT
    % ---------------------------------------------------------

    train_idx = training(cv,fold);
    test_idx  = test(cv,fold);

    X_train = X(train_idx,:);
    y_train = y(train_idx);

    X_test = X(test_idx,:);
    y_test = y(test_idx);

    %% --------------------------------------------------------
    % NORMALIZATION
    % Training statistics ONLY
    % ---------------------------------------------------------

    mu = mean(X_train,1);

    sigma = std(X_train,0,1);

    % Avoid division by zero
    sigma(sigma == 0) = 1;

    X_train_norm = (X_train - mu) ./ sigma;
    X_test_norm  = (X_test - mu) ./ sigma;

    %% ========================================================
    % MODEL 1: LOGISTIC REGRESSION
    % =========================================================

    mdl = fitclinear( ...
        X_train_norm, ...
        y_train, ...
        'Learner','logistic');

    [pred,score] = predict(mdl,X_test_norm);

    [accuracy(fold,1), ...
     precision(fold,1), ...
     recall(fold,1), ...
     f1(fold,1), ...
     auc(fold,1)] = ...
        calculate_metrics(y_test,pred,score,mdl);


    %% ========================================================
    % MODEL 2: k-NN
    % =========================================================

    mdl = fitcknn( ...
        X_train_norm, ...
        y_train, ...
        'NumNeighbors',5, ...
        'Standardize',false);

    [pred,score] = predict(mdl,X_test_norm);

    [accuracy(fold,2), ...
     precision(fold,2), ...
     recall(fold,2), ...
     f1(fold,2), ...
     auc(fold,2)] = ...
        calculate_metrics(y_test,pred,score,mdl);


    %% ========================================================
    % MODEL 3: DECISION TREE
    % =========================================================

    mdl = fitctree( ...
        X_train_norm, ...
        y_train);

    [pred,score] = predict(mdl,X_test_norm);

    [accuracy(fold,3), ...
     precision(fold,3), ...
     recall(fold,3), ...
     f1(fold,3), ...
     auc(fold,3)] = ...
        calculate_metrics(y_test,pred,score,mdl);


    %% ========================================================
    % MODEL 4: SVM-RBF
    % =========================================================

    mdl = fitcsvm( ...
        X_train_norm, ...
        y_train, ...
        'KernelFunction','rbf', ...
        'KernelScale','auto', ...
        'Standardize',false);

    [pred,score] = predict(mdl,X_test_norm);

    [accuracy(fold,4), ...
     precision(fold,4), ...
     recall(fold,4), ...
     f1(fold,4), ...
     auc(fold,4)] = ...
        calculate_metrics(y_test,pred,score,mdl);


    %% ========================================================
    % MODEL 5: RANDOM FOREST
    % =========================================================

    mdl = TreeBagger( ...
        100, ...
        X_train_norm, ...
        y_train, ...
        'Method','classification', ...
        'OOBPrediction','off');

    [pred_cell,score] = predict(mdl,X_test_norm);

    % Convert predicted class from cell/string to numeric
    pred = str2double(string(pred_cell));

    % TreeBagger can store ClassNames as different data types.
    % Convert them to string first.
    class_names = string(mdl.ClassNames);

    % Find column corresponding to seizure class = 1
    seizure_column = find(class_names == "1",1);

    if isempty(seizure_column)

        error('Could not find seizure class (1) in Random Forest ClassNames.');

    end

    seizure_score = score(:,seizure_column);

    [accuracy(fold,5), ...
     precision(fold,5), ...
     recall(fold,5), ...
     f1(fold,5), ...
     auc(fold,5)] = ...
        calculate_metrics_score( ...
        y_test, ...
        pred, ...
        seizure_score);


    %% ========================================================
    % MODEL 6: GRADIENT BOOSTING
    % =========================================================

    mdl = fitcensemble( ...
        X_train_norm, ...
        y_train, ...
        'Method','LogitBoost', ...
        'NumLearningCycles',100);

    [pred,score] = predict(mdl,X_test_norm);

    class_names = string(mdl.ClassNames);

    % Find seizure class = 1
    seizure_column = find(class_names == "1",1);

    if isempty(seizure_column)

        error('Could not find seizure class (1) in Gradient Boosting ClassNames.');

    end

    seizure_score = score(:,seizure_column);

    [accuracy(fold,6), ...
     precision(fold,6), ...
     recall(fold,6), ...
     f1(fold,6), ...
     auc(fold,6)] = ...
        calculate_metrics_score( ...
        y_test, ...
        pred, ...
        seizure_score);


    %% ========================================================
    % DISPLAY FOLD RESULTS
    % =========================================================

    fprintf('\nFold %d results:\n',fold);

    for m = 1:num_models

        fprintf( ...
            '%-22s Accuracy = %6.2f%% | F1 = %6.2f%% | AUC = %.4f\n', ...
            model_names{m}, ...
            accuracy(fold,m)*100, ...
            f1(fold,m)*100, ...
            auc(fold,m));

    end

end

%% ============================================================
% 8. MEAN AND STANDARD DEVIATION
% =============================================================

mean_accuracy = mean(accuracy,1);
std_accuracy  = std(accuracy,0,1);

mean_precision = mean(precision,1);
std_precision  = std(precision,0,1);

mean_recall = mean(recall,1);
std_recall  = std(recall,0,1);

mean_f1 = mean(f1,1);
std_f1  = std(f1,0,1);

mean_auc = mean(auc,1);
std_auc  = std(auc,0,1);

%% ============================================================
% 9. FINAL RESULTS TABLE
% =============================================================

Results = table( ...
    model_names', ...
    mean_accuracy'*100, ...
    std_accuracy'*100, ...
    mean_precision'*100, ...
    std_precision'*100, ...
    mean_recall'*100, ...
    std_recall'*100, ...
    mean_f1'*100, ...
    std_f1'*100, ...
    mean_auc', ...
    std_auc', ...
    'VariableNames', ...
    {'Model', ...
     'AccuracyMean','AccuracyStd', ...
     'PrecisionMean','PrecisionStd', ...
     'RecallMean','RecallStd', ...
     'F1Mean','F1Std', ...
     'AUCMean','AUCStd'});

%% ============================================================
% 10. DISPLAY FINAL RESULTS
% =============================================================

fprintf('\n\n');
fprintf('===============================================\n');
fprintf('FINAL 5-FOLD CROSS-VALIDATION RESULTS\n');
fprintf('===============================================\n');

disp(Results);

%% ============================================================
% 11. SAVE RESULTS
% =============================================================

writetable(Results,'ml_model_comparison.csv');

fprintf('\nResults saved as:\n');
fprintf('ml_model_comparison.csv\n');

%% ============================================================
% 12. ACCURACY COMPARISON
% =============================================================

figure;

bar(mean_accuracy*100);

xticks(1:num_models);
xticklabels(model_names);
xtickangle(35);

ylabel('Mean Accuracy (%)');
title('ML Model Comparison - Accuracy');

grid on;

%% ============================================================
% 13. F1 COMPARISON
% =============================================================

figure;

bar(mean_f1*100);

xticks(1:num_models);
xticklabels(model_names);
xtickangle(35);

ylabel('Mean F1 Score (%)');
title('ML Model Comparison - F1 Score');

grid on;

%% ============================================================
% 14. ROC-AUC COMPARISON
% =============================================================

figure;

bar(mean_auc);

xticks(1:num_models);
xticklabels(model_names);
xtickangle(35);

ylabel('Mean ROC-AUC');
title('ML Model Comparison - ROC-AUC');

grid on;


%% ============================================================
% FUNCTION 1: METRICS FOR MOST MATLAB CLASSIFIERS
% =============================================================

function [acc,prec,rec,f1_score,auc_value] = ...
    calculate_metrics(y_true,y_pred,score,model)

    % Confusion matrix components

    TP = sum((y_true == 1) & (y_pred == 1));

    TN = sum((y_true == 0) & (y_pred == 0));

    FP = sum((y_true == 0) & (y_pred == 1));

    FN = sum((y_true == 1) & (y_pred == 0));


    % Accuracy

    acc = (TP + TN) / length(y_true);


    % Precision

    if (TP + FP) == 0

        prec = 0;

    else

        prec = TP / (TP + FP);

    end


    % Recall

    if (TP + FN) == 0

        rec = 0;

    else

        rec = TP / (TP + FN);

    end


    % F1 Score

    if (prec + rec) == 0

        f1_score = 0;

    else

        f1_score = 2 * prec * rec / (prec + rec);

    end


    % ROC-AUC

    try

        class_names = string(model.ClassNames);

        seizure_column = find(class_names == "1",1);

        if isempty(seizure_column)

            auc_value = NaN;

        else

            seizure_score = score(:,seizure_column);

            [~,~,~,auc_value] = ...
                perfcurve(y_true,seizure_score,1);

        end

    catch

        auc_value = NaN;

    end

end


%% ============================================================
% FUNCTION 2: METRICS WHEN SEIZURE SCORE IS ALREADY PROVIDED
% =============================================================

function [acc,prec,rec,f1_score,auc_value] = ...
    calculate_metrics_score(y_true,y_pred,seizure_score)

    % Confusion matrix components

    TP = sum((y_true == 1) & (y_pred == 1));

    TN = sum((y_true == 0) & (y_pred == 0));

    FP = sum((y_true == 0) & (y_pred == 1));

    FN = sum((y_true == 1) & (y_pred == 0));


    % Accuracy

    acc = (TP + TN) / length(y_true);


    % Precision

    if (TP + FP) == 0

        prec = 0;

    else

        prec = TP / (TP + FP);

    end


    % Recall

    if (TP + FN) == 0

        rec = 0;

    else

        rec = TP / (TP + FN);

    end


    % F1 Score

    if (prec + rec) == 0

        f1_score = 0;

    else

        f1_score = 2 * prec * rec / (prec + rec);

    end


    % ROC-AUC

    try

        [~,~,~,auc_value] = ...
            perfcurve(y_true,seizure_score,1);

    catch

        auc_value = NaN;

    end

end