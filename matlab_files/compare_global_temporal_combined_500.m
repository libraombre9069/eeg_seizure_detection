clc;
clear;
close all;

fprintf('========================================\n');
fprintf('GLOBAL vs TEMPORAL vs COMBINED\n');
fprintf('500-RECORDING EEG DATASET\n');
fprintf('========================================\n\n');

%% Reproducibility
rng(42);

%% Load datasets
global_file = 'nonseizure_vs_seizure_features.csv';
temporal_file = 'temporal_nonseizure_vs_seizure_features.csv';

G = readtable(global_file);
T = readtable(temporal_file);

%% ========================================
% EXTRACT LABELS
% ========================================

% Global dataset uses "Class"
Y = G.Class;

% Temporal dataset uses "Label"
Y_temporal = T.Label;

% Make sure labels match
if ~isequal(Y, Y_temporal)
    error('Global and temporal labels do not match.');
end

%% ========================================
% EXTRACT FEATURES
% ========================================

% Global:
% Columns 1-10 are the 10 global features
X_global = G{:,1:10};

% Temporal:
% Columns 1-15 are the 15 temporal features
X_temporal = T{:,1:15};

% Combined:
X_combined = [X_global X_temporal];

%% ========================================
% CHECK DATASET
% ========================================

fprintf('Dataset loaded successfully.\n');
fprintf('Total recordings: %d\n', length(Y));

fprintf('Global features:   %d\n', size(X_global,2));
fprintf('Temporal features: %d\n', size(X_temporal,2));
fprintf('Combined features: %d\n\n', size(X_combined,2));

fprintf('Class distribution:\n');
fprintf('Non-seizure: %d\n', sum(Y == 0));
fprintf('Seizure:     %d\n\n', sum(Y == 1));

%% ========================================
% FEATURE SETS
% ========================================

feature_sets = {
    'Global'
    'Temporal'
    'Combined'
};

X_sets = {
    X_global
    X_temporal
    X_combined
};

%% ========================================
% CROSS VALIDATION
% ========================================

K = 5;

cv = cvpartition(Y,'KFold',K);

%% ========================================
% MODEL NAMES
% ========================================

model_names = {
    'Logistic Regression'
    'k-NN'
    'Decision Tree'
    'SVM-RBF'
    'Random Forest'
    'Bagged Trees'
};

num_models = length(model_names);
num_sets = length(feature_sets);

%% ========================================
% RESULTS STORAGE
% ========================================

results = [];

%% ========================================
% LOOP THROUGH FEATURE SETS
% ========================================

for s = 1:num_sets

    X = X_sets{s};

    fprintf('\n========================================\n');
    fprintf('%s FEATURES\n',upper(feature_sets{s}));
    fprintf('========================================\n');

    %% Metric storage

    acc_all  = zeros(K,num_models);
    prec_all = zeros(K,num_models);
    rec_all  = zeros(K,num_models);
    f1_all   = zeros(K,num_models);
    auc_all  = zeros(K,num_models);

    %% ========================================
    % 5-FOLD CROSS VALIDATION
    % ========================================

    for fold = 1:K

        fprintf('\nFold %d / %d\n',fold,K);

        train_idx = training(cv,fold);
        test_idx  = test(cv,fold);

        Xtrain = X(train_idx,:);
        Xtest  = X(test_idx,:);

        Ytrain = Y(train_idx);
        Ytest  = Y(test_idx);

        %% ====================================
        % TRAINING-ONLY NORMALIZATION
        % ====================================

        mu = mean(Xtrain,1);
        sigma = std(Xtrain,[],1);

        sigma(sigma == 0) = 1;

        Xtrain_norm = (Xtrain - mu) ./ sigma;
        Xtest_norm  = (Xtest - mu) ./ sigma;

        %% ====================================
        % 1. LOGISTIC REGRESSION
        % ====================================

        fprintf('  Logistic Regression...\n');

        mdl = fitclinear( ...
            Xtrain_norm, ...
            Ytrain, ...
            'Learner','logistic');

        [pred,score] = predict(mdl,Xtest_norm);

        acc_all(fold,1) = mean(pred == Ytest);

        [prec_all(fold,1), ...
         rec_all(fold,1), ...
         f1_all(fold,1)] = ...
            binary_metrics(Ytest,pred);

        auc_all(fold,1) = calculate_auc(Ytest,score(:,2));

        %% ====================================
        % 2. k-NN
        % ====================================

        fprintf('  k-NN...\n');

        mdl = fitcknn( ...
            Xtrain_norm, ...
            Ytrain, ...
            'NumNeighbors',5, ...
            'Standardize',false);

        [pred,score] = predict(mdl,Xtest_norm);

        acc_all(fold,2) = mean(pred == Ytest);

        [prec_all(fold,2), ...
         rec_all(fold,2), ...
         f1_all(fold,2)] = ...
            binary_metrics(Ytest,pred);

        auc_all(fold,2) = calculate_auc(Ytest,score(:,2));

        %% ====================================
        % 3. DECISION TREE
        % ====================================

        fprintf('  Decision Tree...\n');

        mdl = fitctree( ...
            Xtrain_norm, ...
            Ytrain);

        [pred,score] = predict(mdl,Xtest_norm);

        acc_all(fold,3) = mean(pred == Ytest);

        [prec_all(fold,3), ...
         rec_all(fold,3), ...
         f1_all(fold,3)] = ...
            binary_metrics(Ytest,pred);

        auc_all(fold,3) = calculate_auc(Ytest,score(:,2));

        %% ====================================
        % 4. SVM-RBF
        % ====================================

        fprintf('  SVM-RBF...\n');

        template = templateSVM( ...
            'KernelFunction','rbf', ...
            'Standardize',false);

        mdl = fitcecoc( ...
            Xtrain_norm, ...
            Ytrain, ...
            'Learners',template);

        [pred,score] = predict(mdl,Xtest_norm);

        acc_all(fold,4) = mean(pred == Ytest);

        [prec_all(fold,4), ...
         rec_all(fold,4), ...
         f1_all(fold,4)] = ...
            binary_metrics(Ytest,pred);

        auc_all(fold,4) = calculate_auc(Ytest,score(:,2));

        %% ====================================
        % 5. RANDOM FOREST
        % ====================================

        fprintf('  Random Forest...\n');

        mdl = TreeBagger( ...
            200, ...
            Xtrain_norm, ...
            categorical(Ytrain), ...
            'Method','classification', ...
            'OOBPrediction','off');

        [pred_cell,score] = predict(mdl,Xtest_norm);

        pred = str2double(pred_cell);

        acc_all(fold,5) = mean(pred == Ytest);

        [prec_all(fold,5), ...
         rec_all(fold,5), ...
         f1_all(fold,5)] = ...
            binary_metrics(Ytest,pred);

        class_names = mdl.ClassNames;

        class_one_idx = find( ...
            str2double(string(class_names)) == 1);

        auc_all(fold,5) = calculate_auc( ...
            Ytest, ...
            score(:,class_one_idx));

        %% ====================================
        % 6. BAGGED TREES
        % ====================================

        fprintf('  Bagged Trees...\n');

        mdl = fitcensemble( ...
            Xtrain_norm, ...
            Ytrain, ...
            'Method','Bag', ...
            'NumLearningCycles',100);

        [pred,score] = predict(mdl,Xtest_norm);

        acc_all(fold,6) = mean(pred == Ytest);

        [prec_all(fold,6), ...
         rec_all(fold,6), ...
         f1_all(fold,6)] = ...
            binary_metrics(Ytest,pred);

        auc_all(fold,6) = calculate_auc( ...
            Ytest,score(:,2));

    end

    %% ========================================
    % STORE RESULTS
    % ========================================

    for m = 1:num_models

        row = table;

        row.FeatureSet = string(feature_sets{s});
        row.Model = string(model_names{m});

        row.AccuracyMean = mean(acc_all(:,m))*100;
        row.AccuracyStd  = std(acc_all(:,m))*100;

        row.PrecisionMean = mean(prec_all(:,m))*100;
        row.PrecisionStd  = std(prec_all(:,m))*100;

        row.RecallMean = mean(rec_all(:,m))*100;
        row.RecallStd  = std(rec_all(:,m))*100;

        row.F1Mean = mean(f1_all(:,m))*100;
        row.F1Std  = std(f1_all(:,m))*100;

        row.AUCMean = mean(auc_all(:,m));
        row.AUCStd  = std(auc_all(:,m));

        results = [results;row];

        fprintf('\n%s\n',model_names{m});

        fprintf('Accuracy : %.2f +/- %.2f %%\n', ...
            row.AccuracyMean,row.AccuracyStd);

        fprintf('Precision: %.2f +/- %.2f %%\n', ...
            row.PrecisionMean,row.PrecisionStd);

        fprintf('Recall   : %.2f +/- %.2f %%\n', ...
            row.RecallMean,row.RecallStd);

        fprintf('F1 Score : %.2f +/- %.2f %%\n', ...
            row.F1Mean,row.F1Std);

        fprintf('ROC-AUC  : %.4f +/- %.4f\n', ...
            row.AUCMean,row.AUCStd);

    end

end

%% ========================================
% SAVE FINAL RESULTS
% ========================================

output_file = ...
    'global_temporal_combined_500_results.csv';

writetable(results,output_file);

fprintf('\n========================================\n');
fprintf('FINAL RESULTS SAVED\n');
fprintf('========================================\n');

fprintf('%s\n',output_file);


%% ========================================
% LOCAL FUNCTIONS
% ========================================

function [precision,recall,f1] = binary_metrics(Ytrue,Ypred)

    TP = sum((Ytrue == 1) & (Ypred == 1));

    FP = sum((Ytrue == 0) & (Ypred == 1));

    FN = sum((Ytrue == 1) & (Ypred == 0));

    precision = TP / max(TP + FP,1);

    recall = TP / max(TP + FN,1);

    f1 = 2 * precision * recall / ...
        max(precision + recall,eps);

end


function auc = calculate_auc(Ytrue,score)

    try

        [~,~,~,auc] = ...
            perfcurve(Ytrue,score,1);

    catch

        auc = NaN;

    end

end