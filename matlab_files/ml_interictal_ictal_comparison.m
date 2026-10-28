%% ml_interictal_ictal_comparison.m
% F + N (Interictal) vs S (Ictal/Seizure)
% Six-model comparison using 5-fold cross-validation

clear;
clc;
close all;

rng(42);

%% Load dataset
data = readtable('interictal_vs_ictal_features.csv');

% Features
X = data{:, 1:10};

% Labels
y = data.Class;

fprintf('========================================\n');
fprintf('INTERICTAL vs ICTAL ML COMPARISON\n');
fprintf('========================================\n');

fprintf('Total samples : %d\n', size(X,1));
fprintf('Total features: %d\n', size(X,2));
fprintf('Interictal    : %d\n', sum(y == 0));
fprintf('Ictal         : %d\n', sum(y == 1));

%% 5-Fold Cross Validation
cv = cvpartition(y, 'KFold', 5);

%% Model names
model_names = { ...
    'Logistic Regression'
    'k-NN'
    'Decision Tree'
    'SVM-RBF'
    'Random Forest'
    'Gradient Boosting'};

num_models = length(model_names);

%% Storage
accuracy_all  = zeros(num_models, 5);
precision_all = zeros(num_models, 5);
recall_all    = zeros(num_models, 5);
f1_all        = zeros(num_models, 5);
auc_all       = zeros(num_models, 5);

%% Cross-validation loop
for model = 1:num_models

    fprintf('\n========================================\n');
    fprintf('%s\n', model_names{model});
    fprintf('========================================\n');

    for fold = 1:5

        %% Training and testing indices
        train_idx = training(cv, fold);
        test_idx  = test(cv, fold);

        X_train = X(train_idx,:);
        X_test  = X(test_idx,:);

        y_train = y(train_idx);
        y_test  = y(test_idx);

        %% Training-only normalization
        mu = mean(X_train, 1);
        sigma = std(X_train, 0, 1);

        % Avoid division by zero
        sigma(sigma == 0) = 1;

        X_train_norm = (X_train - mu) ./ sigma;
        X_test_norm  = (X_test - mu) ./ sigma;

        %% Train model
        switch model

            case 1
                % Logistic Regression
                mdl = fitglm(X_train_norm, y_train, ...
                    'Distribution', 'binomial');

                scores = predict(mdl, X_test_norm);
                y_pred = double(scores >= 0.5);

            case 2
                % k-NN
                mdl = fitcknn(X_train_norm, y_train, ...
                    'NumNeighbors', 5, ...
                    'Standardize', false);

                [y_pred, scores] = predict(mdl, X_test_norm);

                y_pred = double(y_pred);

                % Probability of class 1
                class_names = string(mdl.ClassNames);
                seizure_column = find(class_names == "1", 1);

                scores = scores(:, seizure_column);

            case 3
                % Decision Tree
                mdl = fitctree(X_train_norm, y_train);

                [y_pred, scores] = predict(mdl, X_test_norm);

                y_pred = double(y_pred);

                class_names = string(mdl.ClassNames);
                seizure_column = find(class_names == "1", 1);

                scores = scores(:, seizure_column);

            case 4
                % SVM with RBF kernel
                mdl = fitcsvm(X_train_norm, y_train, ...
                    'KernelFunction', 'rbf', ...
                    'Standardize', false, ...
                    'ClassNames', [0 1]);

                mdl = fitPosterior(mdl, ...
                    X_train_norm, y_train);

                [y_pred, scores] = predict(mdl, X_test_norm);

                y_pred = double(y_pred);

                class_names = string(mdl.ClassNames);
                seizure_column = find(class_names == "1", 1);

                scores = scores(:, seizure_column);

            case 5
                % Random Forest
                mdl = TreeBagger(100, ...
                    X_train_norm, y_train, ...
                    'Method', 'classification', ...
                    'OOBPrediction', 'off');

                [predicted, scores] = predict(mdl, X_test_norm);

                y_pred = str2double(predicted);

                class_names = string(mdl.ClassNames);
                seizure_column = find(class_names == "1", 1);

                scores = scores(:, seizure_column);

            case 6
                % Gradient Boosting
                mdl = fitcensemble( ...
                    X_train_norm, y_train, ...
                    'Method', 'LogitBoost', ...
                    'NumLearningCycles', 100);

                [y_pred, scores] = predict(mdl, X_test_norm);

                y_pred = double(y_pred);

                class_names = string(mdl.ClassNames);
                seizure_column = find(class_names == "1", 1);

                scores = scores(:, seizure_column);
        end

        %% Confusion matrix values
        TP = sum((y_test == 1) & (y_pred == 1));
        TN = sum((y_test == 0) & (y_pred == 0));
        FP = sum((y_test == 0) & (y_pred == 1));
        FN = sum((y_test == 1) & (y_pred == 0));

        %% Metrics
        accuracy = (TP + TN) / length(y_test);

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
            f1 = 0;
        else
            f1 = 2 * precision * recall / ...
                (precision + recall);
        end

        %% ROC-AUC
        try
            [~,~,~,auc] = perfcurve(y_test, scores, 1);
        catch
            auc = NaN;
        end

        %% Store
        accuracy_all(model,fold)  = accuracy;
        precision_all(model,fold) = precision;
        recall_all(model,fold)    = recall;
        f1_all(model,fold)        = f1;
        auc_all(model,fold)       = auc;

        fprintf(['Fold %d: Accuracy = %.2f%% | ' ...
                 'Precision = %.2f%% | Recall = %.2f%% | ' ...
                 'F1 = %.2f%% | AUC = %.4f\n'], ...
                 fold, ...
                 accuracy*100, ...
                 precision*100, ...
                 recall*100, ...
                 f1*100, ...
                 auc);
    end
end

%% Calculate mean and standard deviation
mean_accuracy  = mean(accuracy_all, 2);
std_accuracy   = std(accuracy_all, 0, 2);

mean_precision = mean(precision_all, 2);
std_precision  = std(precision_all, 0, 2);

mean_recall    = mean(recall_all, 2);
std_recall     = std(recall_all, 0, 2);

mean_f1        = mean(f1_all, 2);
std_f1         = std(f1_all, 0, 2);

mean_auc       = mean(auc_all, 2);
std_auc        = std(auc_all, 0, 2);

%% Display final comparison
fprintf('\n\n');
fprintf('========================================\n');
fprintf('5-FOLD CROSS-VALIDATION RESULTS\n');
fprintf('========================================\n\n');

for model = 1:num_models

    fprintf('%s\n', model_names{model});

    fprintf('Accuracy : %.2f +/- %.2f %%\n', ...
        mean_accuracy(model)*100, ...
        std_accuracy(model)*100);

    fprintf('Precision: %.2f +/- %.2f %%\n', ...
        mean_precision(model)*100, ...
        std_precision(model)*100);

    fprintf('Recall   : %.2f +/- %.2f %%\n', ...
        mean_recall(model)*100, ...
        std_recall(model)*100);

    fprintf('F1 Score : %.2f +/- %.2f %%\n', ...
        mean_f1(model)*100, ...
        std_f1(model)*100);

    fprintf('ROC-AUC  : %.4f +/- %.4f\n\n', ...
        mean_auc(model), ...
        std_auc(model));
end

%% Create results table
Results = table( ...
    string(model_names), ...
    mean_accuracy * 100, ...
    std_accuracy * 100, ...
    mean_precision * 100, ...
    std_precision * 100, ...
    mean_recall * 100, ...
    std_recall * 100, ...
    mean_f1 * 100, ...
    std_f1 * 100, ...
    mean_auc, ...
    std_auc, ...
    'VariableNames', { ...
    'Model', ...
    'MeanAccuracy', ...
    'StdAccuracy', ...
    'MeanPrecision', ...
    'StdPrecision', ...
    'MeanRecall', ...
    'StdRecall', ...
    'MeanF1', ...
    'StdF1', ...
    'MeanAUC', ...
    'StdAUC'});

%% Display table
fprintf('FINAL MODEL COMPARISON:\n');
disp(Results);

%% Save results
writetable(Results, ...
    'interictal_ictal_model_comparison.csv');

fprintf('\nResults saved as:\n');
fprintf('interictal_ictal_model_comparison.csv\n');
