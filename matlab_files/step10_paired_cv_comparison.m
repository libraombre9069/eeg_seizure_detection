%% STEP 10 — PAIRED REPEATED-CV COMPARISON
clear;
clc;
close all;

%% ============================================================
% SETTINGS
% =============================================================

rng(42);

k = 5;
n_repeats = 10;
n_trees = 200;

%% ============================================================
% LOAD DATA
% =============================================================

G = readtable('nonseizure_vs_seizure_features.csv');
T = readtable('temporal_nonseizure_vs_seizure_features.csv');

fprintf('\nGlobal dataset:\n');
fprintf('%d rows x %d columns\n',height(G),width(G));

fprintf('\nTemporal dataset:\n');
fprintf('%d rows x %d columns\n',height(T),width(T));

%% ============================================================
% CHECK LABELS
% =============================================================

Y = G.Class;
Y_temporal = T.Label;

% Convert labels safely
if iscell(Y)
    Y = str2double(string(Y));
elseif isstring(Y)
    Y = str2double(Y);
end

if iscell(Y_temporal)
    Y_temporal = str2double(string(Y_temporal));
elseif isstring(Y_temporal)
    Y_temporal = str2double(Y_temporal);
end

Y = double(Y);
Y_temporal = double(Y_temporal);

% Check that both datasets have identical row ordering
if ~isequal(Y,Y_temporal)

    error(['Global and temporal datasets do not have identical ', ...
           'row ordering or labels.']);

end

fprintf('\nClass distribution:\n');
fprintf('Non-seizure: %d\n',sum(Y == 0));
fprintf('Seizure:     %d\n',sum(Y == 1));

%% ============================================================
% EXTRACT FEATURES
% =============================================================

% 10 global features
X_global = G{:,1:10};

% 15 temporal features
X_temporal = T{:,1:15};

% Combined 25 features
X_combined = [X_global X_temporal];

fprintf('\nFeature dimensions:\n');
fprintf('Global:   %d\n',size(X_global,2));
fprintf('Temporal: %d\n',size(X_temporal,2));
fprintf('Combined: %d\n',size(X_combined,2));

%% ============================================================
% FEATURE SETS
% ============================================================

% Temporal feature order:
%
% 1-5   = MeanTransition
% 6-10  = MaxTransition
% 11-15 = Drift

X_sets = {
    X_global
    [X_global X_temporal(:,1:5)]
    [X_global X_temporal(:,6:10)]
    X_combined
};

set_names = {
    'Global'
    'Global + MeanTransition'
    'Global + MaxTransition'
    'Global + All Temporal'
};

n_sets = length(X_sets);

%% ============================================================
% STORAGE
% =============================================================

Accuracy  = zeros(n_repeats,k,n_sets);
Precision = zeros(n_repeats,k,n_sets);
Recall    = zeros(n_repeats,k,n_sets);
F1        = zeros(n_repeats,k,n_sets);
AUC       = zeros(n_repeats,k,n_sets);

%% ============================================================
% REPEATED STRATIFIED 5-FOLD CROSS-VALIDATION
% =============================================================

fprintf('\n============================================\n');
fprintf('STEP 10 — PAIRED REPEATED CROSS-VALIDATION\n');
fprintf('============================================\n');

for r = 1:n_repeats

    fprintf('\nRepetition %d / %d\n',r,n_repeats);

    % Same folds used for every feature set
    cv = cvpartition(Y,'KFold',k);

    for fold = 1:k

        fprintf('  Fold %d / %d\n',fold,k);

        train_idx = training(cv,fold);
        test_idx  = test(cv,fold);

        Y_train = Y(train_idx);
        Y_test  = Y(test_idx);

        %% ====================================================
        % EACH FEATURE SET
        % ====================================================

        for s = 1:n_sets

            X = X_sets{s};

            X_train = X(train_idx,:);
            X_test  = X(test_idx,:);

            %% ------------------------------------------------
            % TRAINING-FOLD-ONLY NORMALIZATION
            % ------------------------------------------------

            mu = mean(X_train,1);
            sigma = std(X_train,0,1);

            % Prevent division by zero
            sigma(sigma == 0) = 1;

            X_train_norm = ...
                (X_train - mu) ./ sigma;

            X_test_norm = ...
                (X_test - mu) ./ sigma;

            %% ------------------------------------------------
            % RANDOM FOREST
            % ------------------------------------------------

            RF = TreeBagger( ...
                n_trees, ...
                X_train_norm, ...
                Y_train, ...
                'Method','classification', ...
                'OOBPrediction','off');

            %% ------------------------------------------------
            % PREDICTION
            % ------------------------------------------------

            [predicted_class,scores] = ...
                predict(RF,X_test_norm);

            %% ------------------------------------------------
            % CONVERT PREDICTED CLASS SAFELY
            % ------------------------------------------------

            if iscell(predicted_class)

                predicted_class = ...
                    str2double(string(predicted_class));

            elseif isstring(predicted_class)

                predicted_class = ...
                    str2double(predicted_class);

            elseif iscategorical(predicted_class)

                predicted_class = ...
                    str2double(string(predicted_class));

            else

                predicted_class = double(predicted_class);

            end

            predicted_class = predicted_class(:);

            %% ------------------------------------------------
            % FIND SEIZURE PROBABILITY COLUMN
            % ------------------------------------------------

            class_names = RF.ClassNames;

            % TreeBagger commonly returns cell array:
            % {'0'; '1'}

            if iscell(class_names)

                class_names = string(class_names);

            elseif iscategorical(class_names)

                class_names = string(class_names);

            elseif isstring(class_names)

                class_names = class_names;

            elseif isnumeric(class_names)

                class_names = string(class_names);

            elseif ischar(class_names)

                class_names = string(cellstr(class_names));

            end

            % Find class "1" = seizure
            seizure_column = ...
                find(class_names == "1",1);

            if isempty(seizure_column)

                error('Could not identify seizure probability column.');

            end

            seizure_score = scores(:,seizure_column);

            %% ------------------------------------------------
            % CONFUSION MATRIX
            % ------------------------------------------------

            TP = sum( ...
                (Y_test == 1) & ...
                (predicted_class == 1));

            TN = sum( ...
                (Y_test == 0) & ...
                (predicted_class == 0));

            FP = sum( ...
                (Y_test == 0) & ...
                (predicted_class == 1));

            FN = sum( ...
                (Y_test == 1) & ...
                (predicted_class == 0));

            %% ------------------------------------------------
            % ACCURACY
            % ------------------------------------------------

            Accuracy(r,fold,s) = ...
                (TP + TN) / length(Y_test) * 100;

            %% ------------------------------------------------
            % PRECISION
            % ------------------------------------------------

            if (TP + FP) == 0

                precision_value = 0;

            else

                precision_value = ...
                    TP / (TP + FP) * 100;

            end

            Precision(r,fold,s) = precision_value;

            %% ------------------------------------------------
            % RECALL
            % ------------------------------------------------

            if (TP + FN) == 0

                recall_value = 0;

            else

                recall_value = ...
                    TP / (TP + FN) * 100;

            end

            Recall(r,fold,s) = recall_value;

            %% ------------------------------------------------
            % F1 SCORE
            % ------------------------------------------------

            if (precision_value + recall_value) == 0

                f1_value = 0;

            else

                f1_value = ...
                    2 * precision_value * recall_value / ...
                    (precision_value + recall_value);

            end

            F1(r,fold,s) = f1_value;

            %% ------------------------------------------------
            % ROC-AUC
            % ------------------------------------------------

            try

                [~,~,~,auc_value] = ...
                    perfcurve( ...
                    Y_test, ...
                    seizure_score, ...
                    1);

            catch

                auc_value = NaN;

            end

            AUC(r,fold,s) = auc_value;

        end

    end

end

%% ============================================================
% RESHAPE RESULTS
% =============================================================

Accuracy_all = ...
    reshape(Accuracy,[],n_sets);

Precision_all = ...
    reshape(Precision,[],n_sets);

Recall_all = ...
    reshape(Recall,[],n_sets);

F1_all = ...
    reshape(F1,[],n_sets);

AUC_all = ...
    reshape(AUC,[],n_sets);

%% ============================================================
% SUMMARY STATISTICS
% =============================================================

AccuracyMean = ...
    mean(Accuracy_all,1,'omitnan');

AccuracyStd = ...
    std(Accuracy_all,0,1,'omitnan');

PrecisionMean = ...
    mean(Precision_all,1,'omitnan');

PrecisionStd = ...
    std(Precision_all,0,1,'omitnan');

RecallMean = ...
    mean(Recall_all,1,'omitnan');

RecallStd = ...
    std(Recall_all,0,1,'omitnan');

F1Mean = ...
    mean(F1_all,1,'omitnan');

F1Std = ...
    std(F1_all,0,1,'omitnan');

AUCMean = ...
    mean(AUC_all,1,'omitnan');

AUCStd = ...
    std(AUC_all,0,1,'omitnan');

%% ============================================================
% PRINT SUMMARY
% =============================================================

fprintf('\n============================================\n');
fprintf('REPEATED CV RESULTS\n');
fprintf('============================================\n\n');

for s = 1:n_sets

    fprintf('%s\n',set_names{s});

    fprintf('Accuracy  : %.3f +/- %.3f %%\n', ...
        AccuracyMean(s),AccuracyStd(s));

    fprintf('Precision : %.3f +/- %.3f %%\n', ...
        PrecisionMean(s),PrecisionStd(s));

    fprintf('Recall    : %.3f +/- %.3f %%\n', ...
        RecallMean(s),RecallStd(s));

    fprintf('F1 Score  : %.3f +/- %.3f %%\n', ...
        F1Mean(s),F1Std(s));

    fprintf('ROC-AUC   : %.5f +/- %.5f\n\n', ...
        AUCMean(s),AUCStd(s));

end

%% ============================================================
% PAIRED WILCOXON TESTS VS GLOBAL
% =============================================================

fprintf('\n============================================\n');
fprintf('PAIRED COMPARISON VS GLOBAL\n');
fprintf('============================================\n');

acc_diff_mean = zeros(n_sets-1,1);
acc_p = zeros(n_sets-1,1);

f1_diff_mean = zeros(n_sets-1,1);
f1_p = zeros(n_sets-1,1);

auc_diff_mean = zeros(n_sets-1,1);
auc_p = zeros(n_sets-1,1);

for s = 2:n_sets

    fprintf('\n%s vs Global\n',set_names{s});

    %% --------------------------------------------------------
    % ACCURACY
    % --------------------------------------------------------

    diff_acc = ...
        Accuracy_all(:,s) - Accuracy_all(:,1);

    acc_diff_mean(s-1) = ...
        mean(diff_acc,'omitnan');

    acc_p(s-1) = signrank( ...
        Accuracy_all(:,s), ...
        Accuracy_all(:,1));

    fprintf('Accuracy difference: %.4f pp\n', ...
        acc_diff_mean(s-1));

    fprintf('Accuracy p-value: %.8f\n', ...
        acc_p(s-1));

    %% --------------------------------------------------------
    % F1
    % --------------------------------------------------------

    diff_f1 = ...
        F1_all(:,s) - F1_all(:,1);

    f1_diff_mean(s-1) = ...
        mean(diff_f1,'omitnan');

    f1_p(s-1) = signrank( ...
        F1_all(:,s), ...
        F1_all(:,1));

    fprintf('F1 difference: %.4f pp\n', ...
        f1_diff_mean(s-1));

    fprintf('F1 p-value: %.8f\n', ...
        f1_p(s-1));

    %% --------------------------------------------------------
    % AUC
    % --------------------------------------------------------

    valid_auc = ...
        ~isnan(AUC_all(:,s)) & ...
        ~isnan(AUC_all(:,1));

    diff_auc = ...
        AUC_all(valid_auc,s) - ...
        AUC_all(valid_auc,1);

    auc_diff_mean(s-1) = ...
        mean(diff_auc,'omitnan');

    auc_p(s-1) = signrank( ...
        AUC_all(valid_auc,s), ...
        AUC_all(valid_auc,1));

    fprintf('AUC difference: %.6f\n', ...
        auc_diff_mean(s-1));

    fprintf('AUC p-value: %.8f\n', ...
        auc_p(s-1));

end

%% ============================================================
% SUMMARY TABLE
% =============================================================

Summary = table( ...
    string(set_names), ...
    AccuracyMean', ...
    AccuracyStd', ...
    PrecisionMean', ...
    PrecisionStd', ...
    RecallMean', ...
    RecallStd', ...
    F1Mean', ...
    F1Std', ...
    AUCMean', ...
    AUCStd', ...
    'VariableNames',{ ...
    'FeatureSet', ...
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

%% ============================================================
% SAVE SUMMARY
% =============================================================

writetable( ...
    Summary, ...
    'step10_paired_cv_summary.csv');

%% ============================================================
% PAIRED COMPARISON TABLE
% =============================================================

Comparison = table();

Comparison.FeatureSet = ...
    string(set_names(2:end));

Comparison.AccuracyDifference_pp = ...
    acc_diff_mean;

Comparison.Accuracy_p = ...
    acc_p;

Comparison.F1Difference_pp = ...
    f1_diff_mean;

Comparison.F1_p = ...
    f1_p;

Comparison.AUCDifference = ...
    auc_diff_mean;

Comparison.AUC_p = ...
    auc_p;

%% ============================================================
% SAVE COMPARISON
% =============================================================

writetable( ...
    Comparison, ...
    'step10_paired_cv_comparison.csv');

%% ============================================================
% FOLD-LEVEL RESULTS
% ============================================================

FoldResults = table();

FoldResults.Repeat = ...
    repelem((1:n_repeats)',k);

FoldResults.Fold = ...
    repmat((1:k)',n_repeats,1);

for s = 1:n_sets

    FoldResults.(sprintf('Accuracy_%d',s)) = ...
        reshape(Accuracy(:,:,s),[],1);

    FoldResults.(sprintf('Precision_%d',s)) = ...
        reshape(Precision(:,:,s),[],1);

    FoldResults.(sprintf('Recall_%d',s)) = ...
        reshape(Recall(:,:,s),[],1);

    FoldResults.(sprintf('F1_%d',s)) = ...
        reshape(F1(:,:,s),[],1);

    FoldResults.(sprintf('AUC_%d',s)) = ...
        reshape(AUC(:,:,s),[],1);

end

%% ============================================================
% SAVE FOLD RESULTS
% ============================================================

writetable( ...
    FoldResults, ...
    'step10_paired_cv_fold_results.csv');

%% ============================================================
% FINAL MESSAGE
% ============================================================

fprintf('\n============================================\n');
fprintf('STEP 10 COMPLETED SUCCESSFULLY\n');
fprintf('============================================\n');

fprintf('\nSaved files:\n');
fprintf('1. step10_paired_cv_summary.csv\n');
fprintf('2. step10_paired_cv_comparison.csv\n');
fprintf('3. step10_paired_cv_fold_results.csv\n');

fprintf('\nFeature sets:\n');

for s = 1:n_sets

    fprintf('%d. %s\n',s,set_names{s});

end

fprintf('\n============================================\n');