%% ============================================================
% EEG SIGNAL PROCESSING + FEATURE EXTRACTION + SVM CLASSIFICATION
%
% Dataset:
%   Non-Seizure -> Z001 to Z100
%   Seizure     -> S001 to S100
%
% Sampling Frequency:
%   Fs = 173.61 Hz
%
% Features:
%   1. Standard Deviation
%   2. Peak-to-Peak
%   3. Relative Delta Power
%   4. Relative Theta Power
%   5. Relative Alpha Power
%   6. Relative Beta Power
%   7. Relative Gamma Power
%   8. Dominant Frequency
%   9. Spectral Centroid
%  10. Spectral Entropy
%% ============================================================


clear;
clc;
close all;


%% ============================================================
% 1. PARAMETERS
%% ============================================================

Fs = 173.61;          % Sampling frequency

low_cutoff = 0.5;     % Lower cutoff frequency
high_cutoff = 40;     % Upper cutoff frequency
filter_order = 4;     % Butterworth filter order


%% ============================================================
% 2. FIND EEG FILES
%% ============================================================

non_seizure_files = dir('non_seizure/*.txt');
seizure_files = dir('seizure/*.txt');

fprintf('=============================================\n');
fprintf('EEG DATASET INFORMATION\n');
fprintf('=============================================\n');

fprintf('Non-Seizure files : %d\n', length(non_seizure_files));
fprintf('Seizure files     : %d\n', length(seizure_files));


%% ============================================================
% 3. LOAD ONE EEG SIGNAL FOR INITIAL ANALYSIS
%% ============================================================

% Load first non-seizure recording
eeg = readmatrix('non_seizure/Z001.txt');

% Convert to column vector
eeg = eeg(:);

N = length(eeg);

fprintf('Number of samples : %d\n', N);
fprintf('Duration          : %.2f seconds\n', N/Fs);


%% ============================================================
% 4. TIME VECTOR
%% ============================================================

t = (0:N-1) / Fs;


%% ============================================================
% 5. PLOT RAW EEG SIGNAL
%% ============================================================

figure;

plot(t, eeg);

xlabel('Time (seconds)');
ylabel('EEG Amplitude');

title('Raw EEG Signal - Z001');

grid on;


%% ============================================================
% 6. BUTTERWORTH BANDPASS FILTER
%% ============================================================

[b, a] = butter(filter_order, ...
    [low_cutoff high_cutoff] / (Fs/2), ...
    'bandpass');

% Apply zero-phase filtering
eeg_filtered = filtfilt(b, a, eeg);


%% ============================================================
% 7. PLOT RAW VS FILTERED SIGNAL
%% ============================================================

figure;

plot(t, eeg);
hold on;

plot(t, eeg_filtered);

xlabel('Time (seconds)');
ylabel('EEG Amplitude');

title('Raw vs Filtered EEG Signal');

legend('Raw EEG', 'Filtered EEG');

grid on;


%% ============================================================
% 8. FFT OF RAW SIGNAL
%% ============================================================

Y = fft(eeg);

% Two-sided magnitude spectrum
P2 = abs(Y/N);

% One-sided spectrum
P1 = P2(1:floor(N/2)+1);

% Double non-DC/non-Nyquist components
P1(2:end-1) = 2 * P1(2:end-1);

% Frequency vector
f_fft = Fs * (0:floor(N/2)) / N;


%% ============================================================
% 9. FFT OF FILTERED SIGNAL
%% ============================================================

Y_filtered = fft(eeg_filtered);

P2_filtered = abs(Y_filtered/N);

P1_filtered = P2_filtered(1:floor(N/2)+1);

P1_filtered(2:end-1) = ...
    2 * P1_filtered(2:end-1);


%% ============================================================
% 10. PLOT FFT
%% ============================================================

figure;

plot(f_fft, P1);

xlabel('Frequency (Hz)');
ylabel('Magnitude');

title('FFT of Raw EEG Signal');

xlim([0 50]);

grid on;


figure;

plot(f_fft, P1_filtered);

xlabel('Frequency (Hz)');
ylabel('Magnitude');

title('FFT of Filtered EEG Signal');

xlim([0 50]);

grid on;


%% ============================================================
% 11. ABSOLUTE BAND POWER FOR Z001
%
% This is only for visualization.
% These values are NOT used directly as ML features.
%% ============================================================

delta_power = bandpower(eeg_filtered, Fs, [0.5 4]);

theta_power = bandpower(eeg_filtered, Fs, [4 8]);

alpha_power = bandpower(eeg_filtered, Fs, [8 13]);

beta_power = bandpower(eeg_filtered, Fs, [13 30]);

gamma_power = bandpower(eeg_filtered, Fs, [30 40]);


fprintf('\n=============================================\n');
fprintf('Z001 ABSOLUTE BAND POWERS\n');
fprintf('=============================================\n');

fprintf('Delta Power : %.4f\n', delta_power);
fprintf('Theta Power : %.4f\n', theta_power);
fprintf('Alpha Power : %.4f\n', alpha_power);
fprintf('Beta Power  : %.4f\n', beta_power);
fprintf('Gamma Power : %.4f\n', gamma_power);


%% ============================================================
% 12. PLOT BAND POWERS
%% ============================================================

band_names = {'Delta','Theta','Alpha','Beta','Gamma'};

band_powers = [ ...
    delta_power, ...
    theta_power, ...
    alpha_power, ...
    beta_power, ...
    gamma_power];


figure;

bar(band_powers);

set(gca, 'XTick', 1:5);
set(gca, 'XTickLabel', band_names);

xlabel('EEG Frequency Band');
ylabel('Power');

title('EEG Band Powers - Z001');

grid on;


%% ============================================================
% 13. EXTRACT FEATURES FROM ONE SIGNAL
%% ============================================================

test_features = extract_features( ...
    'non_seizure/Z001.txt', Fs);


%% ============================================================
% 14. FEATURE NAMES
%% ============================================================

feature_names = { ...
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
};


%% ============================================================
% 15. DISPLAY FEATURES OF Z001
%% ============================================================

fprintf('\n=============================================\n');
fprintf('FEATURES OF Z001\n');
fprintf('=============================================\n');

for i = 1:length(feature_names)

    fprintf('%-20s : %.6f\n', ...
        feature_names{i}, ...
        test_features(i));

end


%% ============================================================
% 16. EXTRACT FEATURES FROM ALL RECORDINGS
%% ============================================================

num_non_seizure = length(non_seizure_files);
num_seizure = length(seizure_files);

num_total = num_non_seizure + num_seizure;

num_features = length(feature_names);


% Feature matrix
X = zeros(num_total, num_features);


% Labels
% 0 = Non-Seizure
% 1 = Seizure
y = zeros(num_total, 1);


fprintf('\n=============================================\n');
fprintf('EXTRACTING FEATURES\n');
fprintf('=============================================\n');


%% ---------- Non-Seizure ----------

for i = 1:num_non_seizure

    filename = fullfile( ...
        'non_seizure', ...
        non_seizure_files(i).name);

    X(i,:) = extract_features(filename, Fs);

    y(i) = 0;

end


%% ---------- Seizure ----------

for i = 1:num_seizure

    idx = num_non_seizure + i;

    filename = fullfile( ...
        'seizure', ...
        seizure_files(i).name);

    X(idx,:) = extract_features(filename, Fs);

    y(idx) = 1;

end


fprintf('Feature extraction completed.\n');

fprintf('Total recordings : %d\n', num_total);

fprintf('Total features   : %d\n', num_features);


%% ============================================================
% 17. CREATE FEATURE TABLE
%% ============================================================

feature_table = array2table(X, ...
    'VariableNames', feature_names);

% Add class label
feature_table.Class = categorical(y, ...
    [0 1], ...
    {'NonSeizure','Seizure'});


%% ============================================================
% 18. DISPLAY FEATURE TABLE
%% ============================================================

fprintf('\n=============================================\n');
fprintf('FEATURE TABLE\n');
fprintf('=============================================\n');

disp(feature_table(1:10,:));


%% ============================================================
% 19. SAVE FEATURES
%% ============================================================

writetable(feature_table, 'eeg_features.csv');

fprintf('\nFeature table saved as:\n');
fprintf('eeg_features.csv\n');


%% ============================================================
% 20. CLASS-WISE FEATURE AVERAGES
%% ============================================================

non_seizure_mean = mean( ...
    X(y == 0,:), ...
    1);

seizure_mean = mean( ...
    X(y == 1,:), ...
    1);


fprintf('\n=============================================\n');
fprintf('CLASS-WISE FEATURE AVERAGES\n');
fprintf('=============================================\n');

for i = 1:num_features

    fprintf('\n%s\n', feature_names{i});

    fprintf('Non-Seizure : %.6f\n', ...
        non_seizure_mean(i));

    fprintf('Seizure     : %.6f\n', ...
        seizure_mean(i));

end


%% ============================================================
% 21. CLASS-WISE FEATURE TABLE
%% ============================================================

average_table = table( ...
    non_seizure_mean', ...
    seizure_mean', ...
    'VariableNames', ...
    {'NonSeizure','Seizure'}, ...
    'RowNames', feature_names);

disp(average_table);


%% ============================================================
% 22. TRAIN / TEST SPLIT
%% ============================================================

rng(42);

cv = cvpartition(y, ...
    'HoldOut', 0.20);


train_idx = training(cv);

test_idx = test(cv);


X_train = X(train_idx,:);

X_test = X(test_idx,:);

y_train = y(train_idx);

y_test = y(test_idx);


fprintf('\n=============================================\n');
fprintf('TRAIN / TEST SPLIT\n');
fprintf('=============================================\n');

fprintf('Training samples : %d\n', ...
    size(X_train,1));

fprintf('Testing samples  : %d\n', ...
    size(X_test,1));


%% ============================================================
% 23. CHECK CLASS DISTRIBUTION
%% ============================================================

fprintf('\nTraining class distribution:\n');

fprintf('Non-Seizure : %d\n', ...
    sum(y_train == 0));

fprintf('Seizure     : %d\n', ...
    sum(y_train == 1));


fprintf('\nTesting class distribution:\n');

fprintf('Non-Seizure : %d\n', ...
    sum(y_test == 0));

fprintf('Seizure     : %d\n', ...
    sum(y_test == 1));


%% ============================================================
% 24. FEATURE NORMALIZATION
%
% IMPORTANT:
% Mean and standard deviation are calculated ONLY
% from training data.
%% ============================================================

mu = mean(X_train, 1);

sigma = std(X_train, 0, 1);

% Prevent division by zero
sigma(sigma == 0) = 1;


% Normalize training data
X_train_norm = ...
    (X_train - mu) ./ sigma;


% Apply SAME parameters to test data
X_test_norm = ...
    (X_test - mu) ./ sigma;


%% ============================================================
% 25. TRAIN SVM CLASSIFIER
%% ============================================================

fprintf('\n=============================================\n');
fprintf('TRAINING SVM\n');
fprintf('=============================================\n');


SVMModel = fitcsvm( ...
    X_train_norm, ...
    y_train, ...
    'KernelFunction', 'rbf', ...
    'Standardize', false);


%% ============================================================
% 26. TEST SVM
%% ============================================================

y_pred = predict( ...
    SVMModel, ...
    X_test_norm);


%% ============================================================
% 27. CONFUSION MATRIX
%% ============================================================

figure;

confusionchart( ...
    y_test, ...
    y_pred, ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');

title('SVM Confusion Matrix');


%% ============================================================
% 28. CALCULATE PERFORMANCE METRICS
%% ============================================================

TP = sum((y_test == 1) & (y_pred == 1));

TN = sum((y_test == 0) & (y_pred == 0));

FP = sum((y_test == 0) & (y_pred == 1));

FN = sum((y_test == 1) & (y_pred == 0));


%% Accuracy

accuracy = ...
    (TP + TN) / ...
    (TP + TN + FP + FN);


%% Precision

if (TP + FP) == 0

    precision = 0;

else

    precision = TP / (TP + FP);

end


%% Recall

if (TP + FN) == 0

    recall = 0;

else

    recall = TP / (TP + FN);

end


%% F1 Score

if (precision + recall) == 0

    F1 = 0;

else

    F1 = ...
        2 * precision * recall / ...
        (precision + recall);

end


%% ============================================================
% 29. DISPLAY PERFORMANCE
%% ============================================================

fprintf('\n=============================================\n');
fprintf('SVM TEST PERFORMANCE\n');
fprintf('=============================================\n');

fprintf('Accuracy  : %.2f %%\n', ...
    accuracy * 100);

fprintf('Precision : %.2f %%\n', ...
    precision * 100);

fprintf('Recall    : %.2f %%\n', ...
    recall * 100);

fprintf('F1 Score  : %.2f %%\n', ...
    F1 * 100);


fprintf('\nConfusion Matrix Values:\n');

fprintf('TP = %d\n', TP);
fprintf('TN = %d\n', TN);
fprintf('FP = %d\n', FP);
fprintf('FN = %d\n', FN);


%% ============================================================
% 30. 5-FOLD CROSS VALIDATION
%
% IMPORTANT:
% Normalization is performed INSIDE every fold.
% This prevents data leakage.
%% ============================================================

fprintf('\n=============================================\n');
fprintf('5-FOLD CROSS VALIDATION\n');
fprintf('=============================================\n');


rng(42);

cv5 = cvpartition(y, ...
    'KFold', 5);


fold_accuracy = zeros(5,1);


for fold = 1:5

    %% -----------------------------------------
    % Training / validation indices
    %% -----------------------------------------

    train_fold_idx = training(cv5, fold);

    test_fold_idx = test(cv5, fold);


    %% -----------------------------------------
    % Split data
    %% -----------------------------------------

    X_train_fold = X(train_fold_idx,:);

    X_test_fold = X(test_fold_idx,:);

    y_train_fold = y(train_fold_idx);

    y_test_fold = y(test_fold_idx);


    %% -----------------------------------------
    % Normalize using TRAINING fold only
    %% -----------------------------------------

    mu_fold = mean(X_train_fold, 1);

    sigma_fold = ...
        std(X_train_fold, 0, 1);

    sigma_fold(sigma_fold == 0) = 1;


    X_train_fold_norm = ...
        (X_train_fold - mu_fold) ./ sigma_fold;


    X_test_fold_norm = ...
        (X_test_fold - mu_fold) ./ sigma_fold;


    %% -----------------------------------------
    % Train SVM
    %% -----------------------------------------

    model_fold = fitcsvm( ...
        X_train_fold_norm, ...
        y_train_fold, ...
        'KernelFunction', 'rbf', ...
        'Standardize', false);


    %% -----------------------------------------
    % Predict
    %% -----------------------------------------

    y_pred_fold = predict( ...
        model_fold, ...
        X_test_fold_norm);


    %% -----------------------------------------
    % Accuracy
    %% -----------------------------------------

    fold_accuracy(fold) = ...
        mean(y_pred_fold == y_test_fold);


    fprintf('Fold %d Accuracy: %.2f %%\n', ...
        fold, ...
        fold_accuracy(fold) * 100);

end


%% ============================================================
% 31. CROSS-VALIDATION SUMMARY
%% ============================================================

mean_accuracy = mean(fold_accuracy);

std_accuracy = std(fold_accuracy);


fprintf('\n=============================================\n');
fprintf('5-FOLD CROSS-VALIDATION RESULTS\n');
fprintf('=============================================\n');

fprintf('Mean Accuracy : %.2f %%\n', ...
    mean_accuracy * 100);

fprintf('Std Deviation : %.2f %%\n', ...
    std_accuracy * 100);


%% ============================================================
% 32. DISPLAY CROSS-VALIDATION RESULTS
%% ============================================================

cv_results = table( ...
    (1:5)', ...
    fold_accuracy * 100, ...
    'VariableNames', ...
    {'Fold','AccuracyPercent'});


disp(cv_results);


%% ============================================================
% 33. FEATURE DISTRIBUTION VISUALIZATION
%% ============================================================

figure;

boxplot( ...
    X_train_norm, ...
    'Labels', feature_names);

ylabel('Standardized Feature Value');

title('Normalized Feature Distribution - Training Data');

grid on;

xtickangle(45);


%% ============================================================
% 34. FEATURE COMPARISON
%% ============================================================

figure;

bar([non_seizure_mean; seizure_mean]');

xlabel('Feature');

ylabel('Mean Feature Value');

title('Feature Comparison: Non-Seizure vs Seizure');

legend( ...
    'Non-Seizure', ...
    'Seizure');

set(gca, ...
    'XTick', 1:num_features, ...
    'XTickLabel', feature_names);

xtickangle(45);

grid on;


%% ============================================================
% 35. FINAL SUMMARY
%% ============================================================

fprintf('\n=============================================\n');
fprintf('FINAL PROJECT SUMMARY\n');
fprintf('=============================================\n');

fprintf('Total recordings : %d\n', num_total);

fprintf('Features         : %d\n', num_features);

fprintf('Classifier       : RBF SVM\n');

fprintf('Test Accuracy    : %.2f %%\n', ...
    accuracy * 100);

fprintf('Test Precision   : %.2f %%\n', ...
    precision * 100);

fprintf('Test Recall      : %.2f %%\n', ...
    recall * 100);

fprintf('Test F1 Score    : %.2f %%\n', ...
    F1 * 100);

fprintf('5-Fold Mean Acc. : %.2f %%\n', ...
    mean_accuracy * 100);

fprintf('5-Fold Std.      : %.2f %%\n', ...
    std_accuracy * 100);

fprintf('=============================================\n');

%% ============================================================
% 32. LOGISTIC REGRESSION
%
% Uses the SAME train/test split as SVM.
% Uses the SAME training-only normalization.
%% ============================================================

fprintf('\n=============================================\n');
fprintf('LOGISTIC REGRESSION\n');
fprintf('=============================================\n');


%% Train Logistic Regression

LogisticModel = fitclinear( ...
    X_train_norm, ...
    y_train, ...
    'Learner', 'logistic');


%% Predict Test Set

y_pred_logistic = predict( ...
    LogisticModel, ...
    X_test_norm);


%% ============================================================
% 33. LOGISTIC REGRESSION CONFUSION MATRIX
%% ============================================================

figure;

confusionchart( ...
    y_test, ...
    y_pred_logistic, ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');

title('Logistic Regression Confusion Matrix');


%% ============================================================
% 34. LOGISTIC REGRESSION METRICS
%% ============================================================

TP_log = sum((y_test == 1) & ...
             (y_pred_logistic == 1));

TN_log = sum((y_test == 0) & ...
             (y_pred_logistic == 0));

FP_log = sum((y_test == 0) & ...
             (y_pred_logistic == 1));

FN_log = sum((y_test == 1) & ...
             (y_pred_logistic == 0));


%% Accuracy

accuracy_log = ...
    (TP_log + TN_log) / ...
    (TP_log + TN_log + FP_log + FN_log);


%% Precision

if (TP_log + FP_log) == 0
    precision_log = 0;
else
    precision_log = ...
        TP_log / (TP_log + FP_log);
end


%% Recall

if (TP_log + FN_log) == 0
    recall_log = 0;
else
    recall_log = ...
        TP_log / (TP_log + FN_log);
end


%% F1 Score

if (precision_log + recall_log) == 0
    F1_log = 0;
else
    F1_log = ...
        2 * precision_log * recall_log / ...
        (precision_log + recall_log);
end


%% Display Results

fprintf('\nLogistic Regression Test Performance:\n');

fprintf('Accuracy  : %.2f %%\n', ...
    accuracy_log * 100);

fprintf('Precision : %.2f %%\n', ...
    precision_log * 100);

fprintf('Recall    : %.2f %%\n', ...
    recall_log * 100);

fprintf('F1 Score  : %.2f %%\n', ...
    F1_log * 100);

fprintf('\nConfusion Matrix Values:\n');

fprintf('TP = %d\n', TP_log);
fprintf('TN = %d\n', TN_log);
fprintf('FP = %d\n', FP_log);
fprintf('FN = %d\n', FN_log);


%% ============================================================
% 35. LOGISTIC REGRESSION - 5-FOLD CROSS VALIDATION
%
% Normalization is performed separately inside each fold.
%% ============================================================

fprintf('\n=============================================\n');
fprintf('LOGISTIC REGRESSION - 5-FOLD CV\n');
fprintf('=============================================\n');


rng(42);

cv5_log = cvpartition(y, 'KFold', 5);

fold_accuracy_log = zeros(5,1);


for fold = 1:5

    %% Training / validation indices

    train_fold_idx = training(cv5_log, fold);

    test_fold_idx = test(cv5_log, fold);


    %% Split data

    X_train_fold = X(train_fold_idx,:);

    X_test_fold = X(test_fold_idx,:);

    y_train_fold = y(train_fold_idx);

    y_test_fold = y(test_fold_idx);


    %% Normalize using training fold only

    mu_fold = mean(X_train_fold, 1);

    sigma_fold = std(X_train_fold, 0, 1);

    sigma_fold(sigma_fold == 0) = 1;


    X_train_fold_norm = ...
        (X_train_fold - mu_fold) ./ sigma_fold;

    X_test_fold_norm = ...
        (X_test_fold - mu_fold) ./ sigma_fold;


    %% Train Logistic Regression

    model_fold = fitclinear( ...
        X_train_fold_norm, ...
        y_train_fold, ...
        'Learner', 'logistic');


    %% Predict

    y_pred_fold = predict( ...
        model_fold, ...
        X_test_fold_norm);


    %% Accuracy

    fold_accuracy_log(fold) = ...
        mean(y_pred_fold == y_test_fold);


    fprintf('Fold %d Accuracy: %.2f %%\n', ...
        fold, ...
        fold_accuracy_log(fold) * 100);

end


%% Logistic Regression CV Summary

mean_accuracy_log = mean(fold_accuracy_log);

std_accuracy_log = std(fold_accuracy_log);


fprintf('\nLogistic Regression CV Results:\n');

fprintf('Mean Accuracy : %.2f %%\n', ...
    mean_accuracy_log * 100);

fprintf('Std Deviation : %.2f %%\n', ...
    std_accuracy_log * 100);


%% ============================================================
% 36. RANDOM FOREST
%
% MATLAB implementation:
% TreeBagger = ensemble of decision trees.
%% ============================================================

fprintf('\n=============================================\n');
fprintf('RANDOM FOREST\n');
fprintf('=============================================\n');


%% Number of trees

num_trees = 100;


%% Train Random Forest

RandomForestModel = TreeBagger( ...
    num_trees, ...
    X_train_norm, ...
    categorical(y_train), ...
    'Method', 'classification');


%% Predict Test Set

y_pred_rf_cell = predict( ...
    RandomForestModel, ...
    X_test_norm);


%% Convert cell predictions to numeric

y_pred_rf = str2double(y_pred_rf_cell);


%% ============================================================
% 37. RANDOM FOREST CONFUSION MATRIX
%% ============================================================

figure;

confusionchart( ...
    y_test, ...
    y_pred_rf, ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');

title('Random Forest Confusion Matrix');


%% ============================================================
% 38. RANDOM FOREST METRICS
%% ============================================================

TP_rf = sum((y_test == 1) & ...
            (y_pred_rf == 1));

TN_rf = sum((y_test == 0) & ...
            (y_pred_rf == 0));

FP_rf = sum((y_test == 0) & ...
            (y_pred_rf == 1));

FN_rf = sum((y_test == 1) & ...
            (y_pred_rf == 0));


%% Accuracy

accuracy_rf = ...
    (TP_rf + TN_rf) / ...
    (TP_rf + TN_rf + FP_rf + FN_rf);


%% Precision

if (TP_rf + FP_rf) == 0
    precision_rf = 0;
else
    precision_rf = ...
        TP_rf / (TP_rf + FP_rf);
end


%% Recall

if (TP_rf + FN_rf) == 0
    recall_rf = 0;
else
    recall_rf = ...
        TP_rf / (TP_rf + FN_rf);
end


%% F1 Score

if (precision_rf + recall_rf) == 0
    F1_rf = 0;
else
    F1_rf = ...
        2 * precision_rf * recall_rf / ...
        (precision_rf + recall_rf);
end


%% Display Results

fprintf('\nRandom Forest Test Performance:\n');

fprintf('Accuracy  : %.2f %%\n', ...
    accuracy_rf * 100);

fprintf('Precision : %.2f %%\n', ...
    precision_rf * 100);

fprintf('Recall    : %.2f %%\n', ...
    recall_rf * 100);

fprintf('F1 Score  : %.2f %%\n', ...
    F1_rf * 100);

fprintf('\nConfusion Matrix Values:\n');

fprintf('TP = %d\n', TP_rf);
fprintf('TN = %d\n', TN_rf);
fprintf('FP = %d\n', FP_rf);
fprintf('FN = %d\n', FN_rf);


%% ============================================================
% 39. RANDOM FOREST - 5-FOLD CROSS VALIDATION
%% ============================================================

fprintf('\n=============================================\n');
fprintf('RANDOM FOREST - 5-FOLD CV\n');
fprintf('=============================================\n');


rng(42);

cv5_rf = cvpartition(y, 'KFold', 5);

fold_accuracy_rf = zeros(5,1);


for fold = 1:5

    %% Training / validation indices

    train_fold_idx = training(cv5_rf, fold);

    test_fold_idx = test(cv5_rf, fold);


    %% Split data

    X_train_fold = X(train_fold_idx,:);

    X_test_fold = X(test_fold_idx,:);

    y_train_fold = y(train_fold_idx);

    y_test_fold = y(test_fold_idx);


    %% Normalize using training fold only

    mu_fold = mean(X_train_fold, 1);

    sigma_fold = std(X_train_fold, 0, 1);

    sigma_fold(sigma_fold == 0) = 1;


    X_train_fold_norm = ...
        (X_train_fold - mu_fold) ./ sigma_fold;

    X_test_fold_norm = ...
        (X_test_fold - mu_fold) ./ sigma_fold;


    %% Train Random Forest

    model_fold = TreeBagger( ...
        num_trees, ...
        X_train_fold_norm, ...
        categorical(y_train_fold), ...
        'Method', 'classification');


    %% Predict

    y_pred_fold_cell = predict( ...
        model_fold, ...
        X_test_fold_norm);


    %% Convert prediction to numeric

    y_pred_fold = str2double(y_pred_fold_cell);


    %% Accuracy

    fold_accuracy_rf(fold) = ...
        mean(y_pred_fold == y_test_fold);


    fprintf('Fold %d Accuracy: %.2f %%\n', ...
        fold, ...
        fold_accuracy_rf(fold) * 100);

end


%% Random Forest CV Summary

mean_accuracy_rf = mean(fold_accuracy_rf);

std_accuracy_rf = std(fold_accuracy_rf);


fprintf('\nRandom Forest CV Results:\n');

fprintf('Mean Accuracy : %.2f %%\n', ...
    mean_accuracy_rf * 100);

fprintf('Std Deviation : %.2f %%\n', ...
    std_accuracy_rf * 100);


%% ============================================================
% 40. MODEL COMPARISON
%% ============================================================

fprintf('\n=============================================\n');
fprintf('MODEL COMPARISON\n');
fprintf('=============================================\n');


model_names = { ...
    'SVM'
    'Logistic Regression'
    'Random Forest'
};


test_accuracy = [ ...
    accuracy * 100
    accuracy_log * 100
    accuracy_rf * 100
];


test_precision = [ ...
    precision * 100
    precision_log * 100
    precision_rf * 100
];


test_recall = [ ...
    recall * 100
    recall_log * 100
    recall_rf * 100
];


test_F1 = [ ...
    F1 * 100
    F1_log * 100
    F1_rf * 100
];


cv_mean = [ ...
    mean_accuracy * 100
    mean_accuracy_log * 100
    mean_accuracy_rf * 100
];


cv_std = [ ...
    std_accuracy * 100
    std_accuracy_log * 100
    std_accuracy_rf * 100
];


%% Create comparison table

comparison_table = table( ...
    model_names, ...
    test_accuracy, ...
    test_precision, ...
    test_recall, ...
    test_F1, ...
    cv_mean, ...
    cv_std, ...
    'VariableNames', { ...
    'Model', ...
    'TestAccuracy', ...
    'TestPrecision', ...
    'TestRecall', ...
    'TestF1', ...
    'CVMeanAccuracy', ...
    'CVStdAccuracy'});


disp(comparison_table);


%% ============================================================
% 41. SAVE MODEL COMPARISON
%% ============================================================

writetable( ...
    comparison_table, ...
    'model_comparison.csv');


fprintf('\nModel comparison saved as:\n');

fprintf('model_comparison.csv\n');


%% ============================================================
% 42. MODEL ACCURACY VISUALIZATION
%% ============================================================

figure;

bar([ ...
    test_accuracy, ...
    cv_mean]);

set(gca, ...
    'XTick', 1:3, ...
    'XTickLabel', model_names);

ylabel('Accuracy (%)');

xlabel('Machine Learning Model');

title('Model Accuracy Comparison');

legend( ...
    'Test Accuracy', ...
    '5-Fold CV Mean Accuracy');

ylim([0 105]);

grid on;

xtickangle(20);


%% ============================================================
% 43. F1 SCORE COMPARISON
%% ============================================================

figure;

bar(test_F1);

set(gca, ...
    'XTick', 1:3, ...
    'XTickLabel', model_names);

ylabel('F1 Score (%)');

xlabel('Machine Learning Model');

title('F1 Score Comparison');

ylim([0 105]);

grid on;

xtickangle(20);


%% ============================================================
% 44. FINAL ML SUMMARY
%% ============================================================

fprintf('\n=============================================\n');
fprintf('FINAL MACHINE LEARNING RESULTS\n');
fprintf('=============================================\n');


fprintf('\nSVM:\n');

fprintf('Test Accuracy : %.2f %%\n', ...
    accuracy * 100);

fprintf('Test F1       : %.2f %%\n', ...
    F1 * 100);

fprintf('5-Fold CV     : %.2f +/- %.2f %%\n', ...
    mean_accuracy * 100, ...
    std_accuracy * 100);


fprintf('\nLogistic Regression:\n');

fprintf('Test Accuracy : %.2f %%\n', ...
    accuracy_log * 100);

fprintf('Test F1       : %.2f %%\n', ...
    F1_log * 100);

fprintf('5-Fold CV     : %.2f +/- %.2f %%\n', ...
    mean_accuracy_log * 100, ...
    std_accuracy_log * 100);


fprintf('\nRandom Forest:\n');

fprintf('Test Accuracy : %.2f %%\n', ...
    accuracy_rf * 100);

fprintf('Test F1       : %.2f %%\n', ...
    F1_rf * 100);

fprintf('5-Fold CV     : %.2f +/- %.2f %%\n', ...
    mean_accuracy_rf * 100, ...
    std_accuracy_rf * 100);


fprintf('\n=============================================\n');