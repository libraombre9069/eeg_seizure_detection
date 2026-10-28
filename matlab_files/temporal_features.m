%% ============================================================
% TEMPORAL FEATURE EXTRACTION
% 5 EEG bands x 5 temporal statistics = 25 features
%
% Windows:
%   Window length = 4 seconds
%   Overlap       = 50%
%
% Temporal statistics:
%   1. Mean
%   2. Temporal Standard Deviation
%   3. Mean Absolute Transition
%   4. Maximum Transition
%   5. Drift
%
% Additional analysis:
%   Correlation with existing global features
%% ============================================================

clear;
clc;
close all;

%% ============================================================
% 1. PARAMETERS
% ============================================================

Fs = 173.61;

window_duration = 4;        % seconds
overlap_fraction = 0.50;    % 50% overlap

window_samples = round(window_duration * Fs);
hop_samples = round(window_samples * (1 - overlap_fraction));

fprintf('===============================================\n');
fprintf('TEMPORAL EEG FEATURE EXTRACTION\n');
fprintf('===============================================\n');

fprintf('Sampling frequency : %.2f Hz\n', Fs);
fprintf('Window duration    : %.2f seconds\n', window_duration);
fprintf('Window samples     : %d\n', window_samples);
fprintf('Overlap            : %.0f%%\n', overlap_fraction*100);
fprintf('Hop size           : %d samples\n', hop_samples);


%% ============================================================
% 2. FILE LISTS
% ============================================================

non_seizure_files = dir('non_seizure/Z*.txt');
seizure_files = dir('seizure/S*.txt');

num_non_seizure = length(non_seizure_files);
num_seizure = length(seizure_files);

num_total = num_non_seizure + num_seizure;

fprintf('\nNon-seizure recordings : %d\n', num_non_seizure);
fprintf('Seizure recordings     : %d\n', num_seizure);
fprintf('Total recordings       : %d\n', num_total);


%% ============================================================
% 3. INITIALIZE FEATURE MATRIX
% ============================================================

% 5 bands x 5 temporal statistics
% = 25 features

X_temporal = zeros(num_total, 25);

% Class labels
% 0 = Non-Seizure
% 1 = Seizure

y = zeros(num_total,1);


%% ============================================================
% 4. BANDPASS FILTER
% ============================================================

low_cutoff = 0.5;
high_cutoff = 40;
filter_order = 4;

[b,a] = butter(filter_order, ...
    [low_cutoff high_cutoff] / (Fs/2), ...
    'bandpass');


%% ============================================================
% 5. PROCESS NON-SEIZURE RECORDINGS
% ============================================================

fprintf('\n===============================================\n');
fprintf('PROCESSING NON-SEIZURE RECORDINGS\n');
fprintf('===============================================\n');

for i = 1:num_non_seizure

    filename = fullfile( ...
        non_seizure_files(i).folder, ...
        non_seizure_files(i).name);

    %% Load EEG
    eeg = readmatrix(filename);
    eeg = eeg(:);

    %% Filter
    eeg_filtered = filtfilt(b,a,eeg);

    %% Calculate temporal band powers
    band_power = calculate_temporal_bands( ...
        eeg_filtered, ...
        Fs, ...
        window_samples, ...
        hop_samples);

    %% Extract temporal statistics
    X_temporal(i,:) = ...
        extract_temporal_statistics(band_power);

    %% Label
    y(i) = 0;

    fprintf('Processed %d/%d : %s\n', ...
        i, num_non_seizure, ...
        non_seizure_files(i).name);

end


%% ============================================================
% 6. PROCESS SEIZURE RECORDINGS
% ============================================================

fprintf('\n===============================================\n');
fprintf('PROCESSING SEIZURE RECORDINGS\n');
fprintf('===============================================\n');

for i = 1:num_seizure

    filename = fullfile( ...
        seizure_files(i).folder, ...
        seizure_files(i).name);

    %% Load EEG
    eeg = readmatrix(filename);
    eeg = eeg(:);

    %% Filter
    eeg_filtered = filtfilt(b,a,eeg);

    %% Calculate temporal band powers
    band_power = calculate_temporal_bands( ...
        eeg_filtered, ...
        Fs, ...
        window_samples, ...
        hop_samples);

    %% Store features
    row = num_non_seizure + i;

    X_temporal(row,:) = ...
        extract_temporal_statistics(band_power);

    %% Label
    y(row) = 1;

    fprintf('Processed %d/%d : %s\n', ...
        i, num_seizure, ...
        seizure_files(i).name);

end


%% ============================================================
% 7. FEATURE NAMES
% ============================================================

bands = { ...
    'Delta', ...
    'Theta', ...
    'Alpha', ...
    'Beta', ...
    'Gamma'};

feature_names = {};

for bnd = 1:5

    feature_names{end+1} = ...
        [bands{bnd} '_Mean'];

    feature_names{end+1} = ...
        [bands{bnd} '_TemporalStd'];

    feature_names{end+1} = ...
        [bands{bnd} '_MeanTransition'];

    feature_names{end+1} = ...
        [bands{bnd} '_MaxTransition'];

    feature_names{end+1} = ...
        [bands{bnd} '_Drift'];

end


%% ============================================================
% 8. CREATE FEATURE TABLE
% ============================================================

T = array2table( ...
    X_temporal, ...
    'VariableNames', feature_names);

T.Label = y;

fprintf('\n===============================================\n');
fprintf('TEMPORAL FEATURE TABLE\n');
fprintf('===============================================\n');

disp(T);


%% ============================================================
% 9. CLASS-WISE FEATURE MEANS
% ============================================================

non_mean = mean( ...
    X_temporal(y==0,:), ...
    1);

seizure_mean = mean( ...
    X_temporal(y==1,:), ...
    1);

comparison = table( ...
    feature_names', ...
    non_mean', ...
    seizure_mean', ...
    'VariableNames', ...
    {'Feature','NonSeizure','Seizure'});

fprintf('\n===============================================\n');
fprintf('CLASS-WISE FEATURE MEANS\n');
fprintf('===============================================\n');

disp(comparison);


%% ============================================================
% 10. SAVE TEMPORAL FEATURES
% ============================================================

writetable( ...
    T, ...
    'eeg_temporal_features.csv');

fprintf('\nTemporal features saved as:\n');
fprintf('eeg_temporal_features.csv\n');


%% ============================================================
% 11. BOX PLOTS
% ============================================================

figure;

for k = 1:25

    subplot(5,5,k);

    boxplot( ...
        X_temporal(:,k), ...
        y, ...
        'Labels', ...
        {'Non-Seizure','Seizure'});

    title( ...
        feature_names{k}, ...
        'Interpreter','none');

    grid on;

end

sgtitle('Temporal Feature Distributions');


%% ============================================================
% 12. CORRELATION WITH EXISTING GLOBAL FEATURES
% ============================================================

fprintf('\n');
fprintf('===============================================\n');
fprintf('CORRELATION WITH EXISTING GLOBAL FEATURES\n');
fprintf('===============================================\n');


%% ------------------------------------------------------------
% Load existing global feature file
% ------------------------------------------------------------

global_table = readtable('eeg_features.csv');

fprintf('\nColumns found in eeg_features.csv:\n');
disp(global_table.Properties.VariableNames');


%% ------------------------------------------------------------
% IMPORTANT:
% Select ONLY the 10 numeric global features.
%
% Do NOT use:
%   Filename
%   Class
%   Label
%
% because these may be cell/string/categorical variables.
% ------------------------------------------------------------

global_names = { ...
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


%% ------------------------------------------------------------
% Check that all required columns exist
% ------------------------------------------------------------

available_names = global_table.Properties.VariableNames;

for i = 1:length(global_names)

    if ~ismember(global_names{i}, available_names)

        error( ...
            'Global feature "%s" was not found in eeg_features.csv.', ...
            global_names{i});

    end

end


%% ------------------------------------------------------------
% Extract ONLY numeric global features
% ------------------------------------------------------------

X_global = zeros(height(global_table), ...
                 length(global_names));

for i = 1:length(global_names)

    X_global(:,i) = ...
        global_table.(global_names{i});

end


%% ------------------------------------------------------------
% Check dimensions
% ------------------------------------------------------------

fprintf('\nGlobal feature matrix size:\n');
fprintf('%d recordings x %d features\n', ...
    size(X_global,1), ...
    size(X_global,2));

fprintf('Temporal feature matrix size:\n');
fprintf('%d recordings x %d features\n', ...
    size(X_temporal,1), ...
    size(X_temporal,2));


%% ------------------------------------------------------------
% Check recording count
% ------------------------------------------------------------

if size(X_global,1) ~= size(X_temporal,1)

    error(['Number of recordings in eeg_features.csv ' ...
           'does not match temporal feature matrix.']);

end


%% ============================================================
% 13. SELECT TEMPORAL FEATURES OF INTEREST
% ============================================================

candidate_names = { ...
    'Delta_MeanTransition', ...
    'Delta_MaxTransition', ...
    'Alpha_MeanTransition', ...
    'Alpha_MaxTransition', ...
    'Theta_TemporalStd', ...
    'Theta_Drift', ...
    'Gamma_MeanTransition', ...
    'Gamma_MaxTransition', ...
    'Beta_MeanTransition'};


candidate_idx = zeros( ...
    length(candidate_names),1);


for i = 1:length(candidate_names)

    idx = find( ...
        strcmp(feature_names, ...
        candidate_names{i}));

    if isempty(idx)

        error( ...
            'Temporal feature "%s" was not found.', ...
            candidate_names{i});

    end

    candidate_idx(i) = idx;

end


X_candidates = ...
    X_temporal(:,candidate_idx);


%% ============================================================
% 14. CORRELATION MATRIX
% ============================================================

R = corr( ...
    X_global, ...
    X_candidates, ...
    'Rows','complete');


%% ============================================================
% 15. DISPLAY CORRELATION MATRIX
% ============================================================

R_table = array2table( ...
    R, ...
    'VariableNames', candidate_names, ...
    'RowNames', global_names);


fprintf('\n===============================================\n');
fprintf('CORRELATION MATRIX\n');
fprintf('===============================================\n');

disp(R_table);


%% ============================================================
% 16. MAXIMUM ABSOLUTE CORRELATION
% ============================================================

max_corr = zeros( ...
    length(candidate_names),1);


for j = 1:length(candidate_names)

    max_corr(j) = ...
        max(abs(R(:,j)));

end


redundancy_table = table( ...
    candidate_names', ...
    max_corr, ...
    'VariableNames', ...
    {'TemporalFeature', ...
     'MaxAbsCorrelationWithGlobal'});


%% ============================================================
% 17. SORT BY CORRELATION
% ============================================================

redundancy_table = sortrows( ...
    redundancy_table, ...
    'MaxAbsCorrelationWithGlobal', ...
    'ascend');


fprintf('\n===============================================\n');
fprintf('TEMPORAL FEATURE REDUNDANCY ANALYSIS\n');
fprintf('===============================================\n');

disp(redundancy_table);


%% ============================================================
% 18. SAVE CORRELATION RESULTS
% ============================================================

writetable( ...
    redundancy_table, ...
    'temporal_feature_redundancy.csv');

fprintf('\nSaved:\n');
fprintf('temporal_feature_redundancy.csv\n');


%% ============================================================
% 19. CORRELATION HEATMAP
% ============================================================

figure;

imagesc(R);

colorbar;

xticks(1:length(candidate_names));
xticklabels(candidate_names);

yticks(1:length(global_names));
yticklabels(global_names);

xtickangle(45);

title('Correlation: Global Features vs Temporal Features');

xlabel('Temporal Features');
ylabel('Existing Global Features');


%% ============================================================
% 20. TEMPORAL FEATURE EFFECT SIZE
% ============================================================

fprintf('\n');
fprintf('===============================================\n');
fprintf('COHEN''S d FOR TEMPORAL FEATURES\n');
fprintf('===============================================\n');


d_values = zeros(25,1);

for k = 1:25

    x0 = X_temporal(y==0,k);
    x1 = X_temporal(y==1,k);

    mean0 = mean(x0);
    mean1 = mean(x1);

    std0 = std(x0);
    std1 = std(x1);

    n0 = length(x0);
    n1 = length(x1);

    %% Pooled standard deviation

    pooled_std = sqrt( ...
        ((n0-1)*std0^2 + ...
         (n1-1)*std1^2) / ...
        (n0+n1-2));

    %% Cohen's d

    d_values(k) = ...
        (mean1 - mean0) / pooled_std;

end


effect_table = table( ...
    feature_names', ...
    d_values, ...
    abs(d_values), ...
    'VariableNames', ...
    {'Feature','CohensD','AbsoluteCohensD'});


effect_table = sortrows( ...
    effect_table, ...
    'AbsoluteCohensD', ...
    'descend');


disp(effect_table);


%% ============================================================
% 21. SAVE EFFECT SIZE RESULTS
% ============================================================

writetable( ...
    effect_table, ...
    'temporal_feature_effect_sizes.csv');

fprintf('\nSaved:\n');
fprintf('temporal_feature_effect_sizes.csv\n');


%% ============================================================
% LOCAL FUNCTION 1
% CALCULATE TEMPORAL BAND POWERS
% ============================================================

function band_powers = ...
    calculate_temporal_bands( ...
    eeg, Fs, window_samples, hop_samples)


N = length(eeg);


%% Number of windows

num_windows = ...
    floor((N-window_samples) / hop_samples) + 1;


%% Initialize

band_powers = ...
    zeros(num_windows,5);


%% Process each window

for k = 1:num_windows

    %% Window indices

    start_idx = ...
        (k-1)*hop_samples + 1;

    end_idx = ...
        start_idx + window_samples - 1;


    %% Extract segment

    segment = ...
        eeg(start_idx:end_idx);


    %% Welch PSD

    [pxx,f] = ...
        pwelch(segment,[],[],[],Fs);


    %% Select 0.5-40 Hz

    idx = ...
        (f >= 0.5) & (f <= 40);

    f_band = f(idx);
    pxx_band = pxx(idx);


    %% Total power

    total_power = ...
        trapz(f_band,pxx_band);


    %% Prevent division by zero

    if total_power == 0

        continue;

    end


    %% Frequency bands

    delta_idx = ...
        (f_band >= 0.5) & ...
        (f_band < 4);

    theta_idx = ...
        (f_band >= 4) & ...
        (f_band < 8);

    alpha_idx = ...
        (f_band >= 8) & ...
        (f_band < 13);

    beta_idx = ...
        (f_band >= 13) & ...
        (f_band < 30);

    gamma_idx = ...
        (f_band >= 30) & ...
        (f_band <= 40);


    %% Relative powers

    band_powers(k,1) = ...
        trapz( ...
        f_band(delta_idx), ...
        pxx_band(delta_idx)) ...
        / total_power;


    band_powers(k,2) = ...
        trapz( ...
        f_band(theta_idx), ...
        pxx_band(theta_idx)) ...
        / total_power;


    band_powers(k,3) = ...
        trapz( ...
        f_band(alpha_idx), ...
        pxx_band(alpha_idx)) ...
        / total_power;


    band_powers(k,4) = ...
        trapz( ...
        f_band(beta_idx), ...
        pxx_band(beta_idx)) ...
        / total_power;


    band_powers(k,5) = ...
        trapz( ...
        f_band(gamma_idx), ...
        pxx_band(gamma_idx)) ...
        / total_power;

end

end


%% ============================================================
% LOCAL FUNCTION 2
% EXTRACT TEMPORAL STATISTICS
% ============================================================

function features = ...
    extract_temporal_statistics(band_power)


features = [];


%% Five EEG bands

for b = 1:5

    %% Current band's temporal trajectory

    x = band_power(:,b);


    %% 1. Mean

    temporal_mean = ...
        mean(x);


    %% 2. Temporal standard deviation

    temporal_std = ...
        std(x);


    %% 3. Mean absolute transition

    transitions = ...
        abs(diff(x));

    mean_transition = ...
        mean(transitions);


    %% 4. Maximum transition

    max_transition = ...
        max(transitions);


    %% 5. Drift

    drift = ...
        x(end) - x(1);


    %% Add to feature vector

    features = [ ...
        features, ...
        temporal_mean, ...
        temporal_std, ...
        mean_transition, ...
        max_transition, ...
        drift];

end

end