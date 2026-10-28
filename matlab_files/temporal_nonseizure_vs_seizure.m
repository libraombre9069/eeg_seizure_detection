clc;
clear;
close all;

fprintf('========================================\n');
fprintf('TEMPORAL FEATURES: NON-SEIZURE vs SEIZURE\n');
fprintf('========================================\n\n');

%% Parameters
Fs = 173.61;

% Frequency bands
bands = [
    0.5 4;    % Delta
    4   8;    % Theta
    8   13;   % Alpha
    13  30;   % Beta
    30  40    % Gamma
];

band_names = {'Delta','Theta','Alpha','Beta','Gamma'};

% Temporal window parameters
window_duration = 4;      % seconds
overlap = 0.50;           % 50% overlap

window_samples = round(window_duration * Fs);
step_samples = round(window_samples * (1 - overlap));

%% Folders
folders = {'non_seizure','O','F','N','seizure'};

% Class labels
% Z,O,F,N = non-seizure = 0
% S = seizure = 1
classes = [0 0 0 0 1];

%% Storage
all_features = [];
all_labels = [];
all_ids = {};

%% Feature names
feature_names = {};

for b = 1:length(band_names)
    feature_names{end+1} = [band_names{b} '_MeanTransition'];
    feature_names{end+1} = [band_names{b} '_MaxTransition'];
    feature_names{end+1} = [band_names{b} '_Drift'];
end

%% Butterworth filter
[b_filter, a_filter] = butter(4, [0.5 40]/(Fs/2), 'bandpass');

%% Process each folder
for f = 1:length(folders)

    folder = folders{f};

    fprintf('\nProcessing folder: %s\n', folder);

    % Find TXT files
    files = dir(fullfile(folder, '*.txt'));

    if isempty(files)
        files = dir(fullfile(folder, '*.TXT'));
    end

    fprintf('%d files found\n', length(files));

    for k = 1:length(files)

        filename = fullfile(folder, files(k).name);

        fprintf('Processing %s/%s\n', folder, files(k).name);

        %% Load EEG
        eeg = readmatrix(filename);

        % Make sure EEG is a column vector
        eeg = eeg(:);

        %% Filter
        eeg_filtered = filtfilt(b_filter, a_filter, eeg);

        %% Number of windows
        num_windows = floor((length(eeg_filtered) - window_samples) ...
                            / step_samples) + 1;

        %% Store band powers for every window
        window_band_power = zeros(num_windows, length(bands));

        %% Calculate band power for each window
        for w = 1:num_windows

            start_idx = (w-1)*step_samples + 1;
            end_idx = start_idx + window_samples - 1;

            segment = eeg_filtered(start_idx:end_idx);

            % Welch PSD
            [pxx, freq] = pwelch(segment, [], [], [], Fs);

            for b = 1:length(bands)

                idx = freq >= bands(b,1) & freq < bands(b,2);

                window_band_power(w,b) = trapz( ...
                    freq(idx), ...
                    pxx(idx) ...
                );

            end
        end

        %% Extract temporal features
        recording_features = [];

        for b = 1:length(bands)

            power_series = window_band_power(:,b);

            %% Change between consecutive windows
            transitions = diff(power_series);

            %% Mean absolute transition
            mean_transition = mean(abs(transitions));

            %% Maximum absolute transition
            max_transition = max(abs(transitions));

            %% Drift across recording
            x = (1:length(power_series))';

            if length(power_series) > 1
                p = polyfit(x, power_series, 1);
                drift = p(1);
            else
                drift = 0;
            end

            recording_features = [
                recording_features, ...
                mean_transition, ...
                max_transition, ...
                drift
            ];

        end

        %% Store
        all_features = [
            all_features;
            recording_features
        ];

        all_labels = [
            all_labels;
            classes(f)
        ];

        all_ids = [
            all_ids;
            {sprintf('%s_%s', folder, files(k).name)}
        ];

    end
end

%% Create table
T = array2table(all_features, ...
    'VariableNames', feature_names);

T.Label = all_labels;
T.ID = all_ids;

%% Save
output_file = 'temporal_nonseizure_vs_seizure_features.csv';

writetable(T, output_file);

%% Summary
fprintf('\n========================================\n');
fprintf('DATASET SUMMARY\n');
fprintf('========================================\n');

fprintf('Total recordings: %d\n', height(T));
fprintf('Total temporal features: %d\n', length(feature_names));

fprintf('\nClass distribution:\n');
fprintf('Non-seizure: %d\n', sum(all_labels == 0));
fprintf('Seizure:     %d\n', sum(all_labels == 1));

fprintf('\nSaved as:\n');
fprintf('%s\n', output_file);

fprintf('\nTemporal feature extraction completed.\n');