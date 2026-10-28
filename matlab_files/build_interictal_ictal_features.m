%% build_interictal_ictal_features.m
% Build feature dataset for:
% F + N = Interictal (Class 0)
% S     = Ictal/Seizure (Class 1)
%
% Output:
% interictal_vs_ictal_features.csv
%
% Requires:
% extract_features.m

clear;
clc;

%% Settings
Fs = 173.61;

folders = {'F', 'N', 'seizure'};
classes = [0, 0, 1];

X = [];
y = [];

%% Process folders
for f = 1:length(folders)

    current_folder = folders{f};

    % Look for uppercase .TXT
    files = dir(fullfile(current_folder, '*.TXT'));

    % Also check lowercase .txt if needed
    if isempty(files)
        files = dir(fullfile(current_folder, '*.txt'));
    end

    fprintf('%s folder: %d TXT files found\n', ...
        current_folder, length(files));

    if isempty(files)
        warning('No TXT files found in folder: %s', current_folder);
        continue;
    end

    %% Process each EEG file
    for k = 1:length(files)

        filepath = fullfile(files(k).folder, files(k).name);

        fprintf('Processing %s/%s\n', ...
            current_folder, files(k).name);

        % extract_features expects the filename
        features = extract_features(filepath, Fs);

        X = [X; features];
        y = [y; classes(f)];

    end
end

%% Check dataset
fprintf('\n========================================\n');
fprintf('Feature extraction completed.\n');
fprintf('Total recordings: %d\n', size(X,1));
fprintf('Total features: %d\n', size(X,2));
fprintf('Interictal (F + N): %d\n', sum(y == 0));
fprintf('Ictal (S): %d\n', sum(y == 1));
fprintf('========================================\n');

%% Feature names
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

%% Create table
T = array2table(X, 'VariableNames', feature_names);
T.Class = y;

%% Save
output_file = 'interictal_vs_ictal_features.csv';

writetable(T, output_file);

fprintf('\nSaved dataset as:\n%s\n', output_file);