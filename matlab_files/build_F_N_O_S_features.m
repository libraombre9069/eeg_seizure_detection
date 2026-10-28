%% build_F_N_O_S_features.m
clear; clc;

Fs = 173.61;

folders = {'F','N','O','seizure'};
groups  = {'F','N','O','S'};

X = [];
Group = {};

fprintf('========================================\n');
fprintf('F + N + O + S FEATURE EXTRACTION\n');
fprintf('========================================\n');

for f = 1:length(folders)

    folder = folders{f};

    files = dir(fullfile(folder,'*.TXT'));
    if isempty(files)
        files = dir(fullfile(folder,'*.txt'));
    end

    fprintf('\n%s: %d files found\n', groups{f}, length(files));

    for k = 1:length(files)

        filepath = fullfile(files(k).folder,files(k).name);

        features = extract_features(filepath,Fs);

        X = [X; features];
        Group{end+1,1} = groups{f};

    end
end


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

T = array2table(X,'VariableNames',feature_names);

T.Group = categorical(Group);


%% Save

writetable(T,'f_n_o_s_features.csv');

fprintf('\n========================================\n');
fprintf('COMPLETE\n');
fprintf('Total recordings: %d\n',height(T));
fprintf('F: %d\n',sum(T.Group=="F"));
fprintf('N: %d\n',sum(T.Group=="N"));
fprintf('O: %d\n',sum(T.Group=="O"));
fprintf('S: %d\n',sum(T.Group=="S"));
fprintf('Saved: f_n_o_s_features.csv\n');
fprintf('========================================\n');