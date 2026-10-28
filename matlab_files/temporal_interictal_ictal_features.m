%% temporal_interictal_ictal_features.m
% Temporal spectral features:
% F + N = Interictal (0)
% S     = Ictal (1)

clear;
clc;

Fs = 173.61;

folders = {'F','N','seizure'};
classes = [0 0 1];

% 4-second windows, 50% overlap
win_sec = 4;
win = round(win_sec * Fs);
step = round(win/2);

X = [];
y = [];

%% Process folders
for f = 1:length(folders)

    files = dir(fullfile(folders{f},'*.TXT'));

    if isempty(files)
        files = dir(fullfile(folders{f},'*.txt'));
    end

    fprintf('%s: %d files\n',folders{f},length(files));

    for k = 1:length(files)

        filepath = fullfile(files(k).folder,files(k).name);
        eeg = readmatrix(filepath);
        eeg = eeg(:);

        %% Bandpass filter
        [b,a] = butter(4,[0.5 40]/(Fs/2),'bandpass');
        eeg = filtfilt(b,a,eeg);

        %% Number of windows
        starts = 1:step:(length(eeg)-win+1);

        % Store band powers for every window
        BP = zeros(length(starts),5);

        for w = 1:length(starts)

            segment = eeg(starts(w):starts(w)+win-1);

            [pxx,freq] = pwelch(segment,[],[],[],Fs);

            bands = [0.5 4;
                     4   8;
                     8  13;
                    13  30;
                    30  40];

            for bnd = 1:5
                idx = freq >= bands(bnd,1) & ...
                      freq <  bands(bnd,2);

                BP(w,bnd) = trapz(freq(idx),pxx(idx));
            end
        end

        %% Temporal changes between consecutive windows

        if size(BP,1) > 1

            transition = abs(diff(BP));

            mean_transition = mean(transition,1);
            max_transition  = max(transition,[],1);

            % Linear drift
            t = (1:size(BP,1))';

            drift = zeros(1,5);

            for bnd = 1:5
                p = polyfit(t,BP(:,bnd),1);
                drift(bnd) = p(1);
            end

        else
            mean_transition = zeros(1,5);
            max_transition  = zeros(1,5);
            drift = zeros(1,5);
        end

        %% Final 15 temporal features

        features = [ ...
            mean_transition, ...
            max_transition, ...
            drift];

        X = [X; features];
        y = [y; classes(f)];

    end
end

%% Feature names

names = { ...
    'Delta_MeanTransition'
    'Theta_MeanTransition'
    'Alpha_MeanTransition'
    'Beta_MeanTransition'
    'Gamma_MeanTransition'
    'Delta_MaxTransition'
    'Theta_MaxTransition'
    'Alpha_MaxTransition'
    'Beta_MaxTransition'
    'Gamma_MaxTransition'
    'Delta_Drift'
    'Theta_Drift'
    'Alpha_Drift'
    'Beta_Drift'
    'Gamma_Drift'};

T = array2table(X,'VariableNames',names);
T.Class = y;

%% Save

writetable(T,'temporal_interictal_ictal_features.csv');

fprintf('\n====================================\n');
fprintf('Temporal feature extraction complete\n');
fprintf('Recordings: %d\n',size(X,1));
fprintf('Features: %d\n',size(X,2));
fprintf('Interictal: %d\n',sum(y==0));
fprintf('Ictal: %d\n',sum(y==1));
fprintf('Saved: temporal_interictal_ictal_features.csv\n');
fprintf('====================================\n');