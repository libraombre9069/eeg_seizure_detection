%% STEP 11 — FINAL FIGURES
clear;
clc;
close all;

Fs = 173.61;

%% ============================================================
% FIGURE 1 — REPRESENTATIVE EEG SIGNALS
% =============================================================

% Automatically search for Z001 and S001 anywhere under MATLAB Drive
z_files = dir(fullfile('**','Z001*'));
s_files = dir(fullfile('**','S001*'));

if isempty(z_files)
    error('Could not find Z001. Check that the Z dataset is uploaded.');
end

if isempty(s_files)
    error('Could not find S001. Check that the S dataset is uploaded.');
end

nonseizure_file = fullfile(z_files(1).folder,z_files(1).name);
seizure_file    = fullfile(s_files(1).folder,s_files(1).name);

fprintf('\nFigure 1 files:\n');
fprintf('Non-seizure: %s\n',nonseizure_file);
fprintf('Seizure:     %s\n',seizure_file);

% Load signals
nonseizure = readmatrix(nonseizure_file);
seizure    = readmatrix(seizure_file);

nonseizure = nonseizure(:);
seizure    = seizure(:);

% Time vectors
t1 = (0:length(nonseizure)-1)/Fs;
t2 = (0:length(seizure)-1)/Fs;

% Plot
figure;

plot(t1,nonseizure,'LineWidth',1);
hold on;
plot(t2,seizure,'LineWidth',1);

xlabel('Time (s)');
ylabel('Amplitude');
title('Representative EEG Signals');

legend('Non-seizure','Seizure','Location','best');

grid on;
box on;


%% ============================================================
% FIGURE 2 — GLOBAL FEATURE DISTRIBUTIONS
% =============================================================

G = readtable('nonseizure_vs_seizure_features.csv');

fprintf('\nGlobal feature columns:\n');
disp(G.Properties.VariableNames');

% First 10 columns are the global EEG features
feature_names = G.Properties.VariableNames(1:10);

X = G{:,1:10};
Y = G.Class;

figure;

for i = 1:10

    subplot(2,5,i);

    boxchart(categorical(Y),X(:,i));

    title(feature_names{i},'Interpreter','none');
    xlabel('Class');
    ylabel('Value');

    grid on;
    box on;

end

sgtitle('Global EEG Feature Distributions');


%% ============================================================
% FIGURE 3 — TEMPORAL FEATURE DISTRIBUTIONS
% =============================================================

T = readtable('temporal_nonseizure_vs_seizure_features.csv');

fprintf('\nTemporal feature columns:\n');
disp(T.Properties.VariableNames');

% First 15 columns are temporal features
temporal_names = T.Properties.VariableNames(1:15);

Xtemporal = T{:,1:15};
Ytemporal = T.Label;

figure;

for i = 1:15

    subplot(3,5,i);

    boxchart(categorical(Ytemporal),Xtemporal(:,i));

    title(temporal_names{i},'Interpreter','none');
    xlabel('Class');
    ylabel('Value');

    grid on;
    box on;

end

sgtitle('Temporal Spectral-Dynamics Feature Distributions');


%% ============================================================
% FIGURE 4 — REPEATED CV PERFORMANCE
% =============================================================

% Step 10 results
feature_sets = categorical({ ...
    'Global', ...
    'Global + MeanTransition', ...
    'Global + MaxTransition', ...
    'Global + All Temporal'});

accuracy = [98.340 98.500 98.180 98.260];

f1 = [95.825 96.210 95.399 95.607];

auc = [0.99863 0.99845 0.99797 0.99733];

figure;

% Accuracy
subplot(3,1,1);

bar(feature_sets,accuracy);

ylabel('Accuracy (%)');
title('Accuracy');

grid on;
box on;


% F1
subplot(3,1,2);

bar(feature_sets,f1);

ylabel('F1 Score (%)');
title('F1 Score');

grid on;
box on;


% AUC
subplot(3,1,3);

bar(feature_sets,auc);

ylabel('ROC-AUC');
title('ROC-AUC');

grid on;
box on;

sgtitle('Repeated Cross-Validation Performance');


%% ============================================================
% FIGURE 5 — RANDOM FOREST FEATURE IMPORTANCE
% =============================================================

I = readtable('rf_combined_25_feature_importance.csv');

fprintf('\nRF importance columns:\n');
disp(I.Properties.VariableNames');

% ------------------------------------------------------------
% Automatically identify feature and importance columns
% ------------------------------------------------------------

var_names = I.Properties.VariableNames;

feature_col = [];
importance_col = [];

for i = 1:length(var_names)

    name = lower(var_names{i});

    if contains(name,'feature')
        feature_col = i;
    end

    if contains(name,'importance')
        importance_col = i;
    end

end

% If automatic detection fails, assume:
% Column 1 = Feature
% Column 2 = Importance

if isempty(feature_col)
    feature_col = 1;
end

if isempty(importance_col)
    importance_col = 2;
end

feature_names_rf = I{:,feature_col};
importance_values = I{:,importance_col};

% Convert feature names if necessary
if iscell(feature_names_rf)
    feature_names_rf = string(feature_names_rf);
elseif ischar(feature_names_rf)
    feature_names_rf = string(cellstr(feature_names_rf));
end

% Sort importance from smallest to largest
[importance_sorted,idx] = sort(importance_values,'ascend');

feature_sorted = feature_names_rf(idx);

% Plot
figure;

barh(importance_sorted);

yticks(1:length(feature_sorted));
yticklabels(feature_sorted);

xlabel('OOB Permuted Predictor Importance');
ylabel('Feature');

title('Random Forest Importance — Global + Temporal Features');

grid on;
box on;


%% ============================================================
% FIGURE 6 — REPEATED CV ROBUSTNESS
% =============================================================

feature_sets = categorical({ ...
    'Global', ...
    'Global + MeanTransition', ...
    'Global + MaxTransition', ...
    'Global + All Temporal'});

% Mean values from Step 10
accuracy_mean = [98.340 98.500 98.180 98.260];
accuracy_std  = [1.154 0.995 1.190 1.226];

f1_mean = [95.825 96.210 95.399 95.607];
f1_std  = [2.914 2.520 3.039 3.080];

auc_mean = [0.99863 0.99845 0.99797 0.99733];
auc_std  = [0.00208 0.00260 0.00277 0.00435];


figure;

% Accuracy
subplot(3,1,1);

errorbar(1:4,accuracy_mean,accuracy_std,'o-');

xticks(1:4);
xticklabels(feature_sets);

ylabel('Accuracy (%)');
title('Repeated-CV Accuracy');

grid on;
box on;


% F1
subplot(3,1,2);

errorbar(1:4,f1_mean,f1_std,'o-');

xticks(1:4);
xticklabels(feature_sets);

ylabel('F1 Score (%)');
title('Repeated-CV F1 Score');

grid on;
box on;


% AUC
subplot(3,1,3);

errorbar(1:4,auc_mean,auc_std,'o-');

xticks(1:4);
xticklabels(feature_sets);

ylabel('ROC-AUC');
title('Repeated-CV ROC-AUC');

grid on;
box on;

sgtitle('Robustness Across 10 × 5-Fold Cross-Validation');


%% ============================================================
% DONE
% =============================================================

fprintf('\n============================================\n');
fprintf('STEP 11 FIGURES COMPLETED SUCCESSFULLY\n');
fprintf('============================================\n');

fprintf('\nGenerated figures:\n');
fprintf('Figure 1: Representative EEG Signals\n');
fprintf('Figure 2: Global Feature Distributions\n');
fprintf('Figure 3: Temporal Feature Distributions\n');
fprintf('Figure 4: Repeated CV Performance\n');
fprintf('Figure 5: Random Forest Feature Importance\n');
fprintf('Figure 6: Repeated CV Robustness\n');