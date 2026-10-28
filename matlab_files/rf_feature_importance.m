%% rf_feature_importance.m
% Random Forest feature importance
% F + N = Interictal
% S     = Ictal

clear;
clc;
rng(42);

%% Load data
T = readtable('interictal_vs_ictal_features.csv');

X = T{:,1:end-1};
y = T.Class;

feature_names = T.Properties.VariableNames(1:end-1);

%% Normalize
mu = mean(X,1);
sigma = std(X,[],1);
sigma(sigma == 0) = 1;

X = (X-mu)./sigma;

%% Train Random Forest
fprintf('Training Random Forest...\n');

Mdl = TreeBagger(200,X,y,...
    'Method','classification',...
    'OOBPrediction','On',...
    'OOBPredictorImportance','On');

%% Feature importance
importance = Mdl.OOBPermutedPredictorDeltaError;

%% Create table
ImportanceTable = table( ...
    feature_names(:), ...
    importance(:), ...
    'VariableNames',{'Feature','Importance'});

%% Sort
ImportanceTable = sortrows( ...
    ImportanceTable,'Importance','descend');

disp(ImportanceTable);

%% Save
writetable(ImportanceTable,...
    'rf_feature_importance.csv');

%% Plot
figure;

bar(ImportanceTable.Importance);

set(gca,...
    'XTick',1:length(feature_names),...
    'XTickLabel',ImportanceTable.Feature,...
    'XTickLabelRotation',45);

xlabel('Feature');
ylabel('OOB Permuted Predictor Importance');
title('Random Forest Feature Importance');

grid on;

fprintf('\n========================================\n');
fprintf('FEATURE IMPORTANCE COMPLETE\n');
fprintf('Saved: rf_feature_importance.csv\n');
fprintf('========================================\n');