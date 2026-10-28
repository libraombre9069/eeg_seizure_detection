%% feature_ablation_rf.m
% Feature ablation using Random Forest
% F + N = Interictal (0)
% S     = Ictal (1)

clear;
clc;
rng(42);

%% Load data
T = readtable('interictal_vs_ictal_features.csv');

X = T{:,1:end-1};
y = T.Class;

feature_names = T.Properties.VariableNames(1:end-1);

%% 5-fold cross-validation
cv = cvpartition(y,'KFold',5);

%% Number of features
num_features = size(X,2);

%% Store results
AccuracyMean = zeros(num_features+1,1);
AccuracyStd  = zeros(num_features+1,1);
F1Mean       = zeros(num_features+1,1);
AUCMean      = zeros(num_features+1,1);

FeatureSet = cell(num_features+1,1);

%% ==========================================
% Baseline: all features
% ===========================================

FeatureSet{1} = 'All Features';

[AccuracyMean(1),AccuracyStd(1),F1Mean(1),AUCMean(1)] = ...
    evaluate_rf(X,y,cv);

%% ==========================================
% Remove one feature at a time
% ===========================================

for f = 1:num_features

    fprintf('Removing feature: %s\n',feature_names{f});

    X_ablation = X;
    X_ablation(:,f) = [];

    FeatureSet{f+1} = ...
        ['Without ' feature_names{f}];

    [AccuracyMean(f+1),AccuracyStd(f+1),...
        F1Mean(f+1),AUCMean(f+1)] = ...
        evaluate_rf(X_ablation,y,cv);

end

%% ==========================================
% Results table
% ===========================================

Results = table( ...
    FeatureSet,...
    AccuracyMean,...
    AccuracyStd,...
    F1Mean,...
    AUCMean);

disp(Results);

%% Save
writetable(Results,'feature_ablation_rf.csv');

fprintf('\n========================================\n');
fprintf('FEATURE ABLATION COMPLETE\n');
fprintf('Saved: feature_ablation_rf.csv\n');
fprintf('========================================\n');


%% ==========================================
% Local function
% ==========================================

function [acc_mean,acc_std,f1_mean,auc_mean] = ...
    evaluate_rf(X,y,cv)

    nfolds = cv.NumTestSets;

    acc = zeros(nfolds,1);
    f1  = zeros(nfolds,1);
    auc = zeros(nfolds,1);

    for fold = 1:nfolds

        trainIdx = training(cv,fold);
        testIdx  = test(cv,fold);

        Xtrain = X(trainIdx,:);
        Xtest  = X(testIdx,:);

        ytrain = y(trainIdx);
        ytest  = y(testIdx);

        %% Training-only normalization
        mu = mean(Xtrain,1);
        sigma = std(Xtrain,[],1);

        sigma(sigma == 0) = 1;

        Xtrain = (Xtrain-mu)./sigma;
        Xtest  = (Xtest-mu)./sigma;

        %% Random Forest
        mdl = TreeBagger(100,...
            Xtrain,...
            ytrain,...
            'Method','classification');

        [predCell,score] = predict(mdl,Xtest);

        pred = str2double(predCell);

        %% Accuracy
        acc(fold) = mean(pred == ytest);

        %% F1
        TP = sum((ytest==1) & (pred==1));
        FP = sum((ytest==0) & (pred==1));
        FN = sum((ytest==1) & (pred==0));

        if TP+FP == 0
            precision = 0;
        else
            precision = TP/(TP+FP);
        end

        if TP+FN == 0
            recall = 0;
        else
            recall = TP/(TP+FN);
        end

        if precision+recall == 0
            f1(fold) = 0;
        else
            f1(fold) = ...
                2*precision*recall/(precision+recall);
        end

        %% ROC-AUC
        [~,~,~,auc(fold)] = ...
            perfcurve(ytest,score(:,2),1);

    end

    acc_mean = mean(acc)*100;
    acc_std  = std(acc)*100;
    f1_mean  = mean(f1)*100;
    auc_mean = mean(auc);

end