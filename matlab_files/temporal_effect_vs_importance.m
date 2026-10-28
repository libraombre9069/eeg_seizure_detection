clear;
clc;

fprintf('Loading statistical temporal analysis...\n');

T = readtable('statistical_temporal_analysis.csv');

% ------------------------------------------------------------
% Absolute Cohen's d
% ------------------------------------------------------------

T.AbsCohensD = abs(T.CohensD);

% ------------------------------------------------------------
% Correlation between effect size and RF importance
% ------------------------------------------------------------

[r,p] = corr(T.AbsCohensD, T.RF_Importance, ...
    'Type','Spearman');

fprintf('\n');
fprintf('============================================================\n');
fprintf('TEMPORAL EFFECT SIZE vs RF IMPORTANCE\n');
fprintf('============================================================\n');

fprintf('\nSpearman correlation:\n');
fprintf('rho = %.4f\n',r);
fprintf('p   = %.6e\n',p);

% ------------------------------------------------------------
% Display sorted table
% ------------------------------------------------------------

[~,idx] = sort(T.RF_Importance,'descend');

SortedRF = T(idx,:);

fprintf('\n');
fprintf('============================================================\n');
fprintf('FEATURES SORTED BY RF IMPORTANCE\n');
fprintf('============================================================\n');

disp(SortedRF(:, ...
    {'Feature','CohensD','AbsCohensD','FDR_PValue','RF_Importance'}));

% ------------------------------------------------------------
% Save
% ------------------------------------------------------------

writetable(SortedRF,'temporal_effect_vs_importance.csv');

fprintf('\nResults saved to:\n');
fprintf('temporal_effect_vs_importance.csv\n');

% ------------------------------------------------------------
% Scatter plot
% ------------------------------------------------------------

figure;

scatter(T.AbsCohensD,T.RF_Importance,70,'filled');

xlabel('|Cohen''s d|');
ylabel('Random Forest Importance');

title('Temporal Feature Effect Size vs RF Importance');

grid on;

% Add feature labels
for i = 1:height(T)

    text(T.AbsCohensD(i), ...
         T.RF_Importance(i), ...
         ['  ' T.Feature{i}], ...
         'FontSize',8);

end

saveas(gcf,'temporal_effect_vs_rf_importance.png');

fprintf('\nPlot saved to:\n');
fprintf('temporal_effect_vs_rf_importance.png\n');

% ------------------------------------------------------------
% Identify high-effect / high-importance features
% ------------------------------------------------------------

fprintf('\n');
fprintf('============================================================\n');
fprintf('HIGH EFFECT + HIGH RF IMPORTANCE\n');
fprintf('============================================================\n');

% Thresholds chosen only for exploratory grouping:
% |d| >= 1
% RF importance >= median

importanceMedian = median(T.RF_Importance);

for i = 1:height(T)

    if T.AbsCohensD(i) >= 1 && ...
       T.RF_Importance(i) >= importanceMedian

        fprintf('%-30s  |d| = %.4f   RF = %.4f\n', ...
            T.Feature{i}, ...
            T.AbsCohensD(i), ...
            T.RF_Importance(i));

    end

end

fprintf('\n');
fprintf('Median RF importance = %.4f\n',importanceMedian);

fprintf('\n');
fprintf('============================================================\n');
fprintf('STEP 8 COMPLETE\n');
fprintf('============================================================\n');
