%% EEG-Based Academic Stress Detection - SAM40
clear; clc; close all;
fs  = 256;
data_path   = 'BPSS - Assignment 2/fresh_data/filtered_data/';
scales_path = 'BPSS - Assignment 2/fresh_data/scales.xls';
task_names  = {'Arithmetic', 'Symmetry', 'Stroop'};
scale_cols  = [1 2 3; 4 5 6; 7 8 9];

%% STEP 1: Load Self-Report Scales
scales_raw  = readmatrix(scales_path, 'NumHeaderLines', 2);
scales_data = scales_raw(:, 2:10);  % 40 x 9, drop subject number col
fprintf('Scales loaded: %d subjects x %d scores\n', size(scales_data));
fprintf('First subject scores: '); disp(scales_data(1,:));

%% STEP 2: Feature Extraction Loop
all_features = [];
all_labels   = [];
all_sam      = [];
fprintf('Extracting features...\n');
for sub = 1:40
    for trial = 1:3
        for t = 1:3
            fname = sprintf('%s%s_sub_%d_trial%d.mat', ...
                            data_path, task_names{t}, sub, trial);
            if ~isfile(fname), continue; end
            tmp = load(fname);
            eeg = tmp.Clean_data;  % 32 x 3200
            alpha_pow = 0; beta_pow = 0; sp_ent = 0;
            for ch = 1:32
                sig = double(eeg(ch, :));
                alpha_pow = alpha_pow + bandpower(sig, fs, [8  13]);
                beta_pow  = beta_pow  + bandpower(sig, fs, [13 30]);
                [pxx, ~] = pwelch(sig, 128, 64, 256, fs);
                pn       = pxx / (sum(pxx) + eps);
                sp_ent   = sp_ent + (-sum(pn .* log2(pn + eps)));
            end
            alpha_pow = alpha_pow / 32;
            beta_pow  = beta_pow  / 32;
            sp_ent    = sp_ent    / 32;
            ab_ratio  = alpha_pow / (beta_pow + eps);
            sam_score = scales_data(sub, scale_cols(trial, t));
            label     = double(sam_score >= 5);
            all_features = [all_features; ab_ratio, sp_ent, alpha_pow, beta_pow];
            all_labels   = [all_labels;   label];
            all_sam      = [all_sam;      sam_score];
        end
    end
end
fprintf('Total samples extracted: %d\n', length(all_labels));
fprintf('Stress samples: %d | Baseline samples: %d\n', ...
        sum(all_labels==1), sum(all_labels==0));
% --- SAM Debug Check ---
fprintf('\n--- SAM Debug ---\n');
fprintf('SAM unique values: '); disp(unique(all_sam)');
fprintf('SAM std:  %.4f\n', std(all_sam));
fprintf('SAM min:  %d  |  SAM max:  %d\n', min(all_sam), max(all_sam));
fprintf('-----------------\n\n');
%% OUTPUT 1: Stress Biomarker Metrics Table
stress_idx   = all_labels == 1;
baseline_idx = all_labels == 0;
feat_names = {'Alpha/Beta Ratio', 'Spectral Entropy', ...
              'Alpha Power',      'Beta Power'};
stress_means   = mean(all_features(stress_idx,   :))';
baseline_means = mean(all_features(baseline_idx, :))';
stress_stds    = std(all_features(stress_idx,    :))';
baseline_stds  = std(all_features(baseline_idx,  :))';
fig1 = figure('Name','Output 1: Biomarker Metrics Table', ...
              'Position',[100 100 860 300], 'Color','white');
% --- Layout constants (all in figure-normalised units) ---
left     = 0.02;
fig_w    = 0.96;
hdr_bot  = 0.70; hdr_h  = 0.16;
row_h    = 0.14;
row_bots = [0.54, 0.38, 0.22, 0.06];
col_x    = [0.03, 0.24, 0.42, 0.58, 0.76];
col_w    = [0.20, 0.17, 0.15, 0.17, 0.17];
row_bg   = {[0.92 0.96 1.0], [1 1 1], [0.92 0.96 1.0], [1 1 1]};
% Title
annotation('textbox', [left, 0.87, fig_w, 0.12], ...
    'String', 'Stress Biomarker Metrics: Stress vs Baseline (SAM40)', ...
    'FontSize', 11, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
    'BackgroundColor', 'white', 'EdgeColor', 'none');
% Header background
annotation('rectangle', [left, hdr_bot, fig_w, hdr_h], ...
    'FaceColor', [0.18 0.35 0.60], 'EdgeColor', 'none');
% Header text — one textbox per cell
col_headers = {'Metric', 'Stress Mean', 'Stress STD', 'Baseline Mean', 'Baseline STD'};
for c = 1:5
    annotation('textbox', [col_x(c), hdr_bot, col_w(c), hdr_h], ...
        'String', col_headers{c}, ...
        'FontSize', 9, 'FontWeight', 'bold', 'Color', 'white', ...
        'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', ...
        'EdgeColor', 'none', 'Interpreter', 'none');
end
% Data rows — one textbox per cell
for r = 1:4
    rb = row_bots(r);
    annotation('rectangle', [left, rb, fig_w, row_h], ...
        'FaceColor', row_bg{r}, 'EdgeColor', [0.75 0.75 0.75]);
    row_vals = {feat_names{r}, ...
        sprintf('%.5f', stress_means(r)), ...
        sprintf('%.5f', stress_stds(r)), ...
        sprintf('%.5f', baseline_means(r)), ...
        sprintf('%.5f', baseline_stds(r))};
    for c = 1:5
        fw = 'normal'; if c == 1, fw = 'bold'; end
        annotation('textbox', [col_x(c), rb, col_w(c), row_h], ...
            'String', row_vals{c}, ...
            'FontSize', 8.5, 'FontWeight', fw, 'Color', [0.1 0.1 0.1], ...
            'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', ...
            'EdgeColor', 'none', 'Interpreter', 'none');
    end
end
saveas(fig1, 'Output1_Biomarker_Table.png');
fprintf('Output 1 saved.\n');

%% OUTPUT 2: EEG Spectrograms (Stress vs Baseline)
sam_arith_t1       = scales_data(:, 1);
[~, best_stress]   = max(sam_arith_t1);
[~, best_baseline] = min(sam_arith_t1);
fprintf('Spectrogram: Stress subject = %d (SAM=%d), Baseline subject = %d (SAM=%d)\n', ...
    best_stress,   sam_arith_t1(best_stress), ...
    best_baseline, sam_arith_t1(best_baseline));
s_data = load(sprintf('%sArithmetic_sub_%d_trial1.mat', data_path, best_stress));
b_data = load(sprintf('%sArithmetic_sub_%d_trial1.mat', data_path, best_baseline));
sig_stress   = mean(double(s_data.Clean_data(1:6, :)), 1);
sig_baseline = mean(double(b_data.Clean_data(1:6, :)), 1);
fig2 = figure('Name','Output 2: EEG Spectrograms', ...
               'Position',[100 100 1100 430]);
subplot(1,2,1);
spectrogram(sig_baseline, 256, 128, 256, fs, 'yaxis');
title(sprintf('Baseline EEG - Subject %d (SAM=%d)', ...
      best_baseline, sam_arith_t1(best_baseline)));
ylim([0 50]); colormap jet; clim([-120 0]);
subplot(1,2,2);
spectrogram(sig_stress, 256, 128, 256, fs, 'yaxis');
title(sprintf('Exam/Stress EEG - Subject %d (SAM=%d)', ...
      best_stress, sam_arith_t1(best_stress)));
ylim([0 50]); colormap jet; clim([-120 0]);
saveas(fig2, 'Output2_Spectrograms.png');
fprintf('Output 2 saved.\n');
%% Output 2 - Spectrogram Summary Table
[s_b, f_b, t_b] = spectrogram(sig_baseline, 256, 128, 256, fs);
[s_s, f_s, t_s] = spectrogram(sig_stress,   256, 128, 256, fs);

pow_b = 10*log10(abs(s_b).^2 + eps);
pow_s = 10*log10(abs(s_s).^2 + eps);

bands      = {'Delta','Theta','Alpha','Beta','Gamma'};
band_ranges = [0 4; 4 8; 8 13; 13 30; 30 50];

fig5 = figure('Name','Output 2 Table: Spectrogram Band Summary', ...
              'Position',[100 100 860 260], 'Color','white');

% Layout constants
left    = 0.02; fig_w = 0.96;
hdr_bot = 0.72; hdr_h = 0.16;
row_h   = 0.12;
row_bots = [0.58, 0.44, 0.30, 0.16, 0.02];
col_x   = [0.03, 0.20, 0.36, 0.56, 0.76];
col_w   = [0.16, 0.15, 0.19, 0.19, 0.18];
row_bg  = {[0.92 0.96 1.0],[1 1 1],[0.92 0.96 1.0],[1 1 1],[0.92 0.96 1.0]};

annotation('textbox',[left,0.88,fig_w,0.11], ...
    'String','Spectrogram Band Power Summary: Stress vs Baseline (SAM40)', ...
    'FontSize',11,'FontWeight','bold', ...
    'HorizontalAlignment','center','VerticalAlignment','middle', ...
    'BackgroundColor','white','EdgeColor','none');

annotation('rectangle',[left,hdr_bot,fig_w,hdr_h], ...
    'FaceColor',[0.18 0.35 0.60],'EdgeColor','none');

col_headers = {'Band','Range (Hz)','Baseline (dB/Hz)','Stress (dB/Hz)','Difference'};
for c = 1:5
    annotation('textbox',[col_x(c),hdr_bot,col_w(c),hdr_h], ...
        'String',col_headers{c}, ...
        'FontSize',9,'FontWeight','bold','Color','white', ...
        'HorizontalAlignment','left','VerticalAlignment','middle', ...
        'EdgeColor','none','Interpreter','none');
end

for r = 1:5
    fmin = band_ranges(r,1);
    fmax = band_ranges(r,2);
    idx  = f_b >= fmin & f_b <= fmax;

    avg_b   = mean(mean(pow_b(idx,:)));
    avg_s   = mean(mean(pow_s(idx,:)));
    diff_bs = avg_s - avg_b;

    rb = row_bots(r);
    annotation('rectangle',[left,rb,fig_w,row_h], ...
        'FaceColor',row_bg{r},'EdgeColor',[0.75 0.75 0.75]);

    row_vals = {bands{r}, ...
        sprintf('%d–%d', fmin, fmax), ...
        sprintf('%.2f', avg_b), ...
        sprintf('%.2f', avg_s), ...
        sprintf('%+.2f', diff_bs)};

    for c = 1:5
        fw = 'normal'; if c==1, fw='bold'; end
        annotation('textbox',[col_x(c),rb,col_w(c),row_h], ...
            'String',row_vals{c}, ...
            'FontSize',8.5,'FontWeight',fw,'Color',[0.1 0.1 0.1], ...
            'HorizontalAlignment','left','VerticalAlignment','middle', ...
            'EdgeColor','none','Interpreter','none');
    end
end

saveas(fig5,'Output2_Spectrogram_Table.png');
fprintf('Output 2 table saved.\n');
%% OUTPUT 3: ROC Curve
X_feat   = normalize(all_features);
SVMModel = fitcsvm(X_feat, all_labels, ...
    'KernelFunction', 'rbf', ...
    'Standardize',    true, ...
    'ClassNames',     [0, 1]);
[~, scores]             = resubPredict(SVMModel);
[X_roc, Y_roc, ~, AUC] = perfcurve(all_labels, scores(:,2), 1);
fig3 = figure('Name','Output 3: ROC Curve', ...
               'Position',[100 100 600 500]);
plot(X_roc, Y_roc, 'b-', 'LineWidth', 2.5); hold on;
plot([0 1], [0 1], 'k--', 'LineWidth', 1.2);
xlabel('False Positive Rate', 'FontSize', 12);
ylabel('True Positive Rate',  'FontSize', 12);
title(sprintf('ROC Curve \x2014 Stress Classification (AUC = %.3f)', AUC), ...
      'FontSize', 13, 'FontWeight', 'bold');
legend(sprintf('SVM RBF (AUC = %.3f)', AUC), 'Random Classifier', ...
       'Location', 'SouthEast', 'FontSize', 11);
grid on;
saveas(fig3, 'Output3_ROC_Curve.png');
fprintf('Output 3 saved. AUC = %.3f\n', AUC);

%% OUTPUT 4: Comparison with Self-Reported Scales
if std(all_sam) > 0 && std(scores(:,2)) > 0
    [rho, pval] = corr(scores(:,2), all_sam, 'Type', 'Spearman');
else
    rho  = NaN;
    pval = NaN;
    warning('SAM scores or SVM scores have zero variance — correlation is undefined.');
end
fig4 = figure('Name','Output 4: SVM Score vs SAM Rating', ...
               'Position',[100 100 650 520]);
scatter(all_sam(baseline_idx), scores(baseline_idx,2), ...
        50, [0.2 0.5 1], 'filled', 'DisplayName','Baseline (SAM < 5)');
hold on;
scatter(all_sam(stress_idx), scores(stress_idx,2), ...
        50, [1 0.2 0.2], 'filled', 'DisplayName','Stress (SAM \geq 5)');
if ~isnan(rho)
    p      = polyfit(all_sam, scores(:,2), 1);
    x_line = linspace(min(all_sam), max(all_sam), 100);
    plot(x_line, polyval(p, x_line), 'k-', 'LineWidth', 2, ...
         'DisplayName', 'Trend line');
end
xlabel('Self-Reported SAM Stress Score', 'FontSize', 12);
ylabel('SVM Decision Score',             'FontSize', 12);
if ~isnan(rho)
    title(sprintf('SVM Score vs SAM Rating\nSpearman r = %.3f,  p = %.4f', ...
          rho, pval), 'FontSize', 13, 'FontWeight', 'bold');
    fprintf('Output 4 saved. Spearman r = %.3f, p = %.4f\n', rho, pval);
else
    title(sprintf('SVM Score vs SAM Rating\n(Correlation undefined - check SAM variance)'), ...
          'FontSize', 13, 'FontWeight', 'bold');
    fprintf('Output 4: Correlation was NaN - SAM std = %.4f\n', std(all_sam));
end
legend('Location','NorthWest', 'FontSize', 10);
grid on;
saveas(fig4, 'Output4_SAM_Comparison.png');
fprintf('Output 4 saved.\n');
fprintf('\n All 4 outputs generated successfully!\n');
