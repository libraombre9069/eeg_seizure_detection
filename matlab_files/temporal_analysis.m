%% ============================================================
% TEMPORAL SPECTRAL ANALYSIS OF EEG
% Stage 1: Window-wise EEG band-power trajectories
% ============================================================

clear;
clc;
close all;

%% 1. Parameters

Fs = 173.61;              % Sampling frequency (Hz)

window_duration = 4;      % Window length in seconds
overlap_fraction = 0.50;  % 50% overlap

window_samples = round(window_duration * Fs);
hop_samples = round(window_samples * (1 - overlap_fraction));

%% 2. Load EEG recordings

z_file = 'non_seizure/Z001.txt';
s_file = 'seizure/S001.txt';

eeg_z = readmatrix(z_file);
eeg_s = readmatrix(s_file);

eeg_z = eeg_z(:);
eeg_s = eeg_s(:);

%% 3. Filter EEG signals

low_cutoff = 0.5;
high_cutoff = 40;
filter_order = 4;

[b,a] = butter(filter_order, ...
    [low_cutoff high_cutoff]/(Fs/2), ...
    'bandpass');

eeg_z_filtered = filtfilt(b,a,eeg_z);
eeg_s_filtered = filtfilt(b,a,eeg_s);

%% 4. Calculate temporal band-power trajectories

[z_times, z_bands] = calculate_temporal_bands( ...
    eeg_z_filtered, Fs, window_samples, hop_samples);

[s_times, s_bands] = calculate_temporal_bands( ...
    eeg_s_filtered, Fs, window_samples, hop_samples);

%% 5. Plot Z001

figure;

tiledlayout(5,1);

nexttile;
plot(z_times,z_bands(:,1),'o-','LineWidth',1.5);
ylabel('Delta');
grid on;
title('Z001 - Temporal Relative Band Power');

nexttile;
plot(z_times,z_bands(:,2),'o-','LineWidth',1.5);
ylabel('Theta');
grid on;

nexttile;
plot(z_times,z_bands(:,3),'o-','LineWidth',1.5);
ylabel('Alpha');
grid on;

nexttile;
plot(z_times,z_bands(:,4),'o-','LineWidth',1.5);
ylabel('Beta');
grid on;

nexttile;
plot(z_times,z_bands(:,5),'o-','LineWidth',1.5);
ylabel('Gamma');
xlabel('Time (seconds)');
grid on;


%% 6. Plot S001

figure;

tiledlayout(5,1);

nexttile;
plot(s_times,s_bands(:,1),'o-','LineWidth',1.5);
ylabel('Delta');
grid on;
title('S001 - Temporal Relative Band Power');

nexttile;
plot(s_times,s_bands(:,2),'o-','LineWidth',1.5);
ylabel('Theta');
grid on;

nexttile;
plot(s_times,s_bands(:,3),'o-','LineWidth',1.5);
ylabel('Alpha');
grid on;

nexttile;
plot(s_times,s_bands(:,4),'o-','LineWidth',1.5);
ylabel('Beta');
grid on;

nexttile;
plot(s_times,s_bands(:,5),'o-','LineWidth',1.5);
ylabel('Gamma');
xlabel('Time (seconds)');
grid on;


%% 7. Print results

disp(' ');
disp('Z001 temporal band powers:');

Z_table = array2table(z_bands, ...
    'VariableNames', ...
    {'Delta','Theta','Alpha','Beta','Gamma'});

Z_table.Time = z_times;

disp(Z_table);


disp(' ');
disp('S001 temporal band powers:');

S_table = array2table(s_bands, ...
    'VariableNames', ...
    {'Delta','Theta','Alpha','Beta','Gamma'});

S_table.Time = s_times;

disp(S_table);


%% ============================================================
% LOCAL FUNCTION
% ============================================================

function [times, band_powers] = calculate_temporal_bands( ...
    eeg, Fs, window_samples, hop_samples)

    N = length(eeg);

    % Number of complete windows
    num_windows = floor((N - window_samples)/hop_samples) + 1;

    times = zeros(num_windows,1);

    band_powers = zeros(num_windows,5);

    for k = 1:num_windows

        %% Window boundaries

        start_idx = (k-1)*hop_samples + 1;
        end_idx = start_idx + window_samples - 1;

        segment = eeg(start_idx:end_idx);

        %% Time corresponding to center of window

        times(k) = ((start_idx + end_idx)/2 - 1)/Fs;

        %% PSD

        [pxx,f] = pwelch(segment,[],[],[],Fs);

        %% Select 0.5-40 Hz

        freq_idx = (f >= 0.5) & (f <= 40);

        f_band = f(freq_idx);
        pxx_band = pxx(freq_idx);

        %% Total power

        total_power = trapz(f_band,pxx_band);

        %% Frequency bands

        delta_idx = (f_band >= 0.5) & (f_band < 4);
        theta_idx = (f_band >= 4) & (f_band < 8);
        alpha_idx = (f_band >= 8) & (f_band < 13);
        beta_idx  = (f_band >= 13) & (f_band < 30);
        gamma_idx = (f_band >= 30) & (f_band <= 40);

        %% Band powers

        delta_power = trapz( ...
            f_band(delta_idx), ...
            pxx_band(delta_idx));

        theta_power = trapz( ...
            f_band(theta_idx), ...
            pxx_band(theta_idx));

        alpha_power = trapz( ...
            f_band(alpha_idx), ...
            pxx_band(alpha_idx));

        beta_power = trapz( ...
            f_band(beta_idx), ...
            pxx_band(beta_idx));

        gamma_power = trapz( ...
            f_band(gamma_idx), ...
            pxx_band(gamma_idx));

        %% Relative powers

        band_powers(k,1) = delta_power / total_power;
        band_powers(k,2) = theta_power / total_power;
        band_powers(k,3) = alpha_power / total_power;
        band_powers(k,4) = beta_power  / total_power;
        band_powers(k,5) = gamma_power / total_power;

    end

end