function features = extract_features(filename, Fs)

%% ============================================================
%  EEG FEATURE EXTRACTION
%  Features:
%  1. Standard Deviation
%  2. Peak-to-Peak Amplitude
%  3. Relative Delta Power
%  4. Relative Theta Power
%  5. Relative Alpha Power
%  6. Relative Beta Power
%  7. Relative Gamma Power
%  8. Dominant Frequency
%  9. Spectral Centroid
% 10. Spectral Entropy
%% ============================================================


%% 1. Load EEG Signal
eeg = readmatrix(filename);

% Make sure signal is a column vector
eeg = eeg(:);


%% 2. Bandpass Filtering
low_cutoff = 0.5;
high_cutoff = 40;
filter_order = 4;

[b, a] = butter(filter_order, ...
    [low_cutoff high_cutoff] / (Fs/2), ...
    'bandpass');

% Zero-phase filtering
eeg_filtered = filtfilt(b, a, eeg);


%% 3. Time-Domain Features

% Standard deviation
std_value = std(eeg_filtered);

% Peak-to-peak amplitude
peak_to_peak = max(eeg_filtered) - min(eeg_filtered);


%% 4. Power Spectral Density using Welch Method

[pxx, f] = pwelch(eeg_filtered, [], [], [], Fs);


%% 5. Select EEG Frequency Range: 0.5–40 Hz

freq_idx = (f >= 0.5) & (f <= 40);

f_band = f(freq_idx);
pxx_band = pxx(freq_idx);


%% 6. Total Power

total_power = trapz(f_band, pxx_band);


%% 7. Frequency Band Masks

% Delta: 0.5–4 Hz
delta_idx = (f_band >= 0.5) & (f_band < 4);

% Theta: 4–8 Hz
theta_idx = (f_band >= 4) & (f_band < 8);

% Alpha: 8–13 Hz
alpha_idx = (f_band >= 8) & (f_band < 13);

% Beta: 13–30 Hz
beta_idx = (f_band >= 13) & (f_band < 30);

% Gamma: 30–40 Hz
gamma_idx = (f_band >= 30) & (f_band <= 40);


%% 8. Calculate Absolute Band Powers

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


%% 9. Calculate Relative Band Powers

relative_delta = delta_power / total_power;

relative_theta = theta_power / total_power;

relative_alpha = alpha_power / total_power;

relative_beta = beta_power / total_power;

relative_gamma = gamma_power / total_power;


%% 10. Dominant Frequency

% Find frequency corresponding to maximum PSD
[~, max_idx] = max(pxx_band);

dominant_frequency = f_band(max_idx);


%% 11. Spectral Centroid

spectral_centroid = ...
    sum(f_band .* pxx_band) / sum(pxx_band);


%% 12. Spectral Entropy

% Normalize PSD
normalized_psd = pxx_band / sum(pxx_band);

% Shannon spectral entropy
spectral_entropy = ...
    -sum(normalized_psd .* log2(normalized_psd + eps));


%% 13. Final Feature Vector

features = [ ...
    std_value, ...
    peak_to_peak, ...
    relative_delta, ...
    relative_theta, ...
    relative_alpha, ...
    relative_beta, ...
    relative_gamma, ...
    dominant_frequency, ...
    spectral_centroid, ...
    spectral_entropy ...
];

end