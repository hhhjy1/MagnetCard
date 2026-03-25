% =========================================================================
% Physiological Metric Validation: Heart Rate and R-R Interval PCC
% 
% This script evaluates the clinical accuracy of the Magneto-Mechanical 
% bio-interface by comparing its extracted R-R intervals and Heart Rate (HR) 
% against a gold-standard ECG reference.
% =========================================================================

clc; clear; close all;

%% ========== Configuration & Style Settings ==========
set(0,'defaultAxesFontName', 'Times New Roman');
set(0,'defaultTextFontName', 'Times New Roman');

% Paper-specific color palette
color_ecg = [0.22, 0.49, 0.72];     % Nature Blue for ECG Reference
color_bcg = [0.89, 0.10, 0.11];     % Nature Red for Mag-BCG
axis_black = [0.20, 0.20, 0.20];

% File paths (Relative paths for GitHub portability)
FILE_ECG_CSV = 'sample_ecg_reference.csv'; 
FILE_RX_BIN  = 'sample_rx_data';

%% ========================================================================
% Process 1: Load and Process Gold-Standard ECG Data
% =========================================================================
fprintf('Processing Gold-Standard ECG Data...\n');
try
    data = readtable(FILE_ECG_CSV);
    time1 = data.('Time_s_');
    voltage1 = data.('ECG_Voltage_V_');
    
    fs1_original = 1 / mean(diff(time1));
    heartbeat_high_hz = 200 / 60;
    
    % Preprocessing: Remove DC offset
    signal1_processed = voltage1 - mean(voltage1);
    
    % Peak Detection (Time-domain analysis)
    min_peak_distance_samples_1 = round(1 / heartbeat_high_hz * fs1_original);
    min_peak_height_1 = 0.3 * max(signal1_processed);
    
    [peaks1, peak_locs1] = findpeaks(signal1_processed, ...
                                   'MinPeakDistance', min_peak_distance_samples_1, ...
                                   'MinPeakHeight', min_peak_height_1);
    
    if length(peak_locs1) < 2
        warning('Insufficient ECG peaks detected for HR calculation.');
        heart_rate_bpm_1 = NaN; intervals_1 = []; peaks_idx_1 = [];
    else
        intervals_1 = diff(time1(peak_locs1));
        heart_rate_bpm_1 = 60 / mean(intervals_1);
        fprintf('-> ECG Reference Estimated HR: %.2f BPM\n', heart_rate_bpm_1);
        peaks_idx_1 = peak_locs1;
    end
    
catch ME
    error('Failed to process ECG CSV. Ensure %s is in the directory. Error: %s', FILE_ECG_CSV, ME.message);
end

%% ========================================================================
% Process 2: Load and Process Magneto-Mechanical Backscatter Data (Mag-BCG)
% =========================================================================
fprintf('Processing Magneto-Mechanical Backscatter Data...\n');
fi_2 = fopen(FILE_RX_BIN,'rb');
if fi_2 == -1
    error('Cannot open binary file: %s', FILE_RX_BIN);
end
x_inter_2 = fread(fi_2, 'float32');
fclose(fi_2);

len_2 = size(x_inter_2, 1);
re_2 = x_inter_2(1:2:len_2-1);
im_2 = x_inter_2(2:2:len_2);
complex_data_2 = complex(re_2, im_2);
real_complex_data_2 = real(complex_data_2);

% Envelope Extraction (Hilbert Transform)
envelope_hilbert_2 = abs(hilbert(real_complex_data_2));

% Bandpass Filtering for Cardiac Band (0.6 - 3.0 Hz)
heartbeat_low_hz_2 = 0.6;
heartbeat_high_hz_wide_2 = 3.0;
sample_rate_2 = 0.5e6;
target_fs_2 = 200;
ds_factor_2 = round(sample_rate_2 / target_fs_2);
fs_ds_2 = sample_rate_2 / ds_factor_2;

y3_2 = envelope_hilbert_2(:) - median(envelope_hilbert_2(:));
y3_ds_2 = resample(y3_2, 1, ds_factor_2);

[b_2, a_2] = butter(4, [heartbeat_low_hz_2 heartbeat_high_hz_wide_2] / (fs_ds_2/2), 'bandpass');
y3_bp8_2 = filtfilt(b_2, a_2, y3_ds_2);

% Signal Inversion & Smoothing (Savitzky-Golay)
final_processed_signal_2 = -y3_bp8_2;
frame_len_2 = 2 * floor(0.11 * fs_ds_2 / 2) + 1;
final_processed_signal_2 = sgolayfilt(final_processed_signal_2, 3, frame_len_2);

% Peak Detection (Adaptive Thresholding)
min_period_sec = 60 / 200; % Max 200 BPM
min_peak_distance_samples = round(min_period_sec * fs_ds_2);
min_peak_height_2 = 0.23 * max(final_processed_signal_2);

[~, locs] = findpeaks(final_processed_signal_2, ...
                      'MinPeakDistance', min_peak_distance_samples, ...
                      'MinPeakHeight', min_peak_height_2);
                                 
if length(locs) > 1
    % Median Absolute Deviation (MAD) Filtering for valid RR intervals
    intervals_2_temp = diff(locs) / fs_ds_2;
    median_rr = median(intervals_2_temp);
    mad_rr = mad(intervals_2_temp, 1);
    
    min_rr_interval = median_rr - 10.5 * mad_rr;
    max_rr_interval = median_rr + 10.5 * mad_rr;
    valid_intervals_idx = find((intervals_2_temp >= min_rr_interval) & (intervals_2_temp <= max_rr_interval));
    
    if isempty(valid_intervals_idx)
        warning('No valid heart rate intervals found after MAD filtering.');
        heart_rate_bpm_2 = NaN; intervals_2 = [];
    else
        effective_locs = unique(locs([valid_intervals_idx; valid_intervals_idx + 1]));
        if length(effective_locs) < 2
             heart_rate_bpm_2 = NaN; intervals_2 = [];
        else
            intervals_2 = diff(effective_locs) / fs_ds_2;
            heart_rate_bpm_2 = 60 / mean(intervals_2);
            fprintf('-> Mag-BCG Estimated HR: %.2f BPM\n', heart_rate_bpm_2);
            locs = effective_locs; % Update for visualization
        end
    end
else
    heart_rate_bpm_2 = NaN; intervals_2 = [];
    warning('Insufficient peaks detected in Mag-BCG signal.');
end

%% ========================================================================
% Process 3: Evaluation Metrics & Visualization
% =========================================================================
fprintf('\n================ EVALUATION SUMMARY ================\n');
if isnan(heart_rate_bpm_1) || isnan(heart_rate_bpm_2)
    fprintf('Insufficient data to compute HR comparison.\n');
else
    absolute_error = abs(heart_rate_bpm_1 - heart_rate_bpm_2);
    accuracy = (1 - (absolute_error / heart_rate_bpm_1)) * 100;
    fprintf('Absolute HR Error: %.2f BPM\n', absolute_error);
    fprintf('HR Estimation Accuracy: %.2f %%\n', accuracy);
end

if ~isempty(intervals_1) && ~isempty(intervals_2)
    % Truncate sequences to match length for PCC calculation
    min_len_rr = min(length(intervals_1), length(intervals_2));
    rr1_final = intervals_1(1:min_len_rr);
    rr2_final = intervals_2(1:min_len_rr);
    
    % Pearson Correlation Coefficient (PCC)
    R_rr = corrcoef(rr1_final, rr2_final);
    ultimate_best_pcc = R_rr(1,2);
    fprintf('R-R Interval PCC: %.4f\n', ultimate_best_pcc);
    fprintf('====================================================\n');
    
    % --- High-Quality Visualization ---
    figure('Name', 'System Evaluation: R-R Interval PCC', 'Color', 'w', 'Position', [100, 100, 1000, 700]);
    
    % Subplot 1: R-R Interval Tracking Over Time
    subplot(3, 2, 1);
    t_rr = (1:min_len_rr);
    plot(t_rr, rr1_final, '-o', 'Color', color_ecg, 'MarkerFaceColor', color_ecg, 'MarkerSize', 4, 'DisplayName', 'ECG Reference', 'LineWidth', 1.5);
    hold on;
    plot(t_rr, rr2_final, '-s', 'Color', color_bcg, 'MarkerFaceColor', color_bcg, 'MarkerSize', 4, 'DisplayName', 'Mag-BCG', 'LineWidth', 1.5);
    title(sprintf('R-R Interval Sequence (PCC = %.4f)', ultimate_best_pcc), 'FontWeight', 'normal');
    legend('Location', 'best', 'Box', 'off');
    xlabel('Beat Index'); ylabel('R-R Interval (s)');
    set(gca, 'Box', 'on', 'LineWidth', 1, 'XColor', axis_black, 'YColor', axis_black);
    grid on;
    
    % Subplot 2: R-R Interval Correlation Scatter
    subplot(3, 2, 2);
    scatter(rr1_final, rr2_final, 30, color_bcg, 'filled', 'MarkerEdgeColor', 'w');
    hold on;
    % Draw ideal y=x reference line
    min_val = min([rr1_final; rr2_final]); max_val = max([rr1_final; rr2_final]);
    plot([min_val, max_val], [min_val, max_val], 'k--', 'LineWidth', 1);
    xlabel('ECG R-R Interval (s)'); ylabel('Mag-BCG R-R Interval (s)');
    title('Beat-to-Beat Correlation', 'FontWeight', 'normal');
    set(gca, 'Box', 'on', 'LineWidth', 1, 'XColor', axis_black, 'YColor', axis_black);
    grid on;
    
    % Subplot 3: ECG Raw Peaks
    subplot(3, 2, 3);
    plot(time1, signal1_processed, 'Color', color_ecg, 'LineWidth', 1); hold on;
    plot(time1(peaks_idx_1), signal1_processed(peaks_idx_1), 'v', 'MarkerFaceColor', color_ecg, 'MarkerEdgeColor', 'k');
    title('Gold-Standard ECG Peak Detection', 'FontWeight', 'normal');
    xlabel('Time (s)'); ylabel('Amplitude (V)');
    set(gca, 'Box', 'on', 'LineWidth', 1);
    
    % Subplot 4: Mag-BCG Processed Peaks
    subplot(3, 2, 4);
    t2 = (0:length(final_processed_signal_2)-1) / fs_ds_2;
    plot(t2, final_processed_signal_2, 'Color', color_bcg, 'LineWidth', 1); hold on;
    plot(t2(locs), final_processed_signal_2(locs), 'v', 'MarkerFaceColor', color_bcg, 'MarkerEdgeColor', 'k');
    title('Mag-BCG Peak Detection', 'FontWeight', 'normal');
    xlabel('Time (s)'); ylabel('Amplitude (a.u.)');
    set(gca, 'Box', 'on', 'LineWidth', 1);
    
    % Subplot 5: Performance Dashboard
    subplot(3, 2, [5, 6]);
    axis off;
    text(0.1, 0.8, sprintf('System Performance Summary'), 'FontSize', 14, 'FontWeight', 'bold');
    text(0.1, 0.5, sprintf('Heart Rate Accuracy: %.2f%%   |   Absolute Error: %.2f BPM', accuracy, absolute_error), 'FontSize', 12);
    text(0.1, 0.2, sprintf('R-R Interval Pearson Correlation Coefficient (PCC): %.4f', ultimate_best_pcc), 'FontSize', 12);
    
    % Formatting the whole figure
    sgtitle('Validation of Magneto-Mechanical Bio-interface vs. Commercial ECG', 'FontSize', 16, 'FontWeight', 'bold');
end