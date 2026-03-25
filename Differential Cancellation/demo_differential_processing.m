% =========================================================================
% Differential Interference Cancellation and Physiological Signal Extraction
% 
% This script demultiplexes the time-interleaved backscatter signals, 
% performs adaptive common-mode subtraction, and applies bandpass filtering 
% to extract the clean physiological waveform (e.g., BCG).
% =========================================================================

clc; clear; close all;

%% ========== Color Palette (Nature/Science Style) ==========
color_raw = [0.2, 0.3, 0.5];            
color_nature_blue = [0.22, 0.49, 0.72]; 
color_nature_red = [0.89, 0.10, 0.11];  
color_nature_green = [0.30, 0.69, 0.29];
color_nature_purple = [0.48, 0.20, 0.47];

%% ========== Cache System Configuration ==========
cache_dir = './cache_data/';
if ~exist(cache_dir, 'dir'), mkdir(cache_dir); end
cache_file_main = fullfile(cache_dir, 'main_processing_cache.mat');
cache_file_filtered = fullfile(cache_dir, 'filtered_signals_cache.mat');
use_cache = true; force_recalculate = false; 

% File paths (Must match the dataset provided in the repository)
DATA_FILE = 'sample_rx_data';      % Renamed from 'source_file_sink'
TEMPLATE_FILE = 'local_template.bin';  % Renamed from 'alternating_01.bin'

%% ========== Data Loading and Processing ==========
if use_cache && exist(cache_file_main, 'file') && ~force_recalculate
    fprintf('Loading main processing results from cache...\n');
    load(cache_file_main);
else
    fprintf('Processing raw data (this may take a moment)...\n');
    
    % 1. Load Raw I/Q Data
    try
        fi_1 = fopen(DATA_FILE, 'rb');
        if fi_1 == -1, error('Cannot open %s', DATA_FILE); end
        x_inter_1 = fread(fi_1, 'float32'); fclose(fi_1);
    catch
        error('Missing data file: %s', DATA_FILE);
    end
    len = size(x_inter_1,1);
    complex_data = complex(x_inter_1(1:2:len-1), x_inter_1(2:2:len));
    
    % 2. Preamble Synchronization (To verify alignment)
    try
        fid = fopen(TEMPLATE_FILE, 'rb');
        if fid == -1, error('Cannot open %s', TEMPLATE_FILE); end
        allData = fread(fid, 'float32'); fclose(fid);
    catch
        error('Missing template file: %s', TEMPLATE_FILE);
    end
    complex_signal = allData(1:2:end) + 1i * allData(2:2:end);
    local_signal = complex_signal(17:48);
    [rxy, lag] = xcorr(complex_data, local_signal);
    
    % 3. System Parameters
    sample_rate = 1e6; 
    freq = 499.983; % Effective hardware switching frequency (Hz)
    segment_len = sample_rate / (2 * freq); 
    
    % Synchronization index identified from the preamble detection script
    start_idx = 6714910; 
    
    % 4. Demultiplexing Time-Interleaved Signals (Reference vs. Measurement)
    signal1 = []; signal2 = []; idx = start_idx; state = 1;
    real_complex_data = abs(complex_data);
    while idx + segment_len <= length(real_complex_data)
        full_segment = real_complex_data(idx:floor(idx+segment_len-1));
        if state == 1
            signal1 = [signal1; full_segment];
        else
            signal2 = [signal2; full_segment]; 
        end
        idx = idx + segment_len; 
        state = 3 - state; % Toggle between state 1 and 2
    end
    
    % 5. Envelope Extraction and Subtraction
    envelope1_raw = abs(signal1); 
    envelope2_raw = abs(signal2);
    window_size = floor(segment_len); 
    
    % Smoothing envelopes to remove high-frequency switching transients
    envelope1_local_avg = movmean(envelope1_raw, window_size);
    envelope2_local_avg = movmean(envelope2_raw, window_size);
    [b, a] = butter(4, 0.01, 'low');
    envelope1_local_avg = filtfilt(b, a, envelope1_local_avg);
    envelope2_local_avg = filtfilt(b, a, envelope2_local_avg);
    
    min_len = min(length(envelope1_local_avg), length(envelope2_local_avg));
    
    % Differential Subtraction (Common-Mode Rejection)
    diff_local_avg = envelope2_local_avg(1:min_len) - envelope1_local_avg(1:min_len);
    
    % RMS-based alternative calculation
    rms_window = 150;
    envelope1_rms = movmean(envelope1_raw.^2, rms_window).^0.5;
    envelope2_rms = movmean(envelope2_raw.^2, rms_window).^0.5;
    [b_rms, a_rms] = butter(4, 0.001, 'low');
    envelope1_rms = filtfilt(b_rms, a_rms, envelope1_rms);
    envelope2_rms = filtfilt(b_rms, a_rms, envelope2_rms);
    diff_rms = envelope2_rms(1:min_len) - envelope1_rms(1:min_len);
    
    save(cache_file_main, 'complex_data', 'signal1', 'signal2', 'envelope1_raw', 'envelope2_raw', 'envelope1_local_avg', 'envelope2_local_avg', 'diff_local_avg', 'diff_rms', 'real_complex_data', 'sample_rate', 'start_idx', 'min_len');
end

%% ========== Physiological Filtering ==========
if use_cache && exist(cache_file_filtered, 'file') && ~force_recalculate
    fprintf('Loading filtered results from cache...\n');
    load(cache_file_filtered);
else
    fprintf('Applying physiological bandpass filters...\n');
    % Downsample to 200 Hz to isolate physiological band and reduce overhead
    ds_factor = round(1e6 / 200); 
    fs_ds = 1e6 / ds_factor;
    
    y1 = diff_local_avg(:) - median(diff_local_avg);
    y1_ds = resample(y1, 1, ds_factor);
    
    % 0.6 - 20 Hz Bandpass Filter (Heart rate / BCG band)
    [b8, a8] = butter(4, [0.6 20.0]/(fs_ds/2), 'bandpass');
    y1_bp8 = filtfilt(b8, a8, y1_ds);
    
    % Savitzky-Golay filtering for smooth waveform reconstruction
    frame_len = 2*floor(0.11*fs_ds/2)+1; 
    y1_bp8 = sgolayfilt(y1_bp8, 3, frame_len);
    t8 = (0:numel(y1_bp8)-1)/fs_ds;
    
    save(cache_file_filtered, 'y1_bp8', 't8');
end

%% ========== Plotting Section 1: Comprehensive Pipeline (5 Subplots) ==========
fprintf('Generating comprehensive pipeline visualization...\n');

% +++++++ Axis Limit Configuration +++++++
ylim_1 = [-0.2, 3.3];   % Subplot 1 (Raw signal)
ylim_2 = [-0.3, 1.5];   % Subplot 2 (Reference)
ylim_3 = [-0.3, 1.5];   % Subplot 3 (Measured)
ylim_4 = [-0.8, 1.3];   % Subplot 4 (Differential)
ylim_5 = [-0.45, 1];    % Subplot 5 (Filtered)
% ++++++++++++++++++++++++++++++++++++++++

plot_font_size = 14; 
left_margin = 0.08; right_margin = 0.02;
bottom_margin = 0.1; top_margin = 0.05;    
plot_width = 1 - left_margin - right_margin;
total_height = 1 - bottom_margin - top_margin;
num_subplots = 5;
h_unit = total_height / num_subplots; 
border_width = 0.8; 

figure('Color', 'w', 'Position', [100, 100, 500, 300]); 
real_complex_data_display = real_complex_data(start_idx:end);
t_base = (0:length(real_complex_data_display)-1) / sample_rate; 
total_duration = t_base(end); 
x_view_limit = 8.53; 
t_envelope_diff = linspace(0, total_duration, length(envelope1_local_avg)); 
t_filtered = linspace(0, total_duration, length(y1_bp8));

% --- Subplot 1: Raw signal ---
ax1 = axes('Position', [left_margin, bottom_margin + 4*h_unit, plot_width, h_unit]);
p1 = plot(t_base, real_complex_data_display, 'Color', color_raw, 'LineWidth', 0.05); 
lgd1 = legend(p1, {'Raw signal'}, 'Location', 'NorthWest'); 
set(lgd1, 'Box', 'on', 'Color', 'w', 'EdgeColor', 'k', 'FontSize', 10);
ylabel('Amplitude', 'FontSize', plot_font_size);
set(gca, 'FontSize', plot_font_size, 'TickDir', 'in', 'LineWidth', border_width);
set(gca, 'XTickLabel', [], 'YAxisLocation', 'left', 'YTick', [0, 3]); 
xlim([0, x_view_limit]); ylim(ylim_1); 
box off; ax1.XColor = 'none'; ax1.YColor = 'k'; ax1.XAxis.Visible = 'off'; 
line(xlim, [ylim_1(2) ylim_1(2)], 'Color', 'k', 'LineWidth', border_width, 'HandleVisibility', 'off'); 
line([0 0], ylim, 'Color', 'k', 'LineWidth', border_width, 'HandleVisibility', 'off'); 
line([x_view_limit x_view_limit], ylim, 'Color', 'k', 'LineWidth', border_width, 'HandleVisibility', 'off');

% --- Subplot 2: Reference signal ---
ax2 = axes('Position', [left_margin, bottom_margin + 3*h_unit, plot_width, h_unit]);
p2 = plot(t_envelope_diff, 1-envelope2_local_avg, 'Color', color_nature_blue, 'LineWidth', 1.5); 
lgd2 = legend(p2, {'Reference signal'}, 'Location', 'NorthWest');
set(lgd2, 'Box', 'on', 'Color', 'w', 'EdgeColor', 'k', 'FontSize', 10);
ylabel('Amplitude', 'FontSize', plot_font_size); 
set(gca, 'FontSize', plot_font_size, 'TickDir', 'in', 'LineWidth', border_width);
set(gca, 'XTickLabel', [], 'YAxisLocation', 'left', 'YTick', [0, 1]);
xlim([0, x_view_limit]); ylim(ylim_2); 
box off; ax2.XColor = 'none'; ax2.YColor = 'k'; ax2.XAxis.Visible = 'off';
line(xlim, [ylim_2(2) ylim_2(2)], 'Color', 'k', 'LineWidth', border_width, 'HandleVisibility', 'off'); 
line([0 0], ylim, 'Color', 'k', 'LineWidth', border_width, 'HandleVisibility', 'off'); 
line([x_view_limit x_view_limit], ylim, 'Color', 'k', 'LineWidth', border_width, 'HandleVisibility', 'off');

% --- Subplot 3: Measured signal ---
ax3 = axes('Position', [left_margin, bottom_margin + 2*h_unit, plot_width, h_unit]);
p3 = plot(t_envelope_diff, 1-envelope1_local_avg, 'Color', color_nature_red, 'LineWidth', 1.5);
lgd3 = legend(p3, {'Measured signal'}, 'Location', 'NorthWest');
set(lgd3, 'Box', 'on', 'Color', 'w', 'EdgeColor', 'k', 'FontSize', 10);
ylabel('Amplitude', 'FontSize', plot_font_size);
set(gca, 'FontSize', plot_font_size, 'TickDir', 'in', 'LineWidth', border_width);
set(gca, 'XTickLabel', [], 'YAxisLocation', 'left', 'YTick', [0, 1]);
xlim([0, x_view_limit]); ylim(ylim_3); 
box off; ax3.XColor = 'none'; ax3.YColor = 'k'; ax3.XAxis.Visible = 'off';
line(xlim, [ylim_3(2) ylim_3(2)], 'Color', 'k', 'LineWidth', border_width, 'HandleVisibility', 'off'); 
line([0 0], ylim, 'Color', 'k', 'LineWidth', border_width, 'HandleVisibility', 'off'); 
line([x_view_limit x_view_limit], ylim, 'Color', 'k', 'LineWidth', border_width, 'HandleVisibility', 'off');

% --- Subplot 4: Differential signal ---
ax4 = axes('Position', [left_margin, bottom_margin + h_unit, plot_width, h_unit]);
p4 = plot(t_envelope_diff, diff_local_avg, 'Color', color_nature_green, 'LineWidth', 1);
lgd4 = legend(p4, {'Differential signal'}, 'Location', 'NorthWest');
set(lgd4, 'Box', 'on', 'Color', 'w', 'EdgeColor', 'k', 'FontSize', 10);
ylabel('Amplitude', 'FontSize', plot_font_size);
set(gca, 'FontSize', plot_font_size, 'TickDir', 'in', 'LineWidth', border_width);
set(gca, 'XTickLabel', [], 'YAxisLocation', 'left', 'YTick', [-0.3, 0.7]);
xlim([0, x_view_limit]); ylim(ylim_4); 
box off; ax4.XColor = 'none'; ax4.YColor = 'k'; ax4.XAxis.Visible = 'off';
line(xlim, [ylim_4(2) ylim_4(2)], 'Color', 'k', 'LineWidth', border_width, 'HandleVisibility', 'off'); 
line([0 0], ylim, 'Color', 'k', 'LineWidth', border_width, 'HandleVisibility', 'off'); 
line([x_view_limit x_view_limit], ylim, 'Color', 'k', 'LineWidth', border_width, 'HandleVisibility', 'off');

% --- Subplot 5: Filtered signal ---
ax5 = axes('Position', [left_margin, bottom_margin, plot_width, h_unit]);
p5 = plot(t_filtered, y1_bp8, 'Color', color_nature_purple, 'LineWidth', 1.5);
lgd5 = legend(p5, {'Filtered signal'}, 'Location', 'NorthWest');
set(lgd5, 'Box', 'on', 'Color', 'w', 'EdgeColor', 'k', 'FontSize', 10);
xlabel('Time (seconds)', 'FontSize', plot_font_size);
ylabel('Amplitude', 'FontSize', plot_font_size);
set(gca, 'FontSize', plot_font_size, 'TickDir', 'in', 'LineWidth', border_width);
set(gca, 'YAxisLocation', 'left', 'YTick', [-0.3, 0.7]);
xlim([0, x_view_limit]); ylim(ylim_5); 
box off; ax5.XColor = 'k'; ax5.YColor = 'k'; 
line(xlim, [ylim_5(2) ylim_5(2)], 'Color', 'k', 'LineWidth', border_width, 'HandleVisibility', 'off'); 
line([0 0], ylim, 'Color', 'k', 'LineWidth', border_width, 'HandleVisibility', 'off'); 
line([x_view_limit x_view_limit], ylim, 'Color', 'k', 'LineWidth', border_width, 'HandleVisibility', 'off');
line(xlim, [ylim_5(1) ylim_5(1)], 'Color', 'k', 'LineWidth', border_width, 'HandleVisibility', 'off');

linkaxes([ax1, ax2, ax3, ax4, ax5], 'x');

%% ========== Plotting Section 2: Zoomed-in Segment (Figure 2) ==========
fprintf('Generating zoomed-in view (6.849s - 6.8535s)...\n');

% +++++++ Zoom Limits +++++++
ylim_zoom = [-0.1, 2.8]; 
% +++++++++++++++++++++++++++
zoom_start_time = 6.8492; zoom_end_time = 6.8532;
idx_start = find(t_base >= zoom_start_time, 1);
idx_end = find(t_base <= zoom_end_time, 1, 'last');

if ~isempty(idx_start) && ~isempty(idx_end)
    t_zoom = t_base(idx_start:idx_end);
    data_zoom = real_complex_data_display(idx_start:idx_end);
    
    figure('Color', 'w', 'Position', [150, 150, 500, 120]); 
    
    p_zoom = plot(t_zoom, data_zoom, 'Color', color_raw, 'LineWidth', 1.5); 
    
    lgd_zoom = legend(p_zoom, {'Raw signal (Zoomed)'}, 'Location', 'NorthWest');
    set(lgd_zoom, 'Box', 'on', 'Color', 'w', 'EdgeColor', 'k');
    title(sprintf('Zoomed Raw Signal (%.4fs - %.4fs)', zoom_start_time, zoom_end_time), ...
          'FontSize', 16, 'FontWeight', 'normal', 'FontName', 'Arial');
    xlabel('Time (seconds)', 'FontSize', 14);
    ylabel('Amplitude', 'FontSize', 14);
    set(gca, 'FontSize', 14, 'TickDir', 'in', 'LineWidth', border_width);
    
    xlim([zoom_start_time, zoom_end_time]); 
    ylim(ylim_zoom); 
    
    box off; 
    line(xlim, [ylim_zoom(2) ylim_zoom(2)], 'Color', 'k', 'LineWidth', border_width, 'HandleVisibility', 'off'); 
    line(xlim, [ylim_zoom(1) ylim_zoom(1)], 'Color', 'k', 'LineWidth', border_width, 'HandleVisibility', 'off'); 
    line([zoom_start_time zoom_start_time], ylim, 'Color', 'k', 'LineWidth', border_width, 'HandleVisibility', 'off'); 
    line([zoom_end_time zoom_end_time], ylim, 'Color', 'k', 'LineWidth', border_width, 'HandleVisibility', 'off'); 
end