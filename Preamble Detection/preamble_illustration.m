% =========================================================================
% Preamble Detection and Synchronization for Magneto-Mechanical Backscatter
% 
% This script demonstrates the multi-stage preamble detection pipeline 
% described in Algorithm 1 of the paper. It processes raw I/Q samples 
% to locate the synchronization sequence robustly under channel noise.
% =========================================================================

clc; clear; close all;

% -------------------------------------------------------------------------
% 0. Environment & Style Settings
% -------------------------------------------------------------------------
set(0,'defaultAxesFontName', 'Times New Roman');
set(0,'defaultTextFontName', 'Times New Roman');

% Paper-specific color palette
signal_blue = [0.392, 0.478, 0.639];   
template_red = [0.690, 0.118, 0.137];  
axis_black = [0.20, 0.20, 0.20];       
marker_subtle_color = [0.4, 0.4, 0.4]; 

% File paths (Make sure these sample files are in the same directory)
DATA_FILE = 'sample_rx_data';      % Renamed from 'source_file_sink (9)'
TEMPLATE_FILE = 'local_template.bin';  % Renamed from 'alternating_01.bin'
CACHE_FILE = 'cache_preamble.mat';     % Cleaned cache name

% -------------------------------------------------------------------------
% 1. Data Loading and Pre-processing
% -------------------------------------------------------------------------
if exist(CACHE_FILE, 'file')
    fprintf('Loading processed results from cache...\n');
    load(CACHE_FILE);
else
    fprintf('Cache not found. Processing raw data (this may take a moment)...\n');
    
    % Read raw Rx complex data
    fi_1 = fopen(DATA_FILE, 'rb');
    if fi_1 == -1
        error('Cannot open data file. Please ensure %s is in the directory.', DATA_FILE);
    end
    x_inter_1 = fread(fi_1, 'float32');
    fclose(fi_1);
    
    re = x_inter_1(1:2:end);
    im = x_inter_1(2:2:end);
    complex_data = complex(re, im);
    complex_data = complex_data(1:5.5e6); % Use the first 5.5M samples for demo
    
    % Read baseband preamble template
    fid = fopen(TEMPLATE_FILE, 'rb');
    if fid == -1
        error('Cannot open template file. Please ensure %s is in the directory.', TEMPLATE_FILE);
    end
    allData = fread(fid, 'float32');
    fclose(fid);
    
    real_part = allData(1:2:end);
    imag_part = allData(2:2:end);
    complex_signal = real_part + 1i * imag_part;
    local_signal = complex_signal(17:48); % Extract the specific active sequence
    
    % -------------------------------------------------------------------------
    % 2. Preamble Detection Pipeline (Algorithm 1)
    % -------------------------------------------------------------------------
    % Stage 1: Sliding cross-correlation for frame alignment
    [rxy, lag] = xcorr(complex_data, local_signal);
    
    lag_positive = lag(lag >= 0);
    rxy_positive = abs(rxy(lag >= 0));
    
    % Stage 2: Moving average to suppress out-of-band fluctuations
    rxy_smooth = movmean(rxy_positive, 50000); 
    
    % Stage 3: Gain-invariant scaling (Normalization)
    rxy_norm = (rxy_smooth - min(rxy_smooth)) / (max(rxy_smooth) - min(rxy_smooth));
    
    % Stage 4: Template-based timing recovery
    % Constructing ideal envelope template representing the preamble structure
    w1 = 500000; w2 = 250000; 
    template = [zeros(1,w1), ones(1,w1), zeros(1,w1), ones(1,w2), zeros(1,w2)];
    template = (template - mean(template)) / std(template); 
    template = 0.5 * template; 
    
    rxy_norm2 = (rxy_norm - mean(rxy_norm)) / std(rxy_norm); 
    corr_result = conv(rxy_norm2, fliplr(template), 'valid');
    
    % Locate the synchronization peak
    [~, idx] = max(corr_result);
    preamble_end_pos = lag_positive(idx + length(template) - 1);
    
    save(CACHE_FILE, 'complex_data', 'lag_positive', 'rxy_norm', 'rxy_norm2', 'idx', 'template', 'preamble_end_pos');
end

fprintf('Synchronization successful. Preamble end index: %d\n', preamble_end_pos);

% -------------------------------------------------------------------------
% 3. Visualization
% -------------------------------------------------------------------------
figure('Color', 'w', 'Position', [100, 100, 900, 600]);

% Subplot 1: Raw Received Signal
subplot(2,1,1);
plot(real(complex_data), 'LineWidth', 0.8, 'Color', signal_blue); 
set(gca, 'Box', 'on', 'LineWidth', 1.2, 'FontSize', 14);
set(gca, 'XColor', axis_black, 'YColor', axis_black, 'Color', [0.98, 0.98, 0.98]);
xlim([0, 4e6]);
ylabel('Amplitude (a.u.)', 'FontWeight', 'bold');
title('Raw Received I/Q Signal (Real Part)', 'FontWeight', 'normal', 'FontSize', 15);

% Subplot 2: Correlation and Template Matching
subplot(2,1,2);
new_linewidth = 2.5; 

% Plot normalized correlation envelope
plot(lag_positive, rxy_norm - 0.4, 'Color', signal_blue, 'LineWidth', new_linewidth); 
hold on;

% Add synchronization markers
xline(lag_positive(idx), '--', 'Color', marker_subtle_color, 'LineWidth', new_linewidth, 'Alpha', 0.9);
xline(preamble_end_pos, '--', 'Color', marker_subtle_color, 'LineWidth', new_linewidth);

% Overlay ideal template logic
template_start_lag = lag_positive(idx);
template_end_lag = lag_positive(idx + length(template) - 1);
x_template = linspace(template_start_lag, template_end_lag, length(template));
plot(x_template, template, 'Color', template_red, 'LineWidth', new_linewidth);

set(gca, 'Box', 'on', 'LineWidth', 1.2, 'FontSize', 14);
set(gca, 'XColor', axis_black, 'YColor', axis_black, 'Color', [0.98, 0.98, 0.98]);
xlim([0, 4e6]);
ylim([-1.5, 1.5]); 
xlabel('Sample Index', 'FontWeight', 'bold');
ylabel('Correlation Level', 'FontWeight', 'bold');
title('Preamble Synchronization via Template Matching', 'FontWeight', 'normal', 'FontSize', 15);

% Adjust layout for aesthetics
set(gcf, 'PaperPositionMode', 'auto');