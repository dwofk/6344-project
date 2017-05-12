clc; clear; close all;

save_path = fullfile(cd, 'results');
if ~exist(save_path, 'dir')
    mkdir(save_path);
end

input_filename = 'ChineseGarden2.jpg';
highlights = 0; % set to 1 if input image contains highlights

% MATLAB-based exposure bracket
exp_bracket_dir = fullfile(cd, 'exp_bracket_png');
exp_bracket_dir_dn = fullfile(cd, 'exp_bracket_png_dn');

% CNN-based exposure bracket
cnn_exp_bracket_dir = fullfile(cd, 'cnn_exp_bracket');

fprintf('Input image: %s\n', fullfile(cd, 'data', input_filename));

%% Exposure Bracket Generation via Histogram Separation

num_bins = 2; % number of exposure brackets

fprintf('\n(1) Generating exposure bracket for input image...\n');
histSeparate(input_filename, num_bins, 'PNG', exp_bracket_dir);

%% Denoising with Guided Filter

fprintf('\n(2) Denoising images in exposure stack...\n');
denoiseGuided(input_filename, exp_bracket_dir);

%% Exposure Fusion (Mertens)

fprintf('\n(3) Fusing images in exposure stack...\n');
fused_img = fuseExposures(input_filename, exp_bracket_dir, 0);
imwrite(fused_img, [save_path, '/', 'fused_image.png']);

%% HDR Generation using MATLAB's makeHDR Function

fprintf('\n(4) Using the built-in makehdr function...\n'); 
hdr_img = useMakeHDR(input_filename, exp_bracket_dir_dn, 1);
imwrite(hdr_img, [save_path, '/', 'hdr_image.png']);

%% Highlight Removal and Dynamic Range Expansion

if (highlights > 0)
    
    fprintf('\n(5) Removing highlights and expanding dynamic range...\n'); 
    hl_img = imread(fullfile(cd, 'data', input_filename));

    % dehighlighted image
    dehl_img = rmHighlights(double(hl_img));
    % expanded dynamic range
    edr_img = expDynamicRange(dehl_img);
    
    figure('Name','EDR Image'); imshow(edr_img);
    imwrite(edr_img, [save_path, '/', 'edr_image.png']);
    
end

%% Get Output Images from CNN-Based Techniques

%% Calculate Metrics