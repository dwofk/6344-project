clc; clear; close all;

save_path = fullfile(cd, 'results');
if ~exist(save_path, 'dir')
    mkdir(save_path);
end

input_filename = 'cabin.jpg';
highlights = 0; % set to 1 if input image contains highlights

exp_bracket_dir = fullfile(cd, 'exp_bracket_png');
exp_bracket_dir_dn = fullfile(cd, 'exp_bracket_png_dn');

fprintf('Input image: %s\n', fullfile(cd, 'data', input_filename));

%% Exposure Bracket Generation via Histogram Separation

fprintf('\n(1) Generating exposure bracket for input image...\n');
histSeparate(input_filename, 'PNG', exp_bracket_dir);

%% Denoising with Guided Filter

fprintf('\n(2) Denoising images in exposure stack...\n');
denoiseGuided(input_filename, exp_bracket_dir);

%% Exposure Fusion (Mertens)

fprintf('\n(3) Fusing images in exposure stack...\n');
fuseExposures(input_filename, exp_bracket_dir, save_path);

%% HDR Generation using MATLAB's makeHDR Function

fprintf('\n(4) Using the built-in makehdr function...\n'); 
useMakeHDR(input_filename, exp_bracket_dir_dn, save_path);

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