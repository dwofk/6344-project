warning('off','all')
warning

% Metric calculations for:
% 
% --- MATLAB-based approach ---
% (1) MATLAB exposure bracketing (2 brackets)
% (2) exposure fusion from MATLAB outputs
% (3) makeHDR results from MATLAB outputs
% 
% --- CNN-based approach ---
% (4) CNN exposure bracketing (high/low exposure)
% (5) exposure fusion from CNN outputs
% (6) makeHDR results from CNN outputs
% 
% --- CNN HDR generation ---
% (7) CNN-generated tonemapped HDR image

%% Get Images

% original input
input_img = imread('data/ChineseGarden2.jpg');

% reference exposure images
plus_exp_img = imread('data/ChineseGarden1.jpg');
minus_exp_img = imread('data/ChineseGarden3.jpg');

% reference tonemapped HDR image
ref_hdr = imread('data/ChineseGarden.jpg');

% denoised exposure bracket images (MATLAB-generated)
diffmaps_dir = fullfile(cd, 'diff_maps');
exp1_plus_img = imread('exp_bracket_png_dn/exp1.png');
exp2_minus_img = imread('exp_bracket_png_dn/exp2.png');

% exposure fusion result from MATLAB approach
fused_img = imread('results/fused_image.png');

% makeHDR result from MATLAB approach
mkhdr_img = imread('results/hdr_image.png');

% exposure-based CNN results
exp_dir = fullfile('..', 'exposure_cnn', 'vysarge_results', 'ChineseGarden');
cnn_exp_diffmaps_dir = fullfile('..', 'exposure_cnn', 'vysarge_results', 'ChineseGarden', 'diff_maps');
minus_1x1_img = imread(fullfile(exp_dir, 'net_outputs', 'minus_1x1.png'));
plus_1x1_img = imread(fullfile(exp_dir, 'net_outputs', 'plus_1x1.png'));
minus_3x3_img = imread(fullfile(exp_dir, 'net_outputs', 'minus_3x3.png'));
plus_3x3_img = imread(fullfile(exp_dir, 'net_outputs', 'plus_3x3.png'));

% exposure fusion results from CNN outputs
fused_1x1_img = imread(fullfile(exp_dir, 'expfusion', '1x1.png'));
fused_3x3_img = imread(fullfile(exp_dir, 'expfusion', '3x3.png'));

% makeHDR results from CNN outputs
mkhdr_1x1_img = imread(fullfile(exp_dir, 'makehdr', '1x1.png'));
mkhdr_3x3_img = imread(fullfile(exp_dir, 'makehdr', '3x3.png'));

% CNN-generated tonemapped HDR image
cnn_exp_dir = fullfile('..', 'tonemapped_cnn', 'twolayernobnorm_10e-4_results');
cnn_diffmaps_dir = fullfile('..', 'tonemapped_cnn', 'diff_maps');
cnn_hdr_img = imread(fullfile(cnn_exp_dir, 'chinese_garden2.png'));

%% Well-Exposedness (Mertens)

% input and reference images
input_img_we = well_exposedness(mat2gray(input_img));
input_img_we_str = ['(' num2str(mean(input_img_we(:))) ', ' num2str(var(input_img_we(:))) ')'];
plus_exp_img_we = well_exposedness(mat2gray(plus_exp_img));
plus_exp_img_we_str = ['(' num2str(mean(plus_exp_img_we(:))) ', ' num2str(var(plus_exp_img_we(:))) ')'];
minus_exp_img_we = well_exposedness(mat2gray(minus_exp_img));
minus_exp_img_we_str = ['(' num2str(mean(minus_exp_img_we(:))) ', ' num2str(var(minus_exp_img_we(:))) ')'];
ref_hdr_we = well_exposedness(mat2gray(ref_hdr));
ref_hdr_we_str = ['(' num2str(mean(ref_hdr_we(:))) ', ' num2str(var(ref_hdr_we(:))) ')'];

fprintf('\nWell-exposedness of reference images (mean, variance)\n');
disp(['Input image: = ' input_img_we_str]);
disp(['Under-exposed reference: = ' minus_exp_img_we_str]);
disp(['Over-exposed reference: = ' plus_exp_img_we_str]);
disp(['Tonemapped HDR reference: = ' ref_hdr_we_str]);

% denoised exposure bracket images (MATLAB-generated)
exp1_plus_img_we = well_exposedness(mat2gray(exp1_plus_img));
exp1_plus_img_we_str = ['(' num2str(mean(exp1_plus_img_we(:))) ', ' num2str(var(exp1_plus_img_we(:))) ')'];
exp2_minus_img_we = well_exposedness(mat2gray(exp2_minus_img));
exp2_minus_img_we_str = ['(' num2str(mean(exp2_minus_img_we(:))) ', ' num2str(var(exp2_minus_img_we(:))) ')'];

% exposure fusion result from MATLAB approach
fused_img_we = well_exposedness(mat2gray(fused_img));
fused_img_we_str = ['(' num2str(mean(fused_img_we(:))) ', ' num2str(var(fused_img_we(:))) ')'];

% makeHDR result from MATLAB approach
mkhdr_img_we = well_exposedness(mat2gray(mkhdr_img));
mkhdr_img_we_str = ['(' num2str(mean(mkhdr_img_we(:))) ', ' num2str(var(mkhdr_img_we(:))) ')'];

fprintf('\nWell-exposedness of MATLAB-based results (mean, variance)\n');
disp(['Estimated under-exposed: = ' exp2_minus_img_we_str]);
disp(['Estimated over-exposed: = ' exp1_plus_img_we_str]);
disp(['Exposure fusion outputs: = ' fused_img_we_str]);
disp(['makeHDR function outputs: = ' mkhdr_img_we_str]);

% exposure-based CNN results
minus_1x1_we = well_exposedness(mat2gray(minus_1x1_img));
minus_1x1_we_str = ['(' num2str(mean(minus_1x1_we(:))) ', ' num2str(var(minus_1x1_we(:))) ')'];
minus_3x3_we = well_exposedness(mat2gray(minus_3x3_img));
minus_3x3_we_str = ['(' num2str(mean(minus_3x3_we(:))) ', ' num2str(var(minus_3x3_we(:))) ')'];
plus_1x1_we = well_exposedness(mat2gray(plus_1x1_img));
plus_1x1_we_str = ['(' num2str(mean(plus_1x1_we(:))) ', ' num2str(var(plus_1x1_we(:))) ')'];
plus_3x3_we = well_exposedness(mat2gray(plus_3x3_img));
plus_3x3_we_str = ['(' num2str(mean(plus_3x3_we(:))) ', ' num2str(var(plus_3x3_we(:))) ')'];

% exposure fusion results from CNN outputs
fused_1x1_we = well_exposedness(mat2gray(fused_1x1_img));
fused_1x1_we_str = ['(' num2str(mean(fused_1x1_we(:))) ', ' num2str(var(fused_1x1_we(:))) ')'];
fused_3x3_we = well_exposedness(mat2gray(fused_3x3_img));
fused_3x3_we_str = ['(' num2str(mean(fused_3x3_we(:))) ', ' num2str(var(fused_3x3_we(:))) ')'];

% makeHDR results from CNN outputs
mkhdr_1x1_we = well_exposedness(mat2gray(mkhdr_1x1_img));
mkhdr_1x1_we_str = ['(' num2str(mean(mkhdr_1x1_we(:))) ', ' num2str(var(mkhdr_1x1_we(:))) ')'];
mkhdr_3x3_we = well_exposedness(mat2gray(mkhdr_3x3_img));
mkhdr_3x3_we_str = ['(' num2str(mean(mkhdr_3x3_we(:))) ', ' num2str(var(mkhdr_3x3_we(:))) ')'];

fprintf('\nWell-exposedness of CNN-based results (mean, variance)\n');
disp(['Estimated under-exposed: 1x1 = ' minus_1x1_we_str '; 3x3 = ' minus_3x3_we_str]);
disp(['Estimated over-exposed: 1x1 = ' plus_1x1_we_str '; 3x3 = ' plus_3x3_we_str]);
disp(['Exposure fusion outputs: 1x1 = ' fused_1x1_we_str '; 3x3 = ' fused_3x3_we_str]);
disp(['makeHDR function outputs: 1x1 = ' mkhdr_1x1_we_str '; 3x3 = ' mkhdr_3x3_we_str]);

% CNN-generated tonemapped HDR image
cnn_hdr_img_we = well_exposedness(mat2gray(cnn_hdr_img));
cnn_hdr_img_we_str = ['(' num2str(mean(cnn_hdr_img_we(:))) ', ' num2str(var(cnn_hdr_img_we(:))) ')'];

fprintf('\nWell-exposedness of CNN-generated results (mean, variance)\n');
disp(['CNN HDR tonemapped image = ' cnn_hdr_img_we_str]);


%% SSIM

input_ssim = computeSSIMColor(ref_hdr, input_img);

fprintf('\nSSIMs of reference images\n');
disp(['Input image = ' num2str(input_ssim)]);

% MATLAB-based approach
fused_ssim = computeSSIMColor(ref_hdr, fused_img);
mkhdr_ssim = computeSSIMColor(ref_hdr, mkhdr_img);

fprintf('\nSSIMs of MATLAB-based outputs\n');
disp(['Exposure fusion: = ' num2str(fused_ssim)]);
disp(['makeHDR function: = ' num2str(mkhdr_ssim)]);

% CNN-based approach
fused_1x1_ssim = computeSSIMColor(ref_hdr, fused_1x1_img);
fused_3x3_ssim = computeSSIMColor(ref_hdr, fused_3x3_img);
mkhdr_1x1_ssim = computeSSIMColor(ref_hdr, mkhdr_1x1_img);
mkhdr_3x3_ssim = computeSSIMColor(ref_hdr, mkhdr_3x3_img);

fprintf('\nSSIMs of CNN-based outputs\n');
disp(['Exposure fusion: 1x1 = ' num2str(fused_1x1_ssim) '; 3x3 = ' num2str(fused_3x3_ssim)]);
disp(['makeHDR function: 1x1 = ' num2str(mkhdr_1x1_ssim) '; 3x3 = ' num2str(mkhdr_3x3_ssim)]);

% CNN HDR generation
cnn_hdr_ssim = computeSSIMColor(ref_hdr, cnn_hdr_img);

fprintf('\nSSIMs of CNN-generated outputs\n');
disp(['CNN HDR tonemapped image = ' num2str(cnn_hdr_ssim)]);

%% HDR-VDP-2 

% Reference: http://hdrvdp.sourceforge.net/wiki/

addpath(fullfile(cd, '/hdrvdp-2.2.1'));

% input image
input_res = hdrvdp(mat2gray(input_img), mat2gray(ref_hdr), 'sRGB-display', 30);

imwrite(input_res.P_map, fullfile(diffmaps_dir, 'input_pmap.png'));

fprintf('\nHDR-VDP-2 results (Q) for reference images\n');
disp(['Input image: = ' num2str(input_res.Q)]);

% MATLAB-based approach
fused_res = hdrvdp(mat2gray(fused_img), mat2gray(ref_hdr), 'sRGB-display', 30);
mkhdr_res = hdrvdp(mat2gray(mkhdr_img), mat2gray(ref_hdr), 'sRGB-display', 30);

imwrite(fused_res.P_map, fullfile(diffmaps_dir, 'fused_pmap.png'));
imwrite(mkhdr_res.P_map, fullfile(diffmaps_dir, 'mkhdr_pmap.png'));

fprintf('\nHDR-VDP-2 results (Q) for MATLAB-based outputs\n');
disp(['Exposure fusion: = ' num2str(fused_res.Q)]);
disp(['makeHDR function: = ' num2str(mkhdr_res.Q)]);

% CNN-based approach
fused_1x1_res = hdrvdp(mat2gray(fused_1x1_img), mat2gray(ref_hdr), 'sRGB-display', 30);
fused_3x3_res = hdrvdp(mat2gray(fused_3x3_img), mat2gray(ref_hdr), 'sRGB-display', 30);
mkhdr_1x1_res = hdrvdp(mat2gray(mkhdr_1x1_img), mat2gray(ref_hdr), 'sRGB-display', 30);
mkhdr_3x3_res = hdrvdp(mat2gray(mkhdr_3x3_img), mat2gray(ref_hdr), 'sRGB-display', 30);

imwrite(fused_1x1_res.P_map, fullfile(cnn_exp_diffmaps_dir, 'fused_1x1_pmap.png'));
imwrite(fused_3x3_res.P_map, fullfile(cnn_exp_diffmaps_dir, 'fused_3x3_pmap.png'));
imwrite(mkhdr_1x1_res.P_map, fullfile(cnn_exp_diffmaps_dir, 'mkhdr_1x1_pmap.png'));
imwrite(mkhdr_3x3_res.P_map, fullfile(cnn_exp_diffmaps_dir, 'mkhdr_3x3_pmap.png'));

fprintf('\nHDR-VDP-2 results (Q) for CNN-based outputs\n');
disp(['Exposure fusion: 1x1 = ' num2str(fused_1x1_res.Q) '; 3x3 = ' num2str(fused_3x3_res.Q)]);
disp(['makeHDR function: 1x1 = ' num2str(mkhdr_1x1_res.Q) '; 3x3 = ' num2str(mkhdr_3x3_res.Q)]);

% CNN HDR generation
cnn_hdr_res = hdrvdp(mat2gray(cnn_hdr_img), mat2gray(ref_hdr), 'sRGB-display', 30);

imwrite(cnn_hdr_res.P_map, fullfile(cnn_diffmaps_dir, 'cnn_hdr_pmap.png'));

fprintf('\nHDR-VDP-2 results (Q) for CNN-generated outputs\n');
disp(['CNN HDR tonemapped image = ' num2str(cnn_hdr_res.Q)]);