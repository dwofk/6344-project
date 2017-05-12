warning('off','all')
warning

% original input
input_img = imread('data/ChineseGarden2.jpg');

% generated results
fused_img = imread('results/fused_image.png');
mkhdr_img = imread('results/hdr_image.png');

% reference tonemapped HDR image
ref_hdr = imread('data/ChineseGarden.jpg');

% exposure-based CNN results
exp_dir = fullfile('..', 'exposure_cnn', 'vysarge_results', 'ChineseGarden');
exp_outputs_dir = fullfile('..', 'exposure_cnn', 'vysarge_results', 'ChineseGarden', 'diff_maps');
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

%% Well-Exposedness (Mertens)

input_img_we = well_exposedness(mat2gray(input_img));
fused_img_we = well_exposedness(mat2gray(fused_img));
mkhdr_img_we = well_exposedness(mat2gray(mkhdr_img));
ref_hdr_we = well_exposedness(mat2gray(ref_hdr));

mean(input_img_we(:))
mean(fused_img_we(:))
mean(mkhdr_img_we(:))
mean(ref_hdr_we(:))

var(input_img_we(:))
var(fused_img_we(:))
var(mkhdr_img_we(:))
var(ref_hdr_we(:))

minus_1x1_we = well_exposedness(mat2gray(minus_1x1_img));
minus_1x1_we_str = ['(' num2str(mean(minus_1x1_we(:))) ', ' num2str(var(minus_1x1_we(:))) ')'];
minus_3x3_we = well_exposedness(mat2gray(minus_3x3_img));
minus_3x3_we_str = ['(' num2str(mean(minus_3x3_we(:))) ', ' num2str(var(minus_3x3_we(:))) ')'];
plus_1x1_we = well_exposedness(mat2gray(plus_1x1_img));
plus_1x1_we_str = ['(' num2str(mean(plus_1x1_we(:))) ', ' num2str(var(plus_1x1_we(:))) ')'];
plus_3x3_we = well_exposedness(mat2gray(plus_3x3_img));
plus_3x3_we_str = ['(' num2str(mean(plus_3x3_we(:))) ', ' num2str(var(plus_3x3_we(:))) ')'];

fused_1x1_we = well_exposedness(mat2gray(fused_1x1_img));
fused_1x1_we_str = ['(' num2str(mean(fused_1x1_we(:))) ', ' num2str(var(fused_1x1_we(:))) ')'];
fused_3x3_we = well_exposedness(mat2gray(fused_3x3_img));
fused_3x3_we_str = ['(' num2str(mean(fused_3x3_we(:))) ', ' num2str(var(fused_3x3_we(:))) ')'];
mkhdr_1x1_we = well_exposedness(mat2gray(mkhdr_1x1_img));
mkhdr_1x1_we_str = ['(' num2str(mean(mkhdr_1x1_we(:))) ', ' num2str(var(mkhdr_1x1_we(:))) ')'];
mkhdr_3x3_we = well_exposedness(mat2gray(mkhdr_3x3_img));
mkhdr_3x3_we_str = ['(' num2str(mean(mkhdr_3x3_we(:))) ', ' num2str(var(mkhdr_3x3_we(:))) ')'];

disp('CNN-estimated well-exposedness (mean, variance)');
disp(['Estimated under-exposed: 1x1 = ' minus_1x1_we_str '; 3x3 = ' minus_3x3_we_str]);
disp(['Estimated over-exposed: 1x1 = ' plus_1x1_we_str '; 3x3 = ' plus_3x3_we_str]);
disp(['Exposure fusion outputs: 1x1 = ' fused_1x1_we_str '; 3x3 = ' fused_3x3_we_str]);
disp(['makeHDR function outputs: 1x1 = ' mkhdr_1x1_we_str '; 3x3 = ' mkhdr_3x3_we_str]);


%% SSIM

input_ssim = computeSSIMColor(ref_hdr, input_img)
fused_ssim = computeSSIMColor(ref_hdr, fused_img)
mkhdr_ssim = computeSSIMColor(ref_hdr, mkhdr_img)

fused_1x1_ssim = computeSSIMColor(ref_hdr, fused_1x1_img);
fused_3x3_ssim = computeSSIMColor(ref_hdr, fused_3x3_img);
mkhdr_1x1_ssim = computeSSIMColor(ref_hdr, mkhdr_1x1_img);
mkhdr_3x3_ssim = computeSSIMColor(ref_hdr, mkhdr_3x3_img);
disp('SSIMs of fused CNN outputs');
disp(['Exposure fusion: 1x1 = ' num2str(fused_1x1_ssim) '; 3x3 = ' num2str(fused_3x3_ssim)]);
disp(['makeHDR function: 1x1 = ' num2str(mkhdr_1x1_ssim) '; 3x3 = ' num2str(mkhdr_3x3_ssim)]);


%% HDR-VDP-2 

% Reference: http://hdrvdp.sourceforge.net/wiki/

addpath(fullfile(cd, '/hdrvdp-2.2.1'));

input_res = hdrvdp(mat2gray(input_img), mat2gray(ref_hdr), 'sRGB-display', 30);
fused_res = hdrvdp(mat2gray(fused_img), mat2gray(ref_hdr), 'sRGB-display', 30);
mkhdr_res = hdrvdp(mat2gray(mkhdr_img), mat2gray(ref_hdr), 'sRGB-display', 30);

% CNN (exposure) output images
fused_1x1_res = hdrvdp(mat2gray(fused_1x1_img), mat2gray(ref_hdr), 'sRGB-display', 30);
fused_3x3_res = hdrvdp(mat2gray(fused_3x3_img), mat2gray(ref_hdr), 'sRGB-display', 30);
mkhdr_1x1_res = hdrvdp(mat2gray(mkhdr_1x1_img), mat2gray(ref_hdr), 'sRGB-display', 30);
mkhdr_3x3_res = hdrvdp(mat2gray(mkhdr_3x3_img), mat2gray(ref_hdr), 'sRGB-display', 30);

imwrite(fused_1x1_res.P_map, fullfile(exp_outputs_dir, 'fused_1x1_pmap.png'));
imwrite(fused_3x3_res.P_map, fullfile(exp_outputs_dir, 'fused_3x3_pmap.png'));
imwrite(mkhdr_1x1_res.P_map, fullfile(exp_outputs_dir, 'mkhdr_1x1_pmap.png'));
imwrite(mkhdr_3x3_res.P_map, fullfile(exp_outputs_dir, 'mkhdr_3x3_pmap.png'));

disp('HDR-VDP-2 results (Q) for fused CNN outputs');
disp(['Exposure fusion: 1x1 = ' num2str(fused_1x1_res.Q) '; 3x3 = ' num2str(fused_3x3_res.Q)]);
disp(['makeHDR function: 1x1 = ' num2str(mkhdr_1x1_res.Q) '; 3x3 = ' num2str(mkhdr_3x3_res.Q)]);

