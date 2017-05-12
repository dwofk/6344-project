% original input
input_img = imread('data/ChineseGarden2.jpg');

% generated results
fused_img = imread('results/fused_image.png');
mkhdr_img = imread('results/hdr_image.png');

% reference tonemapped HDR image
ref_hdr = imread('data/ChineseGarden.jpg');

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


%% SSIM

input_ssim = computeSSIMColor(ref_hdr, input_img)
fused_ssim = computeSSIMColor(ref_hdr, fused_img)
mkhdr_ssim = computeSSIMColor(ref_hdr, mkhdr_img)

%% HDR-VDP-2 

% Reference: http://hdrvdp.sourceforge.net/wiki/

addpath(fullfile(cd, '/hdrvdp-2.2.1'));

input_res = hdrvdp(mat2gray(input_img), mat2gray(ref_hdr), 'sRGB-display', 30);
fused_res = hdrvdp(mat2gray(fused_img), mat2gray(ref_hdr), 'sRGB-display', 30);
mkhdr_res = hdrvdp(mat2gray(mkhdr_img), mat2gray(ref_hdr), 'sRGB-display', 30);