addpath(fullfile(cd,'/exp_fusion'));
input_filename = 'cabin.jpg';

exp_bracket_path = fullfile(cd, 'exp_bracket_png_dn');
save_path = fullfile(cd, 'results');

img_stack = load_images(exp_bracket_path);
orig_img = imread(fullfile(cd, 'data', input_filename));
img_stack = cat(4, img_stack, mat2gray(orig_img));

figure('Name','Exposure Stack');
% rows and columns in figure for image stack
rc = [2 ceil(0.5*size(img_stack,4))];
% display image stack; original image is last
for i=1:size(img_stack,4)
    s = subplot(rc(1),rc(2),i); imshow(img_stack(:,:,:,i));
    if (i == size(img_stack,4)) % last image
        title(s, 'Original');
    else % exp1...n (decreasing exposure level)
        title(s, ['exp', num2str(i)]);
    end
end

fused_img = exposure_fusion(img_stack,[1 1 1]);
fused_img = uint8(255*fused_img);
figure, imshow(fused_img)

%imshow(fused_img,'DisplayRange', ...
%    [min(min(min(fused_img))) max(max(max(fused_img)))]);

fused_img_HSV = rgb2hsv(fused_img);
fused_img_HSV(:,:,2) = fused_img_HSV(:,:,2) * 1.3;
fused_img_HSV(:,:,3) = fused_img_HSV(:,:,3) * 0.85;
fused_img_HSV(fused_img_HSV > 1) = 1;
fused_img_RGB = hsv2rgb(fused_img_HSV);

figure, imshow(fused_img_RGB);
imwrite(fused_img_RGB, [save_path, '/', 'fused_image.png']);

