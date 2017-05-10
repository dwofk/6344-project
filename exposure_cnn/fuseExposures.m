clear variables;

repo = '/home/vysarge/Documents/repos/6344-project/';
addpath(fullfile(repo,'matlab','exp_fusion'));

evaluating = '1x1';
os_slash = '/';
defaultTonemapping = 0;

% hack to get this to work on linux; dir is behaving strangely with *
input_files = ls(fullfile(repo, 'exposure_cnn','inputs','*','*.jpg'));
splits = find(input_files==sprintf('\n'));

save_path = fullfile(repo, 'exposure_cnn','matlab_outputs', 'expfusion');

b = 1;
e = 0;
split_index = 0;
while (e < max(size(input_files)))
    b = e + 1;
    split_index = split_index + 1;
    e = splits(split_index);
    input_filename = input_files(b:e-1);
    disp(input_filename);
    slashes = find(input_filename==os_slash);
    file_id = input_filename(slashes(end-1):end); % pull out the part of the file that is the same for other folders
    disp(file_id);
    
    im_orig = imread(input_filename);
    %im_orig = imresize(im_orig, [2000 3000]);
    
    low_filename = fullfile(repo, 'exposure_cnn',['outputs_minus_' evaluating],file_id);
    im_low = imread(low_filename);
    
    high_filename = fullfile(repo, 'exposure_cnn',['outputs_plus_' evaluating],file_id);
    im_high = imread(high_filename);
    
    img_stack = zeros([size(im_orig) 3], 'uint8');
    img_stack(:,:,:,1) = im_high;
    img_stack(:,:,:,2) = im_orig;
    img_stack(:,:,:,3) = im_low;
    
    fused_img = exposure_fusion(double(img_stack)/255,[1 1 1]);
    fused_img = uint8(255*fused_img);
    %figure, imshow(fused_img)

    if (defaultTonemapping)
        imwrite(fused_img, [save_path file_id]);
    else
        fused_img_HSV = rgb2hsv(fused_img);
        fused_img_HSV(:,:,2) = fused_img_HSV(:,:,2) * 1.3;
        fused_img_HSV(:,:,3) = fused_img_HSV(:,:,3) * 0.85;
        fused_img_HSV(fused_img_HSV > 1) = 1;
        fused_img_RGB = hsv2rgb(fused_img_HSV);

        %figure, imshow(fused_img_RGB);
        imwrite(fused_img_RGB, [save_path file_id]);
    end
    
end
% 
% input_filename = 'cabin.jpg';
% 
% exp_bracket_path = fullfile(cd, 'exp_bracket_png_dn');
% save_path = fullfile(cd, 'results');
% 
% img_stack = load_images(exp_bracket_path);
% orig_img = imread(fullfile(cd, 'data', input_filename));
% img_stack = cat(4, img_stack, mat2gray(orig_img));
% 
% figure('Name','Exposure Stack');
% % rows and columns in figure for image stack
% rc = [2 ceil(0.5*size(img_stack,4))];
% % display image stack; original image is last
% for i=1:size(img_stack,4)
%     s = subplot(rc(1),rc(2),i); imshow(img_stack(:,:,:,i));
%     if (i == size(img_stack,4)) % last image
%         title(s, 'Original');
%     else % exp1...n (decreasing exposure level)
%         title(s, ['exp', num2str(i)]);
%     end
% end
% 
% fused_img = exposure_fusion(img_stack,[1 1 1]);
% fused_img = uint8(255*fused_img);
% figure, imshow(fused_img)
% 
% %imshow(fused_img,'DisplayRange', ...
% %    [min(min(min(fused_img))) max(max(max(fused_img)))]);
% 
% fused_img_HSV = rgb2hsv(fused_img);
% fused_img_HSV(:,:,2) = fused_img_HSV(:,:,2) * 1.3;
% fused_img_HSV(:,:,3) = fused_img_HSV(:,:,3) * 0.85;
% fused_img_HSV(fused_img_HSV > 1) = 1;
% fused_img_RGB = hsv2rgb(fused_img_HSV);
% 
% figure, imshow(fused_img_RGB);
% imwrite(fused_img_RGB, [save_path, '/', 'fused_image.png']);
% 
