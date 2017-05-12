clear variables;

repo = '/home/vysarge/Documents/repos/6344-project/';
addpath(fullfile(repo,'matlab','exp_fusion'));

evaluating = '1x1';
os_slash = '/';
defaultTonemapping = 0;
custom = 1;

% hack to get this to work on linux; dir is behaving strangely with *
if (custom)
    input_files = ls(fullfile(repo, 'exposure_cnn', 'inputs', 'ChineseGarden.png'));
else
    input_files = ls(fullfile(repo, 'exposure_cnn','inputs','*','*.png'));
end
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
    if custom
        file_id = input_filename(slashes(end):end); % pull out the part of the file that is the same for other folders
    else
        file_id = input_filename(slashes(end-1):end); % pull out the part of the file that is the same for other folders
    end
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

