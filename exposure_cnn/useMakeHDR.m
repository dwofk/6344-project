clear variables;

repo = '/home/vysarge/Documents/repos/6344-project/';
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

save_path = fullfile(repo, 'exposure_cnn','matlab_outputs', 'makehdr');

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
    
    images = zeros([size(im_orig) 3], 'uint8');
    images(:,:,:,1) = im_high;
    images(:,:,:,2) = im_orig;
    images(:,:,:,3) = im_low;
    
    base_filenames = {high_filename, input_filename, low_filename};
    orig_img_idx = 2;

    rel_EV = [4 1 0.25];
    hdr = makehdr(base_filenames, 'RelativeExposure', rel_EV);
    
    if (defaultTonemapping)
        rgb = tonemap(hdr);
    else % following code entirely from Diana
        hdr_YCbCr = rgb2ycbcr(double(hdr));
        hdr_Y = hdr_YCbCr(:,:,1);

        Lwhite = round(max(hdr_Y(:)), 1-numel(num2str(round(max(hdr_Y(:))))));

        log_lum = log10(0.0001 + hdr_Y);
        log_avg_lum = exp((1/(numel(hdr_Y))) * sum(log_lum(:)));

        scale = (hdr_Y .* (1 + (1/(Lwhite^2))*hdr_Y)) ./ (1+hdr_Y);
        hdr2rgb_scale = cat(3, scale, scale, scale);
        rgb = uint8(double(hdr) .* hdr2rgb_scale);
    end
    
    
    %figure, imshow(rgb);
    imwrite(rgb, [save_path, file_id]);
end
  