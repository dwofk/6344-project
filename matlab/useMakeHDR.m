function useMakeHDR(input_filename, exp_bracket_dir, save_path)
    assert(~exist(exp_bracket_dir, 'dir') == 0, ...
        'Exposure bracket directory does not exist.');

    files = dir([exp_bracket_dir, '/exp*']);
    num_files = length(files);

    if (num_files == 0)
        error('No exposure bracket files found.');
    end

    % read original image
    orig_img = imread(fullfile(cd, 'data', input_filename));
    %figure, imshow(orig_img);

    images = zeros([size(orig_img) num_files+1], 'uint8');
    orig_img_idx = ceil((1+num_files)/2);

    base_filenames = {};

    for i=1:(num_files+1)
        if (i == orig_img_idx)
            images(:,:,:,i) = orig_img;
            base_filenames{end+1} = fullfile(cd, 'data', input_filename);
        elseif (i < orig_img_idx)
            images(:,:,:,i) = imread([exp_bracket_dir, '/', files(i).name]);
            base_filenames{end+1} = [exp_bracket_dir, '/', files(i).name];
        elseif (i > orig_img_idx)
            images(:,:,:,i) = imread([exp_bracket_dir, '/', files(i-1).name]);
            base_filenames{end+1} = [exp_bracket_dir, '/', files(i-1).name];
        end
    end

    [row, col, num_ch, num_img] = size(images);
    assert(num_ch == 3, 'Number of channels in images is not 3.')

    img_luma = zeros([row col num_img], 'uint8');
    img_hist = zeros([256 1 num_img], 'double');
    hist_cdf = zeros([256 1 num_img], 'double');

    Yr = 0.2126 * ones([row col]);
    Yg = 0.7152 * ones([row col]);
    Yb = 0.0722 * ones([row col]);

    Y = cat(num_ch, Yr, Yg, Yb);

    figure('Name', 'Histograms of Images in Exposure Stack');
    rc = [2 ceil(0.5 * num_img)];
    
    for i=1:num_img
        img_luma(:,:,i) = uint8(sum(Y.*double(images(:,:,:,i)),3));
        img_hist(:,:,i) = imhist(img_luma(:,:,i));
        subplot(rc(1),rc(2),i); imhist(img_luma(:,:,i));
        hist_cdf(:,:,i) = cumsum(img_hist(:,:,i)) / numel(img_luma(:,:,i));
    end

    hist_map = zeros([256 1 num_img-1], 'uint8');
    map_diff = zeros([256 1 num_img-1], 'double');

    for i=1:(num_img-1)
        for idx=1:256
            [~, map_idx] = min(abs(hist_cdf(idx, 1, i) - hist_cdf(:, 1, i+1)));
            hist_map(idx,:,i) = map_idx;
            map_diff(idx,:,i) = double((idx - hist_map(idx,:,i)))^2;
        end
    end

    rms_map_diff = zeros([num_img-1 1]);
    for i=1:(num_img-1)
        rms_map_diff(i) = sqrt(mean(map_diff(:,:,i)));
    end

    k = 350;
    mid_EV = 0.35;

    diff_EV = (1/k) * rms_map_diff;
    rel_EV = mid_EV * ones([num_img 1]);

    for i=1:num_img
        if (i < orig_img_idx)
            rel_EV(i) = rel_EV(i) + sum(diff_EV(i:(orig_img_idx-1)));
        elseif (i > orig_img_idx)
            rel_EV(i) = rel_EV(i) - sum(diff_EV(orig_img_idx:(i-1)));
        end
    end

    rel_EV = (1/mid_EV) * rel_EV; % normalize so that orig_img has rel EV = 1

    hdr = makehdr(base_filenames, 'RelativeExposure', rel_EV);

    hdr_YCbCr = rgb2ycbcr(double(hdr));
    hdr_Y = hdr_YCbCr(:,:,1);

    Lwhite = round(max(hdr_Y(:)), 1-numel(num2str(round(max(hdr_Y(:))))));

    log_lum = log10(0.0001 + hdr_Y);
    log_avg_lum = exp((1/(prod(size(hdr_Y)))) * sum(log_lum(:)));

    scale = (hdr_Y .* (1 + (1/(Lwhite^2))*hdr_Y)) ./ (1+hdr_Y);
    hdr2rgb_scale = cat(3, scale, scale, scale);

    rgb = uint8(double(hdr) .* hdr2rgb_scale);
    figure('Name', 'HDR Image'), imshow(rgb, []);
    imwrite(rgb, [save_path, '/', 'hdr_image.png']);

end