function denoiseGuided(input_filename, exp_bracket_dir)

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

    % convert original image to YCbCr color space
    orig_img_YCbCr = rgb2ycbcr(orig_img);

    nhoodSize = 5;
    %smoothValue  = 0.001*diff(getrangefromclass(orig_imgB)).^2; % = 65
    smoothValue = 5;

    save_path = [exp_bracket_dir, '_dn'];
    if (exist(save_path, 'dir'))
        delete([save_path, '/*']);
    else
        mkdir(save_path);
    end

    for f=1:size(files, 1)
        fname = files(f).name;
        fprintf('Processing file: %s \n', [exp_bracket_dir, '/', fname]);

        exp_img = imread([exp_bracket_dir, '/', fname]);
        %figure, imshow(exp_img);

        exp_img_YCbCr = rgb2ycbcr(exp_img);

        exp_img_YCbCr(:,:,1) = imguidedfilter( ...
            exp_img_YCbCr(:,:,1), orig_img_YCbCr(:,:,1), ...
            'NeighborhoodSize', nhoodSize, ...
            'DegreeOfSmoothing', smoothValue);

        out_img_RGB = ycbcr2rgb(exp_img_YCbCr);
        %figure, imshow(out_img_RGB);

        out_fname = [exp_bracket_dir, '_dn/', fname];
        imwrite(out_img_RGB, out_fname);
    end
    
end