function fused_img_RGB = fuseExposures(input_filename, exp_bracket_dir, enhance)

    addpath(fullfile(cd,'/exp_fusion'));

    % obtain the exposure image stack
    img_stack = load_images(exp_bracket_dir);
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

    % perform exposure fusion
    fused_img_RGB = exposure_fusion(img_stack,[1 1 1]);
    fused_img_RGB = uint8(255*fused_img_RGB);
    
    if (enhance == 1)
        % adjust saturation and brightness of fused image
        fused_img_HSV = rgb2hsv(fused_img_RGB);
        fused_img_HSV(:,:,2) = fused_img_HSV(:,:,2) * 1.3;
        fused_img_HSV(:,:,3) = fused_img_HSV(:,:,3) * 0.85;
        fused_img_HSV(fused_img_HSV > 1) = 1;
        fused_img_RGB = hsv2rgb(fused_img_HSV);
    end

    figure('Name', 'Fused Image'), imshow(fused_img_RGB);

end
