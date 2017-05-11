function histSeparate(input_filename, save_format, save_path)

    input_img = imread(fullfile(cd, 'data', input_filename));
    
    if size(input_img, 3) == 1
        img = cat(3, input_img, input_img, input_img);
    else
        img = input_img;
    end

    % before proceeding, assert that img has 3 channels
    assert(size(img, 3) == 3, 'Incorrect number of channels in img.')

    % convert to HSV
    img_HSV = rgb2hsv(img);
    img_H = img_HSV(:,:,1);
    img_S = img_HSV(:,:,2);
    img_V = img_HSV(:,:,3);

    %V = medfilt2(V, [5 5]);

    img_hist = imhist(img_V);

    hist_cdf = img_hist;
    for i=2:size(hist_cdf)
        hist_cdf(i) = hist_cdf(i) + hist_cdf(i-1);
    end

    num_bins = 2; % number of exposure brackets

    bin_edges = prod(size(img_V))*linspace(0, 1, num_bins+1);
    bins = discretize(hist_cdf, bin_edges);

    thr = zeros(1, num_bins+1);
    for i=1:num_bins
        thr((i+1):end) = thr((i+1):end) + sum(bins==i);
    end

    thr = thr / thr(end);

    Vsep = ones(size(img_V,1), size(img_V,2), num_bins);
    for bin=1:num_bins
        for c=1:size(img_V,2)
            for r=1:size(img_V,1)
                if (img_V(r,c) <= thr(bin))
                    Vsep(r,c,bin) = thr(bin);
                elseif (img_V(r,c) > thr(bin)) && (img_V(r,c) <= thr(bin+1))
                    Vsep(r,c,bin) = img_V(r,c);
                elseif (img_V(r,c) > thr(bin+1))
                    Vsep(r,c,bin) = thr(bin+1);
                end 
            end
        end
    end

    % clean or create directory for PNG output
    if (strcmp(save_format, 'PNG'))
        if (exist(save_path, 'dir'))
            delete([save_path, '/*.png']);
        else
            mkdir(save_path);
        end
    end

    % clean or create directory for JPG output
    if (strcmp(save_format, 'JPG'))
        if (exist(save_path, 'dir'))
            delete([save_path, '/*.jpg']);
        else
            mkdir(save_path);
        end
    end

    for bin=1:num_bins
        Vsep(:,:,bin) = (Vsep(:,:,bin) - min(min(Vsep(:,:,bin)))) / ...
            (max(max(Vsep(:,:,bin))) - min(min(Vsep(:,:,bin))));
        Vsep(:,:,bin) = adapthisteq(Vsep(:,:,bin));
        %Vsep(:,:,bin) = imsharpen(Vsep(:,:,bin));

        out_HSV = cat(3, img_H, img_S, Vsep(:,:,bin));
        out_RGB = hsv2rgb(out_HSV);

        if (strcmp(save_format, 'JPG'))
            out_jpg_filename = [save_path, '/exp', num2str(bin), '.jpg'];
            fprintf('Writing file: %s \n', out_jpg_filename);
            imwrite(out_RGB, out_jpg_filename);
        end

        if (strcmp(save_format, 'PNG'))
            out_png_filename = [save_path, '/exp', num2str(bin), '.png'];
            fprintf('Writing file: %s \n', out_png_filename);
            imwrite(out_RGB, out_png_filename);
        end

        figure('Name', out_png_filename); imshow(out_RGB);
    end
    
end