function out_RGB = rmHighlights(img)

    figure('Name','Highlight Removal');
    
    subplot(3,3,1); imshow(uint8(img), []);
    title('Input Image');

    % convert to YCbCr
    img_YCbCr = rgb2ycbcr(img);
    img_Y = img_YCbCr(:,:,1);
    img_Cb = img_YCbCr(:,:,2);
    img_Cr = img_YCbCr(:,:,3);
    
    subplot(3,3,2); imshow(uint8(img_Y), []);
    title('Luma Component');
    
    %% Create Binary Highlight Map

    % estimate channel thresholds
    Y_thresh = graythresh(img_Y);
    Cb_thresh = graythresh(img_Cb);
    Cr_thresh = graythresh(img_Cr);

    % estimate highlight threshold -- luma threshold is experimentally
    % weighted 1.5x as much as each of the chrominance thresholds
    hl_thresh = (0.375)*Y_thresh + (0.3125)*(Cb_thresh+Cr_thresh);

    % generate highlight map
    hl_map = im2bw(mat2gray(img_Y), hl_thresh);
    
    subplot(3,3,3); imshow(uint8(hl_map), []);
    title('Highlight Map');

    %% Highlight Detection and Removal Using PCA

    X = reshape(img, size(img,1)*size(img,2), 3);
    [coeff, ~ , latent] = pca(X); % principal component coefficients

    img_PC = X*coeff;

    % separate principal components of image
    PC1 = reshape(img_PC(:,1), size(img,1), size(img,2));
    PC2 = reshape(img_PC(:,2), size(img,1), size(img,2));
    PC3 = reshape(img_PC(:,3), size(img,1), size(img,2));
    
    subplot(3,3,4); imshow(uint8(PC1), []);
    title('1st Principal Component');
    subplot(3,3,5); imshow(uint8(PC2), []);
    title('2nd Principal Component');
    subplot(3,3,6); imshow(uint8(PC3), []);
    title('3rd Principal Component');

    % calculate fidelity ratios
    FR = latent / (sum(latent(:))); 

    % if FR for 2nd eigenvalue < 0.02, remove PC2
    invcoeffT = inv(coeff).';
    if (FR(2) <= 0.02)
        invcoeffT(:,2) = 0;
    end

    % histogram equalization on PC1
    PC1_heq = 255 * histeq(mat2gray(PC1));
    img_PC(:,1) = reshape(PC1_heq, size(img,1)*size(img,2), 1);

    % reconstruct image
    I = invcoeffT * img_PC.';

    I1 = reshape(I(1,:), size(img,1), size(img,2));
    I2 = reshape(I(2,:), size(img,1), size(img,2));
    I3 = reshape(I(3,:), size(img,1), size(img,2));

    I = cat(3, I1, I2, I3);
    subplot(3,3,7); imshow(uint8(I), []);
    title('Reconstructed Image');

    %% Region Filling for Dehighlighted Areas

    % convert to YCbCr
    I_YCbCr = rgb2ycbcr(mat2gray(I));

    % perform region filling on chrominance channels
    I_YCbCr(:,:,2) = regionfill(I_YCbCr(:,:,2), hl_map);
    I_YCbCr(:,:,3) = regionfill(I_YCbCr(:,:,3), hl_map);

    I_RGB = uint8(255 * ycbcr2rgb(I_YCbCr));
    
    subplot(3,3,8); imshow(uint8(I_RGB), []);
    title('Region-Filled Image');
    
    % increase saturation
    out_HSV = rgb2hsv(I_RGB);
    out_HSV(:,:,2) = out_HSV(:,:,2)*1.5;
    out_RGB = 255 * hsv2rgb(out_HSV);
    
    subplot(3,3,9); imshow(uint8(out_RGB), []);
    title('Output Image');

end

