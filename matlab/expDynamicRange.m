function out_RGB = expDynamicRange(img)
    
    % convert to YCbCr
    img_YCbCr = rgb2ycbcr(mat2gray(img));
    img_Y = img_YCbCr(:,:,1);
    img_Cb = img_YCbCr(:,:,2);
    img_Cr = img_YCbCr(:,:,3);
    
    % minimum and maximum luminance
    minLum = min(img_Y(:));
    maxLum = max(img_Y(:));
       
    % gamma -- non-linear scaling factor
    % > 1, mean luminance will be relatively darker
    % < 1, mean luminance will be relatively lighter
    % = 1, all pixels will be scaled equally
    gamma_exp = 1;
    
    % dynamic range expansion
    img_Y = ((img_Y-minLum)/(1.2*(maxLum-minLum))).^gamma_exp;
    
    % recombine channels
    out_YCbCr = cat(3, img_Y, img_Cb, img_Cr);
    out_RGB = uint8(255 * ycbcr2rgb(out_YCbCr));
    
end